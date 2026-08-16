from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from .contract import CHECKPOINT_FORMAT, REPORT_FORMAT, STATE_CHANNELS, ModelConfig, TrainingConfig, canonical, config_dict, source_sha256
from .corpus import ARRAYS, validate_corpus
from .model import NeuralMacroPatchDynamics


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0" + np.asarray(value.shape, dtype="<i8").tobytes() + value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load(root: Path):
    validate_corpus(root)
    manifest = json.loads((Path(root) / "manifest.json").read_text("utf-8"))
    groups = []
    for record in manifest["shards"]:
        with np.load(Path(root) / record["artifact"]["path"], allow_pickle=False) as archive:
            groups.append((record["index"], {name: archive[name] for name in ARRAYS}))
    return groups, manifest


def _concat(groups):
    return {name: np.concatenate([arrays[name] for _, arrays in groups]) for name in ARRAYS}


def _masked_mae(predicted, target, mask):
    selected = mask.expand_as(predicted) if mask.shape != predicted.shape else mask
    return float((predicted - target).abs()[selected].mean()) if bool(selected.any()) else 0.0


@torch.inference_mode()
def _predict(model, arrays, device, *, batch_size=32):
    outputs = []; global_outputs = []; gates = []; global_gates = []
    for start in range(0, len(arrays["current"]), batch_size):
        stop = start + batch_size
        tensors = [torch.from_numpy(arrays[name][start:stop].astype(np.float32)).to(device) for name in ("current", "previous", "global_state", "previous_global")]
        predicted, predicted_global, gate, _, global_gate, _ = model(*tensors)
        outputs.append(predicted.float().cpu()); global_outputs.append(predicted_global.float().cpu()); gates.append(gate.float().cpu()); global_gates.append(global_gate.float().cpu())
    return torch.cat(outputs), torch.cat(global_outputs), torch.cat(gates), torch.cat(global_gates)


@torch.inference_mode()
def calibrate_thresholds(model, arrays, device) -> dict[str, list[float]]:
    predicted, predicted_global, gate, global_gate = _predict(model, arrays, device)
    current = torch.from_numpy(arrays["current"].astype(np.float32)); target = torch.from_numpy(arrays["target"].astype(np.float32))
    global_state = torch.from_numpy(arrays["global_state"]); target_global = torch.from_numpy(arrays["target_global"])
    candidates = torch.cat((torch.linspace(0, 1, 101), torch.tensor([1.01])))

    def choose(output, present, expected, scores):
        chosen = []
        for channel in range(present.shape[1]):
            losses = []
            for threshold in candidates:
                value = torch.where(scores[:, channel] >= threshold, output[:, channel], present[:, channel])
                losses.append(float(F.l1_loss(value, expected[:, channel])))
            chosen.append(float(candidates[int(np.argmin(losses))]))
        return chosen

    return {
        "spatial": choose(predicted, current, target, gate),
        "global": choose(predicted_global, global_state, target_global, global_gate),
    }


@torch.inference_mode()
def evaluate(model, arrays, device, *, thresholds=None, batch_size=32):
    predicted, predicted_global, gate, global_gate = _predict(model, arrays, device, batch_size=batch_size)
    current = torch.from_numpy(arrays["current"].astype(np.float32)); target = torch.from_numpy(arrays["target"].astype(np.float32))
    global_state = torch.from_numpy(arrays["global_state"]); target_global = torch.from_numpy(arrays["target_global"])
    if thresholds is not None:
        spatial_thresholds = torch.tensor(thresholds["spatial"]).view(1, -1, 1, 1)
        global_thresholds = torch.tensor(thresholds["global"]).view(1, -1)
        predicted = torch.where(gate >= spatial_thresholds, predicted, current)
        predicted_global = torch.where(global_gate >= global_thresholds, predicted_global, global_state)
    changed = (target - current).abs() > 1e-3; global_changed = (target_global - global_state).abs() > 1e-3
    spatial_mae = float(F.l1_loss(predicted, target)); spatial_persistence = float(F.l1_loss(current, target))
    global_mae = float(F.l1_loss(predicted_global, target_global)); global_persistence = float(F.l1_loss(global_state, target_global))
    per_channel = {}
    for index, name in enumerate(STATE_CHANNELS):
        mae = float(F.l1_loss(predicted[:, index], target[:, index])); persistence = float(F.l1_loss(current[:, index], target[:, index]))
        per_channel[name] = {"mae": mae, "persistence_mae": persistence, "improvement": 1 - mae / max(persistence, 1e-8), "change_fraction": float(changed[:, index].float().mean())}
    return {
        "pairs": len(current), "spatial_mae": spatial_mae, "spatial_persistence_mae": spatial_persistence,
        "spatial_improvement": 1 - spatial_mae / max(spatial_persistence, 1e-8),
        "changed_spatial_mae": _masked_mae(predicted, target, changed), "changed_spatial_persistence_mae": _masked_mae(current, target, changed),
        "global_mae": global_mae, "global_persistence_mae": global_persistence, "global_improvement": 1 - global_mae / max(global_persistence, 1e-8),
        "changed_global_mae": _masked_mae(predicted_global, target_global, global_changed), "changed_global_persistence_mae": _masked_mae(global_state, target_global, global_changed),
        "gate_mean": float(gate.mean()), "per_channel": per_channel,
    }


@torch.inference_mode()
def rollout_error(model, groups, device, *, thresholds, horizon=8):
    errors = []; persistence_errors = []; global_errors = []; global_persistence_errors = []
    spatial_thresholds = torch.tensor(thresholds["spatial"], device=device).view(1, -1, 1, 1)
    global_thresholds = torch.tensor(thresholds["global"], device=device).view(1, -1)
    for _, arrays in groups:
        if len(arrays["current"]) < horizon: continue
        previous = torch.from_numpy(arrays["previous"][:1].astype(np.float32)).to(device)
        current = torch.from_numpy(arrays["current"][:1].astype(np.float32)).to(device)
        previous_global = torch.from_numpy(arrays["previous_global"][:1]).to(device)
        global_state = torch.from_numpy(arrays["global_state"][:1]).to(device)
        anchor = current.clone()
        global_anchor = global_state.clone()
        for step in range(horizon):
            predicted, predicted_global, gate, _, global_gate, _ = model(current, previous, global_state, previous_global)
            predicted = torch.where(gate >= spatial_thresholds, predicted, current)
            predicted_global = torch.where(global_gate >= global_thresholds, predicted_global, global_state)
            target = torch.from_numpy(arrays["target"][step:step + 1].astype(np.float32)).to(device)
            target_global = torch.from_numpy(arrays["target_global"][step:step + 1]).to(device)
            errors.append(float(F.l1_loss(predicted, target))); persistence_errors.append(float(F.l1_loss(anchor, target)))
            global_errors.append(float(F.l1_loss(predicted_global, target_global))); global_persistence_errors.append(float(F.l1_loss(global_anchor, target_global)))
            previous, current = current, predicted
            previous_global, global_state = global_state, predicted_global
    mae = float(np.mean(errors)); persistence = float(np.mean(persistence_errors))
    global_mae = float(np.mean(global_errors)); global_persistence = float(np.mean(global_persistence_errors))
    return {
        "horizon": horizon, "mae": mae, "frozen_persistence_mae": persistence, "improvement": 1 - mae / max(persistence, 1e-8),
        "global_mae": global_mae, "global_frozen_persistence_mae": global_persistence,
        "global_improvement": 1 - global_mae / max(global_persistence, 1e-8),
    }


def _save(path: Path, payload: dict):
    stage = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    torch.save(payload, stage); os.replace(stage, path)


def train(corpus: Path, output: Path, *, model_config=ModelConfig(), training=TrainingConfig(), device="cuda"):
    target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if target.type == "cuda": torch.cuda.set_per_process_memory_fraction(.85, torch.cuda.current_device())
    torch.set_num_threads(min(12, max(1, os.cpu_count() or 1))); torch.manual_seed(training.seed); np.random.seed(training.seed & 0xffffffff)
    groups, manifest = _load(corpus)
    train_groups = [row for row in groups if row[0] % 8 < 6]; validation_groups = [row for row in groups if row[0] % 8 == 6]; test_groups = [row for row in groups if row[0] % 8 == 7]
    if not train_groups or not validation_groups or not test_groups: raise ValueError("macro corpus split lacks worlds")
    train_arrays = _concat(train_groups); validation_arrays = _concat(validation_groups); test_arrays = _concat(test_groups)
    spatial_change = np.abs(train_arrays["target"].astype(np.float32) - train_arrays["current"].astype(np.float32)) > 1e-3
    global_change = np.abs(train_arrays["target_global"] - train_arrays["global_state"]) > 1e-3
    spatial_frequency = spatial_change.mean((0, 2, 3), dtype=np.float64); global_frequency = global_change.mean(0, dtype=np.float64)
    spatial_pos = torch.from_numpy(np.clip((1 - spatial_frequency) / np.maximum(spatial_frequency, 1e-8), 1, 128).astype(np.float32)).to(target).view(1, -1, 1, 1)
    global_pos = torch.from_numpy(np.clip((1 - global_frequency) / np.maximum(global_frequency, 1e-8), 1, 128).astype(np.float32)).to(target).view(1, -1)
    model = NeuralMacroPatchDynamics(model_config).to(target); ema = copy.deepcopy(model).eval(); optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=2e-3, fused=target.type == "cuda")
    rng = np.random.default_rng(training.seed); history = []; validation_history = []; best_score = float("inf"); best_state = None; best_step = 0
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for step in range(1, training.steps + 1):
        ids = rng.integers(0, len(train_arrays["current"]), training.batch_size)
        current, previous, global_state, previous_global, target_state, target_global = [torch.from_numpy(train_arrays[name][ids].astype(np.float32)).to(target) for name in ("current", "previous", "global_state", "previous_global", "target", "target_global")]
        changed = (target_state - current).abs() > 1e-3; changed_global = (target_global - global_state).abs() > 1e-3
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
            predicted, predicted_global, gate, gate_logits, global_gate, global_gate_logits = model(current, previous, global_state, previous_global)
            spatial_error = F.smooth_l1_loss(predicted, target_state, reduction="none"); spatial_loss = (spatial_error * (1 + training.changed_weight * changed)).mean()
            global_error = F.smooth_l1_loss(predicted_global, target_global, reduction="none"); global_loss = (global_error * (1 + training.changed_weight * changed_global)).mean()
            gate_bce = F.binary_cross_entropy_with_logits(gate_logits.float(), changed.float(), pos_weight=spatial_pos)
            gate_dice = 1 - ((2 * (gate * changed).sum((2, 3)) + 1) / (gate.sum((2, 3)) + changed.sum((2, 3)) + 1)).mean()
            global_gate_bce = F.binary_cross_entropy_with_logits(global_gate_logits.float(), changed_global.float(), pos_weight=global_pos)
            loss = spatial_loss + training.global_weight * global_loss + training.gate_weight * (gate_bce + gate_dice + global_gate_bce)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad():
            for ema_value, value in zip(ema.parameters(), model.parameters()): ema_value.lerp_(value, 1 - training.ema_decay)
            for ema_value, value in zip(ema.buffers(), model.buffers()): ema_value.copy_(value)
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss), 7), "spatial": round(float(spatial_loss), 7), "global": round(float(global_loss), 7), "gate": round(float(gate_bce + gate_dice), 7)}; history.append(row); print(json.dumps(row), flush=True)
        if step % training.validation_every == 0 or step == training.steps:
            metrics = evaluate(ema, validation_arrays, target); score = (metrics["spatial_mae"] / max(metrics["spatial_persistence_mae"], 1e-8) + metrics["global_mae"] / max(metrics["global_persistence_mae"], 1e-8)) * .5
            validation_history.append({"step": step, "selection": score, **metrics}); print(json.dumps({"validation_step": step, "selection": score, "spatial_improvement": metrics["spatial_improvement"], "global_improvement": metrics["global_improvement"]}), flush=True)
            if score < best_score: best_score, best_step, best_state = score, step, {name: value.detach().cpu().to(torch.bfloat16) if value.is_floating_point() else value.detach().cpu() for name, value in ema.state_dict().items()}
            payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": manifest["manifest_sha256"], "model_config": config_dict(model_config), "training_config": config_dict(training), "step": step, "ema_state": best_state, "ema_sha256": _state_hash(best_state), "history": history, "validation_history": validation_history}
            _save(output / "latest.pt", payload)
    ema.load_state_dict(best_state); ema.to(target).eval()
    thresholds = calibrate_thresholds(ema, validation_arrays, target)
    raw_validation = evaluate(ema, validation_arrays, target); raw_test = evaluate(ema, test_arrays, target)
    validation = evaluate(ema, validation_arrays, target, thresholds=thresholds); test = evaluate(ema, test_arrays, target, thresholds=thresholds)
    rollout = rollout_error(ema, test_groups, target, thresholds=thresholds)
    gates = {
        "spatial_beats_persistence": test["spatial_improvement"] > 0,
        "changed_spatial_beats_persistence": test["changed_spatial_mae"] < test["changed_spatial_persistence_mae"],
        "global_beats_persistence": test["global_improvement"] > 0,
        "changed_global_beats_persistence": test["changed_global_mae"] < test["changed_global_persistence_mae"],
        "rollout_beats_frozen_persistence": rollout["improvement"] > 0,
        "global_rollout_beats_frozen_persistence": rollout["global_improvement"] > 0,
    }; gates["all_passed"] = all(gates.values())
    report = {"format": REPORT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": manifest["manifest_sha256"], "parameters": sum(value.numel() for value in model.parameters()), "device": str(target), "best_step": best_step, "best_selection": best_score, "splits": {"train_worlds": [x for x, _ in train_groups], "validation_worlds": [x for x, _ in validation_groups], "test_worlds": [x for x, _ in test_groups]}, "gate_thresholds": thresholds, "raw_validation": raw_validation, "raw_test": raw_test, "validation": validation, "test": test, "rollout": rollout, "gates": gates, "history": history, "validation_history": validation_history}
    final = {"format": CHECKPOINT_FORMAT, "status": "evaluated", "source_sha256": source_sha256(), "corpus_sha256": manifest["manifest_sha256"], "model_config": config_dict(model_config), "training_config": config_dict(training), "ema_state": best_state, "ema_sha256": _state_hash(best_state), "gate_thresholds": thresholds, "report": report}
    _save(output / "runtime.pt", final); (output / "report.json").write_bytes(canonical(report)); return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--corpus", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--steps", type=int, default=2400); parser.add_argument("--batch-size", type=int, default=24); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    print(json.dumps(train(args.corpus, args.output, training=TrainingConfig(steps=args.steps, batch_size=args.batch_size), device=args.device), indent=2))


if __name__ == "__main__": main()
