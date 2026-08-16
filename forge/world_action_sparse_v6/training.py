from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..action_teacher_v1.contract import ACTIONS
from ..world_action_spatial_v4.data import encode_spatial_episodes
from ..world_action_sparse_v5.contract import CHECKPOINT_FORMAT as V5_CHECKPOINT_FORMAT, ModelConfig, source_sha256 as v5_source_sha256
from ..world_action_sparse_v5.model import SparseActionDiT
from ..world_action_sparse_v5.training import TARGETED_INDICES, _contact, _decode, _masked_latent_mae, _masked_rgb_mae, counterfactual_control, latent_edit_mask
from ..world_frame_vae_refiner import RefinedWorldFrameVAERuntime
from .contract import CHECKPOINT_FORMAT, REPORT_FORMAT, TrainingConfig, canonical, config_dict, source_sha256
from .runtime import SparseWorldActionV6Runtime


def _state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0" + value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _concat(episodes, name):
    return np.concatenate([episode[name] for episode in episodes])


def _group(episodes):
    return {name: _concat(episodes, name) for name in ("current", "target", "control", "action", "state", "current_frame", "target_frame")}


def _predict(model, group, mean, std, device):
    current = torch.from_numpy(group["current"])
    target = torch.from_numpy(group["target"])
    predicted, wrong_action, wrong_control, gate_rows = [], [], [], []
    wrong_actions = (group["action"].astype(np.int64) + 7) % len(ACTIONS)
    wrong_controls = counterfactual_control(group["control"])
    for start in range(0, len(current), 32):
        stop = start + 32
        value = current[start:stop].to(device)
        normalized = (value - mean) / std
        action = torch.from_numpy(group["action"][start:stop].astype(np.int64)).to(device)
        control = torch.from_numpy(group["control"][start:stop]).to(device)
        state = torch.from_numpy(group["state"][start:stop]).to(device)
        time = torch.zeros(len(value), device=device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            edited, gate, _, _ = model.edit(normalized, time, action, control, state)
            wrong_a = model.edit(normalized, time, torch.from_numpy(wrong_actions[start:stop]).to(device), control, state)[0]
            wrong_c = model.edit(normalized, time, action, torch.from_numpy(wrong_controls[start:stop]).to(device), state)[0]
        predicted.append((edited * std + mean).float().cpu()); wrong_action.append((wrong_a * std + mean).float().cpu()); wrong_control.append((wrong_c * std + mean).float().cpu()); gate_rows.append(gate.float().cpu())
    return current, target, torch.cat(predicted), torch.cat(wrong_action), torch.cat(wrong_control), torch.cat(gate_rows)


def _latent_metrics(model, episodes, mean, std, device):
    group = _group(episodes)
    current, target, predicted, wrong_action, wrong_control, gate = _predict(model, group, mean, std, device)
    mask = torch.from_numpy(latent_edit_mask(group["current_frame"], group["target_frame"]).astype(np.float32))
    actions = group["action"].astype(np.int64)
    targeted = torch.from_numpy(np.isin(actions, TARGETED_INDICES))
    persistence = float(F.l1_loss(current, target)); model_mae = float(F.l1_loss(predicted, target)); changed_persistence = _masked_latent_mae(current, target, mask); changed_model = _masked_latent_mae(predicted, target, mask)
    wrong_action_mae = float(F.l1_loss(wrong_action, target)); targeted_model = float(F.l1_loss(predicted[targeted], target[targeted])); targeted_wrong_control = float(F.l1_loss(wrong_control[targeted], target[targeted]))
    gate_binary = gate > 0.5; mask_binary = mask > 0.5
    gate_iou = float((gate_binary & mask_binary).sum() / (gate_binary | mask_binary).sum().clamp_min(1))
    metrics = {"pairs": len(current), "model_latent_mae": model_mae, "persistence_latent_mae": persistence, "changed_model_latent_mae": changed_model, "changed_persistence_latent_mae": changed_persistence, "wrong_action_latent_mae": wrong_action_mae, "targeted_model_latent_mae": targeted_model, "targeted_wrong_control_latent_mae": targeted_wrong_control, "latent_improvement": 1 - model_mae / persistence, "changed_latent_improvement": 1 - changed_model / changed_persistence, "correct_action_advantage": wrong_action_mae - model_mae, "targeted_control_advantage": targeted_wrong_control - targeted_model, "edit_gate_iou": gate_iou, "edit_gate_mean": float(gate.mean()), "target_edit_mask_mean": float(mask.mean())}
    return metrics, (group, current, target, predicted, wrong_action, wrong_control, gate, mask)


def _selection(metrics):
    score = 0.65 * metrics["model_latent_mae"] / metrics["persistence_latent_mae"] + 0.35 * metrics["changed_model_latent_mae"] / metrics["changed_persistence_latent_mae"]
    score += max(0.0, -metrics["correct_action_advantage"]) * 4 + max(0.0, -metrics["targeted_control_advantage"]) * 4
    return float(score)


def train(output: Path, train_paths, validation_paths, test_paths, vae_checkpoint: Path, refiner_checkpoint: Path, warm_start: Path, *, training=TrainingConfig()):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(training.seed); np.random.seed(training.seed & 0xFFFFFFFF); random.seed(training.seed); torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refined = RefinedWorldFrameVAERuntime.from_checkpoints(vae_checkpoint, refiner_checkpoint, device=str(device)); refined.model.requires_grad_(False); refined.refiner.requires_grad_(False)
    paths = tuple(map(Path, train_paths)) + tuple(map(Path, validation_paths)) + tuple(map(Path, test_paths))
    encoded, sources, corpus_sha = encode_spatial_episodes(paths, refined)
    train_count = len(train_paths); validation_count = len(validation_paths)
    train_episodes = encoded[:train_count]; validation_episodes = encoded[train_count:train_count + validation_count]; test_episodes = encoded[train_count + validation_count:]
    if not train_episodes or not validation_episodes or not test_episodes:
        raise ValueError("v6 requires nonempty whole-world train, validation, and test splits")
    warm = torch.load(Path(warm_start), map_location="cpu", weights_only=True)
    if warm.get("format") != V5_CHECKPOINT_FORMAT or warm.get("source_sha256") != v5_source_sha256():
        raise ValueError("v6 warm-start checkpoint provenance drifted")
    config = ModelConfig(**warm["model_config"])
    model = SparseActionDiT(config); model.load_state_dict(warm["ema_state"]); model.to(device)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=2e-3, fused=device.type == "cuda")
    train_group = _group(train_episodes)
    current_raw = train_group["current"]; target_raw = train_group["target"]
    mean = np.mean(np.concatenate((current_raw, target_raw)), axis=(0, 2, 3)); std = np.std(np.concatenate((current_raw, target_raw)), axis=(0, 2, 3)) + 1e-4
    current = (current_raw - mean[None, :, None, None]) / std[None, :, None, None]; target = (target_raw - mean[None, :, None, None]) / std[None, :, None, None]
    edit_mask = latent_edit_mask(train_group["current_frame"], train_group["target_frame"])
    mean_tensor = torch.from_numpy(mean).to(device).view(1, -1, 1, 1); std_tensor = torch.from_numpy(std).to(device).view(1, -1, 1, 1)
    rng = np.random.default_rng(training.seed); history = []; validation_history = []; best_state = None; best_score = float("inf"); best_step = 0
    validation_model = SparseActionDiT(config).to(device).eval()
    for step in range(1, training.steps + 1):
        indices = rng.integers(0, len(current), training.batch_size)
        x0 = torch.from_numpy(current[indices]).to(device); x1 = torch.from_numpy(target[indices]).to(device); desired = x1 - x0; noisy = x0 + torch.randn_like(x0) * training.input_noise
        act = torch.from_numpy(train_group["action"][indices].astype(np.int64)).to(device); ctl = torch.from_numpy(train_group["control"][indices]).to(device); st = torch.from_numpy(train_group["state"][indices]).to(device); mask = torch.from_numpy(edit_mask[indices].astype(np.float32)).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            edited, gate, delta, gate_logits = model.edit(noisy, torch.zeros(len(x0), device=device), act, ctl, st)
            residual = (F.smooth_l1_loss(edited, x1, reduction="none") * (1 + training.changed_weight * mask)).mean()
            delta_error = F.smooth_l1_loss(delta, desired, reduction="none").mean(1, keepdim=True); delta_loss = (delta_error * mask).sum() / mask.sum().clamp_min(1)
            predicted_magnitude = (edited - noisy).abs().mean((1, 2, 3)); target_magnitude = desired.abs().mean((1, 2, 3)); magnitude = F.smooth_l1_loss(torch.log1p(predicted_magnitude * 20), torch.log1p(target_magnitude * 20))
            gate_bce = F.binary_cross_entropy_with_logits(gate_logits.float(), mask, pos_weight=torch.tensor(training.gate_positive_weight, device=device)); intersection = (gate * mask).sum((1, 2, 3)); dice = 1 - ((2 * intersection + 1) / (gate.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)).mean(); gate_loss = gate_bce + dice; leakage = (gate * (1 - mask)).mean()
            pixel_count = min(training.pixel_batch, len(indices)); decoded = refined.refiner(refined.model.decode(edited[:pixel_count] * std_tensor + mean_tensor)); target_pixel = torch.from_numpy(train_group["target_frame"][indices[:pixel_count]]).to(device).permute(0, 3, 1, 2).float().div_(255); pixel_delta = torch.from_numpy(np.abs(train_group["target_frame"][indices[:pixel_count]].astype(np.int16) - train_group["current_frame"][indices[:pixel_count]].astype(np.int16)).mean(3).astype(np.float32) / 255).to(device)[:, None]; pixel_weight = 1 + training.pixel_changed_weight * F.max_pool2d((pixel_delta > 0.025).float(), 9, 1, 4); pixel = (F.l1_loss(decoded, target_pixel, reduction="none") * pixel_weight).mean()
            contrast_count = min(training.contrastive_batch, len(indices)); wrong_actions = (act[:contrast_count] + 7) % len(ACTIONS); wrong_controls = torch.from_numpy(counterfactual_control(train_group["control"][indices[:contrast_count]])).to(device); wrong_a = model.edit(noisy[:contrast_count], torch.zeros(contrast_count, device=device), wrong_actions, ctl[:contrast_count], st[:contrast_count])[0]; wrong_c = model.edit(noisy[:contrast_count], torch.zeros(contrast_count, device=device), act[:contrast_count], wrong_controls, st[:contrast_count])[0]; correct_error = (edited[:contrast_count] - x1[:contrast_count]).abs().mean((1, 2, 3)); wrong_a_error = (wrong_a - x1[:contrast_count]).abs().mean((1, 2, 3)); targeted = torch.isin(act[:contrast_count], torch.as_tensor(TARGETED_INDICES, device=device)); wrong_c_error = (wrong_c - x1[:contrast_count]).abs().mean((1, 2, 3)); contrastive = F.relu(training.contrastive_margin + correct_error - wrong_a_error).mean(); contrastive = contrastive + (F.relu(training.contrastive_margin + correct_error[targeted] - wrong_c_error[targeted]).mean() if bool(targeted.any()) else correct_error.new_zeros(()))
            loss = residual + training.delta_weight * delta_loss + training.magnitude_weight * magnitude + training.gate_weight * gate_loss + training.leakage_weight * leakage + training.pixel_weight * pixel + training.contrastive_weight * contrastive
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items(): ema[name].lerp_(value.detach(), 1 - training.ema_decay)
        if step == 1 or step % 250 == 0 or step == training.steps:
            row = {"step": step, "loss": round(float(loss), 6), "residual": round(float(residual), 6), "delta": round(float(delta_loss), 6), "magnitude": round(float(magnitude), 6), "gate": round(float(gate_loss), 6), "gate_mean": round(float(gate.mean()), 6), "leakage": round(float(leakage), 6), "pixel": round(float(pixel), 6), "contrastive": round(float(contrastive), 6)}; history.append(row); print(json.dumps(row), flush=True)
        if step % training.validate_every == 0 or step == training.steps:
            validation_model.load_state_dict(ema); validation_model.eval(); metrics, _ = _latent_metrics(validation_model, validation_episodes, mean_tensor, std_tensor, device); score = _selection(metrics); validation_history.append({"step": step, "selection": score, **metrics}); print(json.dumps({"validation_step": step, "selection": round(score, 6), "latent_improvement": round(metrics["latent_improvement"], 6), "changed_improvement": round(metrics["changed_latent_improvement"], 6), "action_advantage": round(metrics["correct_action_advantage"], 6), "control_advantage": round(metrics["targeted_control_advantage"], 6)}, sort_keys=True), flush=True)
            if score < best_score:
                best_score = score; best_step = step; best_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
            model.train()
    if best_state is None: raise RuntimeError("v6 did not select a checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    validation_metrics, _ = _latent_metrics(model, validation_episodes, mean_tensor, std_tensor, device); test_metrics, test_payload = _latent_metrics(model, test_episodes, mean_tensor, std_tensor, device)
    group, test_current, test_target, test_predicted, test_wrong_action, test_wrong_control, test_gate, test_mask = test_payload
    predicted_rgb = _decode(refined, test_predicted, device); wrong_action_rgb = _decode(refined, test_wrong_action, device); wrong_control_rgb = _decode(refined, test_wrong_control, device); refined_current_rgb = _decode(refined, test_current, device); target_rgb = torch.from_numpy(group["target_frame"]).permute(0, 3, 1, 2).float().div_(255); raw_current_rgb = torch.from_numpy(group["current_frame"]).permute(0, 3, 1, 2).float().div_(255); changed = F.max_pool2d(((target_rgb - raw_current_rgb).abs().mean(1, keepdim=True) > 0.025).float(), 9, 1, 4).squeeze(1) > 0
    test_metrics.update({"model_rgb_mae": float(F.l1_loss(predicted_rgb, target_rgb)), "raw_persistence_rgb_mae": float(F.l1_loss(raw_current_rgb, target_rgb)), "refined_persistence_rgb_mae": float(F.l1_loss(refined_current_rgb, target_rgb)), "changed_model_rgb_mae": _masked_rgb_mae(predicted_rgb, target_rgb, changed), "changed_refined_persistence_rgb_mae": _masked_rgb_mae(refined_current_rgb, target_rgb, changed)})
    gates = {"latent_beats_persistence": test_metrics["latent_improvement"] > 0, "changed_latent_beats_persistence": test_metrics["changed_latent_improvement"] > 0, "correct_beats_wrong_action": test_metrics["correct_action_advantage"] > 0, "targeted_control_beats_wrong_control": test_metrics["targeted_control_advantage"] > 0, "refined_rgb_beats_refined_persistence": test_metrics["model_rgb_mae"] < test_metrics["refined_persistence_rgb_mae"], "changed_rgb_beats_refined_persistence": test_metrics["changed_model_rgb_mae"] < test_metrics["changed_refined_persistence_rgb_mae"], "edit_gate_iou_over_0_35": test_metrics["edit_gate_iou"] > 0.35}; gates["all_passed"] = all(gates.values())
    split_sources = {"train": list(sources[:train_count]), "validation": list(sources[train_count:train_count + validation_count]), "test": list(sources[train_count + validation_count:])}
    report = {"format": REPORT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "warm_start": {"path": str(warm_start), "ema_sha256": warm["ema_sha256"], "source_sha256": warm["source_sha256"]}, "decoder_source_sha256": refined.report["source_sha256"], "parameters": sum(parameter.numel() for parameter in model.parameters()), "device": str(device), "steps": training.steps, "best_step": best_step, "best_validation_selection": best_score, "split_worlds": {name: len(value) for name, value in split_sources.items()}, "split_pairs": {"train": len(current), "validation": validation_metrics["pairs"], "test": test_metrics["pairs"]}, "sources": split_sources, "validation": validation_metrics, "test": test_metrics, "gates": gates, "history": history, "validation_history": validation_history}
    recovery = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "model_config": config_dict(config), "training_config": config_dict(training), "latent_mean": mean.tolist(), "latent_std": std.tolist(), "ema_state": best_state, "ema_sha256": _state_hash(best_state), "history": history, "validation_history": validation_history, "report": report, "status": "evaluated"}
    torch.save(recovery, output / "checkpoint.pt"); torch.save({**recovery, "ema_state": {name: value.to(torch.bfloat16) if value.is_floating_point() else value for name, value in best_state.items()}, "runtime_precision": "bfloat16"}, output / "runtime.pt"); (output / "report.json").write_bytes(canonical(report))
    predicted_frame = np.clip(predicted_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8); wrong_action_frame = np.clip(wrong_action_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8); wrong_control_frame = np.clip(wrong_control_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8); _contact(output / "test_contact_sheet.png", group["current_frame"], group["target_frame"], predicted_frame, wrong_action_frame, wrong_control_frame, test_gate[:, 0].numpy(), group["action"].astype(np.int64))
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--train-episodes", type=Path, nargs="+", required=True); parser.add_argument("--validation-episodes", type=Path, nargs="+", required=True); parser.add_argument("--test-episodes", type=Path, nargs="+", required=True); parser.add_argument("--vae", type=Path, required=True); parser.add_argument("--refiner", type=Path, required=True); parser.add_argument("--warm-start", type=Path, required=True); parser.add_argument("--steps", type=int, default=12000); parser.add_argument("--batch-size", type=int, default=16); args = parser.parse_args(); print(json.dumps(train(args.output, args.train_episodes, args.validation_episodes, args.test_episodes, args.vae, args.refiner, args.warm_start, training=TrainingConfig(steps=args.steps, batch_size=args.batch_size)), indent=2))


if __name__ == "__main__": main()
