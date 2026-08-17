from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..maps.io import file_sha256
from ..recurrent_action_dit_v2.model import RecurrentActionDiT
from ..safety import require_disk_floor
from ..world_latent_dit.contract import ModelConfig as RecurrentConfig
from .contract import CHECKPOINT_FORMAT, FORMAT, RECURRENT, RECURRENT_SHA256, WORLD_STATE, WORLD_STATE_SHA256, ContextModelConfig, ContextTrainingConfig, canonical, source_sha256
from .data import build_aligned_context
from .model import build_model


def _downstream(model, episode, recurrent, recurrent_payload, device):
    context = torch.from_numpy(episode["context"]).to(device); true_state = torch.from_numpy(episode["state"]).to(device); predicted_state = model(context); mean = torch.tensor(recurrent_payload["normalization"]["mean"], device=device)[None, :, None, None]; std = torch.tensor(recurrent_payload["normalization"]["std"], device=device)[None, :, None, None]; current = (torch.from_numpy(episode["current"]).to(device) - mean) / std; previous = (torch.from_numpy(episode["previous"]).to(device) - mean) / std; target = (torch.from_numpy(episode["target"]).to(device) - mean) / std; action = torch.from_numpy(episode["action"]).long().to(device); control = torch.from_numpy(episode["control"]).to(device); actor = torch.from_numpy(episode["actor_state"]).to(device); desired = target - current
    errors = {}
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for name, state in (("true", true_state), ("context", predicted_state), ("zero", torch.zeros_like(true_state))): errors[name] = float(F.l1_loss(recurrent(current, previous, action, control, state, actor).float(), desired.float()))
    return errors


def train(output: Path, *, config: ContextModelConfig = ContextModelConfig(), plan: ContextTrainingConfig = ContextTrainingConfig(), device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if file_sha256(WORLD_STATE) != WORLD_STATE_SHA256 or file_sha256(RECURRENT) != RECURRENT_SHA256: raise ValueError("World context parent release drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 30); target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"); episodes, corpus_sha = build_aligned_context(device=target); model = build_model(config).to(target); ema = copy.deepcopy(model).eval().requires_grad_(False); optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-3, fused=target.type == "cuda"); rng = np.random.default_rng(plan.seed); history = []; started = time.perf_counter()
    train_context = torch.from_numpy(np.concatenate([item["context"] for item in episodes[:5]])).to(target); train_state = torch.from_numpy(np.concatenate([item["state"] for item in episodes[:5]])).to(target); feature_weights = torch.ones(64, device=target); feature_weights[1:6] = 4; feature_weights[12:22] = 3
    if target.type == "cuda": torch.cuda.reset_peak_memory_stats(target)
    for step in range(1, plan.steps + 1):
        index = torch.from_numpy(rng.integers(0, len(train_context), plan.batch_size)).to(target); optimizer.zero_grad(set_to_none=True); prediction = model(train_context[index]); loss = (F.smooth_l1_loss(prediction, train_state[index], reduction="none") * feature_weights).mean();
        if not bool(torch.isfinite(loss)): raise FloatingPointError("World context loss is non-finite.")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad(): torch._foreach_mul_(list(ema.parameters()), plan.ema_decay); torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
        if step == 1 or step % 100 == 0 or step == plan.steps: history.append({"step": step, "loss": float(loss)})
    model = ema.eval(); validation_context = torch.from_numpy(episodes[4]["context"][-64:]).to(target); validation_state = torch.from_numpy(episodes[4]["state"][-64:]).to(target); test_context = torch.from_numpy(episodes[5]["context"]).to(target); test_state = torch.from_numpy(episodes[5]["state"]).to(target)
    with torch.inference_mode(): validation = model(validation_context); test = model(test_context)
    active = slice(0, 51); validation_mae = float(F.l1_loss(validation[:, active], validation_state[:, active])); test_mae = float(F.l1_loss(test[:, active], test_state[:, active])); family_mae = float(F.l1_loss(test[:, 1:6], test_state[:, 1:6])); resource_mae = float(F.l1_loss(test[:, 12:22], test_state[:, 12:22])); recurrent_payload = torch.load(RECURRENT, map_location="cpu", weights_only=True); recurrent = RecurrentActionDiT(RecurrentConfig(**recurrent_payload["model_config"])); recurrent.load_state_dict(recurrent_payload["state"], strict=True); recurrent.to(target).eval(); downstream = _downstream(model, episodes[5], recurrent, recurrent_payload, target); downstream_advantage = downstream["zero"] - downstream["context"]; gates = {"test_active_mae": test_mae <= .10, "family_mae": family_mae <= .08, "resource_mae": resource_mae <= .12, "downstream_beats_zero": downstream_advantage >= .0003, "downstream_near_true": downstream["context"] <= downstream["true"] * 1.08}; status = "ready" if all(gates.values()) else "quality_failed"; state = {name: value.detach().cpu() for name, value in model.state_dict().items()}; report = {"format": FORMAT, "status": status, "source_sha256": source_sha256(), "world_state_sha256": WORLD_STATE_SHA256, "recurrent_sha256": RECURRENT_SHA256, "corpus_sha256": corpus_sha, "model_config": config.__dict__ if hasattr(config, "__dict__") else {"input_features": config.input_features, "width": config.width, "output_features": config.output_features}, "training_config": {"steps": plan.steps, "batch_size": plan.batch_size, "learning_rate": plan.learning_rate, "ema_decay": plan.ema_decay, "seed": plan.seed}, "parameters": sum(value.numel() for value in model.parameters()), "elapsed_seconds": time.perf_counter() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0, "validation_active_mae": validation_mae, "test_active_mae": test_mae, "family_mae": family_mae, "resource_mae": resource_mae, "downstream_delta_mae": downstream, "downstream_advantage_over_zero": downstream_advantage, "gates": gates, "limitations": ["The adapter is promoted for recurrent conditioning, not as an exact human-readable decoder of all 64 scaffold features."], "history": history}; payload = {"format": CHECKPOINT_FORMAT, "source_sha256": report["source_sha256"], "world_state_sha256": WORLD_STATE_SHA256, "recurrent_sha256": RECURRENT_SHA256, "corpus_sha256": corpus_sha, "model_config": report["model_config"], "state": state, "report": report}; output.mkdir(parents=True); torch.save(payload, output / "runtime.pt"); report["checkpoint_sha256"] = file_sha256(output / "runtime.pt"); (output / "report.json").write_bytes(canonical(report)); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--steps", type=int, default=3000); parser.add_argument("--batch-size", type=int, default=256); args = parser.parse_args(argv); plan = ContextTrainingConfig(steps=args.steps, batch_size=args.batch_size); print(json.dumps(train(args.output, plan=plan, device=args.device), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
