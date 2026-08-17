from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..monolithic_world_model_v1.contract import CHECKPOINT_FORMAT as TEACHER_FORMAT, DirectContextConfig
from ..monolithic_world_model_v1.model import FusedStructuredActionModel
from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..safety import require_disk_floor
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT, DEFAULT_OUTPUT, FORMAT, TEACHER, TEACHER_SHA256, MobileActionConfig, MobileActionPlan, canonical, config_dict, file_sha256, source_sha256
from .data import load_sequences


def _normalization(payload: dict, device: torch.device):
    norm = payload["normalization"]
    return (
        torch.tensor(norm["latent_mean"], device=device)[None, :, None, None],
        torch.tensor(norm["latent_std"], device=device)[None, :, None, None],
        torch.tensor(norm["actor_mean"], device=device)[None],
        torch.tensor(norm["actor_std"], device=device)[None],
    )


def _batch(sequences, rng: np.random.Generator, count: int, *, holdout: bool = False):
    rows = []
    for _ in range(count):
        sequence = sequences[int(rng.integers(0, len(sequences)))]; boundary = len(sequence["latent"]) - 64; low, high = (boundary, len(sequence["latent"])) if holdout else (1, boundary); index = int(rng.integers(low, high)); rows.append((sequence, index))
    result = {}
    for name in ("action", "control", "state", "actor_state", "visibility", "memory"):
        result[name] = np.stack([sequence[name][index] for sequence, index in rows])
    result["current"] = np.stack([sequence["latent"][index] for sequence, index in rows]); result["previous"] = np.stack([sequence["latent"][index - 1] for sequence, index in rows]); result["previous_actor"] = np.stack([sequence["actor_state"][index - 1] for sequence, index in rows]); return result


def _forward(model, batch, norms, device):
    lm, ls, am, ast = norms; current = (torch.from_numpy(batch["current"]).to(device) - lm) / ls; previous = (torch.from_numpy(batch["previous"]).to(device) - lm) / ls; actor = (torch.from_numpy(batch["actor_state"]).to(device) - am) / ast; previous_actor = (torch.from_numpy(batch["previous_actor"]).to(device) - am) / ast; action = torch.from_numpy(batch["action"]).long().to(device); control = torch.from_numpy(batch["control"]).to(device); state = torch.from_numpy(batch["state"]).to(device); visibility = torch.from_numpy(batch["visibility"]).to(device); memory = torch.from_numpy(batch["memory"]).to(device); delta, logits = model.gated_action(current, previous, action, control, state, actor, visibility, memory); actor_result = model.actor(actor, previous_actor, action, control, state, visibility, memory); return delta, logits, actor_result.state, actor_result.gate


@torch.inference_mode()
def _evaluate(student, teacher, sequences, norms, device) -> dict[str, float]:
    rng = np.random.default_rng(0x4556414C4D4F4249); batch = _batch(sequences, rng, 192, holdout=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        target = _forward(teacher, batch, norms, device); predicted = _forward(student, batch, norms, device)
        wrong = dict(batch); wrong["action"] = (wrong["action"] + 7) % 22; target_wrong = _forward(teacher, wrong, norms, device); predicted_wrong = _forward(student, wrong, norms, device)
    teacher_delta, teacher_logits, teacher_actor, teacher_actor_gate = [value.float() for value in target]; delta, logits, actor, actor_gate = [value.float() for value in predicted]
    teacher_sensitivity = float(F.l1_loss(teacher_delta, target_wrong[0].float())); student_sensitivity = float(F.l1_loss(delta, predicted_wrong[0].float())); bias = 1.5; teacher_applied = torch.sigmoid(teacher_logits + bias) * teacher_delta; student_applied = torch.sigmoid(logits + bias) * delta; teacher_wrong_applied = torch.sigmoid(target_wrong[1].float() + bias) * target_wrong[0].float(); student_wrong_applied = torch.sigmoid(predicted_wrong[1].float() + bias) * predicted_wrong[0].float(); teacher_applied_sensitivity = float(F.l1_loss(teacher_applied, teacher_wrong_applied)); student_applied_sensitivity = float(F.l1_loss(student_applied, student_wrong_applied))
    return {"delta_mae": float(F.l1_loss(delta, teacher_delta)), "applied_delta_mae": float(F.l1_loss(student_applied, teacher_applied)), "gate_probability_mae": float(F.l1_loss(torch.sigmoid(logits), torch.sigmoid(teacher_logits))), "actor_state_mae": float(F.l1_loss(actor, teacher_actor)), "actor_gate_mae": float(F.l1_loss(actor_gate, teacher_actor_gate)), "teacher_action_sensitivity": teacher_sensitivity, "student_action_sensitivity": student_sensitivity, "action_sensitivity_ratio": student_sensitivity / max(teacher_sensitivity, 1e-9), "teacher_applied_action_sensitivity": teacher_applied_sensitivity, "student_applied_action_sensitivity": student_applied_sensitivity, "applied_action_sensitivity_ratio": student_applied_sensitivity / max(teacher_applied_sensitivity, 1e-9)}


def train(output: Path = DEFAULT_OUTPUT, *, config: MobileActionConfig = MobileActionConfig(), plan: MobileActionPlan = MobileActionPlan(), device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if file_sha256(TEACHER) != TEACHER_SHA256: raise ValueError("Mobile action teacher drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 << 30); target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if target.type == "cuda": torch.cuda.set_per_process_memory_fraction(.48, 0); torch.cuda.reset_peak_memory_stats(target)
    torch.set_num_threads(4); torch.manual_seed(plan.seed); rng = np.random.default_rng(plan.seed); sequences = load_sequences(); teacher_payload = torch.load(TEACHER, map_location="cpu", weights_only=True)
    if teacher_payload.get("format") != TEACHER_FORMAT or teacher_payload.get("status") != "monolithic_foundation_ready": raise ValueError("Mobile action teacher is not promoted.")
    teacher_model = FusedStructuredActionModel(DirectContextConfig(**teacher_payload["model_config"]), ModelConfig(**teacher_payload["recurrent_config"])); teacher_model.load_state_dict(teacher_payload["state"], strict=True); teacher = teacher_model.recurrent.to(target).eval().requires_grad_(False); norms = _normalization(teacher_payload, target)
    student_config = ModelConfig(**config_dict(config)); student = PerceptionRecurrentWorldStudent(student_config).to(target); ema = copy.deepcopy(student).eval().requires_grad_(False); optimizer = torch.optim.AdamW(student.parameters(), lr=plan.learning_rate, weight_decay=1e-3, fused=target.type == "cuda"); history = []; started = time.perf_counter()
    for step in range(1, plan.steps + 1):
        batch = _batch(sequences, rng, plan.batch_size); wrong = dict(batch); wrong["action"] = (wrong["action"] + 7) % 22; optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"): desired = _forward(teacher, batch, norms, target); desired_wrong = _forward(teacher, wrong, norms, target)
        with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
            prediction = _forward(student, batch, norms, target)
            prediction_wrong = _forward(student, wrong, norms, target)
            delta_weight = 1 + 3 * desired[0].float().abs().mean(1, keepdim=True).clamp(0, 1)
            delta_loss = (F.smooth_l1_loss(prediction[0].float(), desired[0].float(), reduction="none") * delta_weight).mean()
            gate_loss = F.smooth_l1_loss(torch.sigmoid(prediction[1].float()), torch.sigmoid(desired[1].float()))
            actor_loss = F.smooth_l1_loss(prediction[2].float(), desired[2].float())
            actor_gate_loss = F.smooth_l1_loss(prediction[3].float(), desired[3].float())
            action_loss = F.smooth_l1_loss((prediction[0] - prediction_wrong[0]).float(), (desired[0] - desired_wrong[0]).float())
            bias = 1.5
            desired_applied = torch.sigmoid(desired[1].float() + bias) * desired[0].float()
            desired_wrong_applied = torch.sigmoid(desired_wrong[1].float() + bias) * desired_wrong[0].float()
            predicted_applied = torch.sigmoid(prediction[1].float() + bias) * prediction[0].float()
            predicted_wrong_applied = torch.sigmoid(prediction_wrong[1].float() + bias) * prediction_wrong[0].float()
            # These signals are intentionally small after gating. L1 keeps their
            # gradient proportional instead of squaring them into irrelevance.
            applied_loss = F.l1_loss(predicted_applied, desired_applied)
            applied_action_loss = F.l1_loss(
                predicted_applied - predicted_wrong_applied,
                desired_applied - desired_wrong_applied,
            )
            loss = delta_loss + .35 * gate_loss + .4 * actor_loss + .15 * actor_gate_loss + .8 * action_loss + .5 * applied_loss + 1.5 * applied_action_loss
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Mobile action distillation became non-finite.")
        loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1); optimizer.step()
        with torch.no_grad(): torch._foreach_mul_(list(ema.parameters()), plan.ema_decay); torch._foreach_add_(list(ema.parameters()), list(student.parameters()), alpha=1 - plan.ema_decay)
        if step == 1 or step % 100 == 0 or step == plan.steps:
            history.append({"step": step, "loss": float(loss), "delta": float(delta_loss), "gate": float(gate_loss), "actor": float(actor_loss), "action": float(action_loss), "applied": float(applied_loss), "applied_action": float(applied_action_loss)})
    raw_metrics = _evaluate(student.eval(), teacher, sequences, norms, target); ema_metrics = _evaluate(ema.eval(), teacher, sequences, norms, target); score = lambda row: row["applied_delta_mae"] + row["gate_probability_mae"] + row["actor_state_mae"] + max(0, .8 - row["applied_action_sensitivity_ratio"]); selected = "raw" if score(raw_metrics) <= score(ema_metrics) else "ema"; chosen = student if selected == "raw" else ema; metrics = raw_metrics if selected == "raw" else ema_metrics; teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters()); parameters = sum(parameter.numel() for parameter in chosen.parameters()); gates = {"raw_delta_diagnostic": metrics["delta_mae"] <= .04, "applied_delta_mae": metrics["applied_delta_mae"] <= .008, "gate_probability_mae": metrics["gate_probability_mae"] <= .05, "actor_state_mae": metrics["actor_state_mae"] <= .03, "actor_gate_mae": metrics["actor_gate_mae"] <= .04, "action_sensitivity_retained": metrics["action_sensitivity_ratio"] >= .72, "applied_action_sensitivity_retained": metrics["applied_action_sensitivity_ratio"] >= .72, "parameter_reduction": parameters <= teacher_parameters * .35}; status = "mobile_action_ready" if all(gates.values()) else "quality_failed"; state = {name: value.detach().cpu() for name, value in chosen.cpu().state_dict().items()}; report = {"format": FORMAT, "status": status, "source_sha256": source_sha256(), "teacher_sha256": TEACHER_SHA256, "model_config": config_dict(config), "training_plan": config_dict(plan), "parameters": parameters, "teacher_parameters": teacher_parameters, "parameter_ratio": parameters / teacher_parameters, "selection": {"selected": selected, "raw": raw_metrics, "ema": ema_metrics}, "metrics": metrics, "gates": gates, "runtime": {"elapsed_seconds": time.perf_counter() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0}, "history": history, "limitations": ["This mobile action student matches held-out one-step teacher behavior; long-rollout parity and physical-device profiling remain required."]}; output.mkdir(parents=True); temporary = output / f".runtime.pt.tmp-{os.getpid()}"; torch.save({"format": CHECKPOINT_FORMAT, "source_sha256": report["source_sha256"], "status": status, "model_config": report["model_config"], "state": state, "normalization": teacher_payload["normalization"], "inference": teacher_payload["inference"], "report": report}, temporary); os.replace(temporary, output / "runtime.pt"); report["checkpoint_sha256"] = file_sha256(output / "runtime.pt"); (output / "report.json").write_bytes(canonical(report)); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--device", default="cuda"); parser.add_argument("--steps", type=int, default=4500); args = parser.parse_args(argv); print(json.dumps(train(args.output, plan=MobileActionPlan(steps=args.steps), device=args.device), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
