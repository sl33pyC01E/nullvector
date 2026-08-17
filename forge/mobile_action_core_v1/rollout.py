from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..monolithic_world_model_v1.contract import DirectContextConfig
from ..monolithic_world_model_v1.model import FusedStructuredActionModel
from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT, FORMAT, TEACHER, TEACHER_SHA256, canonical, file_sha256
from .data import load_sequences


ROLLOUT_FORMAT = FORMAT + "-rollout-v1"


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _step(model, current, previous, actor, previous_actor, sequence, index, bias):
    device = current.device
    action = torch.from_numpy(sequence["action"][index:index + 1]).long().to(device)
    control = torch.from_numpy(sequence["control"][index:index + 1]).to(device)
    state = torch.from_numpy(sequence["state"][index:index + 1]).to(device)
    visibility = torch.from_numpy(sequence["visibility"][index:index + 1]).to(device)
    memory = torch.from_numpy(sequence["memory"][index:index + 1]).to(device)
    delta, logits = model.gated_action(current, previous, action, control, state, actor, visibility, memory)
    next_latent = current + torch.sigmoid(logits + bias) * delta
    result = model.actor(actor, previous_actor, action, control, state, visibility, memory)
    next_actor = actor + .9 * (result.gate >= .7) * (result.state - actor)
    return next_latent, next_actor


@torch.inference_mode()
def evaluate(checkpoint: Path, output: Path, *, horizons: tuple[int, ...] = (4, 8, 16), starts_per_world: int = 3) -> dict[str, object]:
    checkpoint = Path(checkpoint).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if file_sha256(TEACHER) != TEACHER_SHA256: raise ValueError("Teacher drifted.")
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if candidate.get("format") != CHECKPOINT_FORMAT or candidate.get("status") != "mobile_action_ready": raise ValueError("Mobile action candidate is not promoted.")
    teacher_payload = torch.load(TEACHER, map_location="cpu", weights_only=True)
    teacher_full = FusedStructuredActionModel(DirectContextConfig(**teacher_payload["model_config"]), ModelConfig(**teacher_payload["recurrent_config"]))
    teacher_full.load_state_dict(teacher_payload["state"], strict=True); teacher = teacher_full.recurrent.eval()
    student = PerceptionRecurrentWorldStudent(ModelConfig(**candidate["model_config"])); student.load_state_dict(candidate["state"], strict=True); student.eval()
    norm = teacher_payload["normalization"]
    lm = torch.tensor(norm["latent_mean"])[None, :, None, None]; ls = torch.tensor(norm["latent_std"])[None, :, None, None]
    am = torch.tensor(norm["actor_mean"])[None]; ast = torch.tensor(norm["actor_std"])[None]
    bias_max = float(candidate["inference"]["gate_logit_bias_max"]); ramp = int(candidate["inference"]["gate_logit_bias_ramp_steps"])
    rows = []
    for world, sequence in enumerate(load_sequences()):
        boundary = len(sequence["latent"]) - 64
        starts = np.linspace(boundary, len(sequence["latent"]) - max(horizons) - 1, starts_per_world, dtype=int)
        for start in starts:
            base_previous = (torch.from_numpy(sequence["latent"][start - 1:start]) - lm) / ls
            base_current = (torch.from_numpy(sequence["latent"][start:start + 1]) - lm) / ls
            base_previous_actor = (torch.from_numpy(sequence["actor_state"][start - 1:start]) - am) / ast
            base_actor = (torch.from_numpy(sequence["actor_state"][start:start + 1]) - am) / ast
            for horizon in horizons:
                tp, tc, tpa, ta = base_previous.clone(), base_current.clone(), base_previous_actor.clone(), base_actor.clone()
                sp, sc, spa, sa = tp.clone(), tc.clone(), tpa.clone(), ta.clone(); errors = []; actor_errors = []
                for offset in range(horizon):
                    bias = bias_max * min(offset / ramp, 1.0) if ramp else bias_max
                    tn, tan = _step(teacher, tc, tp, ta, tpa, sequence, start + offset, bias)
                    sn, san = _step(student, sc, sp, sa, spa, sequence, start + offset, bias)
                    errors.append(float(F.l1_loss(sn, tn))); actor_errors.append(float(F.l1_loss(san, tan)))
                    tp, tc, tpa, ta = tc, tn, ta, tan; sp, sc, spa, sa = sc, sn, sa, san
                rows.append({"world": world, "start": int(start), "horizon": horizon, "final_latent_mae": errors[-1], "mean_latent_mae": float(np.mean(errors)), "final_actor_mae": actor_errors[-1], "finite": bool(torch.isfinite(sc).all() and torch.isfinite(sa).all()), "max_abs_latent": float(sc.abs().max())})
    by_horizon = {}
    for horizon in horizons:
        selected = [row for row in rows if row["horizon"] == horizon]
        by_horizon[str(horizon)] = {"final_latent_mae": float(np.mean([row["final_latent_mae"] for row in selected])), "mean_latent_mae": float(np.mean([row["mean_latent_mae"] for row in selected])), "final_actor_mae": float(np.mean([row["final_actor_mae"] for row in selected])), "maximum_abs_latent": max(row["max_abs_latent"] for row in selected)}
    longest = by_horizon[str(max(horizons))]
    gates = {"all_rollouts_finite": all(row["finite"] for row in rows), "horizon_16_latent_mae": longest["final_latent_mae"] <= .35, "horizon_16_actor_mae": longest["final_actor_mae"] <= .15, "latent_bounded": longest["maximum_abs_latent"] <= 12, "all_worlds_covered": len({row["world"] for row in rows}) == 6}
    report = {"format": ROLLOUT_FORMAT, "status": "mobile_rollout_ready" if all(gates.values()) else "quality_failed", "source_sha256": _source_sha256(), "checkpoint_sha256": file_sha256(checkpoint), "teacher_sha256": TEACHER_SHA256, "horizons": list(horizons), "starts_per_world": starts_per_world, "rollout_count": len(rows), "metrics": by_horizon, "gates": gates, "records": rows, "limitations": ["This is teacher-relative latent rollout validation; physical-device rendered rollout profiling remains required."]}
    output.mkdir(parents=True); (output / "rollout_report.json").write_bytes(canonical(report)); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint", type=Path); parser.add_argument("output", type=Path); args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.checkpoint, args.output), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
