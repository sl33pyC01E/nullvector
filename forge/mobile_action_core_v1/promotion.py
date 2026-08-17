from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ..monolithic_world_model_v1.contract import DirectContextConfig
from ..monolithic_world_model_v1.model import FusedStructuredActionModel
from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT, FORMAT, TEACHER, TEACHER_SHA256, canonical, file_sha256, source_sha256
from .data import load_sequences
from .training import _evaluate, _normalization


def promote(candidate: Path, output: Path) -> dict[str, object]:
    candidate = Path(candidate).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if file_sha256(TEACHER) != TEACHER_SHA256: raise ValueError("Teacher drifted.")
    parent_sha = file_sha256(candidate)
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    teacher_payload = torch.load(TEACHER, map_location="cpu", weights_only=True)
    student = PerceptionRecurrentWorldStudent(ModelConfig(**payload["model_config"]))
    student.load_state_dict(payload["state"], strict=True); student.eval()
    teacher_model = FusedStructuredActionModel(DirectContextConfig(**teacher_payload["model_config"]), ModelConfig(**teacher_payload["recurrent_config"]))
    teacher_model.load_state_dict(teacher_payload["state"], strict=True); teacher = teacher_model.recurrent.eval()
    metrics = _evaluate(student, teacher, load_sequences(), _normalization(teacher_payload, torch.device("cpu")), torch.device("cpu"))
    applied_gap = abs(metrics["teacher_applied_action_sensitivity"] - metrics["student_applied_action_sensitivity"])
    parameters = sum(value.numel() for value in student.parameters()); teacher_parameters = sum(value.numel() for value in teacher.parameters())
    gates = {
        "raw_delta_diagnostic": metrics["delta_mae"] <= .04,
        "applied_delta_mae": metrics["applied_delta_mae"] <= .008,
        "gate_probability_mae": metrics["gate_probability_mae"] <= .05,
        "actor_state_mae": metrics["actor_state_mae"] <= .03,
        "actor_gate_mae": metrics["actor_gate_mae"] <= .04,
        "raw_action_sensitivity_retained": metrics["action_sensitivity_ratio"] >= .72,
        "executed_action_nonzero": metrics["student_applied_action_sensitivity"] >= .002,
        "executed_action_absolute_parity": applied_gap <= .00125,
        "parameter_reduction": parameters <= teacher_parameters * .35,
    }
    status = "mobile_action_ready" if all(gates.values()) else "quality_failed"
    report = {
        "format": FORMAT, "status": status, "source_sha256": source_sha256(),
        "teacher_sha256": TEACHER_SHA256, "candidate_checkpoint_sha256": parent_sha,
        "model_config": payload["model_config"], "parameters": parameters,
        "teacher_parameters": teacher_parameters, "parameter_ratio": parameters / teacher_parameters,
        "metrics": {**metrics, "executed_action_sensitivity_absolute_gap": applied_gap}, "gates": gates,
        "calibration": {"reason": "Teacher executed counter-action magnitude is small; require a nonzero student response and bounded absolute parity while retaining the raw ratio gate.", "ratio_retained_as_diagnostic": True},
        "limitations": ["Physical Galaxy S25 Ultra rollout profiling remains required."],
    }
    output.mkdir(parents=True)
    checkpoint = output / "runtime.pt"; temporary = output / f".runtime.pt.tmp-{os.getpid()}"
    torch.save({"format": CHECKPOINT_FORMAT, "source_sha256": report["source_sha256"], "status": status, "model_config": payload["model_config"], "state": payload["state"], "normalization": payload["normalization"], "inference": payload["inference"], "report": report}, temporary)
    os.replace(temporary, checkpoint); report["checkpoint_sha256"] = file_sha256(checkpoint)
    (output / "report.json").write_bytes(canonical(report)); return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("candidate", type=Path); parser.add_argument("output", type=Path); args = parser.parse_args()
    print(json.dumps(promote(args.candidate, args.output), indent=2, sort_keys=True))
