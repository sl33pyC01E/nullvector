from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
import numpy as np
import torch
from torch import Tensor

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import MAX_DISPLACEMENT, source_sha256 as parent_source_sha256
from ..creature_stage_neural_motion.dataset import NativeMotionTeacher
from ..creature_stage_neural_motion.evaluation import (
    CROSS_BACKEND_ABSOLUTE_TOLERANCE,
    CROSS_BACKEND_RELATIVE_TOLERANCE,
    METRIC_NAMES,
    PROMOTION_THRESHOLDS,
    _aggregate,
    _clip_metrics,
    _cross_backend_difference,
)
from ..creature_stage_neural_motion.model import CellularMotionTransformer
from ..creature_stage_neural_motion.training import _canonical, _config, _sha256_file
from .contract import CHECKPOINT_FORMAT, source_sha256
from .training import PRODUCTION_FORMAT, _load_rollout_checkpoint


EVALUATION_FORMAT = "nullvector-creature-stage-neural-motion-rollout-evaluation-v1"
EVALUATION_SCHEMA = PROJECT_ROOT / "shared/schema/creature_stage_neural_motion_rollout_evaluation.schema.json"
EVALUATION_SOURCE_FILES = (
    "forge/creature_stage_neural_motion_rollout/evaluation.py",
    "shared/schema/creature_stage_neural_motion_rollout_evaluation.schema.json",
)
REPORT_NAME = "evaluation_manifest.json"
MAX_REPORT_BYTES = 8 * 1024**2
REPLAY_POLICY = {
    "structure": "exact",
    "provenance_and_gates": "exact",
    "numeric_comparison": "symmetric_isclose",
    "absolute_tolerance": CROSS_BACKEND_ABSOLUTE_TOLERANCE,
    "relative_tolerance": CROSS_BACKEND_RELATIVE_TOLERANCE,
    "reference_backend": "cpu",
}


def evaluation_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-motion-rollout-evaluation-source-v1\0")
    for relative in EVALUATION_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("rollout motion evaluation path must remain inside the project") from error


def _load_authority(checkpoint_path: Path) -> tuple[CellularMotionTransformer, dict[str, Any], NativeMotionTeacher]:
    checkpoint_path = Path(checkpoint_path).resolve()
    contract_path = checkpoint_path.parent / "production_contract.json"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file() or not contract_path.is_file():
        raise ValueError("rollout motion evaluation authority is missing")
    raw = contract_path.read_bytes()
    contract = json.loads(raw)
    if (
        raw != _canonical(contract) or contract.get("format") != PRODUCTION_FORMAT
        or contract.get("source_sha256") != source_sha256()
        or contract.get("semantic_sha256")
        != hashlib.sha256(_canonical({key: value for key, value in contract.items() if key != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("rollout motion evaluation contract drifted")
    stem = checkpoint_path.stem
    prefix = "cell_motion_rollout_"
    if not stem.startswith(prefix) or not stem[len(prefix):].isdigit():
        raise ValueError("rollout motion evaluation checkpoint name drifted")
    update = int(stem[len(prefix):])
    payload = _load_rollout_checkpoint(checkpoint_path, contract, update)
    model = CellularMotionTransformer(_config(contract["model"]))
    model.load_state_dict(payload["ema_state"], strict=True)
    teacher = NativeMotionTeacher(PROJECT_ROOT / contract["teacher"]["path"])
    if teacher.semantic_sha256 != contract["teacher"]["semantic_sha256"]:
        raise ValueError("rollout motion evaluation teacher drifted")
    authority = {
        "kind": "rollout_production_ema",
        "format": CHECKPOINT_FORMAT,
        "path": _relative(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256_file(checkpoint_path),
        "update": update,
        "total_updates": int(contract["total_updates"]),
        "final_checkpoint": update == int(contract["total_updates"]),
        "model_state_sha256": payload["model_state_sha256"],
        "ema_state_sha256": payload["ema_state_sha256"],
        "contract_semantic_sha256": contract["semantic_sha256"],
        "parent_checkpoint_sha256": contract["parent"]["sha256"],
    }
    return model.eval(), authority, teacher


def _tensor_rows(rows: list[dict[str, Any]], name: str, device: torch.device) -> Tensor:
    return torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)


def _gates(payload: dict[str, Any]) -> dict[str, bool]:
    aggregate = payload["aggregate"]
    clips = payload["clips"]
    thresholds = payload["thresholds"]
    full = (
        payload["scope"]["split"] == "validation"
        and payload["scope"]["motion_ids"] == list(range(13))
        and payload["scope"]["rollout_frames"] == 72
        and len(clips) == 65
    )
    return {
        "full_validation_matrix": full,
        "final_rollout_checkpoint": payload["checkpoint"]["final_checkpoint"] is True,
        "all_values_finite": all(
            math.isfinite(float(row["metrics"][name])) for row in clips for name in METRIC_NAMES
        ),
        "outside_cells_exact_zero": payload["diagnostics"]["maximum_outside_abs"] == 0.0,
        "position_accuracy": aggregate["position_mae_px"] <= thresholds["position_mae_px_max"],
        "velocity_accuracy": aggregate["velocity_mae_px"] <= thresholds["velocity_mae_px_max"],
        "beats_copy_previous": aggregate["baseline_improvement"] >= thresholds["baseline_improvement_min"],
        "bond_coherence": aggregate["graph_relative_mae_px"] <= thresholds["graph_relative_mae_px_max"],
        "motion_energy_calibrated": (
            thresholds["energy_ratio_min"] <= aggregate["energy_ratio"] <= thresholds["energy_ratio_max"]
            and min(row["metrics"]["energy_ratio"] for row in clips) >= thresholds["minimum_clip_energy_ratio"]
        ),
        "loop_closure": max(row["metrics"]["loop_closure_px"] for row in clips) <= thresholds["loop_closure_px_max"],
        "family_balance": max(row["metrics"]["position_mae_px"] for row in payload["families"]) <= thresholds["maximum_family_position_mae_px"],
        "motion_balance": max(row["metrics"]["position_mae_px"] for row in payload["motions"]) <= thresholds["maximum_motion_position_mae_px"],
        "bounded_displacement": max(row["metrics"]["max_displacement_px"] for row in clips) <= thresholds["maximum_displacement_px"] + 1e-6,
    }


def _build_payload(
    checkpoint_path: Path,
    *,
    split: str,
    motion_ids: tuple[int, ...],
    rollout_frames: int,
    device: torch.device,
    allow_sealed_test: bool,
) -> dict[str, Any]:
    if split not in ("validation", "test"):
        raise ValueError("rollout motion evaluation only permits held-out splits")
    if split == "test" and not allow_sealed_test:
        raise PermissionError("sealed rollout motion test split requires explicit release")
    if not motion_ids or tuple(sorted(set(motion_ids))) != motion_ids or any(not 0 <= value < 13 for value in motion_ids):
        raise ValueError("rollout motion evaluation selection drifted")
    if type(rollout_frames) is not int or not 2 <= rollout_frames <= 72:
        raise ValueError("rollout motion evaluation frame count drifted")
    model, authority, teacher = _load_authority(checkpoint_path)
    model.to(device).eval()
    chassis_ids = tuple(teacher.split_chassis(split))
    if len(chassis_ids) != 5 or sorted(int(teacher.chassis[index]["family_id"]) for index in chassis_ids) != list(range(5)):
        raise ValueError("rollout motion evaluation family coverage drifted")
    clips: list[dict[str, Any]] = []
    maximum_outside = 0.0
    with torch.inference_mode():
        for motion_id in motion_ids:
            predicted_frames: list[np.ndarray] = []
            target_frames: list[np.ndarray] = []
            baseline_frames: list[np.ndarray] = []
            state: Tensor | None = None
            static: Tensor | None = None
            mask: Tensor | None = None
            adjacency: Tensor | None = None
            for frame_index in range(rollout_frames):
                rows = [teacher.sample(chassis, motion_id, frame_index) for chassis in chassis_ids]
                if frame_index == 0:
                    static = _tensor_rows(rows, "static", device)
                    mask = _tensor_rows(rows, "mask", device)
                    adjacency = _tensor_rows(rows, "adjacency", device)
                    state = _tensor_rows(rows, "state", device)
                assert static is not None and mask is not None and adjacency is not None and state is not None
                target = _tensor_rows(rows, "target", device)
                controls = _tensor_rows(rows, "controls", device)
                family = torch.tensor([int(row["family"]) for row in rows], dtype=torch.long, device=device)
                morphotype = torch.tensor([int(row["morphotype"]) for row in rows], dtype=torch.long, device=device)
                motion = torch.full((5,), motion_id, dtype=torch.long, device=device)
                phase = torch.tensor([float(row["phase"]) for row in rows], dtype=torch.float32, device=device)
                predicted = model(static, state, mask, adjacency, family, morphotype, motion, phase, controls)
                if not bool(torch.isfinite(predicted).all()):
                    raise FloatingPointError("rollout motion evaluation became non-finite")
                maximum_outside = max(maximum_outside, float(predicted[~mask].abs().max()))
                predicted_frames.append(predicted.detach().cpu().numpy())
                target_frames.append(target.detach().cpu().numpy())
                baseline_frames.append(state.detach().cpu().numpy())
                state = predicted.detach()
            predicted_np = np.stack(predicted_frames, axis=1)
            target_np = np.stack(target_frames, axis=1)
            baseline_np = np.stack(baseline_frames, axis=1)
            for row_index, chassis_id in enumerate(chassis_ids):
                sample = teacher.sample(chassis_id, motion_id, 0)
                clip = teacher.clips[chassis_id * 13 + motion_id]
                spec = teacher.manifest["motion_specs"][clip["motion"]]
                clips.append({
                    "chassis_id": chassis_id,
                    "family_id": int(sample["family"]),
                    "morphotype_id": int(sample["morphotype"]),
                    "motion_id": motion_id,
                    "motion": clip["motion"],
                    "cell_count": int(sample["cell_count"]),
                    "loop": bool(spec["loop"]),
                    "metrics": _clip_metrics(
                        predicted_np[row_index], target_np[row_index], baseline_np[row_index],
                        sample["static"], sample["mask"], sample["adjacency"], loop=bool(spec["loop"]),
                    ),
                })
    clips.sort(key=lambda row: (row["family_id"], row["motion_id"], row["chassis_id"]))
    families = [
        {"family_id": family, "clip_count": len(rows), "metrics": _aggregate(rows)}
        for family in range(5)
        for rows in [[row for row in clips if row["family_id"] == family]]
    ]
    motions = [
        {"motion_id": motion, "motion": rows[0]["motion"], "clip_count": len(rows), "metrics": _aggregate(rows)}
        for motion in motion_ids
        for rows in [[row for row in clips if row["motion_id"] == motion]]
    ]
    payload: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "status": "evaluated",
        "evaluation_source_sha256": evaluation_source_sha256(),
        "rollout_source_sha256": source_sha256(),
        "parent_model_source_sha256": parent_source_sha256(),
        "checkpoint": authority,
        "teacher": {
            "path": _relative(teacher.root),
            "semantic_sha256": teacher.semantic_sha256,
            "manifest_sha256": teacher.validation["manifest_sha256"],
            "binary_sha256": teacher.validation["binary_sha256"],
        },
        "scope": {
            "split": split,
            "sealed_test_released": bool(split == "test" and allow_sealed_test),
            "motion_ids": list(motion_ids),
            "rollout_frames": rollout_frames,
            "chassis_count": len(chassis_ids),
            "clip_count": len(clips),
            "prediction_fed": True,
            "state_teacher_forcing_after_frame_zero": False,
        },
        "replay_policy": dict(REPLAY_POLICY),
        "thresholds": dict(PROMOTION_THRESHOLDS),
        "clips": clips,
        "aggregate": _aggregate(clips),
        "families": families,
        "motions": motions,
        "diagnostics": {"maximum_outside_abs": round(maximum_outside, 12)},
    }
    payload["gates"] = _gates(payload)
    payload["promotion_eligible"] = all(payload["gates"].values())
    return payload


def evaluate_checkpoint(
    checkpoint_path: Path,
    output: Path,
    *,
    split: str = "validation",
    motion_ids: Iterable[int] = range(13),
    rollout_frames: int = 72,
    device: str = "cpu",
    allow_sealed_test: bool = False,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=64 * 1024**2)
    payload = _build_payload(
        checkpoint_path, split=split, motion_ids=tuple(int(value) for value in motion_ids),
        rollout_frames=rollout_frames, device=torch.device(device), allow_sealed_test=allow_sealed_test,
    )
    payload["semantic_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    output.mkdir(parents=True)
    from ..creature_stage_neural_motion.training import _atomic_bytes
    _atomic_bytes(output / REPORT_NAME, _canonical(payload))
    return validate_evaluation(output / REPORT_NAME, replay=False)


def _validate_relationships(report: dict[str, Any]) -> None:
    if report["thresholds"] != PROMOTION_THRESHOLDS or report["replay_policy"] != REPLAY_POLICY:
        raise ValueError("rollout motion evaluation policy drifted")
    if set(report["checkpoint"]) != {
        "kind", "format", "path", "bytes", "sha256", "update", "total_updates",
        "final_checkpoint", "model_state_sha256", "ema_state_sha256",
        "contract_semantic_sha256", "parent_checkpoint_sha256",
    }:
        raise ValueError("rollout motion evaluation checkpoint structure drifted")
    scope = report["scope"]
    if set(scope) != {
        "split", "sealed_test_released", "motion_ids", "rollout_frames",
        "chassis_count", "clip_count", "prediction_fed",
        "state_teacher_forcing_after_frame_zero",
    } or scope["chassis_count"] != 5 or scope["prediction_fed"] is not True or scope["state_teacher_forcing_after_frame_zero"] is not False:
        raise ValueError("rollout motion evaluation scope drifted")
    clip_keys = {
        "chassis_id", "family_id", "morphotype_id", "motion_id", "motion",
        "cell_count", "loop", "metrics",
    }
    if any(set(row) != clip_keys or set(row["metrics"]) != set(METRIC_NAMES) for row in report["clips"]):
        raise ValueError("rollout motion evaluation clip structure drifted")
    if len(report["clips"]) != scope["clip_count"] or scope["clip_count"] != scope["chassis_count"] * len(scope["motion_ids"]):
        raise ValueError("rollout motion evaluation clip count drifted")
    expected_keys = {(family, motion) for family in range(5) for motion in scope["motion_ids"]}
    actual_keys = {(row["family_id"], row["motion_id"]) for row in report["clips"]}
    if actual_keys != expected_keys or len(actual_keys) != len(report["clips"]):
        raise ValueError("rollout motion evaluation matrix drifted")
    if set(report["diagnostics"]) != {"maximum_outside_abs"}:
        raise ValueError("rollout motion evaluation diagnostics drifted")
    if report["aggregate"] != _aggregate(report["clips"]):
        raise ValueError("rollout motion evaluation aggregate drifted")
    expected_families = [
        {"family_id": family, "clip_count": len(rows), "metrics": _aggregate(rows)}
        for family in range(5)
        for rows in [[row for row in report["clips"] if row["family_id"] == family]]
    ]
    expected_motions = [
        {"motion_id": motion, "motion": rows[0]["motion"], "clip_count": len(rows), "metrics": _aggregate(rows)}
        for motion in report["scope"]["motion_ids"]
        for rows in [[row for row in report["clips"] if row["motion_id"] == motion]]
    ]
    if report["families"] != expected_families or report["motions"] != expected_motions:
        raise ValueError("rollout motion evaluation macro evidence drifted")
    if report["gates"] != _gates(report) or report["promotion_eligible"] is not all(report["gates"].values()):
        raise ValueError("rollout motion evaluation verdict drifted")


def validate_evaluation(report_path: Path, *, replay: bool = False) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    if report_path.is_symlink() or not report_path.is_file() or not 0 < report_path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("rollout motion evaluation report is missing or oversized")
    raw = report_path.read_bytes()
    report = json.loads(raw)
    required = {
        "format", "status", "evaluation_source_sha256", "rollout_source_sha256",
        "parent_model_source_sha256", "checkpoint", "teacher", "scope", "replay_policy",
        "thresholds", "clips", "aggregate", "families", "motions", "diagnostics", "gates",
        "promotion_eligible", "semantic_sha256",
    }
    if raw != _canonical(report) or set(report) != required:
        raise ValueError("rollout motion evaluation structure drifted")
    errors = sorted(
        Draft202012Validator(json.loads(EVALUATION_SCHEMA.read_bytes())).iter_errors(report),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"rollout motion evaluation schema drifted: {errors[0].message}")
    if (
        report["format"] != EVALUATION_FORMAT or report["status"] != "evaluated"
        or report["evaluation_source_sha256"] != evaluation_source_sha256()
        or report["rollout_source_sha256"] != source_sha256()
        or report["parent_model_source_sha256"] != parent_source_sha256()
        or report["semantic_sha256"]
        != hashlib.sha256(_canonical({key: value for key, value in report.items() if key != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("rollout motion evaluation authority drifted")
    _validate_relationships(report)
    checkpoint_path = PROJECT_ROOT / report["checkpoint"]["path"]
    model, authority, teacher = _load_authority(checkpoint_path)
    del model
    if authority != report["checkpoint"] or teacher.semantic_sha256 != report["teacher"]["semantic_sha256"]:
        raise ValueError("rollout motion evaluation provenance drifted")
    if replay:
        replayed = _build_payload(
            checkpoint_path, split=report["scope"]["split"],
            motion_ids=tuple(report["scope"]["motion_ids"]),
            rollout_frames=report["scope"]["rollout_frames"], device=torch.device("cpu"),
            allow_sealed_test=report["scope"]["sealed_test_released"],
        )
        difference = _cross_backend_difference(
            replayed, {key: value for key, value in report.items() if key != "semantic_sha256"}
        )
        if difference is not None:
            raise ValueError(f"rollout motion evaluation cross-backend replay drifted: {difference}")
    return {
        "passed": True,
        "promotion_eligible": report["promotion_eligible"],
        "update": report["checkpoint"]["update"],
        "split": report["scope"]["split"],
        "clips": report["scope"]["clip_count"],
        "frames_per_clip": report["scope"]["rollout_frames"],
        "semantic_sha256": report["semantic_sha256"],
        "gates": report["gates"],
    }
