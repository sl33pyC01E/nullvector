from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from jsonschema import Draft202012Validator

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, MAX_DISPLACEMENT, source_sha256
from .dataset import NativeMotionTeacher
from .model import CellularMotionTransformer
from .training import (
    PRODUCTION_FORMAT,
    SMOKE_FORMAT,
    _atomic_bytes,
    _canonical,
    _config,
    _load_checkpoint,
    _sha256_file,
    _state_sha256,
    validate_cpu_smoke,
)


EVALUATION_FORMAT = "nullvector-creature-stage-neural-motion-evaluation-v1"
EVALUATION_SCHEMA = PROJECT_ROOT / "shared/schema/creature_stage_neural_motion_evaluation.schema.json"
EVALUATION_SOURCE_FILES = (
    "forge/creature_stage_neural_motion/evaluation.py",
    "shared/schema/creature_stage_neural_motion_evaluation.schema.json",
)
REPORT_NAME = "evaluation_manifest.json"
MAX_REPORT_BYTES = 8 * 1024 * 1024
METRIC_NAMES = (
    "position_mae_px",
    "velocity_mae_px",
    "copy_previous_mae_px",
    "baseline_improvement",
    "graph_relative_mae_px",
    "predicted_energy_px",
    "target_energy_px",
    "energy_ratio",
    "appendage_energy_px",
    "core_energy_px",
    "appendage_core_ratio",
    "loop_closure_px",
    "max_displacement_px",
)
PROMOTION_THRESHOLDS = {
    "position_mae_px_max": 0.75,
    "velocity_mae_px_max": 0.35,
    "baseline_improvement_min": 0.10,
    "graph_relative_mae_px_max": 0.45,
    "energy_ratio_min": 0.55,
    "energy_ratio_max": 1.55,
    "minimum_clip_energy_ratio": 0.20,
    "loop_closure_px_max": 0.85,
    "maximum_family_position_mae_px": 1.0,
    "maximum_motion_position_mae_px": 1.2,
    "maximum_displacement_px": MAX_DISPLACEMENT,
}


def evaluation_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-motion-evaluation-source-v1\0")
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
        raise ValueError("cellular motion evaluation path must remain inside the project") from error


def _load_authority(checkpoint_path: Path) -> tuple[CellularMotionTransformer, dict[str, Any], NativeMotionTeacher]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("cellular motion evaluation checkpoint is missing or linked")
    if checkpoint_path.name == "smoke_checkpoint.pt":
        validation = validate_cpu_smoke(checkpoint_path.parent)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = CellularMotionTransformer(_config(payload["config"]))
        model.load_state_dict(payload["model_state"], strict=True)
        teacher = NativeMotionTeacher(PROJECT_ROOT / json.loads(
            (checkpoint_path.parent / "smoke_manifest.json").read_bytes()
        )["teacher"]["path"])
        authority = {
            "kind": "smoke",
            "format": CHECKPOINT_FORMAT,
            "path": _relative(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256_file(checkpoint_path),
            "step": int(payload["steps"]),
            "model_state_sha256": payload["model_state_sha256"],
            "ema_state_sha256": None,
            "contract_semantic_sha256": None,
            "smoke_semantic_sha256": validation["semantic_sha256"],
        }
        return model.eval(), authority, teacher

    contract_path = checkpoint_path.parent / "production_contract.json"
    if not contract_path.is_file():
        raise ValueError("cellular motion production contract is missing")
    raw_contract = contract_path.read_bytes()
    contract = json.loads(raw_contract)
    if (
        raw_contract != _canonical(contract)
        or contract.get("format") != PRODUCTION_FORMAT
        or contract.get("source_sha256") != source_sha256()
        or contract.get("semantic_sha256")
        != hashlib.sha256(_canonical({k: v for k, v in contract.items() if k != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("cellular motion evaluation production contract drifted")
    stem = checkpoint_path.stem
    if not stem.startswith("cell_motion_") or not stem[12:].isdigit():
        raise ValueError("cellular motion checkpoint filename drifted")
    step = int(stem[12:])
    payload = _load_checkpoint(checkpoint_path, contract, step)
    model = CellularMotionTransformer(_config(contract["model"]))
    model.load_state_dict(payload["ema_state"], strict=True)
    teacher = NativeMotionTeacher(PROJECT_ROOT / contract["teacher"]["path"])
    if teacher.semantic_sha256 != contract["teacher"]["semantic_sha256"]:
        raise ValueError("cellular motion evaluation teacher drifted")
    authority = {
        "kind": "production_ema",
        "format": CHECKPOINT_FORMAT,
        "path": _relative(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256_file(checkpoint_path),
        "step": step,
        "model_state_sha256": payload["model_state_sha256"],
        "ema_state_sha256": payload["ema_state_sha256"],
        "contract_semantic_sha256": contract["semantic_sha256"],
        "smoke_semantic_sha256": None,
    }
    return model.eval(), authority, teacher


def _tensor_rows(rows: list[dict[str, Any]], name: str, device: torch.device) -> Tensor:
    return torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)


def _clip_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    static: np.ndarray,
    mask: np.ndarray,
    adjacency: np.ndarray,
    *,
    loop: bool,
) -> dict[str, float]:
    active = mask.astype(bool)
    pred = predicted[:, active]
    truth = target[:, active]
    copy = baseline[:, active]
    scale = MAX_DISPLACEMENT
    position = float(np.abs(pred[:, :, :2] - truth[:, :, :2]).mean() * scale)
    velocity = float(np.abs(pred[:, :, 2:] - truth[:, :, 2:]).mean() * scale)
    copy_error = float(np.abs(copy[:, :, :2] - truth[:, :, :2]).mean() * scale)
    improvement = (copy_error - position) / max(copy_error, 1e-8)
    edge = adjacency[np.ix_(active, active)].copy()
    np.fill_diagonal(edge, False)
    source, destination = np.nonzero(edge)
    if len(source):
        pred_relative = pred[:, source, :2] - pred[:, destination, :2]
        true_relative = truth[:, source, :2] - truth[:, destination, :2]
        graph = float(np.abs(pred_relative - true_relative).mean() * scale)
    else:
        graph = 0.0
    pred_energy = float(np.sqrt(np.mean(np.square(pred[:, :, :2] * scale))))
    target_energy = float(np.sqrt(np.mean(np.square(truth[:, :, :2] * scale))))
    energy_ratio = pred_energy / max(target_energy, 1e-8)
    appendage = static[active, 50] > 0.5
    core = ~appendage
    appendage_energy = float(np.sqrt(np.mean(np.square(pred[:, appendage, :2] * scale)))) if appendage.any() else 0.0
    core_energy = float(np.sqrt(np.mean(np.square(pred[:, core, :2] * scale)))) if core.any() else 0.0
    appendage_core = appendage_energy / max(core_energy, 1e-8)
    loop_closure = float(np.abs(pred[-1, :, :2] - pred[0, :, :2]).mean() * scale) if loop else 0.0
    maximum = float(np.abs(pred[:, :, :2] * scale).max(initial=0.0))
    values = {
        "position_mae_px": position,
        "velocity_mae_px": velocity,
        "copy_previous_mae_px": copy_error,
        "baseline_improvement": improvement,
        "graph_relative_mae_px": graph,
        "predicted_energy_px": pred_energy,
        "target_energy_px": target_energy,
        "energy_ratio": energy_ratio,
        "appendage_energy_px": appendage_energy,
        "core_energy_px": core_energy,
        "appendage_core_ratio": appendage_core,
        "loop_closure_px": loop_closure,
        "max_displacement_px": maximum,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("cellular motion rollout metric became non-finite")
    return {name: round(float(values[name]), 8) for name in METRIC_NAMES}


def _aggregate(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(records)
    if not rows:
        raise ValueError("cellular motion evaluation aggregation is empty")
    return {
        name: round(sum(float(row["metrics"][name]) for row in rows) / len(rows), 8)
        for name in METRIC_NAMES
    }


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
        raise ValueError("cellular motion evaluation only permits held-out splits")
    if split == "test" and not allow_sealed_test:
        raise PermissionError("sealed cellular motion test split requires explicit release")
    if not motion_ids or tuple(sorted(set(motion_ids))) != motion_ids or any(not 0 <= value < 13 for value in motion_ids):
        raise ValueError("cellular motion evaluation motion selection drifted")
    if type(rollout_frames) is not int or not 2 <= rollout_frames <= 72:
        raise ValueError("cellular motion evaluation frame count drifted")
    model, authority, teacher = _load_authority(checkpoint_path)
    model.to(device).eval()
    chassis_ids = tuple(teacher.split_chassis(split))
    if len(chassis_ids) != 5 or sorted(int(teacher.chassis[index]["family_id"]) for index in chassis_ids) != list(range(5)):
        raise ValueError("cellular motion evaluation family coverage drifted")
    clip_records: list[dict[str, Any]] = []
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
            for frame in range(rollout_frames):
                rows = [teacher.sample(chassis, motion_id, frame) for chassis in chassis_ids]
                if frame == 0:
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
                    raise FloatingPointError("cellular motion prediction-fed rollout became non-finite")
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
                spec = teacher.manifest["motion_specs"][teacher.clips[chassis_id * 13 + motion_id]["motion"]]
                clip_records.append(
                    {
                        "chassis_id": chassis_id,
                        "family_id": int(sample["family"]),
                        "morphotype_id": int(sample["morphotype"]),
                        "motion_id": motion_id,
                        "motion": teacher.clips[chassis_id * 13 + motion_id]["motion"],
                        "cell_count": int(sample["cell_count"]),
                        "loop": bool(spec["loop"]),
                        "metrics": _clip_metrics(
                            predicted_np[row_index], target_np[row_index], baseline_np[row_index],
                            sample["static"], sample["mask"], sample["adjacency"], loop=bool(spec["loop"]),
                        ),
                    }
                )
    clip_records.sort(key=lambda row: (row["family_id"], row["motion_id"], row["chassis_id"]))
    families = [
        {"family_id": family, "clip_count": len(rows), "metrics": _aggregate(rows)}
        for family in range(5)
        for rows in [[row for row in clip_records if row["family_id"] == family]]
    ]
    motions = [
        {
            "motion_id": motion,
            "motion": rows[0]["motion"],
            "clip_count": len(rows),
            "metrics": _aggregate(rows),
        }
        for motion in motion_ids
        for rows in [[row for row in clip_records if row["motion_id"] == motion]]
    ]
    payload: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "status": "evaluated",
        "evaluation_source_sha256": evaluation_source_sha256(),
        "model_source_sha256": source_sha256(),
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
            "clip_count": len(clip_records),
            "prediction_fed": True,
            "state_teacher_forcing_after_frame_zero": False,
        },
        "thresholds": dict(PROMOTION_THRESHOLDS),
        "clips": clip_records,
        "aggregate": _aggregate(clip_records),
        "families": families,
        "motions": motions,
        "diagnostics": {"maximum_outside_abs": round(maximum_outside, 12)},
    }
    payload["gates"] = _gates(payload)
    payload["promotion_eligible"] = all(payload["gates"].values()) and authority["kind"] == "production_ema"
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
    selected = tuple(int(value) for value in motion_ids)
    payload = _build_payload(
        checkpoint_path,
        split=split,
        motion_ids=selected,
        rollout_frames=rollout_frames,
        device=torch.device(device),
        allow_sealed_test=allow_sealed_test,
    )
    payload["semantic_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    output.mkdir(parents=True)
    _atomic_bytes(output / REPORT_NAME, _canonical(payload))
    return validate_evaluation(output / REPORT_NAME, replay=False)


def _validate_relationships(report: dict[str, Any]) -> None:
    if report["thresholds"] != PROMOTION_THRESHOLDS:
        raise ValueError("cellular motion evaluation thresholds drifted")
    clips = report["clips"]
    scope = report["scope"]
    if set(scope) != {
        "split", "sealed_test_released", "motion_ids", "rollout_frames",
        "chassis_count", "clip_count", "prediction_fed",
        "state_teacher_forcing_after_frame_zero",
    } or scope["chassis_count"] != 5 or scope["prediction_fed"] is not True or scope["state_teacher_forcing_after_frame_zero"] is not False:
        raise ValueError("cellular motion evaluation scope drifted")
    clip_keys = {
        "chassis_id", "family_id", "morphotype_id", "motion_id", "motion",
        "cell_count", "loop", "metrics",
    }
    if any(set(row) != clip_keys or set(row["metrics"]) != set(METRIC_NAMES) for row in clips):
        raise ValueError("cellular motion evaluation clip structure drifted")
    if len(clips) != scope["clip_count"] or scope["clip_count"] != scope["chassis_count"] * len(scope["motion_ids"]):
        raise ValueError("cellular motion evaluation clip count drifted")
    expected_keys = {
        (family, motion)
        for family in range(5)
        for motion in scope["motion_ids"]
    }
    actual_keys = {(row["family_id"], row["motion_id"]) for row in clips}
    if actual_keys != expected_keys or len(actual_keys) != len(clips):
        raise ValueError("cellular motion evaluation matrix drifted")
    if report["aggregate"] != _aggregate(clips):
        raise ValueError("cellular motion evaluation aggregate drifted")
    expected_families = [
        {"family_id": family, "clip_count": len(rows), "metrics": _aggregate(rows)}
        for family in range(5)
        for rows in [[row for row in clips if row["family_id"] == family]]
    ]
    if report["families"] != expected_families:
        raise ValueError("cellular motion evaluation family aggregate drifted")
    expected_motions = [
        {
            "motion_id": motion,
            "motion": rows[0]["motion"],
            "clip_count": len(rows),
            "metrics": _aggregate(rows),
        }
        for motion in scope["motion_ids"]
        for rows in [[row for row in clips if row["motion_id"] == motion]]
    ]
    if report["motions"] != expected_motions or report["gates"] != _gates(report):
        raise ValueError("cellular motion evaluation derived evidence drifted")
    expected_promotion = all(report["gates"].values()) and report["checkpoint"]["kind"] == "production_ema"
    if report["promotion_eligible"] is not expected_promotion:
        raise ValueError("cellular motion evaluation promotion verdict drifted")


def validate_evaluation(report_path: Path, *, replay: bool = False) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    if report_path.is_symlink() or not report_path.is_file() or not 0 < report_path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("cellular motion evaluation report is missing or oversized")
    raw = report_path.read_bytes()
    report = json.loads(raw)
    required = {
        "format", "status", "evaluation_source_sha256", "model_source_sha256",
        "checkpoint", "teacher", "scope", "thresholds", "clips", "aggregate",
        "families", "motions", "diagnostics", "gates", "promotion_eligible",
        "semantic_sha256",
    }
    if raw != _canonical(report) or set(report) != required:
        raise ValueError("cellular motion evaluation report structure drifted")
    errors = sorted(Draft202012Validator(json.loads(EVALUATION_SCHEMA.read_bytes())).iter_errors(report), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"cellular motion evaluation schema drifted: {errors[0].message}")
    if (
        report["format"] != EVALUATION_FORMAT
        or report["status"] != "evaluated"
        or report["evaluation_source_sha256"] != evaluation_source_sha256()
        or report["model_source_sha256"] != source_sha256()
        or report["semantic_sha256"]
        != hashlib.sha256(_canonical({k: v for k, v in report.items() if k != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("cellular motion evaluation authority drifted")
    _validate_relationships(report)
    checkpoint_path = PROJECT_ROOT / report["checkpoint"]["path"]
    model, authority, teacher = _load_authority(checkpoint_path)
    del model
    if authority != report["checkpoint"] or teacher.semantic_sha256 != report["teacher"]["semantic_sha256"]:
        raise ValueError("cellular motion evaluation provenance drifted")
    if replay:
        replayed = _build_payload(
            checkpoint_path,
            split=report["scope"]["split"],
            motion_ids=tuple(report["scope"]["motion_ids"]),
            rollout_frames=report["scope"]["rollout_frames"],
            device=torch.device("cpu"),
            allow_sealed_test=report["scope"]["sealed_test_released"],
        )
        if replayed != {k: v for k, v in report.items() if k != "semantic_sha256"}:
            raise ValueError("cellular motion evaluation exact replay drifted")
    return {
        "passed": True,
        "promotion_eligible": report["promotion_eligible"],
        "checkpoint_kind": report["checkpoint"]["kind"],
        "split": report["scope"]["split"],
        "clips": report["scope"]["clip_count"],
        "frames_per_clip": report["scope"]["rollout_frames"],
        "semantic_sha256": report["semantic_sha256"],
        "gates": report["gates"],
    }
