from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import CELL_POSITION_BOUND, DEFAULT_TEACHER, FLUID_SCALAR_BOUND, FLUID_SLOTS, source_sha256
from .dataset import NativeInterventionTeacher
from .model import CellularPhysiologyTransformer
from .training import _atomic_bytes, _canonical, _config, _sha256_file, validate_cpu_smoke


EVALUATION_FORMAT = "nullvector-creature-stage-neural-physiology-evaluation-v1"
REPORT_NAME = "evaluation_manifest.json"
SCHEMA_PATH = PROJECT_ROOT / "shared/schema/creature_stage_neural_physiology_evaluation.schema.json"
SOURCE_FILES = (
    "forge/creature_stage_neural_physiology/evaluation.py",
    "shared/schema/creature_stage_neural_physiology_evaluation.schema.json",
)
MAX_REPORT_BYTES = 8 * 1024 * 1024
METRIC_NAMES = (
    "position_mae_px", "health_mae", "alive_accuracy", "summary_mae",
    "copy_health_mae", "baseline_improvement", "fluid_presence_f1",
    "fluid_count_mae", "fluid_value_mae", "minimum_integrity",
    "maximum_death", "target_capacity_after_event", "off_target_capacity_mae",
    "healing_health_gain", "peak_fluid_count",
)
THRESHOLDS = {
    "position_mae_px_max": 0.75,
    "health_mae_max": 0.08,
    "alive_accuracy_min": 0.98,
    "summary_mae_max": 0.08,
    "baseline_improvement_min": 0.10,
    "fluid_presence_f1_min": 0.75,
    "fluid_count_mae_max": 4.0,
    "fluid_value_mae_max": 0.10,
    "control_minimum_integrity_min": 0.95,
    "control_maximum_death_max": 0.05,
    "ablation_target_capacity_max": 0.15,
    "ablation_off_target_mae_max": 0.15,
    "healing_health_gain_min": 0.001,
    "noncontrol_peak_fluid_count_min": 1.0,
}
TARGET_CAPACITY = {4: 1, 5: 2, 6: 3, 7: 4, 8: 5}


def evaluation_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-physiology-evaluation-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _load_authority(checkpoint_path: Path) -> tuple[CellularPhysiologyTransformer, dict[str, Any], NativeInterventionTeacher]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.name != "smoke_checkpoint.pt":
        raise ValueError("production physiology checkpoint authority is not published yet")
    validation = validate_cpu_smoke(checkpoint_path.parent)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CellularPhysiologyTransformer(_config(payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    manifest = json.loads((checkpoint_path.parent / "smoke_manifest.json").read_bytes())
    teacher = NativeInterventionTeacher(PROJECT_ROOT / manifest["teacher"]["path"])
    authority = {
        "kind": "smoke",
        "path": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256_file(checkpoint_path),
        "step": int(payload["steps"]),
        "model_state_sha256": payload["model_state_sha256"],
        "smoke_semantic_sha256": validation["semantic_sha256"],
    }
    return model.eval(), authority, teacher


def _tensor(rows: list[dict[str, Any]], name: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)


def _metrics(
    cell: np.ndarray,
    summary: np.ndarray,
    fluid: np.ndarray,
    cell_target: np.ndarray,
    summary_target: np.ndarray,
    fluid_target: np.ndarray,
    cell_baseline: np.ndarray,
    mask: np.ndarray,
    intervention: int,
) -> dict[str, float]:
    active = mask.astype(bool)
    predicted_cells = cell[:, active]
    target_cells = cell_target[:, active]
    baseline_cells = cell_baseline[:, active]
    position = float(np.abs(predicted_cells[:, :, :2] - target_cells[:, :, :2]).mean() * CELL_POSITION_BOUND)
    health = float(np.abs(predicted_cells[:, :, 2] - target_cells[:, :, 2]).mean())
    copy_health = float(np.abs(baseline_cells[:, :, 2] - target_cells[:, :, 2]).mean())
    alive_accuracy = float(((predicted_cells[:, :, 3] >= 0.5) == (target_cells[:, :, 3] >= 0.5)).mean())
    summary_mae = float(np.abs(summary - summary_target).mean())
    predicted_presence = fluid[:, :, 6] >= 0.5
    target_presence = fluid_target[:, :, 6] >= 0.5
    true_positive = int(np.logical_and(predicted_presence, target_presence).sum())
    false_positive = int(np.logical_and(predicted_presence, ~target_presence).sum())
    false_negative = int(np.logical_and(~predicted_presence, target_presence).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    count_mae = float(np.abs(predicted_presence.sum(axis=1) - target_presence.sum(axis=1)).mean())
    if target_presence.any():
        scales = np.asarray([CELL_POSITION_BOUND] * 4 + [FLUID_SCALAR_BOUND] * 2, dtype=np.float32)
        fluid_value = float((np.abs(fluid[:, :, :6] - fluid_target[:, :, :6]) * scales)[target_presence].mean())
    else:
        fluid_value = 0.0
    event_frame = min(16, len(summary) - 1)
    target_index = TARGET_CAPACITY.get(intervention)
    target_capacity = float(summary[event_frame, target_index]) if target_index is not None else 1.0
    if target_index is not None:
        indices = [index for index in range(1, 6) if index != target_index]
        off_target = float(np.abs(summary[event_frame, indices] - summary_target[event_frame, indices]).mean())
    else:
        off_target = 0.0
    healing_gain = 0.0
    if intervention == 2 and len(cell) > 76:
        healing_gain = float(predicted_cells[76, :, 2].mean() - predicted_cells[74, :, 2].mean())
    values = {
        "position_mae_px": position,
        "health_mae": health,
        "alive_accuracy": alive_accuracy,
        "summary_mae": summary_mae,
        "copy_health_mae": copy_health,
        "baseline_improvement": (copy_health - health) / max(copy_health, 1e-8),
        "fluid_presence_f1": f1,
        "fluid_count_mae": count_mae,
        "fluid_value_mae": fluid_value,
        "minimum_integrity": float(summary[:, 0].min()),
        "maximum_death": float(summary[:, 8].max()),
        "target_capacity_after_event": target_capacity,
        "off_target_capacity_mae": off_target,
        "healing_health_gain": healing_gain,
        "peak_fluid_count": float(predicted_presence.sum(axis=1).max(initial=0)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("cellular physiology rollout metric became non-finite")
    return {name: round(values[name], 8) for name in METRIC_NAMES}


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    records = list(rows)
    if not records:
        raise ValueError("cellular physiology evaluation aggregation is empty")
    return {
        name: round(sum(float(row["metrics"][name]) for row in records) / len(records), 8)
        for name in METRIC_NAMES
    }


def _gates(payload: dict[str, Any]) -> dict[str, bool]:
    clips, aggregate, threshold = payload["clips"], payload["aggregate"], payload["thresholds"]
    controls = [row for row in clips if row["intervention_id"] == 0]
    ablations = [row for row in clips if row["intervention_id"] in TARGET_CAPACITY]
    healing = [row for row in clips if row["intervention_id"] == 2]
    noncontrol = [row for row in clips if row["intervention_id"] != 0]
    return {
        "full_validation_matrix": (
            payload["scope"]["intervention_ids"] == list(range(9))
            and payload["scope"]["rollout_frames"] == 180 and len(clips) == 45
        ),
        "all_values_finite": all(math.isfinite(row["metrics"][name]) for row in clips for name in METRIC_NAMES),
        "outside_cells_exact_zero": payload["diagnostics"]["maximum_outside_abs"] == 0.0,
        "position_accuracy": aggregate["position_mae_px"] <= threshold["position_mae_px_max"],
        "health_accuracy": aggregate["health_mae"] <= threshold["health_mae_max"],
        "viability_accuracy": aggregate["alive_accuracy"] >= threshold["alive_accuracy_min"],
        "organ_summary_accuracy": aggregate["summary_mae"] <= threshold["summary_mae_max"],
        "beats_copy_health": aggregate["baseline_improvement"] >= threshold["baseline_improvement_min"],
        "fluid_presence": aggregate["fluid_presence_f1"] >= threshold["fluid_presence_f1_min"],
        "fluid_count": aggregate["fluid_count_mae"] <= threshold["fluid_count_mae_max"],
        "fluid_values": aggregate["fluid_value_mae"] <= threshold["fluid_value_mae_max"],
        "control_stability": (
            bool(controls)
            and min(row["metrics"]["minimum_integrity"] for row in controls) >= threshold["control_minimum_integrity_min"]
            and max(row["metrics"]["maximum_death"] for row in controls) <= threshold["control_maximum_death_max"]
        ),
        "targeted_ablation_causal": bool(ablations) and max(row["metrics"]["target_capacity_after_event"] for row in ablations) <= threshold["ablation_target_capacity_max"],
        "off_target_organs_preserved": bool(ablations) and max(row["metrics"]["off_target_capacity_mae"] for row in ablations) <= threshold["ablation_off_target_mae_max"],
        "healing_response": bool(healing) and min(row["metrics"]["healing_health_gain"] for row in healing) >= threshold["healing_health_gain_min"],
        "fluid_emission_response": bool(noncontrol) and min(row["metrics"]["peak_fluid_count"] for row in noncontrol) >= threshold["noncontrol_peak_fluid_count_min"],
    }


def _build_payload(
    checkpoint_path: Path,
    *,
    intervention_ids: tuple[int, ...],
    rollout_frames: int,
    device: torch.device,
) -> dict[str, Any]:
    if not intervention_ids or tuple(sorted(set(intervention_ids))) != intervention_ids or any(not 0 <= value < 9 for value in intervention_ids):
        raise ValueError("cellular physiology intervention selection drifted")
    if type(rollout_frames) is not int or not 2 <= rollout_frames <= 180:
        raise ValueError("cellular physiology rollout length drifted")
    model, authority, teacher = _load_authority(checkpoint_path)
    model.to(device).eval()
    chassis_ids = teacher.split_chassis("validation")
    clip_records: list[dict[str, Any]] = []
    maximum_outside = 0.0
    with torch.inference_mode():
        for intervention_id in intervention_ids:
            cell_frames, summary_frames, fluid_frames = [], [], []
            cell_targets, summary_targets, fluid_targets, baselines = [], [], [], []
            cell_state = summary_state = fluid_state = static = mask = adjacency = None
            for frame in range(rollout_frames):
                rows = [teacher.sample(chassis, intervention_id, frame) for chassis in chassis_ids]
                if frame == 0:
                    static = _tensor(rows, "static", device)
                    mask = _tensor(rows, "mask", device)
                    adjacency = _tensor(rows, "adjacency", device)
                    cell_state = _tensor(rows, "cell_state", device)
                    summary_state = _tensor(rows, "summary_state", device)
                    fluid_state = _tensor(rows, "fluid_state", device)
                assert all(value is not None for value in (static, mask, adjacency, cell_state, summary_state, fluid_state))
                cell_target = _tensor(rows, "cell_target", device)
                summary_target = _tensor(rows, "summary_target", device)
                fluid_target = _tensor(rows, "fluid_target", device)
                family = torch.tensor([int(row["family"]) for row in rows], dtype=torch.long, device=device)
                morphotype = torch.tensor([int(row["morphotype"]) for row in rows], dtype=torch.long, device=device)
                intervention = torch.full((5,), intervention_id, dtype=torch.long, device=device)
                phase = torch.tensor([float(row["phase"]) for row in rows], dtype=torch.float32, device=device)
                events = _tensor(rows, "events", device)
                predicted = model(
                    static, cell_state, summary_state, fluid_state, mask, adjacency,
                    family, morphotype, intervention, phase, events,
                )
                if not all(bool(torch.isfinite(value).all()) for value in predicted):
                    raise FloatingPointError("cellular physiology prediction-fed rollout became non-finite")
                maximum_outside = max(maximum_outside, float(predicted[0][~mask].abs().max()))
                cell_frames.append(predicted[0].cpu().numpy())
                summary_frames.append(predicted[1].cpu().numpy())
                fluid_frames.append(predicted[2].cpu().numpy())
                cell_targets.append(cell_target.cpu().numpy())
                summary_targets.append(summary_target.cpu().numpy())
                fluid_targets.append(fluid_target.cpu().numpy())
                baselines.append(cell_state.cpu().numpy())
                cell_state, summary_state, fluid_state = (value.detach() for value in predicted)
            arrays = [np.stack(values, axis=1) for values in (
                cell_frames, summary_frames, fluid_frames, cell_targets,
                summary_targets, fluid_targets, baselines,
            )]
            for index, chassis_id in enumerate(chassis_ids):
                sample = teacher.sample(chassis_id, intervention_id, 0)
                clip_records.append(
                    {
                        "chassis_id": chassis_id,
                        "family_id": int(sample["family"]),
                        "morphotype_id": int(sample["morphotype"]),
                        "intervention_id": intervention_id,
                        "intervention": teacher.clips[chassis_id * 9 + intervention_id]["intervention"],
                        "cell_count": int(sample["cell_count"]),
                        "metrics": _metrics(
                            arrays[0][index], arrays[1][index], arrays[2][index],
                            arrays[3][index], arrays[4][index], arrays[5][index],
                            arrays[6][index], sample["mask"], intervention_id,
                        ),
                    }
                )
    clip_records.sort(key=lambda row: (row["family_id"], row["intervention_id"]))
    families = [
        {"family_id": family, "clip_count": len(rows), "metrics": _aggregate(rows)}
        for family in range(5)
        for rows in [[row for row in clip_records if row["family_id"] == family]]
    ]
    interventions = [
        {"intervention_id": intervention, "intervention": rows[0]["intervention"], "clip_count": len(rows), "metrics": _aggregate(rows)}
        for intervention in intervention_ids
        for rows in [[row for row in clip_records if row["intervention_id"] == intervention]]
    ]
    payload: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "status": "evaluated",
        "evaluation_source_sha256": evaluation_source_sha256(),
        "model_source_sha256": source_sha256(),
        "checkpoint": authority,
        "teacher": {
            "path": teacher.root.relative_to(PROJECT_ROOT).as_posix(),
            "semantic_sha256": teacher.semantic_sha256,
            "manifest_sha256": teacher.validation["manifest_sha256"],
            "binary_sha256": teacher.validation["binary_sha256"],
        },
        "scope": {
            "split": "validation", "intervention_ids": list(intervention_ids),
            "rollout_frames": rollout_frames, "chassis_count": 5,
            "clip_count": len(clip_records), "prediction_fed": True,
        },
        "thresholds": dict(THRESHOLDS),
        "clips": clip_records,
        "aggregate": _aggregate(clip_records),
        "families": families,
        "interventions": interventions,
        "diagnostics": {"maximum_outside_abs": round(maximum_outside, 12)},
    }
    payload["gates"] = _gates(payload)
    payload["promotion_eligible"] = all(payload["gates"].values()) and authority["kind"] == "production_ema"
    return payload


def evaluate_checkpoint(
    checkpoint_path: Path,
    output: Path,
    *,
    intervention_ids: Iterable[int] = range(9),
    rollout_frames: int = 180,
    device: str = "cpu",
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=64 * 1024**2)
    payload = _build_payload(
        checkpoint_path, intervention_ids=tuple(int(value) for value in intervention_ids),
        rollout_frames=rollout_frames, device=torch.device(device),
    )
    payload["semantic_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    output.mkdir(parents=True)
    _atomic_bytes(output / REPORT_NAME, _canonical(payload))
    return validate_evaluation(output / REPORT_NAME, replay=False)


def _validate_derived(payload: dict[str, Any]) -> None:
    clips = payload["clips"]
    if payload["thresholds"] != THRESHOLDS or payload["aggregate"] != _aggregate(clips):
        raise ValueError("cellular physiology evaluation aggregate drifted")
    expected = {(family, intervention) for family in range(5) for intervention in payload["scope"]["intervention_ids"]}
    if {(row["family_id"], row["intervention_id"]) for row in clips} != expected or len(clips) != len(expected):
        raise ValueError("cellular physiology evaluation matrix drifted")
    families = [
        {"family_id": family, "clip_count": len(rows), "metrics": _aggregate(rows)}
        for family in range(5)
        for rows in [[row for row in clips if row["family_id"] == family]]
    ]
    interventions = [
        {"intervention_id": intervention, "intervention": rows[0]["intervention"], "clip_count": len(rows), "metrics": _aggregate(rows)}
        for intervention in payload["scope"]["intervention_ids"]
        for rows in [[row for row in clips if row["intervention_id"] == intervention]]
    ]
    if payload["families"] != families or payload["interventions"] != interventions or payload["gates"] != _gates(payload):
        raise ValueError("cellular physiology evaluation derived evidence drifted")
    if payload["promotion_eligible"] is not (all(payload["gates"].values()) and payload["checkpoint"]["kind"] == "production_ema"):
        raise ValueError("cellular physiology evaluation promotion verdict drifted")


def validate_evaluation(report_path: Path, *, replay: bool = False) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    if report_path.is_symlink() or not report_path.is_file() or not 0 < report_path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("cellular physiology evaluation report is missing or oversized")
    raw = report_path.read_bytes()
    payload = json.loads(raw)
    errors = sorted(Draft202012Validator(json.loads(SCHEMA_PATH.read_bytes())).iter_errors(payload), key=lambda error: list(error.path))
    if raw != _canonical(payload) or errors:
        detail = errors[0].message if errors else "noncanonical JSON"
        raise ValueError(f"cellular physiology evaluation structure drifted: {detail}")
    if (
        payload["format"] != EVALUATION_FORMAT or payload["status"] != "evaluated"
        or payload["evaluation_source_sha256"] != evaluation_source_sha256()
        or payload["model_source_sha256"] != source_sha256()
        or payload["semantic_sha256"]
        != hashlib.sha256(_canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("cellular physiology evaluation authority drifted")
    _validate_derived(payload)
    checkpoint_path = PROJECT_ROOT / payload["checkpoint"]["path"]
    _, authority, teacher = _load_authority(checkpoint_path)
    if authority != payload["checkpoint"] or teacher.semantic_sha256 != payload["teacher"]["semantic_sha256"]:
        raise ValueError("cellular physiology evaluation provenance drifted")
    if replay:
        expected = _build_payload(
            checkpoint_path,
            intervention_ids=tuple(payload["scope"]["intervention_ids"]),
            rollout_frames=payload["scope"]["rollout_frames"],
            device=torch.device("cpu"),
        )
        if expected != {key: value for key, value in payload.items() if key != "semantic_sha256"}:
            raise ValueError("cellular physiology evaluation exact replay drifted")
    return {
        "passed": True,
        "promotion_eligible": payload["promotion_eligible"],
        "checkpoint_kind": payload["checkpoint"]["kind"],
        "clips": payload["scope"]["clip_count"],
        "frames_per_clip": payload["scope"]["rollout_frames"],
        "semantic_sha256": payload["semantic_sha256"],
        "gates": payload["gates"],
    }
