from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import DEFAULT_TEACHER, MAX_DISPLACEMENT, CellularMotionTransformerConfig
from ..creature_stage_neural_motion.dataset import NativeMotionTeacher
from ..creature_stage_neural_motion.model import CellularMotionTransformer
from ..creature_stage_neural_motion.training import (
    _atomic_bytes,
    _atomic_torch,
    _canonical,
    _config,
    _sha256_file,
    _state_sha256,
)
from ..creature_stage_neural_motion_rollout.contract import source_sha256 as parent_rollout_source_sha256
from ..creature_stage_neural_motion_rollout.contract import RolloutTrainingConfig
from ..creature_stage_neural_motion_rollout.training import _rollout
from .contract import LoopTrainingConfig, source_sha256
from .sampler import LoopAwareRolloutBatchSampler, SequenceCoordinate


SMOKE_FORMAT = "nullvector-creature-stage-neural-motion-loop-smoke-v1"
CHECKPOINT_FORMAT = "nullvector-creature-stage-neural-motion-loop-checkpoint-v1"
SCHEMA = PROJECT_ROOT / "shared/schema/creature_stage_neural_motion_loop_smoke.schema.json"
SEED = 0x4C4F4F50534D4B31
MAX_CHECKPOINT_BYTES = 512 * 1024**2
HISTORY_KEYS = {
    "step", "loss", "position", "velocity", "graph", "appendage", "delta",
    "energy", "outside", "gradient_norm", "lr",
}


def _base_config(config: LoopTrainingConfig) -> RolloutTrainingConfig:
    return RolloutTrainingConfig(
        sequence_frames=config.sequence_frames,
        appendage_weight=config.appendage_weight,
        energy_weight=config.energy_weight,
        delta_weight=config.delta_weight,
        velocity_weight=config.velocity_weight,
        graph_weight=config.graph_weight,
        minimum_energy_epsilon=config.minimum_energy_epsilon,
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("loop motion smoke path must remain inside the project") from error


def _diagnostics(
    frames: list[dict[str, torch.Tensor]],
    coordinates: list[SequenceCoordinate],
    predictions: list[torch.Tensor],
    sampler: LoopAwareRolloutBatchSampler,
) -> dict[str, float | int]:
    predicted = torch.stack(predictions, dim=1)
    target = torch.stack([frame["target"] for frame in frames], dim=1)
    mask = frames[0]["mask"]
    active = mask[:, None, :, None]
    outside = float(predicted.masked_select(~active.expand_as(predicted)).abs().max())
    appendage = (frames[0]["static"][:, :, 50] > 0.5) & mask
    appendage_values = predicted[:, :, :, :2].masked_select(
        appendage[:, None, :, None]
    ).reshape(-1, 2)
    appendage_motion = float(appendage_values.square().mean().sqrt() * MAX_DISPLACEMENT)
    predicted_values = predicted[:, :, :, :2].masked_select(
        active.expand_as(predicted[:, :, :, :2])
    ).reshape(-1, 2)
    target_values = target[:, :, :, :2].masked_select(
        active.expand_as(target[:, :, :, :2])
    ).reshape(-1, 2)
    predicted_energy = float(predicted_values.square().mean().sqrt() * MAX_DISPLACEMENT)
    target_energy = float(target_values.square().mean().sqrt() * MAX_DISPLACEMENT)
    seam_errors: list[torch.Tensor] = []
    seam_transitions = 0
    for row, coordinate in enumerate(coordinates):
        indices = sampler.frame_indices(coordinate)
        for offset in range(1, len(indices)):
            if indices[offset - 1] == 71 and indices[offset] == 0:
                seam_transitions += 1
                active_row = mask[row]
                seam_errors.append(
                    (predicted[row, offset, active_row, :2] - target[row, offset, active_row, :2]).abs()
                )
    if not seam_errors:
        raise ValueError("loop motion smoke did not cross a cyclic seam")
    seam_mae = float(torch.cat([value.reshape(-1) for value in seam_errors]).mean() * MAX_DISPLACEMENT)
    return {
        "families": len(set(int(value) for value in frames[0]["family"].tolist())),
        "frames": len(frames),
        "prediction_fed_frames": len(frames) - 1,
        "forced_seam_sequences": sum(int(coordinate.forced_seam) for coordinate in coordinates),
        "seam_transitions": seam_transitions,
        "seam_position_mae_px": round(seam_mae, 9),
        "outside_max_abs": round(outside, 12),
        "appendage_motion_px": round(appendage_motion, 9),
        "predicted_energy_px": round(predicted_energy, 9),
        "target_energy_px": round(target_energy, 9),
        "energy_ratio": round(predicted_energy / max(target_energy, 1e-8), 9),
    }


def run_cpu_smoke(output: Path, *, steps: int = 8) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(steps) is not int or not 2 <= steps <= 16:
        raise ValueError("loop motion smoke step count drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=512 * 1024**2)
    teacher = NativeMotionTeacher(DEFAULT_TEACHER)
    config = LoopTrainingConfig()
    loss_config = _base_config(config)
    model_config = CellularMotionTransformerConfig(
        width=64, depth=2, heads=4, feedforward_multiplier=3,
        condition_width=128, dropout=0.0,
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFFFFFF)
    model = CellularMotionTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    sampler = LoopAwareRolloutBatchSampler(teacher, config=config)
    frames, coordinates = sampler.sequence(0, force_seam=True)
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pieces, _ = _rollout(model, frames, loss_config, backward=True)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gradient)) or float(gradient) <= 0.0:
            raise FloatingPointError("loop motion smoke gradient drifted")
        optimizer.step()
        history.append({
            "step": step + 1,
            **{name: round(float(value), 9) for name, value in pieces.items()},
            "gradient_norm": round(float(gradient), 9),
            "lr": 0.0008,
        })
    model.eval()
    with torch.inference_mode():
        _, predictions = _rollout(model, frames, loss_config, backward=False)
    diagnostics = _diagnostics(frames, coordinates, predictions, sampler)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source_sha256(),
        "parent_rollout_source_sha256": parent_rollout_source_sha256(),
        "teacher_semantic_sha256": teacher.semantic_sha256,
        "model": model_config.to_dict(),
        "training": config.to_dict(),
        "steps": steps,
        "model_state": state,
        "model_state_sha256": _state_sha256(state),
        "history": history,
    }
    output.mkdir(parents=True)
    checkpoint_path = output / "smoke_checkpoint.pt"
    _atomic_torch(checkpoint_path, checkpoint)
    gates = {
        "all_values_finite": all(
            math.isfinite(float(value)) for row in history for key, value in row.items() if key != "step"
        ),
        "all_five_families": diagnostics["families"] == 5,
        "cyclic_seam_crossed": diagnostics["seam_transitions"] == 5,
        "prediction_fed_after_frame_zero": diagnostics["prediction_fed_frames"] == config.sequence_frames - 1,
        "fixed_sequence_loss_improved": float(history[-1]["loss"]) < float(history[0]["loss"]),
        "outside_cells_exact_zero": diagnostics["outside_max_abs"] == 0.0,
        "appendage_motion_nonzero": diagnostics["appendage_motion_px"] > 0.01,
        "gradient_nonzero": all(float(row["gradient_norm"]) > 0.0 for row in history),
    }
    if not all(gates.values()):
        raise ValueError(f"loop motion smoke failed: {gates}")
    report: dict[str, Any] = {
        "format": SMOKE_FORMAT,
        "status": "passed",
        "source_sha256": source_sha256(),
        "parent_rollout_source_sha256": parent_rollout_source_sha256(),
        "teacher": {
            "path": _relative(teacher.root),
            "semantic_sha256": teacher.semantic_sha256,
            "manifest_sha256": teacher.validation["manifest_sha256"],
            "binary_sha256": teacher.validation["binary_sha256"],
        },
        "model": {"config": model_config.to_dict(), "parameters": model.parameter_count},
        "training": config.to_dict(),
        "steps": steps,
        "history": history,
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256_file(checkpoint_path),
            "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "diagnostics": diagnostics,
        "gates": gates,
    }
    report["semantic_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    _atomic_bytes(output / "smoke_manifest.json", _canonical(report))
    return validate_cpu_smoke(output, replay=True)


def validate_cpu_smoke(output: Path, *, replay: bool = True) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / "smoke_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= 2 * 1024**2:
        raise ValueError("loop motion smoke manifest is missing or oversized")
    raw = manifest_path.read_bytes()
    report = json.loads(raw)
    required = {
        "format", "status", "source_sha256", "parent_rollout_source_sha256",
        "teacher", "model", "training", "steps", "history", "checkpoint",
        "diagnostics", "gates", "semantic_sha256",
    }
    if raw != _canonical(report) or set(report) != required:
        raise ValueError("loop motion smoke structure drifted")
    errors = sorted(
        Draft202012Validator(json.loads(SCHEMA.read_bytes())).iter_errors(report),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"loop motion smoke schema drifted: {errors[0].message}")
    if (
        report["source_sha256"] != source_sha256()
        or report["parent_rollout_source_sha256"] != parent_rollout_source_sha256()
        or report["training"] != LoopTrainingConfig().to_dict()
        or report["semantic_sha256"]
        != hashlib.sha256(_canonical({key: value for key, value in report.items() if key != "semantic_sha256"})).hexdigest()
        or any(set(row) != HISTORY_KEYS or row["step"] != index for index, row in enumerate(report["history"], 1))
        or not all(report["gates"].values())
    ):
        raise ValueError("loop motion smoke authority drifted")
    teacher = NativeMotionTeacher(PROJECT_ROOT / report["teacher"]["path"])
    if teacher.semantic_sha256 != report["teacher"]["semantic_sha256"]:
        raise ValueError("loop motion smoke teacher drifted")
    checkpoint_path = output / report["checkpoint"]["path"]
    if (
        checkpoint_path.is_symlink() or not checkpoint_path.is_file()
        or not 0 < checkpoint_path.stat().st_size <= MAX_CHECKPOINT_BYTES
        or checkpoint_path.stat().st_size != report["checkpoint"]["bytes"]
        or _sha256_file(checkpoint_path) != report["checkpoint"]["sha256"]
    ):
        raise ValueError("loop motion smoke checkpoint bytes drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_keys = {
        "format", "source_sha256", "parent_rollout_source_sha256", "teacher_semantic_sha256",
        "model", "training", "steps", "model_state", "model_state_sha256", "history",
    }
    if set(checkpoint) != checkpoint_keys:
        raise ValueError("loop motion smoke checkpoint registry drifted")
    model = CellularMotionTransformer(_config(checkpoint["model"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if (
        checkpoint["format"] != CHECKPOINT_FORMAT
        or checkpoint["source_sha256"] != source_sha256()
        or checkpoint["parent_rollout_source_sha256"] != parent_rollout_source_sha256()
        or checkpoint["teacher_semantic_sha256"] != teacher.semantic_sha256
        or checkpoint["training"] != report["training"]
        or checkpoint["steps"] != report["steps"]
        or checkpoint["history"] != report["history"]
        or _state_sha256(model.state_dict()) != checkpoint["model_state_sha256"]
        or checkpoint["model_state_sha256"] != report["checkpoint"]["model_state_sha256"]
        or model.parameter_count != report["model"]["parameters"]
    ):
        raise ValueError("loop motion smoke checkpoint semantics drifted")
    if replay:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        config = LoopTrainingConfig(**report["training"])
        sampler = LoopAwareRolloutBatchSampler(teacher, config=config)
        frames, coordinates = sampler.sequence(0, force_seam=True)
        model.eval()
        with torch.inference_mode():
            _, predictions = _rollout(model, frames, _base_config(config), backward=False)
        if _diagnostics(frames, coordinates, predictions, sampler) != report["diagnostics"]:
            raise ValueError("loop motion smoke deterministic replay drifted")
    return {
        "passed": True,
        "steps": report["steps"],
        "parameters": report["model"]["parameters"],
        "model_state_sha256": checkpoint["model_state_sha256"],
        "semantic_sha256": report["semantic_sha256"],
        "diagnostics": report["diagnostics"],
    }
