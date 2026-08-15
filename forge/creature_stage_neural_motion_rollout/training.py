from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import (
    MAX_DISPLACEMENT,
    CellularMotionTransformerConfig,
    source_sha256 as parent_source_sha256,
)
from ..creature_stage_neural_motion.dataset import NativeMotionTeacher
from ..creature_stage_neural_motion.model import CellularMotionTransformer
from ..creature_stage_neural_motion.training import (
    PRODUCTION_FORMAT as PARENT_PRODUCTION_FORMAT,
    _atomic_bytes,
    _atomic_torch,
    _canonical,
    _config,
    _load_checkpoint as _load_parent_checkpoint,
    _sha256_file,
    _state_sha256,
)
from .contract import (
    CHECKPOINT_FORMAT,
    DEFAULT_OUTPUT,
    DEFAULT_PARENT,
    FORMAT,
    RolloutTrainingConfig,
    source_sha256,
)


SMOKE_FORMAT = "nullvector-creature-stage-neural-motion-rollout-smoke-v1"
PRODUCTION_FORMAT = "nullvector-creature-stage-neural-motion-rollout-production-v1"
SMOKE_SCHEMA = PROJECT_ROOT / "shared/schema/creature_stage_neural_motion_rollout_smoke.schema.json"
SEED = 0x524F4C4C4F555431
MAX_CHECKPOINT_BYTES = 2 * 1024**3
HISTORY_KEYS = {
    "step", "loss", "position", "velocity", "graph", "appendage", "delta",
    "energy", "outside", "gradient_norm", "lr",
}


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("rollout motion path must remain inside the project") from error


def _mix64(value: int) -> int:
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


class RolloutBatchSampler:
    """Pure-coordinate, family-balanced batches of consecutive teacher frames."""

    def __init__(
        self,
        teacher: NativeMotionTeacher,
        *,
        batch_size: int = 5,
        sequence_frames: int = 4,
        seed: int = SEED,
    ) -> None:
        if type(batch_size) is not int or not 5 <= batch_size <= 30 or batch_size % 5:
            raise ValueError("rollout motion batch must be family balanced")
        if type(sequence_frames) is not int or not 2 <= sequence_frames <= 12:
            raise ValueError("rollout motion sequence length drifted")
        self.teacher = teacher
        self.batch_size = batch_size
        self.sequence_frames = sequence_frames
        self.seed = seed
        self.by_family = {family: teacher.split_chassis("train", family) for family in range(5)}
        if [len(self.by_family[family]) for family in range(5)] != [2] * 5:
            raise ValueError("rollout motion training split drifted")

    def coordinates(self, step: int) -> list[tuple[int, int, int]]:
        if type(step) is not int or step < 0:
            raise ValueError("rollout motion sampler step drifted")
        maximum_start = 72 - self.sequence_frames
        result: list[tuple[int, int, int]] = []
        for slot in range(self.batch_size):
            family = slot % 5
            token = _mix64(self.seed ^ (step * 0xD1342543DE82EF95) ^ (slot * 0xA24BAED4963EE407))
            chassis = self.by_family[family][token % 2]
            motion = _mix64(token ^ 0xC6BC279692B5CC83) % 13
            start = _mix64(token ^ 0xDB4F0B9175AE2165) % (maximum_start + 1)
            result.append((int(chassis), int(motion), int(start)))
        return result

    def sequence(self, step: int, device: str | torch.device = "cpu") -> list[dict[str, Tensor]]:
        coordinates = self.coordinates(step)
        frames: list[dict[str, Tensor]] = []
        for offset in range(self.sequence_frames):
            rows = [self.teacher.sample(chassis, motion, start + offset) for chassis, motion, start in coordinates]
            frame: dict[str, Tensor] = {
                name: torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)
                for name in ("static", "state", "target", "controls")
            }
            frame["mask"] = torch.from_numpy(np.stack([row["mask"] for row in rows]).copy()).to(device)
            frame["adjacency"] = torch.from_numpy(np.stack([row["adjacency"] for row in rows]).copy()).to(device)
            for name in ("family", "morphotype", "motion"):
                frame[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
            frame["phase"] = torch.tensor([float(row["phase"]) for row in rows], dtype=torch.float32, device=device)
            frames.append(frame)
        return frames


def _masked_weighted_mean(value: Tensor, mask: Tensor, weight: Tensor | None = None) -> Tensor:
    active = mask[:, :, None].to(value.dtype)
    if weight is not None:
        active = active * weight[:, :, None].to(value.dtype)
    return (value * active).sum() / (active.sum().clamp_min(1.0) * value.shape[-1])


def rollout_frame_loss(
    predicted: Tensor,
    target: Tensor,
    input_state: Tensor,
    teacher_state: Tensor,
    static: Tensor,
    mask: Tensor,
    adjacency: Tensor,
    config: RolloutTrainingConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    if predicted.shape != target.shape or predicted.shape != input_state.shape or predicted.shape != teacher_state.shape:
        raise ValueError("rollout motion loss tensor shape drifted")
    if predicted.shape[-1] != 4 or mask.shape != predicted.shape[:2] or static.shape[:2] != mask.shape:
        raise ValueError("rollout motion loss interface drifted")
    position_error = F.smooth_l1_loss(predicted[:, :, :2], target[:, :, :2], reduction="none")
    velocity_error = F.smooth_l1_loss(predicted[:, :, 2:], target[:, :, 2:], reduction="none")
    position = _masked_weighted_mean(position_error, mask)
    velocity = _masked_weighted_mean(velocity_error, mask)
    appendage_mask = (static[:, :, 50] > 0.5) & mask
    appendage = _masked_weighted_mean(position_error, appendage_mask)

    adjacency_float = adjacency.to(predicted.dtype)
    degree = adjacency_float.sum(dim=2, keepdim=True).clamp_min(1.0)
    predicted_neighbor = torch.bmm(adjacency_float, predicted[:, :, :2]) / degree
    target_neighbor = torch.bmm(adjacency_float, target[:, :, :2]) / degree
    graph = _masked_weighted_mean(
        F.smooth_l1_loss(
            predicted[:, :, :2] - predicted_neighbor,
            target[:, :, :2] - target_neighbor,
            reduction="none",
        ),
        mask,
    )
    delta = _masked_weighted_mean(
        F.smooth_l1_loss(
            predicted[:, :, :2] - input_state[:, :, :2],
            target[:, :, :2] - teacher_state[:, :, :2],
            reduction="none",
        ),
        mask,
    )
    active = mask[:, :, None].to(predicted.dtype)
    denominator = active.sum(dim=(1, 2)).clamp_min(1.0) * 2.0
    epsilon_squared = config.minimum_energy_epsilon**2
    predicted_energy = (
        (predicted[:, :, :2].square() * active).sum(dim=(1, 2)) / denominator
        + epsilon_squared
    ).sqrt()
    target_energy = (
        (target[:, :, :2].square() * active).sum(dim=(1, 2)) / denominator
        + epsilon_squared
    ).sqrt()
    energy = F.smooth_l1_loss(
        torch.log(predicted_energy),
        torch.log(target_energy),
    )
    outside = (predicted * (~mask)[:, :, None].to(predicted.dtype)).abs().max()
    total = (
        position
        + velocity * config.velocity_weight
        + graph * config.graph_weight
        + appendage * config.appendage_weight
        + delta * config.delta_weight
        + energy * config.energy_weight
        + outside
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("rollout motion loss became non-finite")
    return total, {
        "loss": total.detach(),
        "position": position.detach(),
        "velocity": velocity.detach(),
        "graph": graph.detach(),
        "appendage": appendage.detach(),
        "delta": delta.detach(),
        "energy": energy.detach(),
        "outside": outside.detach(),
    }


def _rollout(
    model: CellularMotionTransformer,
    frames: list[dict[str, Tensor]],
    config: RolloutTrainingConfig,
    *,
    backward: bool,
) -> tuple[dict[str, float], list[Tensor]]:
    state = frames[0]["state"]
    totals = {name: 0.0 for name in ("loss", "position", "velocity", "graph", "appendage", "delta", "energy", "outside")}
    predictions: list[Tensor] = []
    for frame in frames:
        with torch.autocast(
            device_type=state.device.type,
            dtype=torch.bfloat16,
            enabled=state.device.type == "cuda",
        ):
            predicted = model(
                frame["static"], state, frame["mask"], frame["adjacency"],
                frame["family"], frame["morphotype"], frame["motion"],
                frame["phase"], frame["controls"],
            )
        loss, pieces = rollout_frame_loss(
            predicted.float(), frame["target"].float(), state.float(), frame["state"].float(),
            frame["static"].float(), frame["mask"], frame["adjacency"], config,
        )
        if backward:
            (loss / len(frames)).backward()
        for name, value in pieces.items():
            totals[name] += float(value)
        predictions.append(predicted.detach())
        state = predicted.detach()
    return ({name: value / len(frames) for name, value in totals.items()}, predictions)


def _diagnostics(
    frames: list[dict[str, Tensor]], predictions: list[Tensor]
) -> dict[str, float | int | bool]:
    predicted = torch.stack(predictions, dim=1)
    target = torch.stack([frame["target"] for frame in frames], dim=1)
    mask = frames[0]["mask"]
    active = mask[:, None, :, None]
    outside = float(predicted.masked_select(~active.expand_as(predicted)).abs().max())
    appendage = (frames[0]["static"][:, :, 50] > 0.5) & mask
    appendage_values = predicted[:, :, :, :2].masked_select(appendage[:, None, :, None]).reshape(-1, 2)
    appendage_motion = float(appendage_values.square().mean().sqrt() * MAX_DISPLACEMENT)
    predicted_values = predicted[:, :, :, :2].masked_select(active.expand_as(predicted[:, :, :, :2])).reshape(-1, 2)
    target_values = target[:, :, :, :2].masked_select(active.expand_as(target[:, :, :, :2])).reshape(-1, 2)
    predicted_energy = float(predicted_values.square().mean().sqrt() * MAX_DISPLACEMENT)
    target_energy = float(target_values.square().mean().sqrt() * MAX_DISPLACEMENT)
    return {
        "families": len(set(int(value) for value in frames[0]["family"].tolist())),
        "frames": len(frames),
        "prediction_fed_frames": len(frames) - 1,
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
        raise ValueError("rollout motion smoke step count drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=512 * 1024**2)
    from ..creature_stage_neural_motion.contract import DEFAULT_TEACHER

    teacher = NativeMotionTeacher(DEFAULT_TEACHER)
    config = RolloutTrainingConfig()
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
    sampler = RolloutBatchSampler(teacher, sequence_frames=config.sequence_frames)
    frames = sampler.sequence(0)
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pieces, _ = _rollout(model, frames, config, backward=True)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gradient)) or float(gradient) <= 0.0:
            raise FloatingPointError("rollout motion smoke gradient drifted")
        optimizer.step()
        history.append({
            "step": step + 1,
            **{name: round(float(value), 9) for name, value in pieces.items()},
            "gradient_norm": round(float(gradient), 9),
            "lr": 0.0008,
        })
    model.eval()
    with torch.inference_mode():
        _, predictions = _rollout(model, frames, config, backward=False)
    diagnostics = _diagnostics(frames, predictions)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source_sha256(),
        "parent_model_source_sha256": parent_source_sha256(),
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
        "prediction_fed_after_frame_zero": diagnostics["prediction_fed_frames"] == config.sequence_frames - 1,
        "fixed_sequence_loss_improved": float(history[-1]["loss"]) < float(history[0]["loss"]),
        "outside_cells_exact_zero": diagnostics["outside_max_abs"] == 0.0,
        "appendage_motion_nonzero": diagnostics["appendage_motion_px"] > 0.01,
        "energy_ratio_positive": diagnostics["energy_ratio"] > 0.01,
        "gradient_nonzero": all(float(row["gradient_norm"]) > 0.0 for row in history),
    }
    if not all(gates.values()):
        raise ValueError(f"rollout motion smoke failed: {gates}")
    report: dict[str, Any] = {
        "format": SMOKE_FORMAT,
        "status": "passed",
        "source_sha256": source_sha256(),
        "parent_model_source_sha256": parent_source_sha256(),
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
        raise ValueError("rollout motion smoke manifest is missing or oversized")
    raw = manifest_path.read_bytes()
    report = json.loads(raw)
    required = {
        "format", "status", "source_sha256", "parent_model_source_sha256",
        "teacher", "model", "training", "steps", "history", "checkpoint",
        "diagnostics", "gates", "semantic_sha256",
    }
    if raw != _canonical(report) or set(report) != required:
        raise ValueError("rollout motion smoke structure drifted")
    errors = sorted(
        Draft202012Validator(json.loads(SMOKE_SCHEMA.read_bytes())).iter_errors(report),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"rollout motion smoke schema drifted: {errors[0].message}")
    if (
        report["source_sha256"] != source_sha256()
        or report["parent_model_source_sha256"] != parent_source_sha256()
        or report["semantic_sha256"]
        != hashlib.sha256(_canonical({key: value for key, value in report.items() if key != "semantic_sha256"})).hexdigest()
        or report["training"] != RolloutTrainingConfig().to_dict()
        or set(report["history"][0]) != HISTORY_KEYS
        or any(set(row) != HISTORY_KEYS or row["step"] != index for index, row in enumerate(report["history"], 1))
        or not all(report["gates"].values())
    ):
        raise ValueError("rollout motion smoke authority drifted")
    teacher = NativeMotionTeacher(PROJECT_ROOT / report["teacher"]["path"])
    if teacher.semantic_sha256 != report["teacher"]["semantic_sha256"]:
        raise ValueError("rollout motion smoke teacher drifted")
    checkpoint_path = output / report["checkpoint"]["path"]
    if (
        checkpoint_path.is_symlink() or not checkpoint_path.is_file()
        or not 0 < checkpoint_path.stat().st_size <= MAX_CHECKPOINT_BYTES
        or checkpoint_path.stat().st_size != report["checkpoint"]["bytes"]
        or _sha256_file(checkpoint_path) != report["checkpoint"]["sha256"]
    ):
        raise ValueError("rollout motion smoke checkpoint bytes drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_keys = {
        "format", "source_sha256", "parent_model_source_sha256", "teacher_semantic_sha256",
        "model", "training", "steps", "model_state", "model_state_sha256", "history",
    }
    if set(checkpoint) != checkpoint_keys:
        raise ValueError("rollout motion smoke checkpoint registry drifted")
    model = CellularMotionTransformer(_config(checkpoint["model"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if (
        checkpoint["format"] != CHECKPOINT_FORMAT
        or checkpoint["source_sha256"] != source_sha256()
        or checkpoint["parent_model_source_sha256"] != parent_source_sha256()
        or checkpoint["teacher_semantic_sha256"] != teacher.semantic_sha256
        or checkpoint["training"] != report["training"]
        or checkpoint["steps"] != report["steps"]
        or checkpoint["history"] != report["history"]
        or _state_sha256(model.state_dict()) != checkpoint["model_state_sha256"]
        or checkpoint["model_state_sha256"] != report["checkpoint"]["model_state_sha256"]
        or model.parameter_count != report["model"]["parameters"]
    ):
        raise ValueError("rollout motion smoke checkpoint semantics drifted")
    if replay:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model.eval()
        sampler = RolloutBatchSampler(teacher, sequence_frames=report["training"]["sequence_frames"])
        frames = sampler.sequence(0)
        with torch.inference_mode():
            _, predictions = _rollout(model, frames, RolloutTrainingConfig(**report["training"]), backward=False)
        if _diagnostics(frames, predictions) != report["diagnostics"]:
            raise ValueError("rollout motion smoke deterministic replay drifted")
    return {
        "passed": True,
        "steps": report["steps"],
        "parameters": report["model"]["parameters"],
        "model_state_sha256": checkpoint["model_state_sha256"],
        "semantic_sha256": report["semantic_sha256"],
        "diagnostics": report["diagnostics"],
    }


def _load_parent(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    contract_path = path.parent / "production_contract.json"
    raw = contract_path.read_bytes()
    contract = json.loads(raw)
    if (
        raw != _canonical(contract)
        or contract.get("format") != PARENT_PRODUCTION_FORMAT
        or contract.get("source_sha256") != parent_source_sha256()
        or contract.get("semantic_sha256")
        != hashlib.sha256(_canonical({key: value for key, value in contract.items() if key != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("rollout motion parent contract drifted")
    stem = path.stem
    if not stem.startswith("cell_motion_") or not stem[12:].isdigit():
        raise ValueError("rollout motion parent checkpoint name drifted")
    checkpoint = _load_parent_checkpoint(path, contract, int(stem[12:]))
    return contract, checkpoint


def prepare_production(
    output: Path = DEFAULT_OUTPUT,
    *,
    parent_checkpoint: Path = DEFAULT_PARENT,
    total_updates: int = 5_000,
    segment_updates: int = 500,
    batch_size: int = 5,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if (
        type(total_updates) is not int or type(segment_updates) is not int
        or not 1_000 <= total_updates <= 50_000 or not 100 <= segment_updates <= 2_000
        or total_updates % segment_updates
        or type(batch_size) is not int or not 5 <= batch_size <= 30 or batch_size % 5
    ):
        raise ValueError("rollout motion production schedule drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=8 * 1024**3)
    parent_contract, parent = _load_parent(parent_checkpoint)
    training = RolloutTrainingConfig()
    contract: dict[str, Any] = {
        "format": PRODUCTION_FORMAT,
        "source_sha256": source_sha256(),
        "parent": {
            "path": _relative(Path(parent_checkpoint)),
            "sha256": _sha256_file(Path(parent_checkpoint)),
            "step": parent["step"],
            "model_state_sha256": parent["model_state_sha256"],
            "ema_state_sha256": parent["ema_state_sha256"],
            "contract_semantic_sha256": parent_contract["semantic_sha256"],
        },
        "teacher": dict(parent_contract["teacher"]),
        "model": dict(parent_contract["model"]),
        "training": training.to_dict(),
        "seed": SEED,
        "total_updates": total_updates,
        "segment_updates": segment_updates,
        "batch_size": batch_size,
        "optimizer": {
            "name": "AdamW", "lr": 8e-5, "weight_decay": 1e-5,
            "warmup_updates": min(250, total_updates // 5), "gradient_clip": 1.0,
        },
        "ema_decay": 0.999,
        "precision": "bf16-autocast-float32-loss",
        "minimum_free_vram_bytes": 16 * 1024**3,
    }
    contract["semantic_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "production_contract.json"
    if destination.exists():
        if destination.read_bytes() != _canonical(contract):
            raise ValueError("rollout motion production contract changed during resume")
    else:
        _atomic_bytes(destination, _canonical(contract))
    return contract


def checkpoint_name(update: int) -> str:
    return f"cell_motion_rollout_{update:07d}.pt"


def _load_rollout_checkpoint(path: Path, contract: dict[str, Any], expected_update: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("rollout motion checkpoint is missing or oversized")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format", "source_sha256", "contract_semantic_sha256", "update", "model_state",
        "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256",
        "cpu_rng_state", "cuda_rng_state", "history", "runtime",
    }
    if (
        set(payload) != required or payload["format"] != CHECKPOINT_FORMAT
        or payload["source_sha256"] != source_sha256()
        or payload["contract_semantic_sha256"] != contract["semantic_sha256"]
        or payload["update"] != expected_update or len(payload["history"]) != expected_update
    ):
        raise ValueError("rollout motion checkpoint contract drifted")
    model = CellularMotionTransformer(_config(contract["model"]))
    model.load_state_dict(payload["model_state"], strict=True)
    if _state_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("rollout motion model state hash drifted")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_sha256(model.state_dict()) != payload["ema_state_sha256"]:
        raise ValueError("rollout motion EMA state hash drifted")
    for index, row in enumerate(payload["history"], 1):
        if set(row) != HISTORY_KEYS or row["step"] != index or any(
            not math.isfinite(float(value)) for key, value in row.items() if key != "step"
        ):
            raise ValueError("rollout motion history drifted")
    return payload


def train_segment(output: Path = DEFAULT_OUTPUT, *, end_update: int) -> dict[str, Any]:
    output = Path(output).resolve()
    raw = (output / "production_contract.json").read_bytes()
    contract = json.loads(raw)
    if (
        raw != _canonical(contract) or contract.get("source_sha256") != source_sha256()
        or contract.get("semantic_sha256")
        != hashlib.sha256(_canonical({key: value for key, value in contract.items() if key != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("rollout motion training authority drifted")
    segment_updates = int(contract["segment_updates"])
    if type(end_update) is not int or end_update % segment_updates or not segment_updates <= end_update <= contract["total_updates"]:
        raise ValueError("rollout motion segment endpoint drifted")
    destination = output / checkpoint_name(end_update)
    if destination.exists():
        checked = _load_rollout_checkpoint(destination, contract, end_update)
        return {"passed": True, "update": end_update, "model_state_sha256": checked["model_state_sha256"], "ema_state_sha256": checked["ema_state_sha256"]}
    previous_update = end_update - segment_updates
    previous_path = output / checkpoint_name(previous_update) if previous_update else None
    if previous_path is not None and not previous_path.exists():
        raise FileNotFoundError("previous rollout motion segment is missing")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        or torch.cuda.mem_get_info(0)[0] < contract["minimum_free_vram_bytes"]
    ):
        raise RuntimeError("rollout motion training requires deterministic CUDA BF16 and 16 GiB free VRAM")
    require_disk_floor(output, floor_gb=100, planned_bytes=1024**3)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(contract["seed"])
    torch.cuda.manual_seed_all(contract["seed"])
    np.random.seed(contract["seed"] & 0xFFFFFFFF)
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats(device)
    model = CellularMotionTransformer(_config(contract["model"])).to(device)
    parent_contract, parent = _load_parent(PROJECT_ROOT / contract["parent"]["path"])
    if parent_contract["semantic_sha256"] != contract["parent"]["contract_semantic_sha256"]:
        raise ValueError("rollout motion parent changed during training")
    model.load_state_dict(parent["ema_state"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=contract["optimizer"]["lr"],
        weight_decay=contract["optimizer"]["weight_decay"],
    )
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    start_update = 0
    if previous_path is not None:
        previous = _load_rollout_checkpoint(previous_path, contract, previous_update)
        start_update = previous_update
        model.load_state_dict(previous["model_state"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state"])
        ema = {name: value.to(device) for name, value in previous["ema_state"].items()}
        torch.set_rng_state(previous["cpu_rng_state"])
        torch.cuda.set_rng_state(previous["cuda_rng_state"], device)
        history = list(previous["history"])
    teacher = NativeMotionTeacher(PROJECT_ROOT / contract["teacher"]["path"])
    if teacher.semantic_sha256 != contract["teacher"]["semantic_sha256"]:
        raise ValueError("rollout motion teacher changed during training")
    training_config = RolloutTrainingConfig(**contract["training"])
    sampler = RolloutBatchSampler(
        teacher, batch_size=contract["batch_size"], sequence_frames=training_config.sequence_frames,
    )
    started = time.perf_counter()
    model.train()
    for update in range(start_update, end_update):
        frames = sampler.sequence(update, device)
        lr = contract["optimizer"]["lr"] * min(
            1.0, (update + 1) / max(1, contract["optimizer"]["warmup_updates"])
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        pieces, _ = _rollout(model, frames, training_config, backward=True)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["optimizer"]["gradient_clip"])
        if not math.isfinite(float(gradient)):
            raise FloatingPointError("rollout motion training became non-finite")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    ema[name].lerp_(value.detach(), 1.0 - contract["ema_decay"])
                else:
                    ema[name].copy_(value)
        history.append({
            "step": update + 1,
            **{name: round(float(value), 9) for name, value in pieces.items()},
            "gradient_norm": round(float(gradient), 9),
            "lr": round(float(lr), 12),
        })
    elapsed = time.perf_counter() - started
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    payload = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source_sha256(),
        "contract_semantic_sha256": contract["semantic_sha256"],
        "update": end_update,
        "model_state": model_state,
        "ema_state": ema_state,
        "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": _state_sha256(model_state),
        "ema_state_sha256": _state_sha256(ema_state),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
        "history": history,
        "runtime": {
            "segment_seconds": round(elapsed, 6),
            "updates_per_second": round(segment_updates / elapsed, 6),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
        },
    }
    _atomic_torch(destination, payload)
    checked = _load_rollout_checkpoint(destination, contract, end_update)
    return {
        "passed": True,
        "update": end_update,
        "model_state_sha256": checked["model_state_sha256"],
        "ema_state_sha256": checked["ema_state_sha256"],
    }
