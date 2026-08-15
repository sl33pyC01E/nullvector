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
from ..creature_stage_neural_motion.training import _state_sha256
from .contract import (
    DEFAULT_CORPUS,
    DEFAULT_PARENT,
    DEFAULT_PRIOR,
    SMOKE_FORMAT,
    SMOKE_SCHEMA,
    DevelopmentalActuatorConfig,
    DevelopmentalTrainingConfig,
    source_sha256,
)
from .dataset import DevelopmentalMotionTeacher, DevelopmentalSequenceSampler, project_path
from .training import (
    _parent_authority,
    atomic_bytes,
    atomic_torch,
    canonical,
    load_successor_state,
    make_model,
    rollout_sequence,
    semantic,
    sha256_file,
    successor_state,
)


SMOKE_SEED = 0x444556534D4F4B32
MAX_CHECKPOINT_BYTES = 1024**3


def _diagnostics(frames, outputs) -> dict[str, float | int]:
    cell = torch.stack([output["cell_state"] for output in outputs], dim=1).float()
    node = torch.stack([output["node_state"] for output in outputs], dim=1).float()
    muscle = torch.stack([output["muscle_activation"] for output in outputs], dim=1).float()
    target = torch.stack([frame["target"] for frame in frames], dim=1).float()
    cell_mask = frames[0]["mask"][:, None, :, None]
    node_mask = frames[0]["node_mask"][:, None, :, None]
    muscle_mask = frames[0]["muscle_mask"][:, None, :]
    cell_values = cell[:, :, :, :2].masked_select(cell_mask.expand_as(cell[:, :, :, :2])).reshape(-1, 2)
    target_values = target[:, :, :, :2].masked_select(cell_mask.expand_as(target[:, :, :, :2])).reshape(-1, 2)
    node_values = node[:, :, :, :2].masked_select(node_mask.expand_as(node[:, :, :, :2])).reshape(-1, 2)
    muscle_values = muscle.masked_select(muscle_mask)
    outside = float(cell.masked_select(~cell_mask.expand_as(cell)).abs().max())
    return {
        "families": len(set(int(value) for value in frames[0]["family"].tolist())),
        "frames": len(frames),
        "prediction_fed_frames": len(frames) - 1,
        "cell_energy_px": round(float(cell_values.square().mean().sqrt() * 12.0), 9),
        "target_energy_px": round(float(target_values.square().mean().sqrt() * 12.0), 9),
        "node_energy_px": round(float(node_values.square().mean().sqrt() * 12.0), 9),
        "muscle_mean": round(float(muscle_values.mean()), 9),
        "muscle_std": round(float(muscle_values.std()), 9),
        "outside_max_abs": round(outside, 12),
    }


def run_parent_adapter_smoke(
    output: Path,
    *,
    corpus: Path = DEFAULT_CORPUS,
    parent: Path = DEFAULT_PARENT,
    prior: Path = DEFAULT_PRIOR,
    steps: int = 8,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(steps) is not int or not 4 <= steps <= 24:
        raise ValueError("developmental actuator smoke step count drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024**3)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SMOKE_SEED)
    np.random.seed(SMOKE_SEED & 0xFFFFFFFF)
    teacher = DevelopmentalMotionTeacher(corpus, prior=prior, replay=True)
    parent_contract, parent_checkpoint = _parent_authority(parent)
    config = DevelopmentalActuatorConfig(
        width=96, depth=2, heads=4, feedforward_multiplier=2,
        condition_width=96, cell_width=64, cell_graph_blocks=2, dropout=0.0,
    )
    training = DevelopmentalTrainingConfig(sequence_frames=6)
    model = make_model(parent_contract, parent_checkpoint, config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=7e-4, weight_decay=1e-5)
    sampler = DevelopmentalSequenceSampler(
        teacher, batch_size=5, sequence_frames=training.sequence_frames,
        seed=SMOKE_SEED, seam_numerator=1, seam_denominator=3,
    )
    frames, coordinates = sampler.sequence(0)
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pieces, _ = rollout_sequence(model, frames, training, teacher_forcing=.35, backward=True)
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not math.isfinite(float(gradient)) or float(gradient) <= 0.0:
            raise FloatingPointError("developmental actuator smoke gradient drifted")
        optimizer.step()
        history.append({
            "step": step + 1,
            **{name: round(float(value), 9) for name, value in pieces.items()},
            "gradient_norm": round(float(gradient), 9),
        })
    model.eval()
    with torch.inference_mode():
        _, outputs = rollout_sequence(model, frames, training, teacher_forcing=0.0, backward=False)
    diagnostics = _diagnostics(frames, outputs)
    state = successor_state(model)
    checkpoint = {
        "format": "nullvector-creature-stage-developmental-actuator-smoke-checkpoint-v2",
        "source_sha256": source_sha256(),
        "teacher_semantic_sha256": teacher.semantic_sha256,
        "parent_ema_state_sha256": parent_checkpoint["ema_state_sha256"],
        "model": config.to_dict(),
        "training": training.to_dict(),
        "steps": steps,
        "model_state": state,
        "model_state_sha256": _state_sha256(state),
        "history": history,
    }
    output.mkdir(parents=True)
    checkpoint_path = output / "smoke_checkpoint.pt"
    atomic_torch(checkpoint_path, checkpoint)
    gates = {
        "all_values_finite": all(
            math.isfinite(float(value)) for row in history for key, value in row.items() if key != "step"
        ),
        "all_five_families": diagnostics["families"] == 5,
        "prediction_fed_after_frame_zero": diagnostics["prediction_fed_frames"] == training.sequence_frames - 1,
        "fixed_sequence_loss_improved": float(history[-1]["loss"]) < float(history[0]["loss"]),
        "outside_cells_exact_zero": diagnostics["outside_max_abs"] == 0.0,
        "cell_motion_nonzero": diagnostics["cell_energy_px"] > .01,
        "node_motion_nonzero": diagnostics["node_energy_px"] > .01,
        "muscle_channels_noncollapsed": diagnostics["muscle_std"] > .001,
        "gradient_nonzero": all(float(row["gradient_norm"]) > 0.0 for row in history),
        "sealed_parent_update_1000": int(parent_checkpoint["update"]) == 1_000,
    }
    if not all(gates.values()):
        raise ValueError(f"developmental actuator smoke failed: {gates}")
    report: dict[str, Any] = {
        "format": SMOKE_FORMAT,
        "status": "passed",
        "source_sha256": source_sha256(),
        "teacher": {"path": project_path(teacher.root), "semantic_sha256": teacher.semantic_sha256},
        "parent": {
            "path": project_path(Path(parent)), "sha256": sha256_file(Path(parent)),
            "update": 1_000, "ema_state_sha256": parent_checkpoint["ema_state_sha256"],
            "contract_semantic_sha256": parent_contract["semantic_sha256"],
        },
        "prior": {"path": project_path(Path(prior)), "semantic_sha256": teacher.prior_semantic_sha256},
        "model": {"config": config.to_dict(), "parameters": model.parameter_count, "trainable_parameters": model.trainable_parameter_count},
        "training": training.to_dict(),
        "steps": steps,
        "coordinates": [
            {"specimen": item.specimen, "start": item.start, "forced_seam": item.forced_seam}
            for item in coordinates
        ],
        "history": history,
        "checkpoint": {
            "path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path), "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "diagnostics": diagnostics,
        "gates": gates,
    }
    report["semantic_sha256"] = semantic(report)
    atomic_bytes(output / "smoke_manifest.json", canonical(report))
    return validate_parent_adapter_smoke(output, replay=True)


def validate_parent_adapter_smoke(output: Path, *, replay: bool = True) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / "smoke_manifest.json"
    raw = manifest_path.read_bytes()
    report = json.loads(raw)
    if raw != canonical(report):
        raise ValueError("developmental actuator smoke manifest is not canonical")
    errors = sorted(
        Draft202012Validator(json.loads(SMOKE_SCHEMA.read_text(encoding="utf-8"))).iter_errors(report),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"developmental actuator smoke schema drifted: {errors[0].message}")
    if (
        report["format"] != SMOKE_FORMAT or report["status"] != "passed"
        or report["source_sha256"] != source_sha256()
        or report["semantic_sha256"]
        != semantic({key: value for key, value in report.items() if key != "semantic_sha256"})
        or not all(report["gates"].values())
    ):
        raise ValueError("developmental actuator smoke authority drifted")
    teacher = DevelopmentalMotionTeacher(
        PROJECT_ROOT / report["teacher"]["path"],
        prior=PROJECT_ROOT / report["prior"]["path"], replay=False,
    )
    parent_contract, parent = _parent_authority(PROJECT_ROOT / report["parent"]["path"])
    if (
        teacher.semantic_sha256 != report["teacher"]["semantic_sha256"]
        or teacher.prior_semantic_sha256 != report["prior"]["semantic_sha256"]
        or parent["ema_state_sha256"] != report["parent"]["ema_state_sha256"]
        or parent_contract["semantic_sha256"] != report["parent"]["contract_semantic_sha256"]
    ):
        raise ValueError("developmental actuator smoke provenance drifted")
    checkpoint_path = output / report["checkpoint"]["path"]
    if (
        checkpoint_path.is_symlink() or not checkpoint_path.is_file()
        or not 0 < checkpoint_path.stat().st_size <= MAX_CHECKPOINT_BYTES
        or checkpoint_path.stat().st_size != report["checkpoint"]["bytes"]
        or sha256_file(checkpoint_path) != report["checkpoint"]["sha256"]
    ):
        raise ValueError("developmental actuator smoke checkpoint bytes drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = DevelopmentalActuatorConfig(**report["model"]["config"])
    model = make_model(parent_contract, parent, config)
    load_successor_state(model, checkpoint["model_state"])
    if (
        checkpoint["model_state_sha256"] != report["checkpoint"]["model_state_sha256"]
        or _state_sha256(successor_state(model)) != checkpoint["model_state_sha256"]
    ):
        raise ValueError("developmental actuator smoke model state drifted")
    if replay:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        training = DevelopmentalTrainingConfig(**report["training"])
        sampler = DevelopmentalSequenceSampler(
            teacher, batch_size=5, sequence_frames=training.sequence_frames,
            seed=SMOKE_SEED, seam_numerator=1, seam_denominator=3,
        )
        frames, coordinates = sampler.sequence(0)
        expected_coordinates = [
            {"specimen": item.specimen, "start": item.start, "forced_seam": item.forced_seam}
            for item in coordinates
        ]
        model.eval()
        with torch.inference_mode():
            _, outputs = rollout_sequence(model, frames, training, teacher_forcing=0.0, backward=False)
        if expected_coordinates != report["coordinates"] or _diagnostics(frames, outputs) != report["diagnostics"]:
            raise ValueError("developmental actuator smoke deterministic replay drifted")
    return {
        "passed": True, "steps": report["steps"],
        "parameters": report["model"]["parameters"],
        "trainable_parameters": report["model"]["trainable_parameters"],
        "model_state_sha256": checkpoint["model_state_sha256"],
        "semantic_sha256": report["semantic_sha256"],
        "diagnostics": report["diagnostics"],
    }
