from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_motion_rollout.contract import CHECKPOINT_FORMAT as PARENT_FORMAT
from .contract import CHECKPOINT_FORMAT, DEFAULT_OUTPUT, ROLLOUT_PARENT, GroundedModelConfig, GroundedTrainingConfig, canonical_json_bytes, sha256_file, source_sha256
from .dataset import GroundedMotionTeacher
from .model import NeuralGroundedMotion, grounded_loss


SEED = 0x47524F554E444544


def _load_parent(path: Path, model: NeuralGroundedMotion) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**3:
        raise ValueError("grounded parent checkpoint missing or oversized")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != PARENT_FORMAT or payload.get("update") != 1000:
        raise ValueError("grounded parent is not sealed rollout update 1000")
    model.backbone.load_state_dict(payload["ema_state"], strict=True)
    if _state_sha256(payload["ema_state"]) != payload["ema_state_sha256"]:
        raise ValueError("grounded parent EMA identity drifted")
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(), "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "update": 1000,
        "ema_state_sha256": payload["ema_state_sha256"],
        "source_sha256": payload["source_sha256"],
    }


def _loss_kwargs(config: GroundedTrainingConfig) -> dict[str, float]:
    return {name: float(getattr(config, name)) for name in (
        "position_weight", "velocity_weight", "appendage_weight", "contact_weight",
        "graph_weight", "body_velocity_weight", "delta_weight",
    )}


def train(output: Path = DEFAULT_OUTPUT, *, updates: int | None = None,
          device: str = "cuda", batch_size: int | None = None) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = GroundedTrainingConfig(
        total_updates=GroundedTrainingConfig().total_updates if updates is None else updates,
        batch_size=GroundedTrainingConfig().batch_size if batch_size is None else batch_size,
    )
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    teacher = GroundedMotionTeacher()
    backbone_config = CellularMotionTransformerConfig()
    model = NeuralGroundedMotion(backbone_config, GroundedModelConfig())
    parent = _load_parent(ROLLOUT_PARENT, model)
    model.to(target_device).train()
    torch.manual_seed(SEED); np.random.seed(SEED & 0xFFFFFFFF)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config.learning_rate * config.backbone_learning_rate_scale},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("backbone.")], "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)
    ema = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for update in range(1, config.total_updates + 1):
        frames = teacher.sequence(update, config.batch_size, config.sequence_frames, target_device, split="all")
        optimizer.zero_grad(set_to_none=True)
        state = frames[0]["state"]
        totals: dict[str, float] = {}
        losses = []
        for batch in frames:
            with torch.autocast(device_type=target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
                prediction = model(batch["static"], state, batch["dynamic"], batch["mask"], batch["adjacency"], batch["family"], batch["morphotype"], batch["motion"], batch["phase"], batch["controls"])
            loss_batch = dict(batch); loss_batch["state"] = state
            loss, pieces = grounded_loss(prediction, loss_batch, **_loss_kwargs(config))
            losses.append(loss)
            for name, value in pieces.items(): totals[name] = totals.get(name, 0.0) + float(value)
            state = prediction.cells.detach()
        loss = torch.stack(losses).mean(); loss.backward()
        pieces = {name: torch.tensor(value / len(frames)) for name, value in totals.items()}
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not math.isfinite(float(gradient)):
            raise FloatingPointError("grounded neural gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].mul_(config.ema_decay).add_(value.detach().cpu(), alpha=1 - config.ema_decay)
        if update == 1 or update % 25 == 0 or update == config.total_updates:
            history.append({"update": update, **{name: round(float(value), 9) for name, value in pieces.items()}, "gradient_norm": round(float(gradient), 9)})
    seconds = time.perf_counter() - started
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(),
        "teacher_semantic_sha256": teacher.semantic_sha256, "parent": parent,
        "model": GroundedModelConfig().to_dict(), "backbone": backbone_config.to_dict(),
        "training": config.to_dict(), "updates": config.total_updates,
        "model_state": model_state, "ema_state": ema,
        "model_state_sha256": _state_sha256(model_state), "ema_state_sha256": _state_sha256(ema),
        "history": history,
        "runtime": {"seconds": round(seconds, 6), "updates_per_second": round(config.total_updates / seconds, 6), "device": str(target_device), "peak_allocated_bytes": torch.cuda.max_memory_allocated() if target_device.type == "cuda" else 0, "peak_reserved_bytes": torch.cuda.max_memory_reserved() if target_device.type == "cuda" else 0},
    }
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    checkpoint_path = stage / f"grounded_motion_{config.total_updates:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    contract = {
        "format": FORMAT, "source_sha256": source_sha256(), "teacher_semantic_sha256": teacher.semantic_sha256,
        "parent": parent, "model": GroundedModelConfig().to_dict(), "backbone": backbone_config.to_dict(),
        "training": config.to_dict(), "checkpoint": {"path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]},
        "history": history, "runtime": checkpoint["runtime"],
    }
    contract["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    (stage / "production_manifest.json").write_bytes(canonical_json_bytes(contract))
    os.replace(stage, output)
    return contract


FORMAT = "nullvector-neural-grounded-cell-motion-production-v1"


def load_model(checkpoint_path: Path, *, ema: bool = True, device: str | torch.device = "cpu") -> tuple[NeuralGroundedMotion, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
        raise ValueError("grounded neural checkpoint provenance drifted")
    model = NeuralGroundedMotion(CellularMotionTransformerConfig(**payload["backbone"]), GroundedModelConfig(**payload["model"]))
    state = payload["ema_state" if ema else "model_state"]
    expected = payload["ema_state_sha256" if ema else "model_state_sha256"]
    if _state_sha256(state) != expected:
        raise ValueError("grounded neural checkpoint state drifted")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload
