from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_motion_rollout.contract import CHECKPOINT_FORMAT as PARENT_FORMAT
from .contract import (
    CHECKPOINT_FORMAT,
    DEFAULT_OUTPUT,
    FORMAT,
    ROLLOUT_PARENT,
    ROLLOUT_PARENT_SHA256,
    CyclicModelConfig,
    CyclicTrainingConfig,
    canonical_json_bytes,
    sha256_file,
    source_sha256,
)
from .curriculum import CurriculumGroundedTeacher
from .dataset import RuntimeHonestGroundedTeacher
from .model import CyclicMotionOutput, NeuralCyclicGroundedMotion


SEED = 0x4359434C49434752


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    active = mask[:, :, None].to(value.dtype)
    return (value * active).sum() / (active.sum().clamp_min(1) * value.shape[-1])


def _forward(model: NeuralCyclicGroundedMotion, batch: dict[str, Tensor], state: Tensor) -> CyclicMotionOutput:
    return model(
        batch["static"], state, batch["dynamic"], batch["mask"], batch["adjacency"],
        batch["family"], batch["morphotype"], batch["motion"], batch["phase"], batch["controls"],
    )


def cyclic_loss(output: CyclicMotionOutput, batch: dict[str, Tensor], state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    mask = batch["mask"]
    target = batch["target"].float()
    predicted = output.cells.float()
    pos_error = (predicted[:, :, :2] - target[:, :, :2]).abs()
    vel_error = (predicted[:, :, 2:] - target[:, :, 2:]).abs()
    position = _masked_mean(pos_error, mask)
    velocity = _masked_mean(vel_error, mask)
    appendage = (batch["static"][:, :, 50] > .5) & mask
    contact = (batch["dynamic"][:, :, 5] > .5) & mask
    appendage_loss = _masked_mean(pos_error, appendage)
    contact_loss = _masked_mean(pos_error, contact) if bool(contact.any()) else position * 0
    direct = _masked_mean((output.direct_cells.float() - target).abs(), mask)
    adjacency = batch["adjacency"].to(predicted.dtype)
    degree = adjacency.sum(2, keepdim=True).clamp_min(1)
    predicted_neighbor = torch.bmm(adjacency, predicted[:, :, :2]) / degree
    target_neighbor = torch.bmm(adjacency, target[:, :, :2]) / degree
    graph = _masked_mean(((predicted[:, :, :2] - predicted_neighbor) - (target[:, :, :2] - target_neighbor)).abs(), mask)
    body = (output.body_velocity.float() - batch["body_target"].float()).abs().mean()
    baseline = _masked_mean((state[:, :, :2].float() - target[:, :, :2]).abs(), mask).detach()
    copy_margin = torch.relu(position - baseline * .85)
    gate = output.direct_gate[mask].mean()
    outside = output.cells[~mask].abs().max()
    total = (
        position * 1.40 + velocity * .35 + appendage_loss * .75 + contact_loss
        + direct * .90 + graph * .25 + body * .30 + copy_margin * .80
        + (1.0 - gate) * .015 + outside
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("cyclic grounded loss became non-finite")
    return total, {
        "loss": total.detach(), "position_l1": position.detach(), "velocity_l1": velocity.detach(),
        "appendage_l1": appendage_loss.detach(), "contact_l1": contact_loss.detach(),
        "direct_l1": direct.detach(), "graph_l1": graph.detach(), "body_l1": body.detach(),
        "copy_margin": copy_margin.detach(), "direct_gate": gate.detach(), "outside": outside.detach(),
    }


def _batch_at(teacher: CurriculumGroundedTeacher, frame: int, variant_slot: int,
              device: torch.device) -> dict[str, Tensor]:
    identities = [family_rows[variant_slot % len(family_rows)] for family_rows in teacher.family_indices]
    rows = [teacher.sample(identity, frame) for identity in identities]
    result = {
        name: torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)
        for name in ("static", "state", "target", "dynamic", "mask", "adjacency", "controls")
    }
    for name in ("family", "morphotype", "motion"):
        result[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
    for name in ("phase", "body_target"):
        result[name] = torch.tensor([float(row[name]) for row in rows], dtype=torch.float32, device=device)
    return result


def _seam_loss(model: NeuralCyclicGroundedMotion, teacher: CurriculumGroundedTeacher,
               variant_slot: int, device: torch.device) -> tuple[Tensor, Tensor]:
    left = _batch_at(teacher, 71, variant_slot, device)
    right = _batch_at(teacher, 0, variant_slot, device)
    out_left = _forward(model, left, left["state"])
    out_right = _forward(model, right, out_left.cells.detach())
    predicted_transition = out_right.cells[:, :, :2].float() - out_left.cells[:, :, :2].float()
    target_transition = right["target"][:, :, :2].float() - left["target"][:, :, :2].float()
    error = (predicted_transition - target_transition).abs()
    active = error[right["mask"][:, :, None].expand_as(error)]
    top_count = max(1, int(active.numel() * .05))
    return active.mean(), torch.topk(active, top_count).values.mean()


def _load_parent(model: NeuralCyclicGroundedMotion) -> dict[str, Any]:
    if sha256_file(ROLLOUT_PARENT) != ROLLOUT_PARENT_SHA256:
        raise ValueError("cyclic rollout parent bytes drifted")
    payload = torch.load(ROLLOUT_PARENT, map_location="cpu", weights_only=True)
    if payload.get("format") != PARENT_FORMAT or payload.get("update") != 1000:
        raise ValueError("cyclic parent is not sealed rollout update 1000")
    model.backbone.load_state_dict(payload["ema_state"], strict=True)
    if _state_sha256(payload["ema_state"]) != payload["ema_state_sha256"]:
        raise ValueError("cyclic parent EMA identity drifted")
    return {
        "path": ROLLOUT_PARENT.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": ROLLOUT_PARENT.stat().st_size,
        "sha256": ROLLOUT_PARENT_SHA256,
        "update": 1000,
        "ema_state_sha256": payload["ema_state_sha256"],
        "source_sha256": payload["source_sha256"],
    }


def train(output: Path = DEFAULT_OUTPUT, *, updates: int | None = None,
          batch_size: int | None = None, device: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    defaults = CyclicTrainingConfig()
    config = CyclicTrainingConfig(
        updates=defaults.updates if updates is None else updates,
        batch_size=defaults.batch_size if batch_size is None else batch_size,
    )
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    teacher = CurriculumGroundedTeacher()
    sentinel = RuntimeHonestGroundedTeacher()
    backbone_config = CellularMotionTransformerConfig()
    model_config = CyclicModelConfig()
    model = NeuralCyclicGroundedMotion(backbone_config, model_config)
    parent = _load_parent(model)
    model.to(target_device).train()
    torch.manual_seed(SEED); np.random.seed(SEED & 0xFFFFFFFF)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config.learning_rate * config.backbone_scale},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("backbone.")], "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)
    ema = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        frames = teacher.sequence(update, config.batch_size, config.sequence_frames, target_device, split="train")
        state = frames[0]["state"]
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        totals: dict[str, float] = {}
        for batch in frames:
            with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
                result = _forward(model, batch, state)
            loss, pieces = cyclic_loss(result, batch, state)
            losses.append(loss)
            for name, value in pieces.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            state = result.cells.detach()
        with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
            seam_mean, seam_tail = _seam_loss(model, teacher, update, target_device)
        loss = torch.stack(losses).mean() + (seam_mean + seam_tail) * config.seam_weight
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not math.isfinite(float(gradient)):
            raise FloatingPointError("cyclic grounded gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].mul_(config.ema_decay).add_(value.detach().cpu(), alpha=1 - config.ema_decay)
        if update == 1 or update % 25 == 0 or update == config.updates:
            history.append({
                "update": update,
                **{name: round(value / config.sequence_frames, 9) for name, value in totals.items()},
                "seam_mean_l1": round(float(seam_mean), 9),
                "seam_top5_l1": round(float(seam_tail), 9),
                "gradient_norm": round(float(gradient), 9),
            })
    seconds = time.perf_counter() - started
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source_sha256(),
        "teacher_semantic_sha256": teacher.semantic_sha256,
        "sentinel_semantic_sha256": sentinel.semantic_sha256,
        "parent": parent,
        "split": {
            "train_count": len(teacher.organisms),
            "train_genome_ids": [item.genome.genome_id for item in teacher.organisms],
            "evaluation": list(sentinel.split_indices("validation")),
            "evaluation_genome_ids": [sentinel.organisms[index].genome.genome_id for index in sentinel.split_indices("validation")],
            "overlap": [],
        },
        "runtime_honest_contacts": True,
        "model": model_config.to_dict(), "backbone": backbone_config.to_dict(), "training": config.to_dict(),
        "updates": config.updates, "model_state": model_state, "ema_state": ema,
        "model_state_sha256": _state_sha256(model_state), "ema_state_sha256": _state_sha256(ema),
        "history": history,
        "runtime": {
            "seconds": round(seconds, 6), "updates_per_second": round(config.updates / seconds, 6),
            "device": str(target_device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if target_device.type == "cuda" else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if target_device.type == "cuda" else 0,
        },
    }
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    checkpoint_path = stage / f"cyclic_grounded_motion_{config.updates:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        key: checkpoint[key] for key in (
            "format", "source_sha256", "teacher_semantic_sha256", "sentinel_semantic_sha256", "parent", "split",
            "runtime_honest_contacts", "model", "backbone", "training", "updates", "history", "runtime",
        )
    }
    report["format"] = FORMAT
    report["checkpoint"] = {
        "path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path), "model_state_sha256": checkpoint["model_state_sha256"],
        "ema_state_sha256": checkpoint["ema_state_sha256"],
    }
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    (stage / "production_manifest.json").write_bytes(canonical_json_bytes(report))
    os.replace(stage, output)
    return report


def load_model(checkpoint_path: Path, *, ema: bool = True,
               device: str | torch.device = "cpu") -> tuple[NeuralCyclicGroundedMotion, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file() or checkpoint_path.stat().st_size > 1024**3:
        raise ValueError("cyclic checkpoint missing or oversized")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
        raise ValueError("cyclic checkpoint provenance drifted")
    if payload.get("runtime_honest_contacts") is not True or payload.get("split", {}).get("overlap") != []:
        raise ValueError("cyclic checkpoint split/runtime contract drifted")
    model = NeuralCyclicGroundedMotion(
        CellularMotionTransformerConfig(**payload["backbone"]), CyclicModelConfig(**payload["model"]),
    )
    state_name = "ema_state" if ema else "model_state"
    hash_name = "ema_state_sha256" if ema else "model_state_sha256"
    if _state_sha256(payload[state_name]) != payload[hash_name]:
        raise ValueError("cyclic checkpoint state drifted")
    model.load_state_dict(payload[state_name], strict=True)
    return model.to(device).eval(), payload
