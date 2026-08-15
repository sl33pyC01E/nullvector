from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_grounded_cyclic.contract import canonical_json_bytes, sha256_file
from ..creature_stage_neural_motion.training import _state_sha256
from .contract import CHECKPOINT_FORMAT, FORMAT, PARENT, PARENT_SHA256, ControllerModelConfig, ControllerTrainingConfig, source_sha256
from .dataset import ControllerCorpus, build_corpus
from .model import ControllerOutput, NeuralGroundedController


def controller_loss(output: ControllerOutput, batch: dict[str, Tensor], positive_weight: Tensor,
                    next_output: ControllerOutput | None = None, next_batch: dict[str, Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    muscle_mask = batch["muscle_mask"]
    muscle = F.smooth_l1_loss(output.muscle_activation, batch["muscle_target"].float(), reduction="none")
    muscle_loss = muscle[muscle_mask].mean()
    contact = F.binary_cross_entropy_with_logits(output.contact_logits, batch["contact_target"].float(), reduction="none", pos_weight=positive_weight)
    contact_loss = contact[batch["owner_mask"]].mean()
    body_loss = F.smooth_l1_loss(output.body_velocity, batch["body_target"].float())
    smooth = muscle_loss * 0
    if next_output is not None and next_batch is not None:
        target_delta = next_batch["muscle_target"].float() - batch["muscle_target"].float()
        predicted_delta = next_output.muscle_activation - output.muscle_activation
        smooth_muscle = F.smooth_l1_loss(predicted_delta, target_delta, reduction="none")[muscle_mask & next_batch["muscle_mask"]].mean()
        target_contact_delta = next_batch["contact_target"].float() - batch["contact_target"].float()
        predicted_contact_delta = torch.sigmoid(next_output.contact_logits) - torch.sigmoid(output.contact_logits)
        smooth_contact = F.smooth_l1_loss(predicted_contact_delta, target_contact_delta, reduction="none")[batch["owner_mask"] & next_batch["owner_mask"]].mean()
        smooth = smooth_muscle + smooth_contact
    total = muscle_loss * 1.35 + contact_loss * 1.8 + body_loss * .25 + smooth * .4
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("grounded controller loss became non-finite")
    return total, {"loss": total.detach(), "muscle": muscle_loss.detach(), "contact": contact_loss.detach(), "body": body_loss.detach(), "temporal": smooth.detach()}


def _forward(model: NeuralGroundedController, batch: dict[str, Tensor]) -> ControllerOutput:
    return model(batch["owner_input"], batch["global_input"], batch["owner_meta"], batch["owner_mask"], batch["muscle_meta"], batch["muscle_owner"], batch["muscle_mask"])


def _validation(model: NeuralGroundedController, corpus: ControllerCorpus, device: torch.device) -> dict[str, float]:
    muscles, muscle_targets, contacts, contact_targets, bodies, body_targets = [], [], [], [], [], []
    with torch.inference_mode():
        for start in range(0, corpus.samples, 72):
            indices = torch.arange(start, min(start + 72, corpus.samples))
            batch = corpus.batch(indices, device); output = _forward(model, batch)
            muscles.append(output.muscle_activation.cpu()); muscle_targets.append(batch["muscle_target"].cpu())
            contacts.append(torch.sigmoid(output.contact_logits).cpu()); contact_targets.append(batch["contact_target"].cpu())
            bodies.append(output.body_velocity.cpu()); body_targets.append(batch["body_target"].cpu())
    muscle = torch.cat(muscles); mt = torch.cat(muscle_targets); contact = torch.cat(contacts); ct = torch.cat(contact_targets)
    mm = corpus.muscle_mask; om = corpus.owner_mask
    hard = contact >= .5; truth = ct >= .5
    tp = int((hard & truth & om).sum()); fp = int((hard & ~truth & om).sum()); fn = int((~hard & truth & om).sum())
    return {
        "muscle_mae": round(float((muscle - mt).abs()[mm].mean()), 9),
        "contact_bce": round(float(F.binary_cross_entropy(contact[om].clamp(1e-6, 1-1e-6), ct[om])), 9),
        "contact_f1": round(2 * tp / max(1, 2 * tp + fp + fn), 9),
        "contact_iou": round(tp / max(1, tp + fp + fn), 9),
        "body_mae": round(float((torch.cat(bodies) - torch.cat(body_targets)).abs().mean()), 9),
    }


def train(output: Path, *, updates: int | None = None, device: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if sha256_file(PARENT) != PARENT_SHA256:
        raise ValueError("grounded controller parent changed")
    defaults = ControllerTrainingConfig(); config = ControllerTrainingConfig(updates=defaults.updates if updates is None else updates)
    target_device = torch.device(device)
    train_corpus = build_corpus(split="train", device=target_device)
    validation_corpus = build_corpus(split="validation", device=target_device)
    torch.manual_seed(0x434F4E54524F4C31); np.random.seed(0x434F4E31)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(0x434F4E54524F4C31); torch.cuda.reset_peak_memory_stats(target_device)
    model_config = ControllerModelConfig(); model = NeuralGroundedController(model_config).to(target_device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ema = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    positive = train_corpus.contact_target[train_corpus.owner_mask].sum()
    negative = train_corpus.owner_mask.sum() - positive
    positive_weight = torch.tensor(float(negative / positive.clamp_min(1)), device=target_device)
    history: list[dict[str, float | int]] = []; started = time.perf_counter()
    for update in range(1, config.updates + 1):
        generator = torch.Generator().manual_seed(0x434F4E54 ^ update * 0x9E3779B1)
        indices = torch.randint(0, train_corpus.samples, (config.batch_size,), generator=generator)
        next_indices = (indices // 72) * 72 + (indices % 72 + 1) % 72
        batch = train_corpus.batch(indices, target_device); next_batch = train_corpus.batch(next_indices, target_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
            current = _forward(model, batch); following = _forward(model, next_batch)
            loss, pieces = controller_loss(current, batch, positive_weight, following, next_batch)
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gradient)):
            raise FloatingPointError("grounded controller gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].mul_(config.ema_decay).add_(value.detach().cpu(), alpha=1-config.ema_decay)
        if update == 1 or update % 50 == 0 or update == config.updates:
            history.append({"update": update, **{name: round(float(value), 9) for name, value in pieces.items()}, "gradient_norm": round(float(gradient), 9)})
    elapsed = time.perf_counter() - started
    raw_validation = _validation(model.eval(), validation_corpus, target_device)
    raw_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema, strict=True); ema_validation = _validation(model, validation_corpus, target_device)
    checkpoint = {
        "format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(),
        "parent": {"path": PARENT.relative_to(PROJECT_ROOT).as_posix(), "sha256": PARENT_SHA256},
        "train_corpus_sha256": train_corpus.semantic_sha256, "validation_corpus_sha256": validation_corpus.semantic_sha256,
        "model": model_config.to_dict(), "training": config.to_dict(), "updates": config.updates,
        "model_state": raw_state, "ema_state": ema,
        "model_state_sha256": _state_sha256(raw_state), "ema_state_sha256": _state_sha256(ema),
        "history": history, "validation": {"raw": raw_validation, "ema": ema_validation},
    }
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    checkpoint_path = stage / f"grounded_controller_{config.updates:07d}.pt"; torch.save(checkpoint, checkpoint_path)
    report = {
        "format": FORMAT, "source_sha256": checkpoint["source_sha256"], "parent": checkpoint["parent"],
        "train_corpus_sha256": checkpoint["train_corpus_sha256"], "validation_corpus_sha256": checkpoint["validation_corpus_sha256"],
        "model": {**checkpoint["model"], "parameters": model.parameter_count}, "training": checkpoint["training"], "updates": config.updates,
        "history": history, "validation": checkpoint["validation"],
        "checkpoint": {"path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]},
        "runtime": {"seconds": round(elapsed, 6), "updates_per_second": round(config.updates / elapsed, 6), "device": str(target_device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else 0},
    }
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    (stage / "production_manifest.json").write_bytes(canonical_json_bytes(report)); os.replace(stage, output)
    return report


def load_model(checkpoint_path: Path, *, ema: bool = True, device: str | torch.device = "cpu") -> tuple[NeuralGroundedController, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).resolve(); payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("parent", {}).get("sha256") != PARENT_SHA256:
        raise ValueError("grounded controller checkpoint provenance drifted")
    model = NeuralGroundedController(ControllerModelConfig(**payload["model"])); name = "ema_state" if ema else "model_state"; hash_name = "ema_state_sha256" if ema else "model_state_sha256"
    if _state_sha256(payload[name]) != payload[hash_name]:
        raise ValueError("grounded controller state drifted")
    model.load_state_dict(payload[name], strict=True)
    return model.to(device).eval(), payload
