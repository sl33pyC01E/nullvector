from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_grounded_cyclic.contract import CHECKPOINT_FORMAT, canonical_json_bytes, sha256_file
from ..creature_stage_neural_grounded_cyclic.curriculum import CurriculumGroundedTeacher
from ..creature_stage_neural_grounded_cyclic.training import _forward, _seam_loss, cyclic_loss, load_model


FORMAT = "nullvector-neural-grounded-cyclic-continuation-v1"
PARENT = PROJECT_ROOT / "outputs/creature_stage_neural_grounded_cyclic/curriculum_pilot_0600_sealed/cyclic_grounded_motion_0000600.pt"
PARENT_SHA256 = "408ae5f8a08d98988953cdc1abe8362712a7241f86e07bedb7cdaffa7441da1d"
SOURCE_FILES = (
    "forge/creature_stage_neural_grounded_cyclic_continue/__init__.py",
    "forge/creature_stage_neural_grounded_cyclic_continue/__main__.py",
    "forge/creature_stage_neural_grounded_cyclic_continue/training.py",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-grounded-cyclic-continuation-v1\0" + PARENT_SHA256.encode("ascii"))
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def train(output: Path, *, updates: int = 600, device: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not 100 <= updates <= 1800: raise ValueError("cyclic continuation update count drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if sha256_file(PARENT) != PARENT_SHA256: raise ValueError("cyclic continuation parent bytes drifted")
    target_device = torch.device(device)
    model, parent = load_model(PARENT, ema=True, device=target_device)
    if parent.get("updates") != 600 or parent.get("runtime_honest_contacts") is not True:
        raise ValueError("cyclic continuation parent contract drifted")
    teacher = CurriculumGroundedTeacher()
    if parent["teacher_semantic_sha256"] != teacher.semantic_sha256:
        raise ValueError("cyclic continuation teacher drifted")
    model.train()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1.2e-6},
        {"params": [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")], "lr": 3e-5},
    ], weight_decay=1e-5)
    ema = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for local_update in range(1, updates + 1):
        total_update = 600 + local_update
        frames = teacher.sequence(total_update, 5, 8, target_device, split="train")
        state = frames[0]["state"]; optimizer.zero_grad(set_to_none=True)
        losses = []; totals: dict[str, float] = {}
        for batch in frames:
            with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
                result = _forward(model, batch, state)
            loss, pieces = cyclic_loss(result, batch, state); losses.append(loss)
            for name, value in pieces.items(): totals[name] = totals.get(name, 0.0) + float(value)
            state = result.cells.detach()
        with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
            seam_mean, seam_tail = _seam_loss(model, teacher, total_update, target_device)
        loss = torch.stack(losses).mean() + (seam_mean + seam_tail) * .60
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gradient)): raise FloatingPointError("cyclic continuation gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items(): ema[name].mul_(.99).add_(value.detach().cpu(), alpha=.01)
        if local_update == 1 or local_update % 25 == 0 or local_update == updates:
            history.append({
                "update": total_update,
                **{name: round(value / 8, 9) for name, value in totals.items()},
                "seam_mean_l1": round(float(seam_mean), 9), "seam_top5_l1": round(float(seam_tail), 9),
                "gradient_norm": round(float(gradient), 9),
            })
    seconds = time.perf_counter() - started
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint = dict(parent)
    checkpoint.update({
        "updates": 600 + updates, "model_state": model_state, "ema_state": ema,
        "model_state_sha256": _state_sha256(model_state), "ema_state_sha256": _state_sha256(ema),
        "history": list(parent["history"]) + history,
        "continuation": {"format": FORMAT, "source_sha256": source_sha256(), "parent_sha256": PARENT_SHA256, "updates": updates},
    })
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    checkpoint_path = stage / f"cyclic_grounded_motion_{600 + updates:07d}.pt"; torch.save(checkpoint, checkpoint_path)
    report = {
        "format": FORMAT, "source_sha256": source_sha256(), "parent_sha256": PARENT_SHA256,
        "teacher_semantic_sha256": teacher.semantic_sha256, "updates": updates, "total_updates": 600 + updates,
        "checkpoint": {"path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]},
        "history": history,
        "runtime": {"seconds": round(seconds, 6), "updates_per_second": round(updates / seconds, 6), "device": str(target_device)},
    }
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    (stage / "production_manifest.json").write_bytes(canonical_json_bytes(report)); os.replace(stage, output)
    return report
