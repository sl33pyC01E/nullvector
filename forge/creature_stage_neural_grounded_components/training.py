from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
import torch
from torch import Tensor

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import CellularMotionTransformerConfig
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_grounded_cyclic.contract import CyclicModelConfig, canonical_json_bytes, sha256_file
from ..creature_stage_neural_grounded_cyclic.training import load_model as load_cyclic_model
from .contract import CHECKPOINT_FORMAT, FORMAT, MAX_APPENDAGES, PARENT, PARENT_SHA256, ComponentModelConfig, ComponentTrainingConfig, source_sha256
from .dataset import ComponentCurriculumTeacher, ComponentSentinelTeacher
from .model import ComponentMotionOutput, NeuralComponentGroundedMotion


def _mean(value: Tensor, mask: Tensor) -> Tensor:
    active = mask[:, :, None].to(value.dtype)
    return (value * active).sum() / (active.sum().clamp_min(1) * value.shape[-1])


def _forward(model: NeuralComponentGroundedMotion, batch: dict[str, Tensor], state: Tensor) -> ComponentMotionOutput:
    return model(batch["static"], state, batch["dynamic"], batch["owner"], batch["mask"], batch["adjacency"], batch["family"], batch["morphotype"], batch["motion"], batch["phase"], batch["controls"])


def component_loss(output: ComponentMotionOutput, batch: dict[str, Tensor], state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    mask = batch["mask"]; target = batch["target"].float(); predicted = output.cells.float()
    pos_error = (predicted[:, :, :2] - target[:, :, :2]).abs(); vel_error = (predicted[:, :, 2:] - target[:, :, 2:]).abs()
    position = _mean(pos_error, mask); velocity = _mean(vel_error, mask)
    appendage = (batch["owner"] >= 0) & mask; contact = (batch["dynamic"][:, :, 5] > .5) & mask
    appendage_loss = _mean(pos_error, appendage); contact_loss = _mean(pos_error, contact) if bool(contact.any()) else position * 0
    one_hot = torch.nn.functional.one_hot(batch["owner"].clamp_min(0), MAX_APPENDAGES).to(predicted.dtype) * appendage[:, :, None]
    count = one_hot.sum(1).clamp_min(1)
    pred_owner = torch.bmm(one_hot.transpose(1, 2), predicted[:, :, :2]) / count[:, :, None]
    target_owner = torch.bmm(one_hot.transpose(1, 2), target[:, :, :2]) / count[:, :, None]
    owner_active = count > 1
    owner_loss = (pred_owner - target_owner).abs()[owner_active].mean() if bool(owner_active.any()) else position * 0
    adjacency = batch["adjacency"].to(predicted.dtype); degree = adjacency.sum(2, keepdim=True).clamp_min(1)
    graph = _mean(((predicted[:, :, :2] - torch.bmm(adjacency, predicted[:, :, :2]) / degree) - (target[:, :, :2] - torch.bmm(adjacency, target[:, :, :2]) / degree)).abs(), mask)
    body = (output.body_velocity.float() - batch["body_target"].float()).abs().mean()
    baseline = _mean((state[:, :, :2].float() - target[:, :, :2]).abs(), mask).detach()
    copy_margin = torch.relu(position - baseline * .84)
    correction = _mean(output.local_correction.abs(), mask) + output.owner_translation.abs().mean()
    outside = output.cells[~mask].abs().max()
    total = position * 1.5 + velocity * .35 + appendage_loss * .9 + contact_loss * 1.25 + owner_loss * .8 + graph * .2 + body * .3 + copy_margin + correction * .015 + outside
    if not bool(torch.isfinite(total)): raise FloatingPointError("component motion loss became non-finite")
    return total, {"loss": total.detach(), "position_l1": position.detach(), "velocity_l1": velocity.detach(), "appendage_l1": appendage_loss.detach(), "contact_l1": contact_loss.detach(), "owner_l1": owner_loss.detach(), "graph_l1": graph.detach(), "body_l1": body.detach(), "copy_margin": copy_margin.detach(), "correction_l1": correction.detach(), "outside": outside.detach()}


def _batch_at(teacher: ComponentCurriculumTeacher, frame: int, slot: int, device: torch.device) -> dict[str, Tensor]:
    identities = [rows[slot % len(rows)] for rows in teacher.family_indices]
    rows = [teacher.sample(identity, frame) for identity in identities]
    result = {name: torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device) for name in ("static", "state", "target", "dynamic", "mask", "adjacency", "controls")}
    result["owner"] = torch.from_numpy(np.stack([teacher.arrays["appendage_owner"][identity] for identity in identities]).astype(np.int64)).to(device)
    for name in ("family", "morphotype", "motion"): result[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
    for name in ("phase", "body_target"): result[name] = torch.tensor([float(row[name]) for row in rows], device=device)
    return result


def _seam_loss(model: NeuralComponentGroundedMotion, teacher: ComponentCurriculumTeacher, slot: int, device: torch.device) -> tuple[Tensor, Tensor]:
    left = _batch_at(teacher, 71, slot, device); right = _batch_at(teacher, 0, slot, device)
    out_left = _forward(model, left, left["state"]); out_right = _forward(model, right, out_left.cells.detach())
    predicted = out_right.cells[:, :, :2].float() - out_left.cells[:, :, :2].float()
    target = right["target"][:, :, :2].float() - left["target"][:, :, :2].float()
    error = (predicted - target).abs(); active = error[right["mask"][:, :, None].expand_as(error)]
    return active.mean(), torch.topk(active, max(1, int(active.numel() * .05))).values.mean()


def train(output: Path, *, updates: int | None = None, device: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    defaults = ComponentTrainingConfig(); config = ComponentTrainingConfig(updates=defaults.updates if updates is None else updates)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if sha256_file(PARENT) != PARENT_SHA256: raise ValueError("component motion parent bytes drifted")
    target_device = torch.device(device); parent_model, parent = load_cyclic_model(PARENT, ema=True, device="cpu")
    teacher = ComponentCurriculumTeacher(); sentinel = ComponentSentinelTeacher()
    if parent["teacher_semantic_sha256"] != teacher.semantic_sha256 or parent["sentinel_semantic_sha256"] != sentinel.semantic_sha256:
        raise ValueError("component motion authority lineage drifted")
    model_config = ComponentModelConfig()
    model = NeuralComponentGroundedMotion(CellularMotionTransformerConfig(**parent["backbone"]), CyclicModelConfig(**parent["model"]), model_config)
    model.base.load_state_dict(parent_model.state_dict(), strict=True); model.to(target_device).train()
    optimizer = torch.optim.AdamW([
        {"params": model.base.backbone.parameters(), "lr": 3e-7},
        {"params": [parameter for name, parameter in model.base.named_parameters() if not name.startswith("backbone.")], "lr": 3e-6},
        {"params": [parameter for name, parameter in model.named_parameters() if not name.startswith("base.")], "lr": config.learning_rate},
    ], weight_decay=1e-5)
    ema = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; history=[]; started=time.perf_counter()
    for update in range(1, config.updates + 1):
        frames = teacher.sequence(update + 1200, config.batch_size, config.sequence_frames, target_device, split="train")
        state = frames[0]["state"]; optimizer.zero_grad(set_to_none=True); losses=[]; totals={}
        for batch in frames:
            with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"): result = _forward(model, batch, state)
            loss, pieces = component_loss(result, batch, state); losses.append(loss)
            for name, value in pieces.items(): totals[name] = totals.get(name, 0.0) + float(value)
            state = result.cells.detach()
        with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"): seam_mean, seam_tail = _seam_loss(model, teacher, update, target_device)
        loss = torch.stack(losses).mean() + (seam_mean + seam_tail) * .7; loss.backward(); gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        if not math.isfinite(float(gradient)): raise FloatingPointError("component motion gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items(): ema[name].mul_(config.ema_decay).add_(value.detach().cpu(), alpha=1-config.ema_decay)
        if update == 1 or update % 25 == 0 or update == config.updates:
            history.append({"update": 1200 + update, **{name: round(value/config.sequence_frames,9) for name,value in totals.items()}, "seam_mean_l1":round(float(seam_mean),9), "seam_top5_l1":round(float(seam_tail),9), "gradient_norm":round(float(gradient),9)})
    seconds=time.perf_counter()-started; model_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    checkpoint={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"parent":{"path":PARENT.relative_to(PROJECT_ROOT).as_posix(),"sha256":PARENT_SHA256,"ema_state_sha256":parent["ema_state_sha256"]},"teacher_semantic_sha256":teacher.semantic_sha256,"sentinel_semantic_sha256":sentinel.semantic_sha256,"split":parent["split"],"runtime_honest_contacts":True,"updates":1200+config.updates,"component_updates":config.updates,"backbone":parent["backbone"],"cyclic_model":parent["model"],"model":model_config.to_dict(),"training":config.to_dict(),"model_state":model_state,"ema_state":ema,"model_state_sha256":_state_sha256(model_state),"ema_state_sha256":_state_sha256(ema),"history":history}
    stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True);cp=stage/f"component_grounded_motion_{1200+config.updates:07d}.pt";torch.save(checkpoint,cp)
    report={key:checkpoint[key] for key in ("format","source_sha256","parent","teacher_semantic_sha256","sentinel_semantic_sha256","split","runtime_honest_contacts","updates","component_updates","backbone","cyclic_model","model","training","history")};report["format"]=FORMAT;report["checkpoint"]={"path":cp.name,"bytes":cp.stat().st_size,"sha256":sha256_file(cp),"model_state_sha256":checkpoint["model_state_sha256"],"ema_state_sha256":checkpoint["ema_state_sha256"]};report["runtime"]={"seconds":round(seconds,6),"updates_per_second":round(config.updates/seconds,6),"device":str(target_device),"peak_allocated_bytes":torch.cuda.max_memory_allocated() if target_device.type=="cuda" else 0};report["semantic_sha256"]=hashlib.sha256(canonical_json_bytes(report)).hexdigest();(stage/"production_manifest.json").write_bytes(canonical_json_bytes(report));os.replace(stage,output);return report


def load_model(checkpoint_path: Path, *, ema: bool = True, device: str | torch.device = "cpu") -> tuple[NeuralComponentGroundedMotion, dict[str, Any]]:
    checkpoint_path=Path(checkpoint_path).resolve();payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256() or payload.get("runtime_honest_contacts") is not True: raise ValueError("component motion checkpoint provenance drifted")
    model=NeuralComponentGroundedMotion(CellularMotionTransformerConfig(**payload["backbone"]),CyclicModelConfig(**payload["cyclic_model"]),ComponentModelConfig(**payload["model"]));name="ema_state" if ema else "model_state";hash_name="ema_state_sha256" if ema else "model_state_sha256"
    if _state_sha256(payload[name])!=payload[hash_name]:raise ValueError("component motion checkpoint state drifted")
    model.load_state_dict(payload[name],strict=True);return model.to(device).eval(),payload
