from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
import uuid

import torch

from ..organism_raster_vae_v3.calibration import _canonical, _sha, _state_hash
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, DynamicsConfig, FORMAT, TrainingPlan
from .corpus import BodyTransitionCorpus, collate_graphs
from .model import LivingBodyDynamicsNet, loss


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "forge/living_body_substrate/__init__.py",
    "forge/living_body_substrate/contract.py",
    "forge/living_body_substrate/state.py",
    "forge/living_body_dynamics_nn/__init__.py",
    "forge/living_body_dynamics_nn/__main__.py",
    "forge/living_body_dynamics_nn/contract.py",
    "forge/living_body_dynamics_nn/corpus.py",
    "forge/living_body_dynamics_nn/model.py",
    "forge/living_body_dynamics_nn/training.py",
    "forge/living_body_dynamics_nn/evaluation.py",
)
VALIDATION_IDENTITIES = {5, 11, 17, 23, 29}


def source_manifest() -> dict[str, str]:
    return {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in SOURCE_FILES}


def source_sha256() -> str:
    return hashlib.sha256(_canonical({"format": FORMAT, "files": source_manifest()})).hexdigest()


def _latest(root: Path) -> Path | None:
    rows = sorted(root.glob("segment_*/checkpoint.pt")) if root.exists() else []
    return rows[-1] if rows else None


def _device_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def evaluate_rows(
    model: LivingBodyDynamicsNet,
    corpus: BodyTransitionCorpus,
    indices: list[int],
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    model.eval()
    sums = {"health_mae": 0.0, "fluid_mae": 0.0, "scar_mae": 0.0, "system_mae": 0.0, "healthy_drift": 0.0, "fluid_total_error": 0.0}
    healthy_nodes = 0
    graphs = 0
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        batch = _device_batch(collate_graphs([corpus[index] for index in chosen]), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cell, systems = model(batch)
        difference = (cell.float() - batch["target"].float()).abs()
        sums["health_mae"] += float(difference[:, 0].sum())
        sums["fluid_mae"] += float(difference[:, 1].sum())
        sums["scar_mae"] += float(difference[:, 2].sum())
        sums["system_mae"] += float((systems.float() - batch["systems"].float()).abs().sum())
        node_count = len(cell)
        graphs += len(chosen)
        healthy_graph = batch["action_kind"] == 0
        healthy_mask = healthy_graph[batch["graph_index"]]
        if bool(healthy_mask.any()):
            drift = (cell[healthy_mask, 0] - batch["features"][healthy_mask, 30]).abs()
            sums["healthy_drift"] += float(drift.sum())
            healthy_nodes += int(healthy_mask.sum())
        for graph in range(len(chosen)):
            mask = batch["graph_index"] == graph
            sums["fluid_total_error"] += abs(float(cell[mask, 1].sum() - batch["target"][mask, 1].sum())) / max(int(mask.sum()), 1)
        sums.setdefault("nodes", 0.0)
        sums["nodes"] += node_count
    nodes = sums.pop("nodes")
    return {
        "health_mae": round(sums["health_mae"] / nodes, 9),
        "fluid_mae": round(sums["fluid_mae"] / nodes, 9),
        "scar_mae": round(sums["scar_mae"] / nodes, 9),
        "system_mae": round(sums["system_mae"] / (graphs * 7), 9),
        "healthy_drift": round(sums["healthy_drift"] / max(healthy_nodes, 1), 9),
        "fluid_total_error_per_cell": round(sums["fluid_total_error"] / graphs, 9),
    }


def train_segment(root: Path, plan: TrainingPlan = TrainingPlan()) -> Path:
    root = root.resolve()
    require_disk_floor(root.parent, floor_gb=100, planned_bytes=1024**3)
    if not torch.cuda.is_available():
        raise RuntimeError("living body dynamics training requires CUDA")
    device = torch.device("cuda")
    corpus = BodyTransitionCorpus(repeats=4)
    train_indices = [index for index, row in enumerate(corpus.rows) if row[0] not in VALIDATION_IDENTITIES]
    validation_indices = [index for index, row in enumerate(corpus.rows) if row[0] in VALIDATION_IDENTITIES]
    source = source_sha256()
    latest = _latest(root)
    if latest is None:
        torch.manual_seed(plan.seed)
        torch.cuda.manual_seed_all(plan.seed)
        model = LivingBodyDynamicsNet(DynamicsConfig()).to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=2e-5, fused=True)
        order_generator = torch.Generator().manual_seed(plan.seed ^ 0x4F52444552)
        order = torch.randperm(len(train_indices), generator=order_generator).tolist()
        cursor = 0
        global_step = 0
        segment = 1
        history: list[dict[str, float | int]] = []
        predecessor = None
    else:
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source:
            raise ValueError("living dynamics resume source drifted")
        if payload["corpus_sha256"] != corpus.semantic_sha256:
            raise ValueError("living dynamics resume corpus drifted")
        model = LivingBodyDynamicsNet(DynamicsConfig(**payload["config"])).to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        model.load_state_dict(payload["model_state"], strict=True)
        ema.load_state_dict(payload["ema_state"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=2e-5, fused=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        order_generator = torch.Generator()
        order_generator.set_state(payload["order_generator_state"])
        order = [int(value) for value in payload["order"]]
        cursor = int(payload["cursor"])
        global_step = int(payload["global_step"])
        segment = int(payload["segment"]) + 1
        history = list(payload["history"])
        predecessor = _sha(latest)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    started = time.perf_counter()
    segment_history = []
    for local_step in range(plan.segment_steps):
        if cursor + plan.batch_size > len(order):
            order = torch.randperm(len(train_indices), generator=order_generator).tolist()
            cursor = 0
        chosen = [train_indices[order[cursor + offset]] for offset in range(plan.batch_size)]
        cursor += plan.batch_size
        batch = _device_batch(collate_graphs([corpus[index] for index in chosen]), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value, metrics = loss(model, batch)
        if not torch.isfinite(value):
            raise FloatingPointError("living dynamics loss became non-finite")
        value.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()), plan.ema_decay)
            torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
        global_step += 1
        if local_step == 0 or (local_step + 1) % 25 == 0:
            row = {"step": global_step, "segment": segment, **{key: round(item, 9) for key, item in metrics.items()}, "gradient_norm": round(gradient, 8)}
            history.append(row)
            segment_history.append(row)
    seconds = time.perf_counter() - started
    validation = evaluate_rows(ema, corpus, validation_indices, device)
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    ema_state = {key: value.detach().cpu() for key, value in ema.state_dict().items()}
    payload = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source,
        "source_manifest": source_manifest(),
        "corpus_sha256": corpus.semantic_sha256,
        "config": model.config_dict(),
        "plan": {"segment_steps": plan.segment_steps, "batch_size": plan.batch_size, "learning_rate": plan.learning_rate, "ema_decay": plan.ema_decay, "seed": plan.seed},
        "segment": segment, "global_step": global_step, "predecessor_sha256": predecessor,
        "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(),
        "order_generator_state": order_generator.get_state(), "order": order, "cursor": cursor,
        "history": history, "segment_history": segment_history, "validation": validation,
        "model_state_sha256": _state_hash(model_state), "ema_state_sha256": _state_hash(ema_state),
        "runtime": {"seconds": round(seconds, 6), "steps_per_second": round(plan.segment_steps / seconds, 6), "device": torch.cuda.get_device_name(device), "parameters": sum(parameter.numel() for parameter in model.parameters()), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
    }
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".segment_{segment:03d}.tmp-{uuid.uuid4().hex}"
    final = root / f"segment_{segment:03d}"
    staging.mkdir()
    checkpoint = staging / "checkpoint.pt"
    torch.save(payload, checkpoint)
    manifest = {
        "format": FORMAT, "status": "training_segment_complete", "source_sha256": source,
        "corpus_sha256": corpus.semantic_sha256, "segment": segment, "global_step": global_step,
        "predecessor_sha256": predecessor,
        "checkpoint": {"path": checkpoint.name, "sha256": _sha(checkpoint), "bytes": checkpoint.stat().st_size, "model_state_sha256": payload["model_state_sha256"], "ema_state_sha256": payload["ema_state_sha256"]},
        "runtime": payload["runtime"], "training": segment_history[-1], "validation": validation,
        "gates": {"finite": True, "healthy_drift_below_005": validation["healthy_drift"] < .005, "system_mae_below_05": validation["system_mae"] < .05, "fluid_total_error_below_02": validation["fluid_total_error_per_cell"] < .02, "production_promotion_allowed": False},
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    (staging / "segment_manifest.json").write_bytes(_canonical(manifest))
    if final.exists():
        raise FileExistsError(final)
    staging.replace(final)
    return final / "checkpoint.pt"
