from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
import uuid

import torch
from torch import Tensor

from ..organism_raster_vae_v3.calibration import _canonical, _sha, _state_hash
from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..organism_raster_vae_v4_graph.calibration import source_sha256 as v4_source_sha256
from ..organism_raster_vae_v4_graph.model import GraphTokenRasterVAE
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FORMAT, TrainingPlan
from .dataset import AnatomicalGraphCorpus
from .model import AnatomicalGraphRasterVAE, loss


ROOT = Path(__file__).resolve().parents[2]
PARENT_CHECKPOINT = ROOT / "outputs/organism_raster_vae_v4_graph/calibration_0100_fresh_replay/graph_token_calibration.pt"
SOURCE_FILES = (
    "forge/organism_raster_vae_v5_anatomical/__init__.py",
    "forge/organism_raster_vae_v5_anatomical/__main__.py",
    "forge/organism_raster_vae_v5_anatomical/contract.py",
    "forge/organism_raster_vae_v5_anatomical/dataset.py",
    "forge/organism_raster_vae_v5_anatomical/model.py",
    "forge/organism_raster_vae_v5_anatomical/training.py",
    "forge/organism_raster_vae_v5_anatomical/evaluation.py",
)


def source_manifest() -> dict[str, str]:
    return {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in SOURCE_FILES}


def source_sha256() -> str:
    payload = {
        "format": FORMAT,
        "files": source_manifest(),
        "parent_checkpoint_sha256": _sha(PARENT_CHECKPOINT),
        "parent_source_sha256": v4_source_sha256(),
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _batch(corpus: AnatomicalGraphCorpus, indices: list[int], device: torch.device) -> dict[str, Tensor]:
    rows = [corpus[index] for index in indices]
    return {key: torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}


def _load_parent(device: torch.device) -> tuple[dict[str, Tensor], dict[str, object]]:
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if payload["source_sha256"] != v4_source_sha256():
        raise ValueError("anatomical VAE parent source drifted")
    if payload["format"] != "nullvector-graph-token-raster-vae-v4-checkpoint/1.0.0":
        raise ValueError("anatomical VAE parent format drifted")
    parent = GraphTokenRasterVAE(RasterVAEV3Config(**payload["config"]))
    parent.load_state_dict(payload["ema_state"], strict=True)
    return parent.state_dict(), payload


def _warm_start(device: torch.device) -> tuple[AnatomicalGraphRasterVAE, int]:
    parent, _ = _load_parent(device)
    model = AnatomicalGraphRasterVAE().to(device)
    state = model.state_dict()
    copied = 0
    for name, value in parent.items():
        if name in state and state[name].shape == value.shape:
            state[name].copy_(value)
            copied += 1
    model.load_state_dict(state, strict=True)
    return model, copied


def _latest(root: Path) -> Path | None:
    checkpoints = sorted(root.glob("segment_*/checkpoint.pt")) if root.exists() else []
    return checkpoints[-1] if checkpoints else None


def _checkpoint_identity(payload: dict[str, object]) -> str:
    selected = {
        "format": payload["format"],
        "source_sha256": payload["source_sha256"],
        "corpus_sha256": payload["corpus_sha256"],
        "segment": payload["segment"],
        "global_step": payload["global_step"],
        "model_state_sha256": payload["model_state_sha256"],
        "ema_state_sha256": payload["ema_state_sha256"],
        "predecessor_sha256": payload["predecessor_sha256"],
        "history": payload["history"],
    }
    return hashlib.sha256(_canonical(selected)).hexdigest()


def train_segment(root: Path, plan: TrainingPlan = TrainingPlan()) -> Path:
    root = root.resolve()
    require_disk_floor(root.parent, floor_gb=100, planned_bytes=3 * 1024**3)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("anatomical production segment requires CUDA")
    corpus = AnatomicalGraphCorpus()
    validation_identities = {5, 11, 17, 23, 29}
    train_indices = [
        index for index, (identity, _) in enumerate(corpus.rows) if identity not in validation_identities
    ]
    latest = _latest(root)
    source = source_sha256()

    if latest is None:
        model, copied = _warm_start(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=plan.learning_rate, weight_decay=1e-5, fused=True
        )
        order_generator = torch.Generator().manual_seed(plan.seed ^ 0x4F52444552)
        latent_generator = torch.Generator(device=device).manual_seed(plan.seed ^ 0x4C4154454E54)
        order = torch.randperm(len(train_indices), generator=order_generator).tolist()
        cursor = 0
        global_step = 0
        segment = 1
        history: list[dict[str, float | int]] = []
        predecessor_sha256 = None
    else:
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source:
            raise ValueError("anatomical checkpoint source or format drifted")
        if payload["corpus_sha256"] != corpus.semantic_sha256:
            raise ValueError("anatomical checkpoint corpus drifted")
        if payload["plan"] != {
            "segment_steps": plan.segment_steps,
            "batch_size": plan.batch_size,
            "learning_rate": plan.learning_rate,
            "ema_decay": plan.ema_decay,
            "seed": plan.seed,
        }:
            raise ValueError("anatomical resume plan drifted")
        model = AnatomicalGraphRasterVAE(RasterVAEV3Config(**payload["config"])).to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        model.load_state_dict(payload["model_state"], strict=True)
        ema.load_state_dict(payload["ema_state"], strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=plan.learning_rate, weight_decay=1e-5, fused=True
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        order_generator = torch.Generator()
        order_generator.set_state(payload["order_generator_state"])
        latent_generator = torch.Generator(device=device)
        latent_generator.set_state(payload["latent_generator_state"])
        order = [int(value) for value in payload["order"]]
        cursor = int(payload["cursor"])
        global_step = int(payload["global_step"])
        segment = int(payload["segment"]) + 1
        history = list(payload["history"])
        predecessor_sha256 = _sha(latest)
        copied = int(payload["warm_start_parameters_copied"])

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    started = time.perf_counter()
    segment_history: list[dict[str, float | int]] = []
    for local_step in range(plan.segment_steps):
        if cursor + plan.batch_size > len(order):
            order = torch.randperm(len(train_indices), generator=order_generator).tolist()
            cursor = 0
        chosen = [train_indices[order[cursor + offset]] for offset in range(plan.batch_size)]
        cursor += plan.batch_size
        batch = _batch(corpus, chosen, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                batch["living"],
                batch["family"],
                batch["traits"],
                batch["phase"],
                batch["tokens"],
                batch["token_mask"],
                generator=latent_generator,
                stochastic=True,
            )
            value, metrics = loss(output, batch, model.config, min(1.0, (global_step + 1) / 240))
        if not torch.isfinite(value):
            raise FloatingPointError("anatomical graph training became non-finite")
        value.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()), plan.ema_decay)
            torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
        global_step += 1
        if local_step == 0 or (local_step + 1) % 20 == 0:
            row: dict[str, float | int] = {
                "step": global_step,
                "segment": segment,
                **{key: round(value, 8) for key, value in metrics.items()},
                "gradient_norm": round(gradient_norm, 8),
                "gate12": round(float(torch.tanh(model.gate12).detach()), 8),
                "gate24": round(float(torch.tanh(model.gate24).detach()), 8),
            }
            history.append(row)
            segment_history.append(row)

    seconds = time.perf_counter() - started
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    ema_state = {key: value.detach().cpu() for key, value in ema.state_dict().items()}
    payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source,
        "source_manifest": source_manifest(),
        "parent_checkpoint_sha256": _sha(PARENT_CHECKPOINT),
        "corpus_sha256": corpus.semantic_sha256,
        "config": model.config.to_dict(),
        "plan": {
            "segment_steps": plan.segment_steps,
            "batch_size": plan.batch_size,
            "learning_rate": plan.learning_rate,
            "ema_decay": plan.ema_decay,
            "seed": plan.seed,
        },
        "segment": segment,
        "global_step": global_step,
        "predecessor_sha256": predecessor_sha256,
        "warm_start_parameters_copied": copied,
        "model_state": model_state,
        "ema_state": ema_state,
        "optimizer_state": optimizer.state_dict(),
        "order_generator_state": order_generator.get_state(),
        "latent_generator_state": latent_generator.get_state(),
        "order": order,
        "cursor": cursor,
        "history": history,
        "segment_history": segment_history,
        "model_state_sha256": _state_hash(model_state),
        "ema_state_sha256": _state_hash(ema_state),
        "runtime": {
            "seconds": round(seconds, 6),
            "steps_per_second": round(plan.segment_steps / seconds, 6),
            "device": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    payload["checkpoint_identity_sha256"] = _checkpoint_identity(payload)
    staging = root / f".segment_{segment:03d}.tmp-{uuid.uuid4().hex}"
    final = root / f"segment_{segment:03d}"
    root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    checkpoint = staging / "checkpoint.pt"
    torch.save(payload, checkpoint)
    summary = {
        "format": FORMAT,
        "status": "training_segment_complete",
        "source_sha256": source,
        "corpus_sha256": corpus.semantic_sha256,
        "segment": segment,
        "global_step": global_step,
        "checkpoint": {
            "path": "checkpoint.pt",
            "sha256": _sha(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "identity_sha256": payload["checkpoint_identity_sha256"],
        },
        "predecessor_sha256": predecessor_sha256,
        "runtime": payload["runtime"],
        "latest_metrics": segment_history[-1],
        "production_promotion_allowed": False,
    }
    summary["summary_sha256"] = hashlib.sha256(_canonical(summary)).hexdigest()
    (staging / "segment_manifest.json").write_bytes(_canonical(summary))
    if final.exists():
        raise FileExistsError(final)
    staging.replace(final)
    return final / "checkpoint.pt"
