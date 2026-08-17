from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time
import uuid

import torch

from ..organism_raster_vae_v3.calibration import _canonical, _sha, _state_hash
from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..organism_raster_vae_v5_anatomical.dataset import AnatomicalGraphCorpus
from ..organism_raster_vae_v5_anatomical.model import AnatomicalGraphRasterVAE, loss
from ..organism_raster_vae_v5_anatomical.training import _batch, source_sha256 as parent_source_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, DEFAULT_OUTPUT, FORMAT, PARENT_CHECKPOINT, SOURCE_FILES, TrainingPlan
from ..config import PROJECT_ROOT


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-current-anatomical-raster-vae-v6\0")
    digest.update(parent_source_sha256().encode() + b"\0")
    digest.update(_sha(PARENT_CHECKPOINT).encode() + b"\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _latest(root: Path) -> Path | None:
    paths = sorted(root.glob("segment_*/checkpoint.pt")) if root.exists() else []
    return paths[-1] if paths else None


def _plan_dict(plan: TrainingPlan) -> dict[str, int | float]:
    return {
        "segment_steps": plan.segment_steps,
        "batch_size": plan.batch_size,
        "learning_rate": plan.learning_rate,
        "ema_decay": plan.ema_decay,
        "seed": plan.seed,
    }


def _identity(payload: dict[str, object]) -> str:
    selected = {
        key: payload[key]
        for key in (
            "format", "source_sha256", "corpus_sha256", "segment", "global_step",
            "model_state_sha256", "ema_state_sha256", "predecessor_sha256", "history",
        )
    }
    return hashlib.sha256(_canonical(selected)).hexdigest()


def _new_model(device: torch.device) -> tuple[AnatomicalGraphRasterVAE, str]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if parent.get("source_sha256") != parent_source_sha256():
        raise ValueError("V6 parent source drifted")
    model = AnatomicalGraphRasterVAE(RasterVAEV3Config(**parent["config"]))
    model.load_state_dict(parent["ema_state"], strict=True)
    return model.to(device), _sha(PARENT_CHECKPOINT)


def train_segment(root: Path = DEFAULT_OUTPUT, plan: TrainingPlan = TrainingPlan()) -> Path:
    root = Path(root).resolve()
    require_disk_floor(root.parent, floor_gb=100, planned_bytes=3 * 1024**3)
    if not torch.cuda.is_available():
        raise RuntimeError("V6 training requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    corpus = AnatomicalGraphCorpus()
    validation_identities = {5, 11, 17, 23, 29}
    train_indices = [index for index, (identity, _) in enumerate(corpus.rows) if identity not in validation_identities]
    latest = _latest(root)
    source = source_sha256()
    if latest is None:
        model, parent_sha = _new_model(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-5, fused=True)
        order_generator = torch.Generator().manual_seed(plan.seed ^ 0x4F52444552)
        latent_generator = torch.Generator(device=device).manual_seed(plan.seed ^ 0x4C4154454E54)
        order = torch.randperm(len(train_indices), generator=order_generator).tolist()
        cursor = global_step = 0
        segment = 1
        history: list[dict[str, float | int]] = []
        predecessor_sha256 = None
    else:
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source:
            raise ValueError("V6 checkpoint source drifted")
        if payload.get("corpus_sha256") != corpus.semantic_sha256 or payload.get("plan") != _plan_dict(plan):
            raise ValueError("V6 checkpoint corpus or plan drifted")
        model = AnatomicalGraphRasterVAE(RasterVAEV3Config(**payload["config"])).to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        model.load_state_dict(payload["model_state"], strict=True)
        ema.load_state_dict(payload["ema_state"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-5, fused=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        order_generator = torch.Generator(); order_generator.set_state(payload["order_generator_state"])
        latent_generator = torch.Generator(device=device); latent_generator.set_state(payload["latent_generator_state"])
        order = [int(value) for value in payload["order"]]
        cursor = int(payload["cursor"]); global_step = int(payload["global_step"])
        segment = int(payload["segment"]) + 1
        history = list(payload["history"])
        predecessor_sha256 = _sha(latest)
        parent_sha = str(payload["parent_checkpoint_sha256"])
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); started = time.perf_counter(); segment_history = []
    for local_step in range(plan.segment_steps):
        if cursor + plan.batch_size > len(order):
            order = torch.randperm(len(train_indices), generator=order_generator).tolist(); cursor = 0
        chosen = [train_indices[order[cursor + offset]] for offset in range(plan.batch_size)]
        cursor += plan.batch_size
        batch = _batch(corpus, chosen, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                batch["living"], batch["family"], batch["traits"], batch["phase"],
                batch["tokens"], batch["token_mask"], generator=latent_generator, stochastic=True,
            )
            value, metrics = loss(output, batch, model.config, 1.0)
        if not torch.isfinite(value):
            raise FloatingPointError("V6 training became non-finite")
        value.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()), plan.ema_decay)
            torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
        global_step += 1
        if local_step == 0 or (local_step + 1) % 20 == 0:
            row = {
                "step": global_step, "segment": segment,
                **{key: round(number, 8) for key, number in metrics.items()},
                "gradient_norm": round(gradient_norm, 8),
            }
            history.append(row); segment_history.append(row)
    seconds = time.perf_counter() - started
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    ema_state = {key: value.detach().cpu() for key, value in ema.state_dict().items()}
    payload = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source,
        "parent_checkpoint_sha256": parent_sha,
        "corpus_sha256": corpus.semantic_sha256,
        "config": model.config.to_dict(),
        "plan": _plan_dict(plan),
        "segment": segment,
        "global_step": global_step,
        "predecessor_sha256": predecessor_sha256,
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
    payload["checkpoint_identity_sha256"] = _identity(payload)
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".segment_{segment:03d}.tmp-{uuid.uuid4().hex}"
    final = root / f"segment_{segment:03d}"
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
        "checkpoint": {"sha256": _sha(checkpoint), "bytes": checkpoint.stat().st_size},
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
