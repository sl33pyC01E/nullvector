from __future__ import annotations

import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from ..sprite_latent.codec import SemanticSpriteFSQ
from ..sprite_latent.training import canonical_state_hash, load_production_training_contract
from .checkpoint import load_checkpoint, save_checkpoint_new
from .contract import (
    CHECKPOINT_FORMAT,
    MIN_FREE_CUDA_BYTES,
    SEGMENT_FORMAT,
    ProductionConfig,
    canonical_json_bytes,
    production_source_hash,
    sha256_bytes,
    sha256_file,
)
from .evaluation import batch_from_indices, evaluate_model
from .loss import deterministic_sprite_codec_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed & 0xFFFFFFFF); torch.manual_seed(seed & 0x7FFFFFFF)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed & 0x7FFFFFFF)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor): state[key] = value.to(device)


def _ema_update(ema: dict[str, torch.Tensor], model: SemanticSpriteFSQ, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            source = value.detach()
            if torch.is_floating_point(source): ema[name].mul_(decay).add_(source, alpha=1.0 - decay)
            else: ema[name].copy_(source)


def _learning_rate(config: ProductionConfig, step: int, total_steps: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (config.learning_rate - config.minimum_learning_rate) * cosine


def _provenance(contract: dict[str, Any]) -> dict[str, str]:
    return {
        "corpus_sha256": contract["corpus"].file_sha256,
        "split_fingerprint": contract["split"].fingerprint,
        "legal_tuple_fingerprint": contract["legal_tuple_fingerprint"],
    }


def _assert_provenance(checkpoint: dict[str, Any], contract: dict[str, Any], config: ProductionConfig) -> None:
    expected = _provenance(contract)
    if checkpoint["config"] != config.metadata(): raise ValueError("resume config mismatch")
    for key, value in expected.items():
        if checkpoint[key] != value: raise ValueError(f"resume {key} mismatch")
    if checkpoint["partial_epoch"] is not None:
        raise ValueError("partial calibration checkpoints are evidence-only and cannot be resumed")


def _make_checkpoint(
    *, model: SemanticSpriteFSQ, ema: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
    config: ProductionConfig, contract: dict[str, Any], epoch: int, global_step: int,
    history: list[dict[str, Any]], partial_epoch: dict[str, Any] | None,
    previous_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    model_state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    ema_state = {name: value.detach().cpu().contiguous() for name, value in ema.items()}
    model_for_hash = SemanticSpriteFSQ(config.codec_config()); model_for_hash.load_state_dict(model_state)
    ema_for_hash = SemanticSpriteFSQ(config.codec_config()); ema_for_hash.load_state_dict(ema_state)
    provenance = _provenance(contract)
    return {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": production_source_hash(),
        **provenance,
        "config": config.metadata(),
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model_state,
        "ema_state": ema_state,
        "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": canonical_state_hash(model_for_hash),
        "ema_state_sha256": canonical_state_hash(ema_for_hash),
        "history": history,
        "partial_epoch": partial_epoch,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "rng": {
            "torch": torch.get_rng_state().cpu(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def run_segment(
    *, corpus_path: Path, output: Path, config: ProductionConfig,
    start_epoch: int, end_epoch: int, resume: Path | None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    if start_epoch < 0 or end_epoch <= start_epoch or end_epoch > config.epochs:
        raise ValueError("invalid production segment epoch bounds")
    if end_epoch - start_epoch > config.segment_epochs and max_steps is None:
        raise ValueError("production worker cannot exceed the declared segment size")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("production calibration max_steps must be positive")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("production sprite latent worker requires CUDA")
    free_cuda_bytes, _ = torch.cuda.mem_get_info()
    if free_cuda_bytes < MIN_FREE_CUDA_BYTES:
        raise RuntimeError("production sprite latent worker lacks the minimum free CUDA memory")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    _seed_everything(config.seed)
    contract = load_production_training_contract(corpus_path)
    corpus = contract["corpus"]; split = contract["split"]
    legal = contract["legal_tuples"].to(device)
    weights = {name: value.to(device) for name, value in contract["class_weights"].items()}
    model = SemanticSpriteFSQ(config.codec_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=True)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    initial_history_length = 0
    global_step = 0; previous_checkpoint_sha256 = None
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location="cpu"); _assert_provenance(checkpoint, contract, config)
        if int(checkpoint["epoch"]) != start_epoch: raise ValueError("resume epoch does not match segment start")
        model.load_state_dict(checkpoint["model_state"], strict=True); optimizer.load_state_dict(checkpoint["optimizer_state"]); _optimizer_to(optimizer, device)
        ema = {name: value.to(device) for name, value in checkpoint["ema_state"].items()}
        history = list(checkpoint["history"]); global_step = int(checkpoint["global_step"]); previous_checkpoint_sha256 = sha256_file(resume)
        initial_history_length = len(history)
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if checkpoint["rng"]["cuda"]: torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    steps_per_epoch = math.ceil(len(split.training) / config.batch_size)
    total_steps = config.epochs * steps_per_epoch
    started = time.perf_counter(); training_steps = 0; stopped_early = False
    segment_history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, end_epoch):
        generator = torch.Generator(device="cpu"); generator.manual_seed((config.seed + epoch * 0x9E3779B1) & 0x7FFFFFFFFFFFFFFF)
        order = torch.randperm(len(split.training), generator=generator).numpy()
        epoch_sums: dict[str, float] = {}; epoch_steps = 0; limit_reached = False
        quantize = epoch >= config.continuous_warmup_epochs
        model.train()
        for begin in range(0, len(order), config.batch_size):
            indices = split.training[order[begin : begin + config.batch_size]]
            batch = batch_from_indices(corpus, indices, device)
            lr = _learning_rate(config, global_step, total_steps)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_values = model(batch["part"], batch["material"], batch["emission"], batch["morphology"], batch["subtype"], batch["role"], batch["genes"], quantize=quantize)
                loss, pieces = deterministic_sprite_codec_loss(
                    output_values,
                    batch["part"],
                    batch["material"],
                    batch["emission"],
                    legal,
                    config=model.config,
                    class_weights=weights,
                )
            if not bool(torch.isfinite(loss)): raise FloatingPointError("production loss became non-finite")
            loss.backward(); gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not bool(torch.isfinite(gradient_norm)): raise FloatingPointError("production gradient became non-finite")
            optimizer.step(); _ema_update(ema, model, config.ema_decay)
            values = {name: float(value) for name, value in pieces.items()}; values["gradient_norm"] = float(gradient_norm); values["learning_rate"] = lr
            for name, value in values.items(): epoch_sums[name] = epoch_sums.get(name, 0.0) + value
            epoch_steps += 1; training_steps += 1; global_step += 1
            epoch_complete = begin + config.batch_size >= len(order)
            if max_steps is not None and training_steps >= max_steps:
                limit_reached = True
                if not epoch_complete:
                    stopped_early = True
                break
        record = {
            "epoch": epoch + 1,
            "complete": not stopped_early,
            "quantized": quantize,
            "steps": epoch_steps,
            **{name: value / max(1, epoch_steps) for name, value in epoch_sums.items()},
        }
        segment_history.append(record)
        if not stopped_early:
            history.append(record)
        if limit_reached:
            if not stopped_early and epoch + 1 < end_epoch:
                stopped_early = True
            break
    completed_epoch = start_epoch + (len(history) - initial_history_length)
    if not stopped_early and completed_epoch != end_epoch:
        raise RuntimeError("production segment did not complete its declared epoch range")
    partial_epoch = (
        segment_history[-1]
        if segment_history and segment_history[-1]["complete"] is False
        else None
    )
    checkpoint_payload = _make_checkpoint(
        model=model,
        ema=ema,
        optimizer=optimizer,
        config=config,
        contract=contract,
        epoch=completed_epoch,
        global_step=global_step,
        history=history,
        partial_epoch=partial_epoch,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
    )
    checkpoint_path = output / "checkpoint.pt"; checkpoint_sha256 = save_checkpoint_new(checkpoint_path, checkpoint_payload)
    ema_model = SemanticSpriteFSQ(config.codec_config()).to(device); ema_model.load_state_dict(ema, strict=True)
    evaluation = evaluate_model(ema_model, corpus, split.validation, legal, batch_size=config.evaluation_batch_size, device=device)
    elapsed = time.perf_counter() - started
    report = {
        "format": SEGMENT_FORMAT,
        "status": "passed",
        "source_sha256": production_source_hash(),
        **_provenance(contract),
        "config": config.metadata(),
        "start_epoch": start_epoch,
        "requested_end_epoch": end_epoch,
        "end_epoch": completed_epoch,
        "global_step": global_step,
        "training_steps": training_steps,
        "stopped_early": stopped_early,
        "elapsed_seconds": elapsed,
        "steps_per_second": training_steps / max(elapsed, 1e-9),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "checkpoint": {"path": "checkpoint.pt", "bytes": checkpoint_path.stat().st_size, "sha256": checkpoint_sha256, "model_state_sha256": checkpoint_payload["model_state_sha256"], "ema_state_sha256": checkpoint_payload["ema_state_sha256"]},
        "history": segment_history,
        "checkpoint_history_length": len(history),
        "evaluation": evaluation,
        "gates": {"finite_training": True, "checkpoint_exactly_reloadable": True, "ema_evaluated": True, "legal_projection_exact": True},
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    report_path = output / "segment_report.json"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(report))
    os.replace(temporary, report_path)
    return report
