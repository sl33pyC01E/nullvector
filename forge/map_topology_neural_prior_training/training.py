from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import torch
import torch.nn.functional as F

from ..map_topology_neural_prior.masking import mask_tokens
from ..map_topology_neural_prior.model import build_prior, masked_token_loss
from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .checkpoint import load_checkpoint, save_checkpoint
from .contract import (
    CALIBRATION_FORMAT, CHECKPOINT_FORMAT, FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
    FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256, FROZEN_LATENT_CORPUS_RELATIVE,
    PROJECT_ROOT, QUALITY_GATES, PriorCalibrationConfig, canonical_json_bytes,
    CLAIM_KEYS, RUNTIME_KEYS, SAFETY_GATE_KEYS, sha256_file, source_manifest,
    training_source_sha256, validate_evaluation, validate_history,
)
from .dataset import PriorTrainingDataset
from .metrics import evaluate_prior


REPORT_NAME: Final[str] = "calibration_report.json"
CHECKPOINT_NAME: Final[str] = "checkpoint_final.pt"
MAX_REPORT_BYTES: Final[int] = 32 * 1024 * 1024
REPORT_KEYS: Final[set[str]] = {
    "format", "status", "source_sha256", "source_manifest",
    "latent_corpus_manifest_file_sha256", "latent_corpus_identity_sha256",
    "config", "history", "evaluation", "quality", "initial_model_sha256",
    "final_model_sha256", "final_ema_sha256", "checkpoint", "runtime",
    "safety_gates", "claim_boundary", "report_sha256",
}


def _configure_cuda(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before prior CUDA startup.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Prior calibration requires CUDA BF16.")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < 4 * 1024**3:
        raise RuntimeError("Prior calibration requires at least 4 GiB free CUDA memory.")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list): return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple): return tuple(_cpu_tree(child) for child in value)
    return value


def _ema_update(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point(): ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else: ema[name].copy_(value.detach())


def _model_from_state(config: PriorCalibrationConfig, state: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    model = build_prior(config.model_config()); model.load_state_dict(state, strict=True); return model.to(device).eval()


def _quality(history: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    window = max(1, min(16, len(history) // 2))
    first = sum(item["loss"] for item in history[:window]) / window
    last = sum(item["loss"] for item in history[-window:]) / window
    improvement = (first - last) / max(abs(first), 1.0e-12)
    checks = {
        "minimum_loss_improvement": improvement >= QUALITY_GATES["minimum_loss_improvement"],
        "validation_accuracy": evaluation["validation"]["ema"]["accuracy"] >= QUALITY_GATES["minimum_validation_accuracy"],
        "test_accuracy": evaluation["test"]["ema"]["accuracy"] >= QUALITY_GATES["minimum_test_accuracy"],
        "validation_macro_mode_accuracy": evaluation["validation"]["ema"]["macro_mode_accuracy"] >= QUALITY_GATES["minimum_validation_macro_mode_accuracy"],
        "test_macro_mode_accuracy": evaluation["test"]["ema"]["macro_mode_accuracy"] >= QUALITY_GATES["minimum_test_macro_mode_accuracy"],
    }
    return {"quality_milestone_reached": all(checks.values()), "checks": checks, "thresholds": QUALITY_GATES, "loss_first_window_mean": first, "loss_last_window_mean": last, "relative_loss_improvement": improvement}


def _validate_runtime(runtime: Any) -> dict[str, Any]:
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_KEYS or not isinstance(runtime["device"], str) or not runtime["device"] or runtime["precision"] != "bf16-autocast-float32-loss":
        raise ValueError("Prior calibration runtime census drifted.")
    for name in ("training_seconds", "evaluation_seconds", "elapsed_seconds"):
        if type(runtime[name]) not in (int, float) or isinstance(runtime[name], bool) or not math.isfinite(runtime[name]) or runtime[name] < 0:
            raise ValueError("Prior calibration runtime duration drifted.")
    if runtime["elapsed_seconds"] + 1.0e-9 < runtime["training_seconds"] + runtime["evaluation_seconds"]:
        raise ValueError("Prior calibration runtime totals drifted.")
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if type(runtime[name]) is not int or runtime[name] < 0:
            raise ValueError("Prior calibration CUDA memory metric drifted.")
    if runtime["peak_reserved_bytes"] < runtime["peak_allocated_bytes"]:
        raise ValueError("Prior calibration CUDA memory totals drifted.")
    return runtime


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_calibration(corpus_root: Path, latent_root: Path, output: Path, *, config: PriorCalibrationConfig) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError("Prior calibration publication is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=512 * 1024 * 1024)
    dataset = PriorTrainingDataset(corpus_root, latent_root)
    validation_refs = dataset.evaluation_refs("validation", config.validation_samples)
    test_refs = dataset.evaluation_refs("test", config.test_samples)
    device = _configure_cuda(config.seed); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter(); model = build_prior(config.model_config()).to(device); initial_sha = tensor_state_sha256({name: value.cpu() for name, value in model.state_dict().items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x545241494E)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    # Baseline is the exact pre-update state, not merely another model that is
    # expected to initialize the same way.
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    baseline = _model_from_state(config, initial_state, device)
    baseline_eval = {
        "validation": evaluate_prior(baseline, dataset, validation_refs, device=device, config=config),
        "test": evaluate_prior(baseline, dataset, test_refs, device=device, config=config),
    }
    history: list[dict[str, Any]] = []; training_started = time.perf_counter(); model.train()
    for step in range(config.steps):
        refs = dataset.training_refs(step, generator, config); batch = dataset.collate(refs)
        masked = mask_tokens(batch["targets"], batch["valid_mask"], generator=generator, config=config.model_config(), step=step)
        inputs = {
            "tokens": masked["tokens"].to(device), "valid_mask": batch["valid_mask"].to(device),
            "point_conditions": batch["point_conditions"].to(device), "global_conditions": batch["global_conditions"].to(device),
            "theme_index": batch["theme_index"].to(device), "mask_fraction": masked["mask_fraction"].to(device),
        }
        targets = batch["targets"].to(device); mask = masked["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): logits = model(inputs)
        loss = masked_token_loss(logits.float(), targets, mask); loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip); optimizer.step(); _ema_update(ema, model, config.model_ema_decay)
        history.append({
            "step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm),
            "batch_size": len(refs), "shape": list(refs[0].shape), "masked_cells": int(mask.sum()),
            "mask_fraction_mean": float(masked["mask_fraction"].mean()), "modes": list(masked["modes"]),
            "sample_registry_sha256": hashlib.sha256(canonical_json_bytes([ref.full_map_identity_sha256 for ref in refs])).hexdigest(),
        })
    training_seconds = time.perf_counter() - training_started
    raw_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; ema_cpu = {name: value.detach().cpu().clone() for name, value in ema.items()}
    raw_model = _model_from_state(config, raw_state, device); ema_model = _model_from_state(config, ema_cpu, device)
    evaluation_started = time.perf_counter()
    evaluation = {
        "validation": {"baseline": baseline_eval["validation"], "raw": evaluate_prior(raw_model, dataset, validation_refs, device=device, config=config), "ema": evaluate_prior(ema_model, dataset, validation_refs, device=device, config=config)},
        "test": {"baseline": baseline_eval["test"], "raw": evaluate_prior(raw_model, dataset, test_refs, device=device, config=config), "ema": evaluate_prior(ema_model, dataset, test_refs, device=device, config=config)},
    }
    evaluation_seconds = time.perf_counter() - evaluation_started; quality = _quality(history, evaluation)
    validate_history(history, config); validate_evaluation(evaluation, config)
    model_sha = tensor_state_sha256(raw_state); ema_sha = tensor_state_sha256(ema_cpu)
    payload = {
        "format": CHECKPOINT_FORMAT, "source_sha256": training_source_sha256(), "source_manifest": source_manifest(),
        "latent_corpus_manifest_file_sha256": FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256, "latent_corpus_identity_sha256": FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
        "config": config.to_dict(), "step": config.steps, "model_state": raw_state, "ema_state": ema_cpu,
        "optimizer_state": _cpu_tree(optimizer.state_dict()), "generator_state": generator.get_state().cpu(), "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_states": [value.cpu() for value in torch.cuda.get_rng_state_all()], "history": history, "evaluation": evaluation,
        "model_state_sha256": model_sha, "ema_state_sha256": ema_sha,
    }
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"; staging.mkdir(parents=True, exist_ok=False)
    sidecar = save_checkpoint(staging / CHECKPOINT_NAME, payload); require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=0)
    safety = {
        "step_count_exact": len(history) == config.steps, "finite_history": all(math.isfinite(item["loss"]) for item in history),
        "model_updated": model_sha != initial_sha, "evaluation_census_exact": all(evaluation[split][mode]["sample_count"] == (config.validation_samples if split == "validation" else config.test_samples) for split in ("validation", "test") for mode in ("baseline", "raw", "ema")),
        "train_split_only": True, "latent_corpus_exact": True, "raw_generation_disabled": True, "compiler_disabled": True,
        "godot_integration_disabled": True, "disk_floor_preserved": True,
    }
    report: dict[str, Any] = {
        "format": CALIBRATION_FORMAT, "status": "passed" if all(safety.values()) else "failed",
        "source_sha256": training_source_sha256(), "source_manifest": source_manifest(),
        "latent_corpus_manifest_file_sha256": FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256, "latent_corpus_identity_sha256": FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
        "config": config.to_dict(), "history": history, "evaluation": evaluation, "quality": quality,
        "initial_model_sha256": initial_sha, "final_model_sha256": model_sha, "final_ema_sha256": ema_sha,
        "checkpoint": {**sidecar, "path": CHECKPOINT_NAME},
        "runtime": {"device": torch.cuda.get_device_name(device), "precision": "bf16-autocast-float32-loss", "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds, "elapsed_seconds": time.perf_counter() - started, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
        "safety_gates": safety,
        "claim_boundary": {"masked_token_calibration_only": True, "quality_milestone_reached": quality["quality_milestone_reached"], "raw_generation_published": False, "compiled_maps_published": False, "godot_integration": False},
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest(); _atomic_json(staging / REPORT_NAME, report); os.replace(staging, output)
    return validate_calibration(corpus_root, latent_root, output, replay_metrics=False)


def _read_report(output: Path) -> dict[str, Any]:
    path = Path(output).resolve() / REPORT_NAME
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES: raise ValueError("Prior calibration report is missing or oversized.")
    encoded = path.read_bytes(); report = json.loads(encoded)
    if not isinstance(report, dict) or encoded != canonical_json_bytes(report): raise ValueError("Prior calibration report is not canonical JSON.")
    stored = report.pop("report_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(report)).hexdigest(): raise ValueError("Prior calibration report self-hash failed.")
    report["report_sha256"] = stored; return report


def validate_calibration(corpus_root: Path, latent_root: Path, output: Path, *, replay_metrics: bool = True) -> dict[str, Any]:
    output = Path(output).resolve(); report = _read_report(output)
    if set(report) != REPORT_KEYS or report["format"] != CALIBRATION_FORMAT or report["status"] != "passed": raise ValueError("Prior calibration report format/status/census failed.")
    if report["source_sha256"] != training_source_sha256() or report["source_manifest"] != source_manifest(): raise ValueError("Prior calibration report source drifted.")
    config = PriorCalibrationConfig.from_dict(report["config"]); dataset = PriorTrainingDataset(corpus_root, latent_root)
    validate_history(report["history"], config); validate_evaluation(report["evaluation"], config); _validate_runtime(report["runtime"])
    if report["latent_corpus_manifest_file_sha256"] != FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256 or report["latent_corpus_identity_sha256"] != FROZEN_LATENT_CORPUS_IDENTITY_SHA256: raise ValueError("Prior calibration latent corpus drifted.")
    checkpoint_path = output / CHECKPOINT_NAME; payload = load_checkpoint(checkpoint_path)
    sidecar = json.loads(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").read_text(encoding="utf-8"))
    if report["checkpoint"] != {**sidecar, "path": CHECKPOINT_NAME} or payload["history"] != report["history"] or payload["evaluation"] != report["evaluation"] or payload["model_state_sha256"] != report["final_model_sha256"] or payload["ema_state_sha256"] != report["final_ema_sha256"]: raise ValueError("Prior calibration checkpoint/report semantics drifted.")
    if report["quality"] != _quality(report["history"], report["evaluation"]): raise ValueError("Prior calibration quality derivation drifted.")
    expected_claim = {"masked_token_calibration_only": True, "quality_milestone_reached": report["quality"]["quality_milestone_reached"], "raw_generation_published": False, "compiled_maps_published": False, "godot_integration": False}
    expected_safety = {
        "step_count_exact": len(report["history"]) == config.steps,
        "finite_history": all(math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"]) for item in report["history"]),
        "model_updated": report["final_model_sha256"] != report["initial_model_sha256"],
        "evaluation_census_exact": all(report["evaluation"][split][mode]["sample_count"] == (config.validation_samples if split == "validation" else config.test_samples) for split in ("validation", "test") for mode in ("baseline", "raw", "ema")),
        "train_split_only": True, "latent_corpus_exact": True, "raw_generation_disabled": True,
        "compiler_disabled": True, "godot_integration_disabled": True, "disk_floor_preserved": True,
    }
    if set(report["claim_boundary"]) != CLAIM_KEYS or report["claim_boundary"] != expected_claim or set(report["safety_gates"]) != SAFETY_GATE_KEYS or report["safety_gates"] != expected_safety or not all(expected_safety.values()): raise ValueError("Prior calibration claim/safety boundary drifted.")
    if replay_metrics:
        device = _configure_cuda(config.seed); validation_refs = dataset.evaluation_refs("validation", config.validation_samples); test_refs = dataset.evaluation_refs("test", config.test_samples)
        baseline = _model_from_state(config, {name: value for name, value in build_prior(config.model_config()).state_dict().items()}, device); raw = _model_from_state(config, payload["model_state"], device); ema = _model_from_state(config, payload["ema_state"], device)
        replay = {"validation": {"baseline": evaluate_prior(baseline, dataset, validation_refs, device=device, config=config), "raw": evaluate_prior(raw, dataset, validation_refs, device=device, config=config), "ema": evaluate_prior(ema, dataset, validation_refs, device=device, config=config)}, "test": {"baseline": evaluate_prior(baseline, dataset, test_refs, device=device, config=config), "raw": evaluate_prior(raw, dataset, test_refs, device=device, config=config), "ema": evaluate_prior(ema, dataset, test_refs, device=device, config=config)}}
        if replay != report["evaluation"]: raise ValueError("Prior calibration exact metric replay failed.")
    return report
