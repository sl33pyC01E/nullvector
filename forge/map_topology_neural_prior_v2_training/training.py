from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import torch

from ..map_topology_neural_prior_training.dataset import LatentRef, PriorTrainingDataset
from ..map_topology_neural_prior_v2.masking import MASK_MODES_V2, mask_tokens_v2
from ..map_topology_neural_prior_v2.model import build_prior_v2, masked_token_loss_v2
from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .checkpoint import load_checkpoint, save_checkpoint
from .contract import (
    FROZEN_AUTHORITY, SEGMENT_FORMAT, PriorV2CalibrationConfig,
    canonical_json_bytes, sha256_file, source_manifest, training_v2_source_sha256,
)
from .metrics import evaluate_free_generation, evaluate_masked


REPORT_NAME: Final[str] = "segment_report.json"
CHECKPOINT_NAME: Final[str] = "checkpoint.pt"
MAX_REPORT_BYTES: Final[int] = 32 * 1024 * 1024
REPORT_KEYS: Final[set[str]] = {
    "format", "status", "source_sha256", "source_manifest", "authority", "config",
    "segment", "history", "evaluation", "free_generation", "model", "checkpoint",
    "runtime", "safety_gates", "calibration", "claim_boundary", "report_sha256",
}


def _configure_device(device_name: str, seed: int) -> torch.device:
    if device_name == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before prior-v2 CUDA startup.")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("Prior-v2 calibration requires CUDA BF16.")
        free, _ = torch.cuda.mem_get_info(0)
        if free < 4 * 1024**3: raise RuntimeError("Prior-v2 calibration requires at least 4 GiB free CUDA memory.")
        device = torch.device("cuda", 0); torch.cuda.reset_peak_memory_stats(device); torch.cuda.manual_seed_all(seed)
    elif device_name == "cpu": device = torch.device("cpu")
    else: raise ValueError("Prior-v2 device must be cpu or cuda.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    return device


def _context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list): return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple): return tuple(_cpu_tree(child) for child in value)
    return value


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor): state[key] = value.to(device)


def _ema_update(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point(): ema[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else: ema[name].copy_(value.detach())


def _training_refs(dataset: PriorTrainingDataset, step: int, generator: torch.Generator, config: PriorV2CalibrationConfig) -> tuple[LatentRef, ...]:
    shapes = tuple(dataset.train_buckets); shape = shapes[step % len(shapes)]; bucket = dataset.train_buckets[shape]
    count = min(config.maximum_batch_size, max(1, config.cell_budget // (shape[0] * shape[1])), len(bucket))
    order = torch.randperm(len(bucket), generator=generator)[:count].tolist(); return tuple(bucket[index] for index in order)


def _registry(refs: tuple[LatentRef, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes([ref.full_map_identity_sha256 for ref in refs])).hexdigest()


def _validate_history(history: Any, config: PriorV2CalibrationConfig, step: int) -> list[dict[str, Any]]:
    keys = {"step", "loss", "gradient_norm", "batch_size", "shape", "masked_cells", "mask_fraction_mean", "modes", "sample_registry_sha256"}
    if not isinstance(history, list) or len(history) != step: raise ValueError("Prior-v2 history census drifted.")
    for expected, row in enumerate(history, 1):
        if not isinstance(row, dict) or set(row) != keys or row["step"] != expected: raise ValueError("Prior-v2 history schema drifted.")
        if any(type(row[name]) not in (int, float) or isinstance(row[name], bool) or not math.isfinite(row[name]) for name in ("loss", "gradient_norm", "mask_fraction_mean")): raise ValueError("Prior-v2 history metric is non-finite.")
        if row["loss"] <= 0 or row["gradient_norm"] < 0 or not 0 < row["mask_fraction_mean"] <= 1: raise ValueError("Prior-v2 history metric domain drifted.")
        if type(row["batch_size"]) is not int or not 1 <= row["batch_size"] <= config.maximum_batch_size or type(row["masked_cells"]) is not int or row["masked_cells"] <= 0: raise ValueError("Prior-v2 history batch census drifted.")
        if not isinstance(row["shape"], list) or len(row["shape"]) != 2 or any(type(value) is not int or value <= 0 for value in row["shape"]): raise ValueError("Prior-v2 history shape drifted.")
        if not isinstance(row["modes"], list) or len(row["modes"]) != row["batch_size"] or any(mode not in MASK_MODES_V2 for mode in row["modes"]): raise ValueError("Prior-v2 history mask modes drifted.")
        if not isinstance(row["sample_registry_sha256"], str) or len(row["sample_registry_sha256"]) != 64: raise ValueError("Prior-v2 history registry hash drifted.")
    return history


def _validate_metric(metric: Any, expected_samples: int) -> dict[str, Any]:
    keys = {"sample_count", "sample_registry_sha256", "masked_cells", "accuracy", "loss", "macro_mode_accuracy", "full_mask_accuracy", "full_mask_loss", "modes", "vocabulary_size"}
    if not isinstance(metric, dict) or set(metric) != keys or metric["sample_count"] != expected_samples or metric["vocabulary_size"] != 512: raise ValueError("Prior-v2 evaluation schema/census drifted.")
    if not isinstance(metric["sample_registry_sha256"], str) or len(metric["sample_registry_sha256"]) != 64 or any(character not in "0123456789abcdef" for character in metric["sample_registry_sha256"]): raise ValueError("Prior-v2 evaluation registry hash drifted.")
    for name in ("accuracy", "loss", "macro_mode_accuracy", "full_mask_accuracy", "full_mask_loss"):
        if type(metric[name]) not in (int, float) or isinstance(metric[name], bool) or not math.isfinite(metric[name]): raise ValueError("Prior-v2 evaluation aggregate metric is non-finite.")
    if not 0 <= metric["accuracy"] <= 1 or not 0 <= metric["macro_mode_accuracy"] <= 1 or not 0 <= metric["full_mask_accuracy"] <= 1 or metric["loss"] <= 0 or metric["full_mask_loss"] <= 0: raise ValueError("Prior-v2 evaluation aggregate metric domain drifted.")
    if not isinstance(metric["modes"], dict) or set(metric["modes"]) != set(MASK_MODES_V2): raise ValueError("Prior-v2 evaluation mask modes drifted.")
    total = 0
    for mode, row in metric["modes"].items():
        if not isinstance(row, dict) or set(row) != {"masked_cells", "accuracy", "loss"} or type(row["masked_cells"]) is not int or row["masked_cells"] <= 0: raise ValueError(f"Prior-v2 evaluation {mode} census drifted.")
        if not 0 <= row["accuracy"] <= 1 or not math.isfinite(row["loss"]) or row["loss"] <= 0: raise ValueError("Prior-v2 evaluation mode metric drifted.")
        total += row["masked_cells"]
    if type(metric["masked_cells"]) is not int or total != metric["masked_cells"] or metric["full_mask_accuracy"] != metric["modes"]["full"]["accuracy"] or metric["full_mask_loss"] != metric["modes"]["full"]["loss"]: raise ValueError("Prior-v2 full-mask metric derivation drifted.")
    if metric["macro_mode_accuracy"] != sum(row["accuracy"] for row in metric["modes"].values()) / len(MASK_MODES_V2): raise ValueError("Prior-v2 macro metric derivation drifted.")
    weighted_accuracy = sum(row["accuracy"] * row["masked_cells"] for row in metric["modes"].values()) / total
    weighted_loss = sum(row["loss"] * row["masked_cells"] for row in metric["modes"].values()) / total
    if not math.isclose(metric["accuracy"], weighted_accuracy, rel_tol=0, abs_tol=1e-12) or not math.isclose(metric["loss"], weighted_loss, rel_tol=0, abs_tol=1e-12): raise ValueError("Prior-v2 aggregate metric derivation drifted.")
    return metric


def _validate_free(free: Any) -> dict[str, Any]:
    keys = {"sample_count", "sample_registry_sha256", "token_accuracy", "unique_samples", "tokens_sha256", "uncertainty_sha256", "all_tokens_revealed"}
    if not isinstance(free, dict) or set(free) != keys or free["sample_count"] != 6: raise ValueError("Prior-v2 free-generation schema/census drifted.")
    for name in ("sample_registry_sha256", "tokens_sha256", "uncertainty_sha256"):
        if not isinstance(free[name], str) or len(free[name]) != 64 or any(character not in "0123456789abcdef" for character in free[name]): raise ValueError("Prior-v2 free-generation hash drifted.")
    if type(free["token_accuracy"]) not in (int, float) or isinstance(free["token_accuracy"], bool) or not math.isfinite(free["token_accuracy"]) or not 0 <= free["token_accuracy"] <= 1: raise ValueError("Prior-v2 free-generation accuracy drifted.")
    if type(free["unique_samples"]) is not int or not 1 <= free["unique_samples"] <= 6 or free["all_tokens_revealed"] is not True: raise ValueError("Prior-v2 free-generation gate drifted.")
    return free


def _expected_safety(report: dict[str, Any], config: PriorV2CalibrationConfig) -> dict[str, bool]:
    segment = report["segment"]; history = report["history"]; free = report["free_generation"]
    modes_seen = {mode for row in history for mode in row["modes"]}
    return {
        "source_and_corpus_bound": True,
        "step_range_exact": segment["end_step"] - segment["start_step"] == min(config.steps_per_segment, config.total_steps - segment["start_step"]),
        "history_cumulative_exact": len(history) == segment["end_step"],
        "all_mask_modes_seen": modes_seen == set(MASK_MODES_V2),
        "full_mask_training_seen": "full" in modes_seen,
        "finite_training": all(math.isfinite(row["loss"]) for row in history),
        "model_updated": report["model"]["raw_sha256"] != report["model"]["initial_sha256"],
        "free_tokens_revealed": free["all_tokens_revealed"],
        "immutable_predecessor_bound": segment["start_step"] == 0 or segment["predecessor"] is not None,
        "disk_floor_preserved": True,
    }


def _calibration(history: list[dict[str, Any]], evaluation: dict[str, Any], free: dict[str, Any]) -> dict[str, Any]:
    window = min(8, max(1, len(history) // 2)); first = sum(row["loss"] for row in history[:window]) / window; last = sum(row["loss"] for row in history[-window:]) / window
    improvement = (first - last) / max(first, 1e-12)
    thresholds = {"relative_loss_improvement": .02, "validation_full_mask_accuracy": .02, "test_full_mask_accuracy": .015, "free_unique_samples": 6}
    checks = {"relative_loss_improvement": improvement >= thresholds["relative_loss_improvement"], "validation_full_mask_accuracy": evaluation["validation"]["ema"]["full_mask_accuracy"] >= thresholds["validation_full_mask_accuracy"], "test_full_mask_accuracy": evaluation["test"]["ema"]["full_mask_accuracy"] >= thresholds["test_full_mask_accuracy"], "free_unique_samples": free["unique_samples"] >= thresholds["free_unique_samples"], "free_tokens_revealed": free["all_tokens_revealed"] is True}
    return {"calibration_gate_passed": all(checks.values()), "checks": checks, "thresholds": thresholds, "loss_first_window_mean": first, "loss_last_window_mean": last, "relative_loss_improvement": improvement, "production_promotion_allowed": False}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(canonical_json_bytes(payload)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def run_segment(corpus_root: Path, latent_root: Path, output: Path, *, config: PriorV2CalibrationConfig, resume: Path | None = None, device_name: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError("Prior-v2 segment publication is immutable.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024 * 1024 * 1024)
    dataset = PriorTrainingDataset(corpus_root, latent_root); device = _configure_device(device_name, config.seed)
    model = build_prior_v2(config.model_config()).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x545241494E); history: list[dict[str, Any]] = []; predecessor = None; start_step = 0
    initial_sha = tensor_state_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()}); ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if resume is not None:
        resume_path = Path(resume).resolve(); payload = load_checkpoint(resume_path)
        if payload["config"] != config.to_dict() or payload["step"] >= config.total_steps: raise ValueError("Prior-v2 resume config/step is incompatible.")
        model.load_state_dict(payload["model_state"], strict=True); ema = {name: value.to(device) for name, value in payload["ema_state"].items()}; optimizer.load_state_dict(payload["optimizer_state"]); _move_optimizer(optimizer, device)
        generator.set_state(payload["mask_generator_state"]); torch.set_rng_state(payload["torch_cpu_rng_state"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])
        history = list(payload["history"]); start_step = payload["step"]; initial_sha = payload["initial_model_sha256"]; predecessor = {"checkpoint_sha256": sha256_file(resume_path), "step": start_step}
    end_step = min(config.total_steps, start_step + config.steps_per_segment)
    if end_step <= start_step: raise ValueError("Prior-v2 segment has no remaining updates.")
    started = time.perf_counter(); model.train()
    for step in range(start_step, end_step):
        refs = _training_refs(dataset, step, generator, config); batch = dataset.collate(refs)
        masked = mask_tokens_v2(batch["targets"], batch["valid_mask"], generator=generator, config=config.model_config(), step=step)
        inputs = {name: batch[name].to(device) for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")}; inputs.update(tokens=masked["tokens"].to(device), mask_fraction=masked["mask_fraction"].to(device))
        optimizer.zero_grad(set_to_none=True)
        with _context(device): logits = model(inputs)
        loss = masked_token_loss_v2(logits.float(), batch["targets"].to(device), masked["mask"].to(device)); loss.backward(); gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip); optimizer.step(); _ema_update(ema, model, config.ema_decay)
        history.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm), "batch_size": len(refs), "shape": list(refs[0].shape), "masked_cells": int(masked["mask"].sum()), "mask_fraction_mean": float(masked["mask_fraction"].mean()), "modes": list(masked["modes"]), "sample_registry_sha256": _registry(refs)})
    training_seconds = time.perf_counter() - started; _validate_history(history, config, end_step)
    raw_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    raw_model = build_prior_v2(config.model_config()).to(device); raw_model.load_state_dict(raw_state, strict=True); raw_model.eval()
    ema_model = build_prior_v2(config.model_config()).to(device); ema_model.load_state_dict(ema_state, strict=True); ema_model.eval()
    validation_refs = dataset.evaluation_refs("validation", config.validation_samples); test_refs = dataset.evaluation_refs("test", config.test_samples); free_refs = dataset.evaluation_refs("validation", 6)
    evaluation_started = time.perf_counter(); evaluation = {
        "validation": {"raw": evaluate_masked(raw_model, dataset, validation_refs, device=device, config=config), "ema": evaluate_masked(ema_model, dataset, validation_refs, device=device, config=config)},
        "test": {"raw": evaluate_masked(raw_model, dataset, test_refs, device=device, config=config), "ema": evaluate_masked(ema_model, dataset, test_refs, device=device, config=config)},
    }; free = evaluate_free_generation(ema_model, dataset, free_refs, device=device, config=config); evaluation_seconds = time.perf_counter() - evaluation_started
    for split, expected in (("validation", config.validation_samples), ("test", config.test_samples)):
        for mode in ("raw", "ema"): _validate_metric(evaluation[split][mode], expected)
    _validate_free(free); calibration = _calibration(history, evaluation, free); model_sha = tensor_state_sha256(raw_state); ema_sha = tensor_state_sha256(ema_state)
    payload = {"format": "nullvector-neural-map-topology-prior-v2-calibration-checkpoint/1.0.0", "source_sha256": training_v2_source_sha256(), "source_manifest": source_manifest(), "authority": FROZEN_AUTHORITY, "config": config.to_dict(), "step": end_step, "initial_model_sha256": initial_sha, "model_state": raw_state, "ema_state": ema_state, "optimizer_state": _cpu_tree(optimizer.state_dict()), "mask_generator_state": generator.get_state().cpu(), "torch_cpu_rng_state": torch.get_rng_state().cpu(), "torch_cuda_rng_states": [value.cpu() for value in torch.cuda.get_rng_state_all()] if device.type == "cuda" else [], "history": history, "evaluation": evaluation, "free_generation": free, "model_state_sha256": model_sha, "ema_state_sha256": ema_sha, "predecessor": predecessor}
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"; staging.mkdir(parents=True, exist_ok=False)
    try:
        sidecar = save_checkpoint(staging / CHECKPOINT_NAME, payload); elapsed = time.perf_counter() - started
        runtime = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "precision": "bf16-autocast-float32-loss" if device.type == "cuda" else "float32", "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds, "elapsed_seconds": elapsed, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0}
        modes_seen = {mode for row in history for mode in row["modes"]}; safety = {"source_and_corpus_bound": True, "step_range_exact": end_step - start_step == min(config.steps_per_segment, config.total_steps - start_step), "history_cumulative_exact": len(history) == end_step, "all_mask_modes_seen": modes_seen == set(MASK_MODES_V2), "full_mask_training_seen": "full" in modes_seen, "finite_training": all(math.isfinite(row["loss"]) for row in history), "model_updated": model_sha != initial_sha, "free_tokens_revealed": free["all_tokens_revealed"], "immutable_predecessor_bound": resume is None or predecessor is not None, "disk_floor_preserved": True}
        report: dict[str, Any] = {"format": SEGMENT_FORMAT, "status": "passed" if all(safety.values()) else "failed", "source_sha256": training_v2_source_sha256(), "source_manifest": source_manifest(), "authority": FROZEN_AUTHORITY, "config": config.to_dict(), "segment": {"start_step": start_step, "end_step": end_step, "updates": end_step - start_step, "predecessor": predecessor}, "history": history, "evaluation": evaluation, "free_generation": free, "model": {"initial_sha256": initial_sha, "raw_sha256": model_sha, "ema_sha256": ema_sha}, "checkpoint": {**sidecar, "path": CHECKPOINT_NAME}, "runtime": runtime, "safety_gates": safety, "calibration": calibration, "claim_boundary": {"segmented_calibration_only": True, "production_promotion_allowed": False, "compiled_maps_published": False, "runtime_integration": False}}
        report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest(); _atomic_json(staging / REPORT_NAME, report); require_disk_floor(output.parent, floor_gb=100, planned_bytes=0); os.replace(staging, output)
    except BaseException:
        if staging.exists(): os.replace(staging, output.parent / f"{staging.name}.failed-{time.time_ns()}")
        raise
    return validate_segment(output)


def _read_report(output: Path) -> dict[str, Any]:
    path = Path(output).resolve() / REPORT_NAME
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES: raise ValueError("Prior-v2 segment report is missing or oversized.")
    encoded = path.read_bytes(); report = json.loads(encoded)
    if not isinstance(report, dict) or encoded != canonical_json_bytes(report): raise ValueError("Prior-v2 segment report is not canonical JSON.")
    stored = report.pop("report_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(report)).hexdigest(): raise ValueError("Prior-v2 segment report self-hash failed.")
    report["report_sha256"] = stored; return report


def validate_segment(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); report = _read_report(output)
    if set(report) != REPORT_KEYS or report["format"] != SEGMENT_FORMAT or report["status"] != "passed": raise ValueError("Prior-v2 segment report contract failed.")
    if report["source_sha256"] != training_v2_source_sha256() or report["source_manifest"] != source_manifest() or report["authority"] != FROZEN_AUTHORITY: raise ValueError("Prior-v2 segment report provenance drifted.")
    config = PriorV2CalibrationConfig.from_dict(report["config"])
    segment = report["segment"]
    if not isinstance(segment, dict) or set(segment) != {"start_step", "end_step", "updates", "predecessor"}: raise ValueError("Prior-v2 segment range schema drifted.")
    if type(segment["start_step"]) is not int or type(segment["end_step"]) is not int or type(segment["updates"]) is not int or not 0 <= segment["start_step"] < segment["end_step"] <= config.total_steps or segment["updates"] != segment["end_step"] - segment["start_step"]: raise ValueError("Prior-v2 segment range drifted.")
    predecessor = segment["predecessor"]
    if segment["start_step"] == 0:
        if predecessor is not None: raise ValueError("Prior-v2 initial segment predecessor drifted.")
    elif not isinstance(predecessor, dict) or set(predecessor) != {"checkpoint_sha256", "step"} or predecessor["step"] != segment["start_step"] or not isinstance(predecessor["checkpoint_sha256"], str) or len(predecessor["checkpoint_sha256"]) != 64: raise ValueError("Prior-v2 resumed segment predecessor drifted.")
    end_step = segment["end_step"]; _validate_history(report["history"], config, end_step)
    if not isinstance(report["evaluation"], dict) or set(report["evaluation"]) != {"validation", "test"}: raise ValueError("Prior-v2 evaluation split census drifted.")
    for split, expected in (("validation", config.validation_samples), ("test", config.test_samples)):
        if set(report["evaluation"][split]) != {"raw", "ema"}: raise ValueError("Prior-v2 evaluation model census drifted.")
        for mode in ("raw", "ema"): _validate_metric(report["evaluation"][split][mode], expected)
    _validate_free(report["free_generation"])
    if not isinstance(report["checkpoint"], dict) or report["checkpoint"].get("path") != CHECKPOINT_NAME: raise ValueError("Prior-v2 segment checkpoint path drifted.")
    checkpoint_path = output / CHECKPOINT_NAME
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    sidecar_bytes = sidecar_path.read_bytes(); sidecar = json.loads(sidecar_bytes)
    if sidecar_bytes != canonical_json_bytes(sidecar) or report["checkpoint"] != {**sidecar, "path": CHECKPOINT_NAME}: raise ValueError("Prior-v2 segment checkpoint descriptor drifted.")
    payload = load_checkpoint(checkpoint_path)
    if payload["config"] != report["config"] or payload["step"] != end_step or payload["history"] != report["history"] or payload["evaluation"] != report["evaluation"] or payload["free_generation"] != report["free_generation"] or payload["initial_model_sha256"] != report["model"]["initial_sha256"] or payload["model_state_sha256"] != report["model"]["raw_sha256"] or payload["ema_state_sha256"] != report["model"]["ema_sha256"] or payload["predecessor"] != predecessor: raise ValueError("Prior-v2 checkpoint/report semantics drifted.")
    if report["calibration"] != _calibration(report["history"], report["evaluation"], report["free_generation"]) or report["calibration"]["production_promotion_allowed"] is not False: raise ValueError("Prior-v2 calibration derivation drifted.")
    if report["safety_gates"] != _expected_safety(report, config) or not all(report["safety_gates"].values()): raise ValueError("Prior-v2 segment safety gates failed.")
    runtime_keys = {"device", "precision", "training_seconds", "evaluation_seconds", "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes"}
    if not isinstance(report["runtime"], dict) or set(report["runtime"]) != runtime_keys or not isinstance(report["runtime"]["device"], str) or report["runtime"]["precision"] not in {"float32", "bf16-autocast-float32-loss"}: raise ValueError("Prior-v2 runtime schema drifted.")
    for name in ("training_seconds", "evaluation_seconds", "elapsed_seconds"):
        if type(report["runtime"][name]) not in (int, float) or isinstance(report["runtime"][name], bool) or not math.isfinite(report["runtime"][name]) or report["runtime"][name] < 0: raise ValueError("Prior-v2 runtime timing drifted.")
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if type(report["runtime"][name]) is not int or report["runtime"][name] < 0: raise ValueError("Prior-v2 runtime memory drifted.")
    if report["claim_boundary"] != {"segmented_calibration_only": True, "production_promotion_allowed": False, "compiled_maps_published": False, "runtime_integration": False}: raise ValueError("Prior-v2 segment claim boundary drifted.")
    return report
