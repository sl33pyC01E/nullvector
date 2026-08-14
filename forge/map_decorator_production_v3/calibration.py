from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Final, Sequence
import uuid

import torch

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.dataset import collate_teacher_samples
from ..map_decorator_ml.legality import TorchLegalMasks
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_production_v2.contract import CALIBRATION_GATES, ForegroundPatchConfig
from ..map_decorator_production_v2.patches import foreground_centered_crop, plan_foreground_batches
from ..map_decorator_production_v2.quality import evaluate_dual_split_gate
from ..map_decorator_production_v2.runner import V2CorpusAuthority
from ..map_decorator_production_v2.training import WarmStartEMA
from ..safety import require_disk_floor
from .checkpoint import (
    checkpoint_source_sha256,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
    tensor_state_sha256,
)
from .contract import LocatorLossConfig, LocatorModelConfig, LocatorTrainingConfig, V3_CONTRACT_SHA256
from .decoding import select_sparse_locator_argmax
from .model import SparseLocatorDecoratorV3
from .training import make_optimizer_v3, train_batch_v3


CALIBRATION_FORMAT: Final[str] = "nullvector-map-decorator-v3-cuda-calibration/1.0.0"
SUPERVISOR_FORMAT: Final[str] = "nullvector-map-decorator-v3-calibration-supervisor/1.0.0"
REPORT_NAME: Final[str] = "calibration_report.json"
SUPERVISOR_REPORT_NAME: Final[str] = "supervisor_report.json"
CALIBRATION_GATE_SHA256: Final[str] = json_sha256(CALIBRATION_GATES)
MAX_REPORT_BYTES: Final[int] = 32 * 1024 * 1024


def _checkpoint_name(steps: int) -> str:
    return f"checkpoint_step_{steps:04d}.pt"


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    steps: int = 100
    validation_batch_size: int = 4
    test_batch_size: int = 4
    model: LocatorModelConfig = LocatorModelConfig()
    training: LocatorTrainingConfig = LocatorTrainingConfig()
    loss: LocatorLossConfig = LocatorLossConfig()
    patch: ForegroundPatchConfig = ForegroundPatchConfig()

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not 1 <= self.steps <= 1_000:
            raise ValueError("Calibration steps must be in [1,1000].")
        for name, value in (
            ("validation_batch_size", self.validation_batch_size),
            ("test_batch_size", self.test_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
                raise ValueError(f"{name} must be an integer in [1,16].")

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "validation_batch_size": self.validation_batch_size,
            "test_batch_size": self.test_batch_size,
            "precision": "bf16",
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "loss": self.loss.to_dict(),
            "patch": self.patch.to_dict(),
        }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Calibration JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _read_report(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("Calibration report is missing, unsafe, or oversized.")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object)
    if not isinstance(value, dict):
        raise ValueError("Calibration report root must be an object.")
    stored = value.pop("report_sha256", None)
    if stored != json_sha256(value):
        raise ValueError("Calibration report self-hash failed.")
    value["report_sha256"] = stored
    return value


def _configure_cuda(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before CUDA startup.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required for v3 calibration.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def _memory_report(device: torch.device) -> dict[str, object]:
    return {
        "name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _copy_ema_model(
    model: SparseLocatorDecoratorV3,
    ema: WarmStartEMA,
    device: torch.device,
) -> SparseLocatorDecoratorV3:
    target = SparseLocatorDecoratorV3(model.config)
    target.load_state_dict(model.state_dict(), strict=True)
    ema.copy_to(target)
    return target.to(device)


def _batches(items: Sequence[object], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _predict_batch(
    model: SparseLocatorDecoratorV3,
    batch: dict[str, object],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, bool]:
    features = batch["features"].to(device)  # type: ignore[union-attr]
    targets = {name: batch["targets"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    legal_masks = {name: batch["legal_masks"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    valid = batch["valid_cells"].to(device)  # type: ignore[union-attr]
    hard_empty = batch["hard_empty"].to(device)  # type: ignore[union-attr]
    theme = batch["theme_index"].to(device)  # type: ignore[union-attr]
    conditions = batch["global_conditions"].to(device)  # type: ignore[union-attr]
    masked = {name: valid.clone() for name in HEAD_NAMES}
    level = torch.ones((features.shape[0],), dtype=torch.float32, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(features, targets, masked, theme, conditions, level)
        legal = TorchLegalMasks(hard_empty=hard_empty, **legal_masks)
        prediction = select_sparse_locator_argmax(output, legal)
    hard_legal = True
    for name in HEAD_NAMES:
        selected_legal = legal_masks[name].gather(1, prediction[name].unsqueeze(1)).squeeze(1)
        hard_legal = hard_legal and bool(selected_legal[valid].all())
        if name != "variant":
            hard_legal = hard_legal and not bool((prediction[name][hard_empty & valid] != 0).any())
    hard_legal = hard_legal and not bool(
        ((prediction["decal"] != 0) & (prediction["prop"] != 0) & valid).any()
    )
    return prediction, targets, valid, hard_legal


def evaluate_full_split_v3(
    model: SparseLocatorDecoratorV3,
    authority: V2CorpusAuthority,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    refs = authority.corpus.epoch_refs(split, 0, seed)
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    identities: list[str] = []
    valid_cells = 0
    hard_legal = True
    was_training = model.training
    model.eval()
    try:
        for group in _batches(refs, batch_size):
            samples = [authority.corpus.sample(ref) for ref in group]
            observed, truth, valid, batch_legal = _predict_batch(
                model,
                collate_teacher_samples(samples),
                device=device,
            )
            hard_legal = hard_legal and batch_legal
            for name in HEAD_NAMES:
                predictions[name].append(observed[name][valid].detach().cpu())
                targets[name].append(truth[name][valid].detach().cpu())
            valid_cells += int(valid.sum().item())
            identities.extend(ref.sample_identity_sha256 for ref in group)
    finally:
        model.train(was_training)
    joined_prediction = {name: torch.cat(values) for name, values in predictions.items()}
    joined_target = {name: torch.cat(values) for name, values in targets.items()}
    metrics = decoration_metrics(
        joined_prediction,
        joined_target,
        torch.ones((valid_cells,), dtype=torch.bool),
    )
    metrics.update(
        {
            "split": split,
            "sample_count": len(identities),
            "sample_set_sha256": json_sha256(sorted(identities)),
            "full_split": len(identities) == len(authority.corpus.refs_by_split[split]),
            "valid_cell_count": valid_cells,
            "hard_legality": 1.0 if hard_legal else 0.0,
            "immutable_semantic_changes": 0,
            "source_provenance_failures": 0,
        }
    )
    return metrics


def _evaluate_pair(
    model: SparseLocatorDecoratorV3,
    authority: V2CorpusAuthority,
    config: CalibrationConfig,
    device: torch.device,
) -> dict[str, object]:
    return {
        "validation": evaluate_full_split_v3(
            model,
            authority,
            "validation",
            batch_size=config.validation_batch_size,
            device=device,
            seed=config.training.seed,
        ),
        "test": evaluate_full_split_v3(
            model,
            authority,
            "test",
            batch_size=config.test_batch_size,
            device=device,
            seed=config.training.seed,
        ),
    }


def _train(
    model: SparseLocatorDecoratorV3,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    authority: V2CorpusAuthority,
    config: CalibrationConfig,
    generator: torch.Generator,
    device: torch.device,
) -> list[dict[str, object]]:
    plan = plan_foreground_batches(
        authority.stats["train"],
        steps=config.steps,
        epoch=0,
        seed=config.training.seed,
        config=config.patch,
    )
    history: list[dict[str, object]] = []
    for step, planned in enumerate(plan):
        crops = []
        source_ids: list[str] = []
        focus_heads: list[str] = []
        for slot, item in enumerate(planned):
            sample = authority.sample_for_stat(item.stat)
            source_ids.append(sample.sample_identity_sha256)
            focus_heads.append(item.focus_head)
            crops.append(
                foreground_centered_crop(
                    sample,
                    focus_head=item.focus_head,
                    epoch=0,
                    step=step,
                    slot=slot,
                    seed=config.training.seed,
                    config=config.patch,
                )
            )
        result = train_batch_v3(
            model,
            optimizer,
            ema,
            collate_teacher_samples(crops),
            generator=generator,
            training_config=config.training,
            loss_config=config.loss,
            device=device,
            autocast_dtype=torch.bfloat16,
        )
        loss = result["loss"]
        if not isinstance(loss, dict):
            raise TypeError("V3 calibration loss payload is malformed.")
        record: dict[str, object] = {
            "step": step + 1,
            "source_sample_sha256": source_ids,
            "focus_heads": focus_heads,
            "total_loss": float(loss["total"]),
            "gradient_norm": float(result["gradient_norm"]),
            "decal_predicted_count": float(loss["decal_predicted_count"]),
            "decal_target_count": float(loss["decal_target_count"]),
            "prop_predicted_count": float(loss["prop_predicted_count"]),
            "prop_target_count": float(loss["prop_target_count"]),
            "full_mask_sample_count": int(result["full_mask_sample_count"]),
        }
        if any(
            not math.isfinite(float(record[key]))
            for key in (
                "total_loss",
                "gradient_norm",
                "decal_predicted_count",
                "decal_target_count",
                "prop_predicted_count",
                "prop_target_count",
            )
        ):
            raise FloatingPointError("V3 CUDA calibration produced non-finite evidence.")
        if float(record["decal_target_count"]) <= 0 or float(record["prop_target_count"]) <= 0:
            raise ValueError("V3 calibration batch lost foreground supervision.")
        history.append(record)
    return history


def _semantic_payload(report: dict[str, object]) -> dict[str, object]:
    return {
        key: report[key]
        for key in (
            "format",
            "status",
            "v3_contract_sha256",
            "calibration_gate_sha256",
            "checkpoint_source_sha256",
            "authority",
            "config",
            "history",
            "raw_evaluation",
            "ema_evaluation",
            "raw_quality_gate",
            "ema_quality_gate",
            "quality_passed",
            "model_tensor_sha256",
            "ema_tensor_sha256",
            "ema_updates",
            "gates",
        )
    }


def run_calibration_worker(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    config: CalibrationConfig = CalibrationConfig(),
) -> dict[str, object]:
    output = Path(output).resolve()
    report_path = output / REPORT_NAME
    if report_path.is_file():
        return validate_calibration(report_path, corpus_root=corpus_root, index_root=index_root)
    if output.exists():
        raise FileExistsError(f"Calibration output exists without a complete report: {output}")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=2 * 1024**3)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    device = _configure_cuda(config.training.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    authority = V2CorpusAuthority.load(Path(corpus_root), Path(index_root))
    model = SparseLocatorDecoratorV3(config.model).to(device)
    optimizer = make_optimizer_v3(model, config.training)
    ema = WarmStartEMA(model, config.training.ema_decay)
    generator = torch.Generator(device=device).manual_seed(config.training.seed)
    training_started = time.perf_counter()
    history = _train(model, optimizer, ema, authority, config, generator, device)
    training_elapsed = time.perf_counter() - training_started
    evaluation_started = time.perf_counter()
    raw_evaluation = _evaluate_pair(model, authority, config, device)
    ema_model = _copy_ema_model(model, ema, device)
    ema_evaluation = _evaluate_pair(ema_model, authority, config, device)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    raw_gate = evaluate_dual_split_gate(
        raw_evaluation["validation"], raw_evaluation["test"], stage="calibration"  # type: ignore[arg-type]
    )
    ema_gate = evaluate_dual_split_gate(
        ema_evaluation["validation"], ema_evaluation["test"], stage="calibration"  # type: ignore[arg-type]
    )
    raw_safety = all(
        float(raw_evaluation[split]["hard_legality"]) == 1.0  # type: ignore[index]
        and int(raw_evaluation[split]["immutable_semantic_changes"]) == 0  # type: ignore[index]
        and int(raw_evaluation[split]["source_provenance_failures"]) == 0  # type: ignore[index]
        for split in ("validation", "test")
    )
    ema_safety = all(
        float(ema_evaluation[split]["hard_legality"]) == 1.0  # type: ignore[index]
        and int(ema_evaluation[split]["immutable_semantic_changes"]) == 0  # type: ignore[index]
        and int(ema_evaluation[split]["source_provenance_failures"]) == 0  # type: ignore[index]
        for split in ("validation", "test")
    )
    hard_safety = raw_safety and ema_safety
    quality_passed = bool(ema_gate["passed"])
    model_sha = tensor_state_sha256(model.state_dict())
    ema_sha = tensor_state_sha256(ema.shadow)
    metrics = {
        "history": history,
        "raw_evaluation": raw_evaluation,
        "ema_evaluation": ema_evaluation,
        "raw_quality_gate": raw_gate,
        "ema_quality_gate": ema_gate,
        "hard_safety": hard_safety,
        "quality_passed": quality_passed,
    }
    checkpoint_name = _checkpoint_name(config.steps)
    checkpoint_path = staging / checkpoint_name
    sidecar = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        ema,
        training_config=config.training,
        loss_config=config.loss,
        patch_config=config.patch,
        schedule={"epoch": 0, "steps": config.steps, "validation_batch_size": config.validation_batch_size, "test_batch_size": config.test_batch_size},
        corpus_sha256=authority.corpus.corpus_sha256,
        corpus_manifest_sha256=authority.corpus.manifest_sha256,
        index_semantic_sha256=authority.index_semantic_sha256,
        index_manifest_sha256=authority.index_manifest_sha256,
        epoch=0,
        global_step=config.steps,
        predecessor_checkpoint_sha256=None,
        training_generator=generator,
        metrics=metrics,
    )
    reload_model = SparseLocatorDecoratorV3(config.model).to(device)
    reload_optimizer = make_optimizer_v3(reload_model, config.training)
    reload_ema = WarmStartEMA(reload_model, config.training.ema_decay)
    reload_generator = torch.Generator(device=device).manual_seed(0)
    loaded = load_checkpoint(
        checkpoint_path,
        reload_model,
        reload_optimizer,
        reload_ema,
        reload_generator,
        expected={
            "v3_contract_sha256": V3_CONTRACT_SHA256,
            "corpus_sha256": authority.corpus.corpus_sha256,
            "index_semantic_sha256": authority.index_semantic_sha256,
            "model_config": config.model.to_dict(),
            "training_config": config.training.to_dict(),
            "loss_config": config.loss.to_dict(),
            "patch_config": config.patch.to_dict(),
            "epoch": 0,
            "global_step": config.steps,
        },
    )
    reload_exact = (
        tensor_state_sha256(reload_model.state_dict()) == model_sha
        and tensor_state_sha256(reload_ema.shadow) == ema_sha
        and loaded["metrics"] == metrics
    )
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "format": CALIBRATION_FORMAT,
        "status": "calibration_quality_passed" if quality_passed else "calibration_failed_quality",
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "calibration_gate_sha256": CALIBRATION_GATE_SHA256,
        "checkpoint_source_sha256": checkpoint_source_sha256(),
        "authority": {
            "corpus_root": Path(corpus_root).resolve().name,
            "corpus_sha256": authority.corpus.corpus_sha256,
            "corpus_manifest_sha256": authority.corpus.manifest_sha256,
            "index_root": Path(index_root).resolve().name,
            "index_semantic_sha256": authority.index_semantic_sha256,
            "index_manifest_sha256": authority.index_manifest_sha256,
            "split_counts": {name: len(authority.corpus.refs_by_split[name]) for name in ("train", "validation", "test")},
        },
        "config": config.to_dict(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "precision": "bf16",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "elapsed_seconds": elapsed,
            "training_seconds": training_elapsed,
            "evaluation_seconds": evaluation_elapsed,
            "training_steps_per_second": config.steps / training_elapsed,
            "memory": _memory_report(device),
        },
        "history": history,
        "raw_evaluation": raw_evaluation,
        "ema_evaluation": ema_evaluation,
        "raw_quality_gate": raw_gate,
        "ema_quality_gate": ema_gate,
        "quality_passed": quality_passed,
        "model_tensor_sha256": model_sha,
        "ema_tensor_sha256": ema_sha,
        "ema_updates": ema.updates,
        "checkpoint": {
            "path": checkpoint_name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
            "sidecar_sha256": sidecar["sidecar_sha256"],
        },
        "gates": {
            "real_corpus_bound": True,
            "foreground_index_bound": True,
            "finite_training": all(math.isfinite(float(item["total_loss"])) for item in history),
            "raw_hard_safety": raw_safety,
            "ema_hard_safety": ema_safety,
            "full_validation_split": bool(raw_evaluation["validation"]["full_split"] and ema_evaluation["validation"]["full_split"]),  # type: ignore[index]
            "full_test_split": bool(raw_evaluation["test"]["full_split"] and ema_evaluation["test"]["full_split"]),  # type: ignore[index]
            "checkpoint_reload_exact": reload_exact,
            "cuda_bf16": str(device) == "cuda:0" and torch.cuda.is_bf16_supported(),
            "quality_is_separate_from_publication_safety": True,
            "no_runtime_integration_without_quality_gate": True,
        },
    }
    if not all(report["gates"].values()):  # type: ignore[union-attr]
        raise RuntimeError(f"V3 calibration publication-safety gate failed: {report['gates']}")
    report["semantic_sha256"] = json_sha256(_semantic_payload(report))
    report["report_sha256"] = json_sha256(report)
    _atomic_json(staging / REPORT_NAME, report)
    os.replace(staging, output)
    return validate_calibration(output / REPORT_NAME, corpus_root=corpus_root, index_root=index_root)


def _config_from_dict(value: object) -> CalibrationConfig:
    if not isinstance(value, dict):
        raise ValueError("Calibration configuration is malformed.")
    config = CalibrationConfig(
        steps=int(value["steps"]),
        validation_batch_size=int(value["validation_batch_size"]),
        test_batch_size=int(value["test_batch_size"]),
        model=LocatorModelConfig(**value["model"]),
        training=LocatorTrainingConfig(**value["training"]),
        loss=LocatorLossConfig(**value["loss"]),
        patch=ForegroundPatchConfig(**value["patch"]),
    )
    if value != config.to_dict():
        raise ValueError("Calibration configuration is noncanonical.")
    return config


def validate_calibration(
    report_path: Path,
    *,
    corpus_root: Path,
    index_root: Path,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    value = _read_report(report_path)
    expected_keys = {
        "format", "status", "v3_contract_sha256", "calibration_gate_sha256",
        "checkpoint_source_sha256", "authority", "config", "runtime", "history",
        "raw_evaluation", "ema_evaluation", "raw_quality_gate", "ema_quality_gate",
        "quality_passed", "model_tensor_sha256", "ema_tensor_sha256", "ema_updates",
        "checkpoint", "gates", "semantic_sha256", "report_sha256",
    }
    if set(value) != expected_keys or value["format"] != CALIBRATION_FORMAT:
        raise ValueError("Calibration report format or members are invalid.")
    if value["status"] not in {"calibration_quality_passed", "calibration_failed_quality"}:
        raise ValueError("Calibration report status is invalid.")
    if value["v3_contract_sha256"] != V3_CONTRACT_SHA256 or value["calibration_gate_sha256"] != CALIBRATION_GATE_SHA256:
        raise ValueError("Calibration contract or gate provenance drifted.")
    if value["checkpoint_source_sha256"] != checkpoint_source_sha256():
        raise ValueError("Calibration source provenance drifted.")
    config = _config_from_dict(value["config"])
    authority = V2CorpusAuthority.load(Path(corpus_root), Path(index_root))
    expected_authority = {
        "corpus_root": Path(corpus_root).resolve().name,
        "corpus_sha256": authority.corpus.corpus_sha256,
        "corpus_manifest_sha256": authority.corpus.manifest_sha256,
        "index_root": Path(index_root).resolve().name,
        "index_semantic_sha256": authority.index_semantic_sha256,
        "index_manifest_sha256": authority.index_manifest_sha256,
        "split_counts": {name: len(authority.corpus.refs_by_split[name]) for name in ("train", "validation", "test")},
    }
    if value["authority"] != expected_authority:
        raise ValueError("Calibration corpus/index authority drifted.")
    runtime = value["runtime"]
    if not isinstance(runtime, dict) or runtime.get("device") != "cuda:0" or runtime.get("precision") != "bf16" or runtime.get("deterministic_algorithms") is not True:
        raise ValueError("Calibration runtime contract failed.")
    for timing in ("elapsed_seconds", "training_seconds", "evaluation_seconds", "training_steps_per_second"):
        if not math.isfinite(float(runtime.get(timing, math.nan))) or float(runtime[timing]) <= 0:
            raise ValueError("Calibration runtime timing evidence is malformed.")
    history = value["history"]
    if not isinstance(history, list) or len(history) != config.steps:
        raise ValueError("Calibration history length differs from its schedule.")
    for index, record in enumerate(history, start=1):
        if not isinstance(record, dict) or record.get("step") != index:
            raise ValueError("Calibration history record is malformed.")
        if any(not math.isfinite(float(record[key])) for key in ("total_loss", "gradient_norm", "decal_target_count", "prop_target_count")):
            raise ValueError("Calibration history contains non-finite evidence.")
    expected_counts = expected_authority["split_counts"]
    for collection_name in ("raw_evaluation", "ema_evaluation"):
        collection = value[collection_name]
        if not isinstance(collection, dict) or set(collection) != {"validation", "test"}:
            raise ValueError("Calibration evaluation split registry is malformed.")
        for split in ("validation", "test"):
            metrics = collection[split]
            expected_set = json_sha256(sorted(ref.sample_identity_sha256 for ref in authority.corpus.refs_by_split[split]))
            if metrics.get("sample_count") != expected_counts[split] or metrics.get("sample_set_sha256") != expected_set or metrics.get("full_split") is not True:
                raise ValueError("Calibration evaluation did not cover the authoritative full split.")
            if metrics.get("hard_legality") != 1.0 or metrics.get("immutable_semantic_changes") != 0 or metrics.get("source_provenance_failures") != 0:
                raise ValueError("Calibration evaluation safety evidence failed.")
    expected_raw_gate = evaluate_dual_split_gate(value["raw_evaluation"]["validation"], value["raw_evaluation"]["test"], stage="calibration")
    expected_ema_gate = evaluate_dual_split_gate(value["ema_evaluation"]["validation"], value["ema_evaluation"]["test"], stage="calibration")
    if value["raw_quality_gate"] != expected_raw_gate or value["ema_quality_gate"] != expected_ema_gate:
        raise ValueError("Calibration quality gate was not exactly replayed.")
    quality_passed = bool(expected_ema_gate["passed"])
    if value["quality_passed"] is not quality_passed or value["status"] != ("calibration_quality_passed" if quality_passed else "calibration_failed_quality"):
        raise ValueError("Calibration quality status is inconsistent.")
    checkpoint = value["checkpoint"]
    checkpoint_name = _checkpoint_name(config.steps)
    if not isinstance(checkpoint, dict) or checkpoint.get("path") != checkpoint_name:
        raise ValueError("Calibration checkpoint artifact record is malformed.")
    checkpoint_path = report_path.parent / checkpoint_name
    if checkpoint_path.stat().st_size != checkpoint["bytes"] or file_sha256(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("Calibration checkpoint artifact hash failed.")
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object)
    if not isinstance(sidecar, dict) or sidecar.get("sidecar_sha256") != checkpoint.get("sidecar_sha256"):
        raise ValueError("Calibration checkpoint sidecar identity failed.")
    payload = inspect_checkpoint(checkpoint_path)
    metrics = {
        "history": value["history"],
        "raw_evaluation": value["raw_evaluation"],
        "ema_evaluation": value["ema_evaluation"],
        "raw_quality_gate": value["raw_quality_gate"],
        "ema_quality_gate": value["ema_quality_gate"],
        "hard_safety": True,
        "quality_passed": quality_passed,
    }
    expected_checkpoint = {
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "source_sha256": value["checkpoint_source_sha256"],
        "corpus_sha256": authority.corpus.corpus_sha256,
        "index_semantic_sha256": authority.index_semantic_sha256,
        "model_config": config.model.to_dict(),
        "training_config": config.training.to_dict(),
        "loss_config": config.loss.to_dict(),
        "patch_config": config.patch.to_dict(),
        "schedule": {
            "epoch": 0,
            "steps": config.steps,
            "validation_batch_size": config.validation_batch_size,
            "test_batch_size": config.test_batch_size,
        },
        "epoch": 0,
        "global_step": config.steps,
        "model_tensor_sha256": value["model_tensor_sha256"],
        "ema_tensor_sha256": value["ema_tensor_sha256"],
        "metrics": metrics,
    }
    for label, expected in expected_checkpoint.items():
        if payload.get(label) != expected:
            raise ValueError(f"Calibration checkpoint mismatch for {label}.")
    if value["ema_updates"] != config.steps:
        raise ValueError("Calibration EMA update count differs from its schedule.")
    expected_gates = {
        "real_corpus_bound": True,
        "foreground_index_bound": True,
        "finite_training": True,
        "raw_hard_safety": True,
        "ema_hard_safety": True,
        "full_validation_split": True,
        "full_test_split": True,
        "checkpoint_reload_exact": True,
        "cuda_bf16": True,
        "quality_is_separate_from_publication_safety": True,
        "no_runtime_integration_without_quality_gate": True,
    }
    if value["gates"] != expected_gates:
        raise ValueError("Calibration publication-safety gates are not exact.")
    if value["semantic_sha256"] != json_sha256(_semantic_payload(value)):
        raise ValueError("Calibration semantic identity failed.")
    return value


def _exit_class(returncode: int | None, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode is None:
        return "missing_exit_code"
    unsigned = returncode & 0xFFFFFFFF
    if unsigned == 0xC0000005:
        return "windows_access_violation"
    if returncode == 0:
        return "success"
    return f"exit_{unsigned:08x}"


def supervise_calibration(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    config: CalibrationConfig = CalibrationConfig(),
    max_attempts: int = 3,
    timeout_seconds: int = 3_600,
) -> dict[str, Any]:
    if isinstance(max_attempts, bool) or not 1 <= max_attempts <= 3:
        raise ValueError("Calibration supervisor attempts must be in [1,3].")
    if isinstance(timeout_seconds, bool) or not 60 <= timeout_seconds <= 7_200:
        raise ValueError("Calibration supervisor timeout must be in [60,7200] seconds.")
    output = Path(output).resolve()
    if (output / SUPERVISOR_REPORT_NAME).is_file():
        return validate_supervised_calibration(output, corpus_root=corpus_root, index_root=index_root)
    if output.exists():
        raise FileExistsError(f"Supervisor output exists without a complete report: {output}")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=3 * 1024**3)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, object]] = []
    successful: Path | None = None
    try:
        for attempt in range(1, max_attempts + 1):
            attempt_root = staging / f"attempt_{attempt:02d}"
            attempt_root.mkdir()
            worker_output = attempt_root / "calibration"
            command = [
                sys.executable,
                "-m",
                "forge.map_decorator_production_v3",
                "calibration-worker",
                "--corpus",
                str(Path(corpus_root).resolve()),
                "--index",
                str(Path(index_root).resolve()),
                "--output",
                str(worker_output),
                "--steps",
                str(config.steps),
                "--validation-batch-size",
                str(config.validation_batch_size),
                "--test-batch-size",
                str(config.test_batch_size),
                "--base-channels",
                str(config.model.base_channels),
            ]
            environment = dict(os.environ)
            environment.update(
                {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
            started = time.perf_counter()
            timed_out = False
            try:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                returncode = completed.returncode
                stdout = completed.stdout[-2 * 1024 * 1024 :]
                stderr = completed.stderr[-2 * 1024 * 1024 :]
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = None
                stdout = (exc.stdout or b"")[-2 * 1024 * 1024 :]
                stderr = (exc.stderr or b"")[-2 * 1024 * 1024 :]
            (attempt_root / "stdout.log").write_bytes(stdout)
            (attempt_root / "stderr.log").write_bytes(stderr)
            record: dict[str, object] = {
                "attempt": attempt,
                "returncode": returncode,
                "exit_class": _exit_class(returncode, timed_out),
                "elapsed_seconds": time.perf_counter() - started,
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            }
            attempts.append(record)
            if returncode == 0:
                validate_calibration(worker_output / REPORT_NAME, corpus_root=corpus_root, index_root=index_root)
                successful = worker_output
                break
        if successful is None:
            failed = output.parent / f"{output.name}.failed-{uuid.uuid4().hex}"
            _atomic_json(staging / "failed_supervisor.json", {"format": SUPERVISOR_FORMAT, "attempts": attempts})
            os.replace(staging, failed)
            raise RuntimeError(f"V3 calibration failed after {len(attempts)} attempts; evidence: {failed}")
        calibration_target = staging / "calibration"
        os.replace(successful, calibration_target)
        calibration_report = validate_calibration(calibration_target / REPORT_NAME, corpus_root=corpus_root, index_root=index_root)
        supervisor: dict[str, object] = {
            "format": SUPERVISOR_FORMAT,
            "status": "published",
            "checkpoint_source_sha256": checkpoint_source_sha256(),
            "attempt_count": len(attempts),
            "attempts": attempts,
            "calibration": {
                "path": f"calibration/{REPORT_NAME}",
                "report_file_sha256": file_sha256(calibration_target / REPORT_NAME),
                "report_sha256": calibration_report["report_sha256"],
                "semantic_sha256": calibration_report["semantic_sha256"],
                "quality_passed": calibration_report["quality_passed"],
                "status": calibration_report["status"],
            },
            "gates": {
                "bounded_attempts": len(attempts) <= max_attempts,
                "successful_attempt": attempts[-1]["exit_class"] == "success",
                "worker_validated_before_publication": True,
                "disk_floor_enforced": True,
            },
        }
        supervisor["report_sha256"] = json_sha256(supervisor)
        _atomic_json(staging / SUPERVISOR_REPORT_NAME, supervisor)
        os.replace(staging, output)
        return validate_supervised_calibration(output, corpus_root=corpus_root, index_root=index_root)
    except BaseException:
        # Preserve unique staging as evidence on this host unless it was already
        # promoted to a named failure directory.
        raise


def validate_supervised_calibration(
    output: Path,
    *,
    corpus_root: Path,
    index_root: Path,
) -> dict[str, Any]:
    output = Path(output).resolve()
    supervisor = _read_report(output / SUPERVISOR_REPORT_NAME)
    if supervisor.get("format") != SUPERVISOR_FORMAT or supervisor.get("status") != "published":
        raise ValueError("Calibration supervisor format/status failed.")
    if supervisor.get("checkpoint_source_sha256") != checkpoint_source_sha256():
        raise ValueError("Calibration supervisor source drifted.")
    attempts = supervisor.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3 or supervisor.get("attempt_count") != len(attempts):
        raise ValueError("Calibration supervisor attempts are malformed.")
    if attempts[-1].get("exit_class") != "success":
        raise ValueError("Calibration supervisor has no successful terminal attempt.")
    calibration = supervisor.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("path") != f"calibration/{REPORT_NAME}":
        raise ValueError("Calibration supervisor artifact record is malformed.")
    report_path = output / "calibration" / REPORT_NAME
    report = validate_calibration(report_path, corpus_root=corpus_root, index_root=index_root)
    expected_calibration = {
        "path": f"calibration/{REPORT_NAME}",
        "report_file_sha256": file_sha256(report_path),
        "report_sha256": report["report_sha256"],
        "semantic_sha256": report["semantic_sha256"],
        "quality_passed": report["quality_passed"],
        "status": report["status"],
    }
    if calibration != expected_calibration:
        raise ValueError("Calibration supervisor artifact closure failed.")
    expected_gates = {
        "bounded_attempts": True,
        "successful_attempt": True,
        "worker_validated_before_publication": True,
        "disk_floor_enforced": True,
    }
    if supervisor.get("gates") != expected_gates:
        raise ValueError("Calibration supervisor gates failed.")
    return supervisor
