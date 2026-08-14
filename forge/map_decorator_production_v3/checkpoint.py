from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final
import math

import torch

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_production_v2.contract import ForegroundPatchConfig
from ..map_decorator_production_v2.training import WarmStartEMA
from ..safety import require_disk_floor
from .contract import (
    LocatorLossConfig,
    LocatorModelConfig,
    LocatorTrainingConfig,
    V3_CHECKPOINT_FORMAT_VERSION,
    V3_CONTRACT_SHA256,
)
from .model import SparseLocatorDecoratorV3
from .smoke import tensor_state_sha256


SOURCE_PACKAGES: Final[tuple[str, ...]] = (
    "forge/map_decorator_ml",
    "forge/map_decorator_production",
    "forge/map_decorator_production_v2",
    "forge/map_decorator_production_v3",
)
MAX_CHECKPOINT_BYTES: Final[int] = 768 * 1024 * 1024
MAX_SIDECAR_BYTES: Final[int] = 2 * 1024 * 1024
PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "format_version", "v3_contract_sha256", "source_manifest", "source_sha256",
        "corpus_sha256", "corpus_manifest_sha256", "index_semantic_sha256", "index_manifest_sha256",
        "model_config", "training_config", "loss_config", "patch_config", "schedule",
        "epoch", "global_step", "predecessor_checkpoint_sha256", "model_state",
        "model_tensor_sha256", "ema_state", "ema_tensor_sha256", "optimizer_state",
        "training_generator_state", "torch_cpu_rng_state", "torch_cuda_rng_states", "metrics",
    }
)
SIDECAR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "checkpoint", "checkpoint_sha256", "format_version", "v3_contract_sha256",
        "source_sha256", "corpus_sha256", "index_semantic_sha256", "epoch", "global_step",
        "predecessor_checkpoint_sha256", "model_tensor_sha256", "ema_tensor_sha256",
        "sidecar_sha256",
    }
)


class V3CheckpointError(ValueError):
    pass


def checkpoint_source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for package_name in SOURCE_PACKAGES:
        package = Path(root) / package_name
        for path in sorted(item for item in package.glob("*.py") if item.is_file()):
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise FileNotFoundError("No v3 checkpoint source files were found.")
    return result


def checkpoint_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return json_sha256(checkpoint_source_manifest(root))


def _is_sha(value: object, *, nullable: bool = False) -> bool:
    if nullable and value is None:
        return True
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _exact(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise V3CheckpointError(f"Checkpoint mismatch for {label}: {observed!r} != {expected!r}")


def _validate_metadata(value: object, *, label: str, depth: int = 0) -> None:
    if depth > 12:
        raise ValueError(f"{label} exceeds the bounded metadata nesting depth.")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float.")
        return
    if isinstance(value, list):
        if len(value) > 100_000:
            raise ValueError(f"{label} list exceeds its bounded size.")
        for item in value:
            _validate_metadata(item, label=label, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 10_000 or any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} mapping violates its bounded string-key contract.")
        for item in value.values():
            _validate_metadata(item, label=label, depth=depth + 1)
        return
    raise TypeError(f"{label} contains unsupported metadata type {type(value).__name__}.")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise V3CheckpointError(f"Checkpoint JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _cpu_clone(value: object, *, label: str, depth: int = 0) -> object:
    """Detach checkpoint state from devices and reject unsupported tensor trees."""
    if depth > 24:
        raise ValueError(f"{label} exceeds the bounded checkpoint nesting depth.")
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
        if result.is_floating_point() and not bool(torch.isfinite(result).all()):
            raise ValueError(f"{label} contains a non-finite tensor.")
        return result
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float.")
        return value
    if isinstance(value, dict):
        return {
            key: _cpu_clone(item, label=f"{label}.{key}", depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_cpu_clone(item, label=label, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item, label=label, depth=depth + 1) for item in value)
    raise TypeError(f"{label} contains unsupported state type {type(value).__name__}.")


def _validate_schedule(value: object) -> None:
    if not isinstance(value, dict) or not value or any(
        not isinstance(key, str) or not key or type(item) is not int or item < 0
        for key, item in value.items()
    ):
        raise V3CheckpointError(
            "Checkpoint schedule must be a nonempty string-to-nonnegative-int map."
        )


def _validate_rng_state(value: object, *, label: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise V3CheckpointError(f"Checkpoint {label} is malformed.")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=512 * 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", suffix=".pt", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_checkpoint(
    path: Path,
    model: SparseLocatorDecoratorV3,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    *,
    training_config: LocatorTrainingConfig,
    loss_config: LocatorLossConfig,
    patch_config: ForegroundPatchConfig,
    schedule: dict[str, int],
    corpus_sha256: str,
    corpus_manifest_sha256: str,
    index_semantic_sha256: str,
    index_manifest_sha256: str,
    epoch: int,
    global_step: int,
    predecessor_checkpoint_sha256: str | None,
    training_generator: torch.Generator,
    metrics: dict[str, object],
) -> dict[str, object]:
    path = Path(path).resolve()
    for label, value in {
        "corpus_sha256": corpus_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "index_semantic_sha256": index_semantic_sha256,
        "index_manifest_sha256": index_manifest_sha256,
    }.items():
        if not _is_sha(value):
            raise ValueError(f"{label} must be a SHA-256 string.")
    if not _is_sha(predecessor_checkpoint_sha256, nullable=True):
        raise ValueError("predecessor checkpoint identity is malformed.")
    if type(epoch) is not int or type(global_step) is not int or epoch < 0 or global_step < 0:
        raise ValueError("Checkpoint counters must be nonnegative integers.")
    if ema.updates != global_step:
        raise ValueError("EMA update count must equal global_step.")
    _validate_schedule(schedule)
    if not isinstance(metrics, dict):
        raise TypeError("Checkpoint metrics must be a dictionary.")
    _validate_metadata(metrics, label="metrics")
    sources = checkpoint_source_manifest()
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    ema_state = _cpu_clone(ema.state_dict(), label="ema_state")
    if not isinstance(ema_state, dict):
        raise TypeError("EMA state is malformed.")
    shadow = ema_state.get("shadow")
    if not isinstance(shadow, dict):
        raise TypeError("EMA shadow is malformed.")
    payload: dict[str, object] = {
        "format_version": V3_CHECKPOINT_FORMAT_VERSION,
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "source_manifest": sources,
        "source_sha256": json_sha256(sources),
        "corpus_sha256": corpus_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "index_semantic_sha256": index_semantic_sha256,
        "index_manifest_sha256": index_manifest_sha256,
        "model_config": model.config.to_dict(),
        "training_config": training_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "patch_config": patch_config.to_dict(),
        "schedule": schedule,
        "epoch": epoch,
        "global_step": global_step,
        "predecessor_checkpoint_sha256": predecessor_checkpoint_sha256,
        "model_state": model_state,
        "model_tensor_sha256": tensor_state_sha256(model_state),
        "ema_state": ema_state,
        "ema_tensor_sha256": tensor_state_sha256(shadow),
        "optimizer_state": _cpu_clone(optimizer.state_dict(), label="optimizer_state"),
        "training_generator_state": training_generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        # A CPU-only run stays CUDA-cold. A future CUDA calibration records every
        # device stream so an interrupted segment can resume exactly.
        "torch_cuda_rng_states": (
            [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else []
        ),
        "metrics": metrics,
    }
    _atomic_torch_save(path, payload)
    sidecar: dict[str, object] = {
        "checkpoint": path.name,
        "checkpoint_sha256": file_sha256(path),
        "format_version": V3_CHECKPOINT_FORMAT_VERSION,
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "source_sha256": payload["source_sha256"],
        "corpus_sha256": corpus_sha256,
        "index_semantic_sha256": index_semantic_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "predecessor_checkpoint_sha256": predecessor_checkpoint_sha256,
        "model_tensor_sha256": payload["model_tensor_sha256"],
        "ema_tensor_sha256": payload["ema_tensor_sha256"],
    }
    sidecar["sidecar_sha256"] = json_sha256(sidecar)
    _atomic_json(path.with_suffix(path.suffix + ".json"), sidecar)
    return sidecar


def _read(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise V3CheckpointError("Checkpoint is missing or exceeds its bounded size.")
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not sidecar_path.is_file() or sidecar_path.stat().st_size > MAX_SIDECAR_BYTES:
        raise V3CheckpointError("Checkpoint sidecar is missing or exceeds its bounded size.")
    try:
        sidecar = json.loads(
            sidecar_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise V3CheckpointError("Checkpoint sidecar JSON is invalid.") from exc
    if not isinstance(sidecar, dict) or set(sidecar) != SIDECAR_KEYS:
        raise V3CheckpointError("Checkpoint sidecar members are invalid.")
    stored = sidecar.pop("sidecar_sha256")
    _exact("sidecar SHA", stored, json_sha256(sidecar))
    sidecar["sidecar_sha256"] = stored
    _exact("sidecar checkpoint name", sidecar["checkpoint"], path.name)
    _exact("checkpoint file hash", sidecar["checkpoint_sha256"], file_sha256(path))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise V3CheckpointError("Checkpoint payload members are invalid.")
    return payload, sidecar


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    payload, sidecar = _read(path)
    for label in (
        "format_version", "v3_contract_sha256", "source_sha256", "corpus_sha256",
        "index_semantic_sha256", "epoch", "global_step", "predecessor_checkpoint_sha256",
        "model_tensor_sha256", "ema_tensor_sha256",
    ):
        _exact(f"sidecar.{label}", sidecar[label], payload[label])
    _exact("checkpoint format", payload["format_version"], V3_CHECKPOINT_FORMAT_VERSION)
    _exact("contract SHA", payload["v3_contract_sha256"], V3_CONTRACT_SHA256)
    sources = checkpoint_source_manifest()
    _exact("source manifest", payload["source_manifest"], sources)
    _exact("source SHA", payload["source_sha256"], json_sha256(sources))
    for label in ("corpus_sha256", "corpus_manifest_sha256", "index_semantic_sha256", "index_manifest_sha256"):
        if not _is_sha(payload[label]):
            raise V3CheckpointError(f"Checkpoint {label} is malformed.")
    if not _is_sha(payload["predecessor_checkpoint_sha256"], nullable=True):
        raise V3CheckpointError("Checkpoint predecessor is malformed.")
    model_config = LocatorModelConfig(**payload["model_config"])
    training_config = LocatorTrainingConfig(**payload["training_config"])
    loss_config = LocatorLossConfig(**payload["loss_config"])
    patch_config = ForegroundPatchConfig(**payload["patch_config"])
    for label, expected in {
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "patch_config": patch_config.to_dict(),
    }.items():
        _exact(label, payload[label], expected)
    model_state, ema_state = payload["model_state"], payload["ema_state"]
    if not isinstance(model_state, dict) or not isinstance(ema_state, dict) or set(ema_state) != {"decay", "updates", "warmup_policy", "shadow"}:
        raise V3CheckpointError("Checkpoint tensor states are malformed.")
    if ema_state["warmup_policy"] != WarmStartEMA.POLICY or ema_state["updates"] != payload["global_step"] or float(ema_state["decay"]) != training_config.ema_decay:
        raise V3CheckpointError("Checkpoint EMA policy/update count drifted.")
    _exact("model tensor SHA", tensor_state_sha256(model_state), payload["model_tensor_sha256"])
    shadow = ema_state["shadow"]
    if not isinstance(shadow, dict):
        raise V3CheckpointError("Checkpoint EMA shadow is malformed.")
    _exact("EMA tensor SHA", tensor_state_sha256(shadow), payload["ema_tensor_sha256"])
    cpu_rng_before_reference = torch.get_rng_state()
    try:
        reference = SparseLocatorDecoratorV3(model_config).state_dict()
    finally:
        torch.set_rng_state(cpu_rng_before_reference)
    for label, state in (("model", model_state), ("EMA", shadow)):
        expected_names = set(reference) if label == "model" else {name for name, value in reference.items() if value.is_floating_point()}
        if set(state) != expected_names:
            raise V3CheckpointError(f"{label} state names drifted.")
        for name, value in state.items():
            expected = reference[name]
            dtype = expected.dtype if label == "model" else torch.float32
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != dtype:
                raise V3CheckpointError(f"{label} tensor {name!r} violates its model contract.")
    if not isinstance(payload["optimizer_state"], dict) or set(payload["optimizer_state"]) != {"state", "param_groups"}:
        raise V3CheckpointError("Checkpoint optimizer state is malformed.")
    for label in ("training_generator_state", "torch_cpu_rng_state"):
        _validate_rng_state(payload[label], label=label)
    cuda_rng_states = payload["torch_cuda_rng_states"]
    if not isinstance(cuda_rng_states, list):
        raise V3CheckpointError("Checkpoint CUDA RNG state collection is malformed.")
    for index, state in enumerate(cuda_rng_states):
        _validate_rng_state(state, label=f"torch_cuda_rng_states[{index}]")
    _validate_schedule(payload["schedule"])
    if not isinstance(payload["metrics"], dict):
        raise V3CheckpointError("Checkpoint metrics are malformed.")
    _validate_metadata(payload["metrics"], label="metrics")
    if type(payload["epoch"]) is not int or type(payload["global_step"]) is not int or payload["epoch"] < 0 or payload["global_step"] < 0:
        raise V3CheckpointError("Checkpoint counters are malformed.")
    return payload


def load_checkpoint(
    path: Path,
    model: SparseLocatorDecoratorV3,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    training_generator: torch.Generator,
    *,
    expected: Mapping[str, object],
) -> dict[str, Any]:
    payload = inspect_checkpoint(path)
    for label, value in expected.items():
        _exact(label, payload.get(label), value)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    ema.load_state_dict(payload["ema_state"])
    training_generator.set_state(payload["training_generator_state"].to(training_generator.device))
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    cuda_rng_states = payload["torch_cuda_rng_states"]
    if cuda_rng_states:
        if not torch.cuda.is_available():
            raise V3CheckpointError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        if len(cuda_rng_states) != torch.cuda.device_count():
            raise V3CheckpointError("Checkpoint CUDA RNG device count does not match this host.")
        torch.cuda.set_rng_state_all(cuda_rng_states)
    elif torch.cuda.is_initialized():
        raise V3CheckpointError(
            "A CUDA-initialized resume requires checkpointed CUDA RNG states."
        )
    return payload
