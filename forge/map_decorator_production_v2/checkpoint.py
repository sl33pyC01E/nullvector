from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import torch

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_ml.training import EMA
from ..safety import require_disk_floor
from .contract import (
    DISK_FLOOR_GIB,
    FactoredLossConfig,
    FactoredModelConfig,
    ForegroundPatchConfig,
    V2_CHECKPOINT_FORMAT_VERSION,
    V2_CONTRACT_SHA256,
    V2TrainingConfig,
)
from .model import FactoredDecoratorV2


SOURCE_PACKAGES: Final[tuple[str, ...]] = (
    "forge/map_decorator_ml",
    "forge/map_decorator_production",
    "forge/map_decorator_production_v2",
)
MAX_CHECKPOINT_BYTES: Final[int] = 768 * 1024 * 1024
MAX_SIDECAR_BYTES: Final[int] = 2 * 1024 * 1024
PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "format_version",
        "v2_contract_sha256",
        "source_manifest",
        "source_sha256",
        "corpus_sha256",
        "corpus_manifest_sha256",
        "index_semantic_sha256",
        "index_manifest_sha256",
        "model_config",
        "training_config",
        "loss_config",
        "patch_config",
        "schedule",
        "epoch",
        "global_step",
        "predecessor_checkpoint_sha256",
        "model_state",
        "model_tensor_sha256",
        "ema_state",
        "ema_tensor_sha256",
        "optimizer_state",
        "training_generator_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "metrics",
    }
)
SIDECAR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "checkpoint",
        "checkpoint_sha256",
        "format_version",
        "v2_contract_sha256",
        "source_sha256",
        "corpus_sha256",
        "index_semantic_sha256",
        "epoch",
        "global_step",
        "predecessor_checkpoint_sha256",
        "model_tensor_sha256",
        "ema_tensor_sha256",
    }
)


class V2CheckpointError(ValueError):
    pass


def source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for package_name in SOURCE_PACKAGES:
        package = Path(root) / package_name
        for path in sorted(item for item in package.glob("*.py") if item.is_file()):
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise FileNotFoundError("No v2 training sources were found.")
    return result


def source_sha256(root: Path = PROJECT_ROOT) -> str:
    return json_sha256(source_manifest(root))


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State member {name!r} is not a tensor.")
        tensor = value.detach().contiguous().cpu()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"State member {name!r} contains a non-finite value.")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        array = tensor.view(torch.uint8).numpy()
        digest.update(memoryview(array))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=len(encoded) + 1024 * 1024)
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


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=1024**3)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", suffix=".pt", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _exact(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise V2CheckpointError(f"Checkpoint mismatch for {label}: {observed!r} != {expected!r}")


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


def save_checkpoint(
    path: Path,
    model: FactoredDecoratorV2,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    *,
    training_config: V2TrainingConfig,
    loss_config: FactoredLossConfig,
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
    for label, value in {
        "corpus_sha256": corpus_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "index_semantic_sha256": index_semantic_sha256,
        "index_manifest_sha256": index_manifest_sha256,
    }.items():
        if not _is_sha(value):
            raise ValueError(f"{label} must be a SHA-256 string.")
    if not _is_sha(predecessor_checkpoint_sha256, nullable=True):
        raise ValueError("predecessor_checkpoint_sha256 is malformed.")
    if epoch < 0 or global_step < 0:
        raise ValueError("Checkpoint counters cannot be negative.")
    sources = source_manifest()
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    ema_state = ema.state_dict()
    shadow = ema_state["shadow"]
    if not isinstance(shadow, dict):
        raise TypeError("EMA shadow is malformed.")
    payload: dict[str, object] = {
        "format_version": V2_CHECKPOINT_FORMAT_VERSION,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
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
        "optimizer_state": optimizer.state_dict(),
        "training_generator_state": training_generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "metrics": metrics,
    }
    _atomic_torch_save(path, payload)
    sidecar = {
        "checkpoint": path.name,
        "checkpoint_sha256": file_sha256(path),
        "format_version": V2_CHECKPOINT_FORMAT_VERSION,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "source_sha256": payload["source_sha256"],
        "corpus_sha256": corpus_sha256,
        "index_semantic_sha256": index_semantic_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "predecessor_checkpoint_sha256": predecessor_checkpoint_sha256,
        "model_tensor_sha256": payload["model_tensor_sha256"],
        "ema_tensor_sha256": payload["ema_tensor_sha256"],
    }
    _atomic_json(path.with_suffix(path.suffix + ".json"), sidecar)
    return sidecar


def _read(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise V2CheckpointError("Checkpoint is missing or exceeds its bounded size.")
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not sidecar_path.is_file() or sidecar_path.stat().st_size > MAX_SIDECAR_BYTES:
        raise V2CheckpointError("Checkpoint sidecar is missing or exceeds its bounded size.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict) or set(sidecar) != SIDECAR_KEYS:
        raise V2CheckpointError("Checkpoint sidecar members are invalid.")
    _exact("sidecar checkpoint name", sidecar["checkpoint"], path.name)
    _exact("checkpoint file hash", sidecar["checkpoint_sha256"], file_sha256(path))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise V2CheckpointError("Checkpoint payload members are invalid.")
    return payload, sidecar


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    payload, sidecar = _read(Path(path))
    for label in (
        "format_version",
        "v2_contract_sha256",
        "source_sha256",
        "corpus_sha256",
        "index_semantic_sha256",
        "epoch",
        "global_step",
        "predecessor_checkpoint_sha256",
        "model_tensor_sha256",
        "ema_tensor_sha256",
    ):
        _exact(f"sidecar.{label}", sidecar[label], payload[label])
    expected_sources = source_manifest()
    _exact("source manifest", payload["source_manifest"], expected_sources)
    _exact("source SHA", payload["source_sha256"], json_sha256(expected_sources))
    _exact("contract SHA", payload["v2_contract_sha256"], V2_CONTRACT_SHA256)
    model_config = FactoredModelConfig(**payload["model_config"])
    training_config = V2TrainingConfig(**payload["training_config"])
    loss_config = FactoredLossConfig(**payload["loss_config"])
    patch_config = ForegroundPatchConfig(**payload["patch_config"])
    for name, value in {
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "patch_config": patch_config.to_dict(),
    }.items():
        _exact(name, payload[name], value)
    model_state = payload["model_state"]
    ema_state = payload["ema_state"]
    if not isinstance(model_state, dict) or not isinstance(ema_state, dict) or set(ema_state) != {"decay", "shadow"}:
        raise V2CheckpointError("Checkpoint tensor states are malformed.")
    _exact("model tensor hash", tensor_state_sha256(model_state), payload["model_tensor_sha256"])
    shadow = ema_state["shadow"]
    if not isinstance(shadow, dict):
        raise V2CheckpointError("EMA shadow is malformed.")
    _exact("EMA tensor hash", tensor_state_sha256(shadow), payload["ema_tensor_sha256"])
    reference = FactoredDecoratorV2(model_config).state_dict()
    for label, state in (("model", model_state), ("EMA", shadow)):
        expected_names = set(reference) if label == "model" else {
            name for name, value in reference.items() if value.is_floating_point()
        }
        if set(state) != expected_names:
            raise V2CheckpointError(f"{label} state names drifted.")
        for name, value in state.items():
            expected = reference[name]
            expected_dtype = expected.dtype if label == "model" else torch.float32
            if value.shape != expected.shape or value.dtype != expected_dtype:
                raise V2CheckpointError(f"{label} tensor {name!r} violates its model contract.")
    if not isinstance(payload["schedule"], dict) or not isinstance(payload["metrics"], dict):
        raise V2CheckpointError("Checkpoint schedule or metrics are malformed.")
    return payload


def load_checkpoint(
    path: Path,
    model: FactoredDecoratorV2,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    training_generator: torch.Generator,
    *,
    expected: dict[str, object],
) -> dict[str, Any]:
    payload = inspect_checkpoint(path)
    for label, value in expected.items():
        _exact(label, payload.get(label), value)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    ema.load_state_dict(payload["ema_state"])
    training_generator.set_state(payload["training_generator_state"].to(training_generator.device))
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    cuda_states = payload["torch_cuda_rng_states"]
    if torch.cuda.is_available():
        if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
            raise V2CheckpointError("Checkpoint CUDA RNG device count does not match the host.")
        torch.cuda.set_rng_state_all(cuda_states)
    return payload
