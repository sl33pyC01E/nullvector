from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import torch

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_ml.contract import ModelConfig
from ..map_decorator_production_v2.training import WarmStartEMA
from ..map_decorator_production_v4.contract import ProposalLocatorConfig, V4_CONTRACT_SHA256
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from ..map_decorator_production_v4.smoke import source_sha256 as v4_source_sha256
from ..safety import require_disk_floor
from .contract import ResidualLossConfig, ResidualTrainingConfig, V4_TRAINING_CONTRACT_SHA256


CHECKPOINT_FORMAT: Final[str] = "nullvector-map-decorator-v4-residual-checkpoint/1.0.0"
MAX_CHECKPOINT_BYTES: Final[int] = 768 * 1024 * 1024


def training_source_manifest(root: Path = PROJECT_ROOT) -> dict[str, object]:
    package = Path(root) / "forge/map_decorator_production_v4_training"
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in package.glob("*.py") if item.is_file())
    }
    return {"v4_source_sha256": v4_source_sha256(root), "training_files": files}


def training_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return json_sha256(training_source_manifest(root))


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _cpu_clone(value: object, depth: int = 0) -> object:
    if depth > 24:
        raise ValueError("V4 checkpoint state exceeds bounded nesting depth.")
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
        if result.is_floating_point() and not bool(torch.isfinite(result).all()):
            raise ValueError("V4 checkpoint contains non-finite tensor state.")
        return result
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("V4 checkpoint contains non-finite metadata.")
        return value
    if isinstance(value, dict):
        return {key: _cpu_clone(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item, depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item, depth + 1) for item in value)
    raise TypeError(f"Unsupported V4 checkpoint state type {type(value).__name__}.")


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


def _atomic_torch(path: Path, payload: dict[str, object]) -> None:
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
    model: ProposalConditionedDecoratorV4,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    generator: torch.Generator,
    *,
    core_config: ModelConfig,
    locator_config: ProposalLocatorConfig,
    training_config: ResidualTrainingConfig,
    loss_config: ResidualLossConfig,
    global_step: int,
    corpus_sha256: str,
    index_semantic_sha256: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    if type(global_step) is not int or global_step < 0 or ema.updates != global_step:
        raise ValueError("V4 checkpoint global step must equal EMA updates.")
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    ema_state = _cpu_clone(ema.state_dict())
    optimizer_state = _cpu_clone(optimizer.state_dict())
    payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "v4_contract_sha256": V4_CONTRACT_SHA256,
        "training_contract_sha256": V4_TRAINING_CONTRACT_SHA256,
        "source_manifest": training_source_manifest(),
        "source_sha256": training_source_sha256(),
        "corpus_sha256": corpus_sha256,
        "index_semantic_sha256": index_semantic_sha256,
        "core_config": core_config.to_dict(),
        "locator_config": locator_config.to_dict(),
        "training_config": training_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "global_step": global_step,
        "model_state": model_state,
        "model_tensor_sha256": tensor_state_sha256(model_state),
        "ema_state": ema_state,
        "ema_tensor_sha256": tensor_state_sha256(ema_state["shadow"]),  # type: ignore[index]
        "optimizer_state": optimizer_state,
        "generator_state": generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "metrics": metrics,
    }
    path = Path(path).resolve()
    _atomic_torch(path, payload)
    sidecar: dict[str, object] = {
        "checkpoint": path.name,
        "checkpoint_sha256": file_sha256(path),
        "format": CHECKPOINT_FORMAT,
        "source_sha256": payload["source_sha256"],
        "v4_contract_sha256": V4_CONTRACT_SHA256,
        "training_contract_sha256": V4_TRAINING_CONTRACT_SHA256,
        "corpus_sha256": corpus_sha256,
        "index_semantic_sha256": index_semantic_sha256,
        "global_step": global_step,
        "model_tensor_sha256": payload["model_tensor_sha256"],
        "ema_tensor_sha256": payload["ema_tensor_sha256"],
    }
    sidecar["sidecar_sha256"] = json_sha256(sidecar)
    _atomic_json(path.with_suffix(path.suffix + ".json"), sidecar)
    return sidecar


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("V4 residual checkpoint is missing, unsafe, or oversized.")
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    stored_sidecar = sidecar.pop("sidecar_sha256", None)
    if stored_sidecar != json_sha256(sidecar):
        raise ValueError("V4 checkpoint sidecar self-hash failed.")
    sidecar["sidecar_sha256"] = stored_sidecar
    if sidecar.get("checkpoint") != path.name or sidecar.get("checkpoint_sha256") != file_sha256(path):
        raise ValueError("V4 checkpoint sidecar artifact identity failed.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("V4 checkpoint payload is not a dictionary.")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("V4 checkpoint format failed.")
    if payload.get("v4_contract_sha256") != V4_CONTRACT_SHA256 or payload.get("training_contract_sha256") != V4_TRAINING_CONTRACT_SHA256:
        raise ValueError("V4 checkpoint contract drifted.")
    if payload.get("source_manifest") != training_source_manifest() or payload.get("source_sha256") != training_source_sha256():
        raise ValueError("V4 checkpoint source provenance drifted.")
    if tensor_state_sha256(payload["model_state"]) != payload.get("model_tensor_sha256"):
        raise ValueError("V4 checkpoint model tensor identity failed.")
    ema_state = payload.get("ema_state")
    if not isinstance(ema_state, dict) or tensor_state_sha256(ema_state["shadow"]) != payload.get("ema_tensor_sha256"):
        raise ValueError("V4 checkpoint EMA tensor identity failed.")
    return payload


def load_checkpoint(
    path: Path,
    model: ProposalConditionedDecoratorV4,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    generator: torch.Generator,
    *,
    expected_step: int,
    expected_corpus_sha256: str,
    expected_index_semantic_sha256: str,
    expected_training_config: ResidualTrainingConfig,
    expected_loss_config: ResidualLossConfig,
) -> dict[str, Any]:
    payload = inspect_checkpoint(path)
    if payload["global_step"] != expected_step:
        raise ValueError("V4 checkpoint step differs from resume contract.")
    expected = {
        "corpus_sha256": expected_corpus_sha256,
        "index_semantic_sha256": expected_index_semantic_sha256,
        "core_config": model.core_config.to_dict(),
        "locator_config": model.locator_config.to_dict(),
        "training_config": expected_training_config.to_dict(),
        "loss_config": expected_loss_config.to_dict(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V4 checkpoint {key} differs from resume authority.")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    ema.load_state_dict(payload["ema_state"])
    generator.set_state(payload["generator_state"].cpu())
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    return payload
