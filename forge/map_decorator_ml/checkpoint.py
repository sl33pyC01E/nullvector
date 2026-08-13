from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from ..config import PROJECT_ROOT
from ..map_decorator.catalog import CATALOG_SHA256
from ..map_decorator.contract import FEATURE_CONTRACT_SHA256
from ..map_decorator.hashing import json_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT_VERSION, MODEL_CONTRACT_SHA256, ModelConfig
from .model import CategoricalRefinementUNet
from .training import EMA, TrainingConfig


class ResumeContractError(ValueError):
    pass


MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_CHECKPOINT_SIDECAR_BYTES = 1024 * 1024
CHECKPOINT_PAYLOAD_KEYS = frozenset(
    {
        "format_version",
        "model_contract_sha256",
        "feature_contract_sha256",
        "catalog_sha256",
        "source_manifest",
        "source_sha256",
        "model_config",
        "training_config",
        "corpus_sha256",
        "epoch",
        "global_step",
        "model_state",
        "model_tensor_sha256",
        "ema_state",
        "ema_tensor_sha256",
        "optimizer_state",
        "training_generator_state",
        "metrics",
    }
)
TRAINING_SOURCE_PACKAGE = "forge/map_decorator_ml"
CHECKPOINT_SIDECAR_KEYS = frozenset(
    {
        "checkpoint",
        "checkpoint_sha256",
        "format_version",
        "model_contract_sha256",
        "source_sha256",
        "corpus_sha256",
        "epoch",
        "global_step",
        "model_tensor_sha256",
        "ema_tensor_sha256",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(
    root: Path = PROJECT_ROOT,
    *,
    packages: tuple[str, ...] = (TRAINING_SOURCE_PACKAGE,),
) -> dict[str, str]:
    files: list[Path] = []
    for relative_package in packages:
        package = Path(root) / relative_package
        files.extend(sorted(path for path in package.glob("*.py") if path.is_file()))
    if not files:
        raise FileNotFoundError("No checkpoint-bound Python sources were found.")
    result: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def source_sha256(
    root: Path = PROJECT_ROOT,
    *,
    packages: tuple[str, ...] = (TRAINING_SOURCE_PACKAGE,),
) -> str:
    return json_sha256(source_manifest(root, packages=packages))


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State member {name!r} is not a tensor.")
        contiguous = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, planned_bytes=256 * 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", suffix=".pt", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def save_checkpoint(
    path: Path,
    model: CategoricalRefinementUNet,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    *,
    training_config: TrainingConfig,
    corpus_sha256: str,
    epoch: int,
    global_step: int,
    training_generator: torch.Generator,
    metrics: dict[str, object] | None = None,
    root: Path = PROJECT_ROOT,
    source_packages: tuple[str, ...] = (TRAINING_SOURCE_PACKAGE,),
) -> dict[str, object]:
    try:
        valid_corpus_hash = len(corpus_sha256) == 64 and int(corpus_sha256, 16) >= 0
    except (TypeError, ValueError):
        valid_corpus_hash = False
    if not valid_corpus_hash:
        raise ValueError("corpus_sha256 must be a SHA-256 string.")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or epoch < 0
        or global_step < 0
    ):
        raise ValueError("epoch and global_step cannot be negative.")
    sources = source_manifest(root, packages=source_packages)
    source_hash = json_sha256(sources)
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    ema_state = ema.state_dict()
    shadow = ema_state["shadow"]
    if not isinstance(shadow, dict):
        raise TypeError("EMA state is malformed.")
    payload: dict[str, object] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_contract_sha256": MODEL_CONTRACT_SHA256,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "source_manifest": sources,
        "source_sha256": source_hash,
        "model_config": model.config.to_dict(),
        "training_config": training_config.to_dict(),
        "corpus_sha256": corpus_sha256,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model_state,
        "model_tensor_sha256": tensor_state_sha256(model_state),
        "ema_state": ema_state,
        "ema_tensor_sha256": tensor_state_sha256(shadow),  # type: ignore[arg-type]
        "optimizer_state": optimizer.state_dict(),
        "training_generator_state": training_generator.get_state().cpu(),
        "metrics": metrics or {},
    }
    _atomic_torch_save(payload, Path(path))
    report: dict[str, object] = {
        "checkpoint": Path(path).name,
        "checkpoint_sha256": file_sha256(path),
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_contract_sha256": MODEL_CONTRACT_SHA256,
        "source_sha256": source_hash,
        "corpus_sha256": corpus_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "model_tensor_sha256": payload["model_tensor_sha256"],
        "ema_tensor_sha256": payload["ema_tensor_sha256"],
    }
    _atomic_json(report, Path(path).with_suffix(Path(path).suffix + ".json"))
    return report


def _exact(name: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ResumeContractError(f"Resume contract mismatch for {name}: {observed!r} != {expected!r}")


def _read_checkpoint_container(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ResumeContractError("Checkpoint exceeds the bounded model foundation contract.")
    report_path = path.with_suffix(path.suffix + ".json")
    if not report_path.is_file():
        raise ResumeContractError("Checkpoint sidecar is required for pre-load file verification.")
    if report_path.stat().st_size > MAX_CHECKPOINT_SIDECAR_BYTES:
        raise ResumeContractError("Checkpoint sidecar exceeds its size bound.")
    try:
        sidecar = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResumeContractError(f"Checkpoint sidecar is malformed JSON: {error}") from error
    if not isinstance(sidecar, dict):
        raise ResumeContractError("Checkpoint sidecar root must be an object.")
    if set(sidecar) != CHECKPOINT_SIDECAR_KEYS:
        raise ResumeContractError("Checkpoint sidecar members are incomplete or unexpected.")
    if sidecar.get("checkpoint") != path.name:
        raise ResumeContractError("Checkpoint sidecar filename does not match the supplied file.")
    _exact("checkpoint_sha256", file_sha256(path), sidecar.get("checkpoint_sha256"))
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        raise ResumeContractError(f"Checkpoint container cannot be loaded safely: {error}") from error
    if not isinstance(payload, dict):
        raise ResumeContractError("Checkpoint root is not a dictionary.")
    if set(payload) != CHECKPOINT_PAYLOAD_KEYS:
        raise ResumeContractError("Checkpoint payload members are incomplete or unexpected.")
    return payload, sidecar


def _validate_checkpoint_provenance(
    payload: dict[str, object],
    sidecar: dict[str, object],
    *,
    root: Path,
    source_packages: tuple[str, ...],
) -> tuple[ModelConfig, TrainingConfig]:
    expected_base = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_contract_sha256": MODEL_CONTRACT_SHA256,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "source_manifest": source_manifest(root, packages=source_packages),
        "source_sha256": source_sha256(root, packages=source_packages),
    }
    for name, expected in expected_base.items():
        _exact(name, payload.get(name), expected)
    for name in (
        "format_version",
        "model_contract_sha256",
        "source_sha256",
        "corpus_sha256",
        "epoch",
        "global_step",
        "model_tensor_sha256",
        "ema_tensor_sha256",
    ):
        _exact(f"sidecar.{name}", sidecar.get(name), payload.get(name))
    if not all(
        _is_sha256(payload.get(name))
        for name in ("source_sha256", "corpus_sha256", "model_tensor_sha256", "ema_tensor_sha256")
    ):
        raise ResumeContractError("Checkpoint provenance contains a malformed SHA-256 value.")
    model_payload = payload.get("model_config")
    training_payload = payload.get("training_config")
    if not isinstance(model_payload, dict) or not isinstance(training_payload, dict):
        raise ResumeContractError("Checkpoint model/training configuration is malformed.")
    try:
        model_config = ModelConfig(**model_payload)
        training_config = TrainingConfig(**training_payload)
    except (TypeError, ValueError) as error:
        raise ResumeContractError(f"Checkpoint configuration violates its contract: {error}") from error
    if model_config.to_dict() != model_payload or training_config.to_dict() != training_payload:
        raise ResumeContractError("Checkpoint configuration members are incomplete or unexpected.")
    model_state = payload.get("model_state")
    ema_state = payload.get("ema_state")
    if not isinstance(model_state, dict) or not isinstance(ema_state, dict):
        raise ResumeContractError("Checkpoint model/EMA state is malformed.")
    if tensor_state_sha256(model_state) != payload.get("model_tensor_sha256"):
        raise ResumeContractError("Checkpoint model tensor hash does not match its payload.")
    if set(ema_state) != {"decay", "shadow"} or ema_state.get("decay") != training_config.ema_decay:
        raise ResumeContractError("Checkpoint EMA configuration does not match training configuration.")
    shadow = ema_state.get("shadow")
    if not isinstance(shadow, dict) or tensor_state_sha256(shadow) != payload.get("ema_tensor_sha256"):
        raise ResumeContractError("Checkpoint EMA tensor hash does not match its payload.")
    with torch.random.fork_rng(devices=[]):
        expected_model_state = CategoricalRefinementUNet(model_config).state_dict()
    for label, observed in (("model", model_state), ("EMA", shadow)):
        if set(observed) != set(expected_model_state):
            raise ResumeContractError(f"Checkpoint {label} tensor names do not match its model config.")
        for name, expected in expected_model_state.items():
            tensor = observed[name]
            expected_dtype = expected.dtype if label == "model" else torch.float32
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected.shape
                or tensor.dtype != expected_dtype
                or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
            ):
                raise ResumeContractError(f"Checkpoint {label} tensor is invalid for {name!r}.")
    epoch = payload.get("epoch")
    global_step = payload.get("global_step")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ResumeContractError("Checkpoint counters are invalid.")
    if not isinstance(payload.get("metrics"), dict):
        raise ResumeContractError("Checkpoint metrics must be an object.")
    return model_config, training_config


def inspect_checkpoint_provenance(
    path: Path,
    *,
    root: Path = PROJECT_ROOT,
    source_packages: tuple[str, ...] = (TRAINING_SOURCE_PACKAGE,),
) -> dict[str, object]:
    """Verify a checkpoint without trusting its sidecar or constructing an optimizer."""
    payload, sidecar = _read_checkpoint_container(Path(path))
    model_config, training_config = _validate_checkpoint_provenance(
        payload, sidecar, root=root, source_packages=source_packages
    )
    return {
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "source_sha256": payload["source_sha256"],
        "corpus_sha256": payload["corpus_sha256"],
        "model_tensor_sha256": payload["model_tensor_sha256"],
        "ema_tensor_sha256": payload["ema_tensor_sha256"],
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
    }


def load_checkpoint(
    path: Path,
    model: CategoricalRefinementUNet,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    *,
    expected_training_config: TrainingConfig,
    expected_corpus_sha256: str,
    training_generator: torch.Generator,
    root: Path = PROJECT_ROOT,
    source_packages: tuple[str, ...] = (TRAINING_SOURCE_PACKAGE,),
) -> dict[str, object]:
    path = Path(path)
    payload, sidecar = _read_checkpoint_container(path)
    _validate_checkpoint_provenance(
        payload, sidecar, root=root, source_packages=source_packages
    )
    expected_values = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_contract_sha256": MODEL_CONTRACT_SHA256,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "source_manifest": source_manifest(root, packages=source_packages),
        "source_sha256": source_sha256(root, packages=source_packages),
        "model_config": model.config.to_dict(),
        "training_config": expected_training_config.to_dict(),
        "corpus_sha256": expected_corpus_sha256,
    }
    for name, expected in expected_values.items():
        _exact(name, payload.get(name), expected)
    model_state = payload.get("model_state")
    ema_state = payload.get("ema_state")
    if not isinstance(model_state, dict) or not isinstance(ema_state, dict):
        raise ResumeContractError("Checkpoint model/EMA state is malformed.")
    if tensor_state_sha256(model_state) != payload.get("model_tensor_sha256"):
        raise ResumeContractError("Checkpoint model tensor hash does not match its payload.")
    shadow = ema_state.get("shadow")
    if not isinstance(shadow, dict) or tensor_state_sha256(shadow) != payload.get("ema_tensor_sha256"):
        raise ResumeContractError("Checkpoint EMA tensor hash does not match its payload.")
    current_model_state = model.state_dict()
    if set(model_state) != set(current_model_state):
        raise ResumeContractError("Checkpoint model tensor names do not exactly match the model.")
    for name, expected_tensor in current_model_state.items():
        observed_tensor = model_state[name]
        if (
            not isinstance(observed_tensor, torch.Tensor)
            or observed_tensor.shape != expected_tensor.shape
            or observed_tensor.dtype != expected_tensor.dtype
        ):
            raise ResumeContractError(f"Checkpoint model tensor contract drifted for {name!r}.")
    if set(shadow) != set(ema.shadow):
        raise ResumeContractError("Checkpoint EMA tensor names do not exactly match the model.")
    for name, expected_tensor in ema.shadow.items():
        observed_tensor = shadow[name]
        if not isinstance(observed_tensor, torch.Tensor) or observed_tensor.shape != expected_tensor.shape:
            raise ResumeContractError(f"Checkpoint EMA tensor contract drifted for {name!r}.")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ResumeContractError("Checkpoint optimizer state is malformed.")
    observed_groups = optimizer_state.get("param_groups")
    if not isinstance(observed_groups, list) or len(observed_groups) != len(optimizer.param_groups):
        raise ResumeContractError("Checkpoint optimizer parameter groups do not match.")
    for observed, expected in zip(observed_groups, optimizer.param_groups, strict=True):
        if not isinstance(observed, dict) or len(observed.get("params", ())) != len(expected["params"]):
            raise ResumeContractError("Checkpoint optimizer parameter membership does not match.")
    generator_state = payload.get("training_generator_state")
    if (
        not isinstance(generator_state, torch.Tensor)
        or generator_state.dtype != torch.uint8
        or generator_state.ndim != 1
    ):
        raise ResumeContractError("training_generator_state must be a one-dimensional uint8 tensor.")
    try:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        ema.load_state_dict(ema_state)
        training_generator.set_state(generator_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ResumeContractError(f"Checkpoint state could not be restored: {error}") from error
    epoch = payload.get("epoch")
    global_step = payload.get("global_step")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ResumeContractError("Checkpoint counters are invalid.")
    return {
        "epoch": epoch,
        "global_step": global_step,
        "metrics": payload.get("metrics", {}),
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "source_sha256": payload["source_sha256"],
        "corpus_sha256": payload["corpus_sha256"],
        "ema_tensor_sha256": payload["ema_tensor_sha256"],
    }
