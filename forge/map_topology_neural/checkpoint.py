from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any

import jsonschema
import torch
from torch import Tensor

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .codec import CODEC_NAME, CODEC_VERSION, CodecConfig, CategoricalTopologyCodec, build_codec
from .contract import CONTRACT_SHA256
from .hashing import file_sha256, json_sha256, require_sha256
from .provenance import source_sha256


CHECKPOINT_FORMAT = "nullvector-neural-topology-codec-checkpoint-v1"
SIDECAR_FORMAT = "nullvector-neural-topology-codec-checkpoint-sidecar-v1"
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_SIDECAR_BYTES = 1024 * 1024
MAX_TENSOR_BYTES = 48 * 1024 * 1024
MAX_TENSOR_COUNT = 4096
MAX_CONTAINER_OBJECTS = 100_000

PAYLOAD_KEYS = {
    "format",
    "authority",
    "codec_name",
    "codec_version",
    "model_config",
    "model_init_seed",
    "step",
    "model_state",
    "ema_state",
    "optimizer_state",
    "training_generator_state",
    "torch_cpu_rng_state",
    "source_sha256",
    "corpus_sha256",
    "tensor_contract_sha256",
    "metrics",
}
SIDECAR_KEYS = {
    "format",
    "file",
    "bytes",
    "sha256",
    "payload_identity_sha256",
    "model_state_sha256",
    "ema_state_sha256",
    "tensor_count",
    "tensor_elements",
    "tensor_bytes",
    "source_sha256",
    "corpus_sha256",
    "tensor_contract_sha256",
    "model_config_sha256",
    "model_init_seed",
    "step",
    "device",
}


def _state_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _audit_safe_payload(value: object) -> dict[str, int]:
    counts = {"objects": 0, "tensors": 0, "elements": 0, "bytes": 0}

    def visit(item: object, depth: int) -> None:
        counts["objects"] += 1
        if depth > 16 or counts["objects"] > MAX_CONTAINER_OBJECTS:
            raise ValueError("Checkpoint container exceeds its structural safety bound.")
        if isinstance(item, Tensor):
            if item.device.type != "cpu" or item.layout != torch.strided or item.ndim > 8:
                raise ValueError("Checkpoint tensor uses an unsafe device/layout/rank.")
            counts["tensors"] += 1
            counts["elements"] += item.numel()
            counts["bytes"] += item.numel() * item.element_size()
            if counts["tensors"] > MAX_TENSOR_COUNT or counts["bytes"] > MAX_TENSOR_BYTES:
                raise ValueError("Checkpoint tensors exceed their strict count/byte bound.")
            return
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Checkpoint contains a non-finite scalar.")
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, (str, int)) or isinstance(key, bool):
                    raise ValueError("Checkpoint dictionary key type is unsafe.")
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        raise ValueError(f"Checkpoint contains unsafe object type {type(item).__name__}.")

    visit(value, 0)
    return {key: counts[key] for key in ("tensors", "elements", "bytes")}


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_codec_checkpoint(
    path: Path,
    *,
    model: CategoricalTopologyCodec,
    model_init_seed: int,
    step: int,
    optimizer_state: dict[str, object],
    ema_state: dict[str, Tensor],
    training_generator_state: Tensor,
    torch_cpu_rng_state: Tensor,
    corpus_sha256: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    path = Path(path).resolve()
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        raise FileExistsError("Neural topology checkpoint publication is immutable.")
    require_sha256(corpus_sha256, "corpus_sha256")
    if step < 1 or step > 2:
        raise ValueError("Foundation checkpoint step must be one or two.")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=MAX_CHECKPOINT_BYTES + MAX_SIDECAR_BYTES)
    source_hash = source_sha256()
    payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "authority": "representation_only_not_generative",
        "codec_name": CODEC_NAME,
        "codec_version": CODEC_VERSION,
        "model_config": model.config.to_dict(),
        "model_init_seed": int(model_init_seed),
        "step": int(step),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "ema_state": {name: value.detach().cpu() for name, value in ema_state.items()},
        "optimizer_state": optimizer_state,
        "training_generator_state": training_generator_state.detach().cpu(),
        "torch_cpu_rng_state": torch_cpu_rng_state.detach().cpu(),
        "source_sha256": source_hash,
        "corpus_sha256": corpus_sha256,
        "tensor_contract_sha256": CONTRACT_SHA256,
        "metrics": metrics,
    }
    audit = _audit_safe_payload(payload)
    payload_identity = {
        "format": payload["format"],
        "authority": payload["authority"],
        "codec_name": payload["codec_name"],
        "codec_version": payload["codec_version"],
        "model_config": payload["model_config"],
        "model_init_seed": payload["model_init_seed"],
        "step": payload["step"],
        "source_sha256": payload["source_sha256"],
        "corpus_sha256": payload["corpus_sha256"],
        "tensor_contract_sha256": payload["tensor_contract_sha256"],
        "metrics": payload["metrics"],
    }
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(payload, temporary)
        if not 0 < temporary.stat().st_size <= MAX_CHECKPOINT_BYTES:
            raise ValueError("Serialized checkpoint exceeds its strict size bound.")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    sidecar: dict[str, object] = {
        "format": SIDECAR_FORMAT,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "payload_identity_sha256": json_sha256(payload_identity),
        "model_state_sha256": _state_sha256(payload["model_state"]),  # type: ignore[arg-type]
        "ema_state_sha256": _state_sha256(payload["ema_state"]),  # type: ignore[arg-type]
        "tensor_count": audit["tensors"],
        "tensor_elements": audit["elements"],
        "tensor_bytes": audit["bytes"],
        "source_sha256": source_hash,
        "corpus_sha256": corpus_sha256,
        "tensor_contract_sha256": CONTRACT_SHA256,
        "model_config_sha256": json_sha256(model.config.to_dict()),
        "model_init_seed": int(model_init_seed),
        "step": int(step),
        "device": "cpu",
    }
    _atomic_json(path.with_suffix(path.suffix + ".json"), sidecar)
    return sidecar


def load_codec_checkpoint(
    path: Path,
    *,
    expected_corpus_sha256: str,
    expected_source_sha256: str | None = None,
) -> tuple[CategoricalTopologyCodec, dict[str, object], dict[str, object]]:
    path = Path(path).resolve()
    sidecar_path = path.with_suffix(path.suffix + ".json")
    require_sha256(expected_corpus_sha256, "expected_corpus_sha256")
    expected_source = expected_source_sha256 or source_sha256()
    require_sha256(expected_source, "expected_source_sha256")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint is missing, empty, or exceeds its strict byte bound.")
    if not sidecar_path.is_file() or not 0 < sidecar_path.stat().st_size <= MAX_SIDECAR_BYTES:
        raise ValueError("Checkpoint sidecar is missing, empty, or exceeds its strict byte bound.")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint sidecar is malformed JSON.") from error
    if not isinstance(sidecar, dict) or set(sidecar) != SIDECAR_KEYS:
        raise ValueError("Checkpoint sidecar members are incomplete or unexpected.")
    schema = json.loads(
        (PROJECT_ROOT / "shared" / "schema" / "map_topology_neural_checkpoint.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(sidecar)
    if (
        sidecar.get("format") != SIDECAR_FORMAT
        or sidecar.get("file") != path.name
        or sidecar.get("bytes") != path.stat().st_size
        or sidecar.get("sha256") != file_sha256(path)
        or sidecar.get("device") != "cpu"
    ):
        raise ValueError("Checkpoint sidecar does not match its bounded artifact.")
    if sidecar.get("source_sha256") != expected_source or sidecar.get("corpus_sha256") != expected_corpus_sha256:
        raise ValueError("Checkpoint sidecar source/corpus provenance drifted.")
    if sidecar.get("tensor_contract_sha256") != CONTRACT_SHA256:
        raise ValueError("Checkpoint tensor contract provenance drifted.")
    try:
        payload = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except Exception as error:
        raise ValueError(f"Checkpoint safe load failed: {error}") from error
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError("Checkpoint payload members are incomplete or unexpected.")
    audit = _audit_safe_payload(payload)
    if (
        audit["tensors"] != sidecar.get("tensor_count")
        or audit["elements"] != sidecar.get("tensor_elements")
        or audit["bytes"] != sidecar.get("tensor_bytes")
    ):
        raise ValueError("Checkpoint bounded tensor census drifted from sidecar.")
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("authority") != "representation_only_not_generative"
        or payload.get("codec_name") != CODEC_NAME
        or payload.get("codec_version") != CODEC_VERSION
        or payload.get("source_sha256") != expected_source
        or payload.get("corpus_sha256") != expected_corpus_sha256
        or payload.get("tensor_contract_sha256") != CONTRACT_SHA256
    ):
        raise ValueError("Checkpoint payload provenance/authority drifted.")
    model_config_payload = payload.get("model_config")
    if not isinstance(model_config_payload, dict):
        raise ValueError("Checkpoint model config is malformed.")
    config = CodecConfig.from_dict(model_config_payload)
    if json_sha256(config.to_dict()) != sidecar.get("model_config_sha256"):
        raise ValueError("Checkpoint model config hash drifted.")
    model_state = payload.get("model_state")
    ema_state = payload.get("ema_state")
    if not isinstance(model_state, dict) or not isinstance(ema_state, dict):
        raise ValueError("Checkpoint model/EMA state is malformed.")
    if _state_sha256(model_state) != sidecar.get("model_state_sha256") or _state_sha256(ema_state) != sidecar.get("ema_state_sha256"):
        raise ValueError("Checkpoint model/EMA tensor identity drifted.")
    init_seed = payload.get("model_init_seed")
    if not isinstance(init_seed, int) or isinstance(init_seed, bool) or init_seed != sidecar.get("model_init_seed"):
        raise ValueError("Checkpoint model initialization seed drifted.")
    if payload.get("step") != sidecar.get("step"):
        raise ValueError("Checkpoint step drifted.")
    payload_identity = {
        "format": payload["format"],
        "authority": payload["authority"],
        "codec_name": payload["codec_name"],
        "codec_version": payload["codec_version"],
        "model_config": payload["model_config"],
        "model_init_seed": payload["model_init_seed"],
        "step": payload["step"],
        "source_sha256": payload["source_sha256"],
        "corpus_sha256": payload["corpus_sha256"],
        "tensor_contract_sha256": payload["tensor_contract_sha256"],
        "metrics": payload["metrics"],
    }
    if json_sha256(payload_identity) != sidecar.get("payload_identity_sha256"):
        raise ValueError("Checkpoint non-tensor payload identity drifted.")
    model = build_codec(config, init_seed=init_seed)
    try:
        model.load_state_dict(model_state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Checkpoint model state is incompatible: {error}") from error
    model.eval()
    return model, payload, sidecar
