from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import torch

from ..map_topology_neural.corpus import FROZEN_CORPUS_MANIFEST_FILE_SHA256, FROZEN_CORPUS_SHA256
from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import (
    CHECKPOINT_FORMAT,
    FROZEN_CODEC_CHECKPOINT_SHA256,
    FROZEN_CODEC_EMA_SHA256,
    FROZEN_CODEC_SOURCE_SHA256,
    MaskedPriorConfig,
    canonical_json_bytes,
    prior_source_sha256,
    sha256_file,
    source_manifest,
)
from .model import build_prior


MAX_CHECKPOINT_BYTES: Final[int] = 64 * 1024 * 1024
MAX_TENSOR_BYTES: Final[int] = 48 * 1024 * 1024
PAYLOAD_KEYS: Final[set[str]] = {
    "format", "source_sha256", "source_manifest", "codec_checkpoint_sha256",
    "codec_source_sha256", "codec_ema_sha256", "corpus_sha256",
    "corpus_manifest_file_sha256", "config", "step", "model_state",
    "ema_state", "optimizer_state", "generator_state", "torch_rng_state",
    "history", "model_state_sha256", "ema_state_sha256",
}


def _audit(value: Any) -> None:
    total = 0
    objects = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal total, objects
        objects += 1
        if depth > 18 or objects > 100_000:
            raise ValueError("Masked-prior checkpoint container exceeds its structural bound.")
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.layout != torch.strided or item.ndim > 8:
                raise ValueError("Masked-prior checkpoint tensor is unsafe.")
            total += item.numel() * item.element_size()
            if total > MAX_TENSOR_BYTES:
                raise ValueError("Masked-prior checkpoint tensor budget exceeded.")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()):
                raise ValueError("Masked-prior checkpoint contains non-finite tensors.")
            return
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Masked-prior checkpoint contains non-finite scalars.")
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, (str, int)) or isinstance(key, bool):
                    raise ValueError("Masked-prior checkpoint key type is unsafe.")
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        raise ValueError(f"Masked-prior checkpoint contains unsafe {type(item).__name__}.")

    visit(value, 0)


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError("Masked-prior checkpoint key census drifted.")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Masked-prior checkpoint format drifted.")
    if payload["source_sha256"] != prior_source_sha256() or payload["source_manifest"] != source_manifest():
        raise ValueError("Masked-prior checkpoint source drifted.")
    if (
        payload["codec_checkpoint_sha256"] != FROZEN_CODEC_CHECKPOINT_SHA256
        or payload["codec_source_sha256"] != FROZEN_CODEC_SOURCE_SHA256
        or payload["codec_ema_sha256"] != FROZEN_CODEC_EMA_SHA256
    ):
        raise ValueError("Masked-prior checkpoint codec authority drifted.")
    if (
        payload["corpus_sha256"] != FROZEN_CORPUS_SHA256
        or payload["corpus_manifest_file_sha256"] != FROZEN_CORPUS_MANIFEST_FILE_SHA256
    ):
        raise ValueError("Masked-prior checkpoint corpus authority drifted.")
    config = MaskedPriorConfig.from_dict(payload["config"])
    if type(payload["step"]) is not int or payload["step"] != config.steps:
        raise ValueError("Masked-prior checkpoint step drifted.")
    if not isinstance(payload["history"], list) or len(payload["history"]) != config.steps:
        raise ValueError("Masked-prior checkpoint history is incomplete.")
    model = build_prior(config)
    model.load_state_dict(payload["model_state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("Masked-prior checkpoint model hash failed.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != payload["ema_state_sha256"]:
        raise ValueError("Masked-prior checkpoint EMA hash failed.")
    if not isinstance(payload["optimizer_state"], dict):
        raise ValueError("Masked-prior optimizer state is malformed.")
    for name in ("generator_state", "torch_rng_state"):
        value = payload[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.device.type != "cpu":
            raise ValueError(f"Masked-prior {name} is malformed.")
    _audit(payload)
    return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or sidecar_path.exists():
        raise FileExistsError("Masked-prior checkpoint publication is immutable.")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=MAX_CHECKPOINT_BYTES)
    validated = validate_payload(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(validated, temporary)
        if not 0 < temporary.stat().st_size <= MAX_CHECKPOINT_BYTES:
            raise ValueError("Masked-prior checkpoint serialized outside its byte bound.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = {
        "format": "nullvector-neural-map-topology-masked-prior-checkpoint-sidecar/1.0.0",
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_sha256": prior_source_sha256(),
        "codec_checkpoint_sha256": FROZEN_CODEC_CHECKPOINT_SHA256,
        "step": validated["step"],
        "model_state_sha256": validated["model_state_sha256"],
        "ema_state_sha256": validated["ema_state_sha256"],
    }
    sidecar["sidecar_sha256"] = hashlib.sha256(canonical_json_bytes(sidecar)).hexdigest()
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))
    load_checkpoint(path)
    return sidecar


def load_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Masked-prior checkpoint is missing or oversized.")
    if not sidecar_path.is_file() or sidecar_path.is_symlink() or not 0 < sidecar_path.stat().st_size <= 1024 * 1024:
        raise ValueError("Masked-prior checkpoint sidecar is missing or oversized.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    stored = sidecar.pop("sidecar_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(sidecar)).hexdigest():
        raise ValueError("Masked-prior checkpoint sidecar self-hash failed.")
    expected = {
        "format", "file", "bytes", "sha256", "source_sha256",
        "codec_checkpoint_sha256", "step", "model_state_sha256", "ema_state_sha256",
    }
    if set(sidecar) != expected:
        raise ValueError("Masked-prior checkpoint sidecar census drifted.")
    if (
        sidecar["file"] != path.name
        or sidecar["bytes"] != path.stat().st_size
        or sidecar["sha256"] != sha256_file(path)
        or sidecar["source_sha256"] != prior_source_sha256()
        or sidecar["codec_checkpoint_sha256"] != FROZEN_CODEC_CHECKPOINT_SHA256
    ):
        raise ValueError("Masked-prior checkpoint sidecar identity failed.")
    payload = validate_payload(torch.load(path, map_location="cpu", weights_only=True))
    if (
        sidecar["step"] != payload["step"]
        or sidecar["model_state_sha256"] != payload["model_state_sha256"]
        or sidecar["ema_state_sha256"] != payload["ema_state_sha256"]
    ):
        raise ValueError("Masked-prior checkpoint sidecar semantics drifted.")
    return payload
