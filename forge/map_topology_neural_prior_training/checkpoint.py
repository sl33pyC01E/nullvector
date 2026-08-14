from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import torch

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import (
    CHECKPOINT_FORMAT,
    FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
    FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256,
    PriorCalibrationConfig,
    canonical_json_bytes,
    sha256_file,
    source_manifest,
    training_source_sha256,
    validate_evaluation,
    validate_history,
)
from ..map_topology_neural_prior.model import build_prior


MAX_BYTES: Final[int] = 128 * 1024 * 1024
PAYLOAD_KEYS: Final[set[str]] = {
    "format", "source_sha256", "source_manifest", "latent_corpus_manifest_file_sha256",
    "latent_corpus_identity_sha256", "config", "step", "model_state", "ema_state",
    "optimizer_state", "generator_state", "torch_cpu_rng_state", "torch_cuda_rng_states",
    "history", "evaluation", "model_state_sha256", "ema_state_sha256",
}


def _audit(value: Any) -> None:
    total = 0
    objects = 0
    def visit(item: Any, depth: int) -> None:
        nonlocal total, objects
        objects += 1
        if depth > 20 or objects > 200_000:
            raise ValueError("Prior training checkpoint structure is oversized.")
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.layout != torch.strided or item.ndim > 8:
                raise ValueError("Prior training checkpoint tensor is unsafe.")
            total += item.numel() * item.element_size()
            if total > 96 * 1024 * 1024:
                raise ValueError("Prior training checkpoint tensors are oversized.")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()):
                raise ValueError("Prior training checkpoint contains non-finite tensors.")
            return
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Prior training checkpoint contains non-finite scalars.")
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, (str, int)) or isinstance(key, bool):
                    raise ValueError("Prior training checkpoint key type is unsafe.")
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        raise ValueError("Prior training checkpoint container type is unsafe.")
    visit(value, 0)


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS or payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Prior training checkpoint format/census drifted.")
    if payload["source_sha256"] != training_source_sha256() or payload["source_manifest"] != source_manifest():
        raise ValueError("Prior training checkpoint source drifted.")
    if payload["latent_corpus_manifest_file_sha256"] != FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256 or payload["latent_corpus_identity_sha256"] != FROZEN_LATENT_CORPUS_IDENTITY_SHA256:
        raise ValueError("Prior training checkpoint latent corpus drifted.")
    config = PriorCalibrationConfig.from_dict(payload["config"])
    if type(payload["step"]) is not int or payload["step"] != config.steps:
        raise ValueError("Prior training checkpoint step/history drifted.")
    validate_history(payload["history"], config)
    validate_evaluation(payload["evaluation"], config)
    model = build_prior(config.model_config())
    model.load_state_dict(payload["model_state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("Prior training model hash failed.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != payload["ema_state_sha256"]:
        raise ValueError("Prior training EMA hash failed.")
    if not isinstance(payload["optimizer_state"], dict):
        raise ValueError("Prior training optimizer/evaluation is malformed.")
    for key in ("generator_state", "torch_cpu_rng_state"):
        if not isinstance(payload[key], torch.Tensor) or payload[key].dtype != torch.uint8:
            raise ValueError("Prior training RNG state is malformed.")
    if not isinstance(payload["torch_cuda_rng_states"], list) or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 for value in payload["torch_cuda_rng_states"]):
        raise ValueError("Prior training CUDA RNG state is malformed.")
    _audit(payload)
    return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve(); sidecar_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or sidecar_path.exists():
        raise FileExistsError("Prior training checkpoint publication is immutable.")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=MAX_BYTES)
    payload = validate_payload(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        if not 0 < temporary.stat().st_size <= MAX_BYTES:
            raise ValueError("Prior training checkpoint serialized outside its bound.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = {
        "format": "nullvector-neural-map-topology-prior-production-checkpoint-sidecar/1.0.0",
        "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "source_sha256": training_source_sha256(), "step": payload["step"],
        "model_state_sha256": payload["model_state_sha256"], "ema_state_sha256": payload["ema_state_sha256"],
    }
    sidecar["sidecar_sha256"] = hashlib.sha256(canonical_json_bytes(sidecar)).hexdigest()
    sidecar_path.write_bytes(canonical_json_bytes(sidecar)); load_checkpoint(path)
    return sidecar


def load_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(); sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Prior training checkpoint is missing or oversized.")
    if not sidecar_path.is_file() or sidecar_path.is_symlink() or not 0 < sidecar_path.stat().st_size <= 1024 * 1024:
        raise ValueError("Prior training checkpoint sidecar is missing or oversized.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")); stored = sidecar.pop("sidecar_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(sidecar)).hexdigest():
        raise ValueError("Prior training checkpoint sidecar self-hash failed.")
    expected = {"format", "file", "bytes", "sha256", "source_sha256", "step", "model_state_sha256", "ema_state_sha256"}
    if set(sidecar) != expected or sidecar["file"] != path.name or sidecar["bytes"] != path.stat().st_size or sidecar["sha256"] != sha256_file(path) or sidecar["source_sha256"] != training_source_sha256():
        raise ValueError("Prior training checkpoint sidecar identity failed.")
    payload = validate_payload(torch.load(path, map_location="cpu", weights_only=True))
    if sidecar["step"] != payload["step"] or sidecar["model_state_sha256"] != payload["model_state_sha256"] or sidecar["ema_state_sha256"] != payload["ema_state_sha256"]:
        raise ValueError("Prior training checkpoint sidecar semantics drifted.")
    return payload
