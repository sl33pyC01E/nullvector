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
from ..map_topology_neural_prior_v2.model import build_prior_v2
from ..safety import require_disk_floor
from .contract import (
    CHECKPOINT_FORMAT, FROZEN_AUTHORITY, PriorV2CalibrationConfig,
    canonical_json_bytes, sha256_file, source_manifest, training_v2_source_sha256,
)


MAX_BYTES: Final[int] = 512 * 1024 * 1024
PAYLOAD_KEYS: Final[set[str]] = {
    "format", "source_sha256", "source_manifest", "authority", "config", "step",
    "initial_model_sha256", "model_state", "ema_state", "optimizer_state",
    "mask_generator_state", "torch_cpu_rng_state", "torch_cuda_rng_states",
    "history", "evaluation", "free_generation", "model_state_sha256",
    "ema_state_sha256", "predecessor",
}


def _audit(value: Any) -> None:
    tensor_bytes = 0; objects = 0
    def visit(item: Any, depth: int) -> None:
        nonlocal tensor_bytes, objects
        objects += 1
        if depth > 24 or objects > 500_000: raise ValueError("Prior-v2 checkpoint structure is oversized.")
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.layout != torch.strided or item.ndim > 8: raise ValueError("Prior-v2 checkpoint tensor is unsafe.")
            tensor_bytes += item.numel() * item.element_size()
            if tensor_bytes > 420 * 1024 * 1024: raise ValueError("Prior-v2 checkpoint tensors are oversized.")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()): raise ValueError("Prior-v2 checkpoint has non-finite tensors.")
            return
        if isinstance(item, float) and not math.isfinite(item): raise ValueError("Prior-v2 checkpoint has non-finite scalars.")
        if item is None or isinstance(item, (bool, int, float, str)): return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, (str, int)) or isinstance(key, bool): raise ValueError("Prior-v2 checkpoint key is unsafe.")
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item: visit(child, depth + 1)
            return
        raise ValueError("Prior-v2 checkpoint container type is unsafe.")
    visit(value, 0)


def validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS or payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Prior-v2 checkpoint format/census drifted.")
    if payload["source_sha256"] != training_v2_source_sha256() or payload["source_manifest"] != source_manifest() or payload["authority"] != FROZEN_AUTHORITY:
        raise ValueError("Prior-v2 checkpoint provenance drifted.")
    config = PriorV2CalibrationConfig.from_dict(payload["config"])
    if type(payload["step"]) is not int or not 1 <= payload["step"] <= config.total_steps or len(payload["history"]) != payload["step"]:
        raise ValueError("Prior-v2 checkpoint step/history drifted.")
    if not isinstance(payload["predecessor"], (dict, type(None))) or payload["predecessor"] is not None and set(payload["predecessor"]) != {"checkpoint_sha256", "step"}:
        raise ValueError("Prior-v2 checkpoint predecessor drifted.")
    model = build_prior_v2(config.model_config())
    model.load_state_dict(payload["model_state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != payload["model_state_sha256"]: raise ValueError("Prior-v2 model-state hash failed.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != payload["ema_state_sha256"]: raise ValueError("Prior-v2 EMA-state hash failed.")
    for name in ("mask_generator_state", "torch_cpu_rng_state"):
        if not isinstance(payload[name], torch.Tensor) or payload[name].dtype != torch.uint8: raise ValueError("Prior-v2 RNG state is malformed.")
    if not isinstance(payload["torch_cuda_rng_states"], list) or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 for value in payload["torch_cuda_rng_states"]): raise ValueError("Prior-v2 CUDA RNG states are malformed.")
    if not isinstance(payload["optimizer_state"], dict) or not isinstance(payload["evaluation"], dict) or not isinstance(payload["free_generation"], dict): raise ValueError("Prior-v2 checkpoint training state is malformed.")
    _audit(payload); return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve(); sidecar_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or sidecar_path.exists(): raise FileExistsError("Prior-v2 checkpoint publication is immutable.")
    path.parent.mkdir(parents=True, exist_ok=True); require_disk_floor(path.parent, floor_gb=100, planned_bytes=MAX_BYTES)
    validate_payload(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); os.close(descriptor); temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        if not 0 < temporary.stat().st_size <= MAX_BYTES: raise ValueError("Prior-v2 checkpoint serialized outside bound.")
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)
    sidecar = {"format": "nullvector-neural-map-topology-prior-v2-checkpoint-sidecar/1.0.0", "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "source_sha256": training_v2_source_sha256(), "step": payload["step"], "model_state_sha256": payload["model_state_sha256"], "ema_state_sha256": payload["ema_state_sha256"]}
    sidecar["sidecar_sha256"] = hashlib.sha256(canonical_json_bytes(sidecar)).hexdigest(); sidecar_path.write_bytes(canonical_json_bytes(sidecar)); load_checkpoint(path); return sidecar


def load_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(); sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_BYTES: raise ValueError("Prior-v2 checkpoint is missing or oversized.")
    if not sidecar_path.is_file() or sidecar_path.is_symlink() or not 0 < sidecar_path.stat().st_size <= 1024 * 1024: raise ValueError("Prior-v2 checkpoint sidecar is missing or oversized.")
    sidecar = json.loads(sidecar_path.read_text("utf-8")); stored = sidecar.pop("sidecar_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(sidecar)).hexdigest(): raise ValueError("Prior-v2 checkpoint sidecar self-hash failed.")
    expected = {"format", "file", "bytes", "sha256", "source_sha256", "step", "model_state_sha256", "ema_state_sha256"}
    if set(sidecar) != expected or sidecar["file"] != path.name or sidecar["bytes"] != path.stat().st_size or sidecar["sha256"] != sha256_file(path) or sidecar["source_sha256"] != training_v2_source_sha256(): raise ValueError("Prior-v2 checkpoint sidecar identity failed.")
    payload = validate_payload(torch.load(path, map_location="cpu", weights_only=True))
    if sidecar["step"] != payload["step"] or sidecar["model_state_sha256"] != payload["model_state_sha256"] or sidecar["ema_state_sha256"] != payload["ema_state_sha256"]: raise ValueError("Prior-v2 checkpoint sidecar semantics drifted.")
    return payload
