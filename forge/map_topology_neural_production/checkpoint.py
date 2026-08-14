from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from ..map_topology_neural.codec import build_codec
from ..map_topology_neural.contract import CONTRACT_SHA256
from ..map_topology_neural.corpus import FROZEN_CORPUS_MANIFEST_FILE_SHA256, FROZEN_CORPUS_SHA256
from ..safety import require_disk_floor
from .contract import (
    CHECKPOINT_FORMAT,
    TopologyCodecCalibrationConfig,
    canonical_json_bytes,
    production_source_manifest,
    production_source_sha256,
    sha256_file,
)


MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_TENSOR_BYTES = 192 * 1024 * 1024
PAYLOAD_KEYS = {
    "format", "source_sha256", "source_manifest", "tensor_contract_sha256",
    "corpus_sha256", "corpus_manifest_file_sha256", "dataset_registry_sha256",
    "config", "step", "model_state", "ema_state", "optimizer_state",
    "training_generator_state", "torch_cpu_rng_state", "torch_cuda_rng_states",
    "history", "evaluation", "model_state_sha256", "ema_state_sha256",
}


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(memoryview(value.numpy()))
    return digest.hexdigest()


def _audit(value: Any) -> int:
    total = 0
    objects = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal total, objects
        objects += 1
        if depth > 20 or objects > 200_000:
            raise ValueError("Topology production checkpoint container exceeds its structural bound.")
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.layout != torch.strided or item.ndim > 8:
                raise ValueError("Topology production checkpoint tensor is unsafe.")
            total += item.numel() * item.element_size()
            if total > MAX_TENSOR_BYTES:
                raise ValueError("Topology production checkpoint tensors exceed their byte bound.")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()):
                raise ValueError("Topology production checkpoint contains non-finite tensors.")
            return
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Topology production checkpoint contains non-finite scalars.")
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, (str, int)) or isinstance(key, bool):
                    raise ValueError("Topology production checkpoint key type is unsafe.")
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        raise ValueError(f"Topology production checkpoint contains unsafe {type(item).__name__}.")

    visit(value, 0)
    return total


def validate_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError("Topology production checkpoint key census drifted.")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Topology production checkpoint format drifted.")
    if payload["source_sha256"] != production_source_sha256() or payload["source_manifest"] != production_source_manifest():
        raise ValueError("Topology production checkpoint source drifted.")
    if payload["tensor_contract_sha256"] != CONTRACT_SHA256:
        raise ValueError("Topology production checkpoint tensor contract drifted.")
    if payload["corpus_sha256"] != FROZEN_CORPUS_SHA256 or payload["corpus_manifest_file_sha256"] != FROZEN_CORPUS_MANIFEST_FILE_SHA256:
        raise ValueError("Topology production checkpoint corpus authority drifted.")
    config = TopologyCodecCalibrationConfig.from_dict(payload["config"])
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step != config.steps:
        raise ValueError("Topology production checkpoint step count drifted.")
    if not isinstance(payload["history"], list) or len(payload["history"]) != step:
        raise ValueError("Topology production checkpoint history is not exact.")
    if not isinstance(payload["dataset_registry_sha256"], str) or len(payload["dataset_registry_sha256"]) != 64:
        raise ValueError("Topology production checkpoint dataset identity is malformed.")
    model = build_codec(config.codec_config(), init_seed=config.seed)
    model.load_state_dict(payload["model_state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("Topology production checkpoint model state hash failed.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != payload["ema_state_sha256"]:
        raise ValueError("Topology production checkpoint EMA state hash failed.")
    if not isinstance(payload["optimizer_state"], dict) or not isinstance(payload["evaluation"], dict):
        raise ValueError("Topology production checkpoint optimizer/evaluation payload is malformed.")
    if not isinstance(payload["training_generator_state"], torch.Tensor) or payload["training_generator_state"].dtype != torch.uint8:
        raise ValueError("Topology production checkpoint generator state is malformed.")
    if not isinstance(payload["torch_cpu_rng_state"], torch.Tensor) or payload["torch_cpu_rng_state"].dtype != torch.uint8:
        raise ValueError("Topology production checkpoint CPU RNG state is malformed.")
    cuda_states = payload["torch_cuda_rng_states"]
    if not isinstance(cuda_states, list) or len(cuda_states) > 16 or any(not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 for item in cuda_states):
        raise ValueError("Topology production checkpoint CUDA RNG states are malformed.")
    _audit(payload)
    return payload


def load_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Topology production checkpoint is missing or oversized.")
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not sidecar_path.is_file() or sidecar_path.is_symlink() or not 0 < sidecar_path.stat().st_size <= 1024 * 1024:
        raise ValueError("Topology production checkpoint sidecar is missing or oversized.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        raise ValueError("Topology production checkpoint sidecar is malformed.")
    stored = sidecar.pop("sidecar_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(sidecar)).hexdigest():
        raise ValueError("Topology production checkpoint sidecar self-hash failed.")
    expected_sidecar_keys = {
        "format", "file", "bytes", "sha256", "source_sha256", "step",
        "model_state_sha256", "ema_state_sha256",
    }
    if set(sidecar) != expected_sidecar_keys:
        raise ValueError("Topology production checkpoint sidecar key census drifted.")
    if (
        sidecar["file"] != path.name
        or sidecar["bytes"] != path.stat().st_size
        or sidecar["sha256"] != sha256_file(path)
        or sidecar["source_sha256"] != production_source_sha256()
    ):
        raise ValueError("Topology production checkpoint sidecar artifact identity failed.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validated = validate_checkpoint(payload)
    if (
        sidecar["step"] != validated["step"]
        or sidecar["model_state_sha256"] != validated["model_state_sha256"]
        or sidecar["ema_state_sha256"] != validated["ema_state_sha256"]
    ):
        raise ValueError("Topology production checkpoint sidecar semantics drifted.")
    return validated


def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = Path(path).resolve()
    if target.exists() or target.with_suffix(target.suffix + ".json").exists():
        raise FileExistsError("Topology production checkpoint publication is immutable.")
    target.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(target.parent, floor_gb=100.0, planned_bytes=MAX_CHECKPOINT_BYTES)
    validated = validate_checkpoint(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(validated, temporary)
        if not 0 < temporary.stat().st_size <= MAX_CHECKPOINT_BYTES:
            raise ValueError("Topology production checkpoint serialized outside its byte bound.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = {
        "format": "nullvector-neural-map-topology-codec-production-checkpoint-sidecar/1.0.0",
        "file": target.name,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "source_sha256": production_source_sha256(),
        "step": validated["step"],
        "model_state_sha256": validated["model_state_sha256"],
        "ema_state_sha256": validated["ema_state_sha256"],
    }
    sidecar["sidecar_sha256"] = hashlib.sha256(canonical_json_bytes(sidecar)).hexdigest()
    target.with_suffix(target.suffix + ".json").write_bytes(canonical_json_bytes(sidecar))
    load_checkpoint(target)
    return sidecar
