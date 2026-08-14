from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..organism_raster_vae_v2.contract import sha256_file as vae_sha256_file
from ..organism_raster_vae_v2.dataset import OrganismRasterCorpusV2
from ..organism_raster_vae_v2.smoke import CHECKPOINT_NAME as VAE_CHECKPOINT_NAME
from ..organism_raster_vae_v2.smoke import MANIFEST_NAME as VAE_MANIFEST_NAME
from ..organism_raster_vae_v2.smoke import _load_checkpoint as load_vae_checkpoint
from .contract import CORPUS_FORMAT, VAE_CHECKPOINT_SHA256, VAE_MANIFEST_SHA256, VAE_OUTPUT, VAE_SOURCE_SHA256, canonical_json_bytes


TENSOR_KEYS = (
    "coarse_mean", "coarse_log_variance", "fine_mean", "fine_log_variance",
    "condition", "family", "subtype", "role", "genes", "style",
)


def _tensor_hash(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes({"dtype": str(value.dtype), "shape": list(value.shape)})
    return hashlib.sha256(header + value.numpy().tobytes(order="C")).hexdigest()


def build_latent_corpus() -> dict[str, Any]:
    checkpoint_path = VAE_OUTPUT / VAE_CHECKPOINT_NAME
    manifest_path = VAE_OUTPUT / VAE_MANIFEST_NAME
    encoded_manifest = manifest_path.read_bytes()
    vae_manifest = json.loads(encoded_manifest)
    if encoded_manifest != canonical_json_bytes(vae_manifest) or vae_sha256_file(checkpoint_path) != VAE_CHECKPOINT_SHA256 or vae_manifest.get("manifest_sha256") != VAE_MANIFEST_SHA256:
        raise ValueError("Frozen organism VAE v2 authority changed.")
    model, checkpoint = load_vae_checkpoint(checkpoint_path)
    if checkpoint["source_sha256"] != VAE_SOURCE_SHA256:
        raise ValueError("Frozen organism VAE v2 source changed.")
    model.eval()
    corpus = OrganismRasterCorpusV2()
    collected: dict[str, list[Tensor]] = {key: [] for key in TENSOR_KEYS}
    sample_ids: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(corpus), 5):
            rows = [corpus[index] for index in range(start, min(len(corpus), start + 5))]
            batch = default_collate(rows)
            condition = model.condition_vector(batch["family"], batch["subtype"], batch["role"], batch["genes"], batch["style"])
            coarse_mean, coarse_log_variance, fine_mean, fine_log_variance = model.encode(batch["living_field"], condition)
            values = {
                "coarse_mean": coarse_mean, "coarse_log_variance": coarse_log_variance,
                "fine_mean": fine_mean, "fine_log_variance": fine_log_variance,
                "condition": condition, "family": batch["family"], "subtype": batch["subtype"],
                "role": batch["role"], "genes": batch["genes"], "style": batch["style"],
            }
            for key, value in values.items():
                collected[key].append(value.detach().cpu().float() if value.dtype.is_floating_point else value.detach().cpu().long())
            sample_ids.extend(str(row["sample_id"]) for row in rows)
    tensors = {key: torch.cat(values, dim=0).contiguous() for key, values in collected.items()}
    coarse_center = tensors["coarse_mean"].mean((0, 2, 3), keepdim=True)
    coarse_scale = tensors["coarse_mean"].std((0, 2, 3), correction=0, keepdim=True).clamp_min(.08)
    fine_center = tensors["fine_mean"].mean((0, 2, 3), keepdim=True)
    fine_scale = tensors["fine_mean"].std((0, 2, 3), correction=0, keepdim=True).clamp_min(.08)
    tensors.update({"coarse_center": coarse_center, "coarse_scale": coarse_scale, "fine_center": fine_center, "fine_scale": fine_scale})
    tensor_hashes = {key: _tensor_hash(value) for key, value in tensors.items()}
    semantic = {
        "format": CORPUS_FORMAT,
        "vae_source_sha256": VAE_SOURCE_SHA256,
        "vae_manifest_sha256": VAE_MANIFEST_SHA256,
        "vae_checkpoint_sha256": VAE_CHECKPOINT_SHA256,
        "sample_ids": sample_ids,
        "tensor_hashes": tensor_hashes,
        "sample_count": len(sample_ids),
        "family_census": [int((tensors["family"] == family).sum()) for family in range(5)],
    }
    semantic["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
    return {"semantic": semantic, "tensors": tensors}


def save_latent_corpus(path: Path, corpus: dict[str, Any]) -> None:
    payload = {"format": CORPUS_FORMAT, "semantic": corpus["semantic"], "tensors": corpus["tensors"]}
    torch.save(payload, path)


def load_latent_corpus(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("Organism latent corpus missing or oversized.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"format", "semantic", "tensors"} or payload["format"] != CORPUS_FORMAT:
        raise ValueError("Organism latent corpus envelope drifted.")
    semantic, tensors = payload["semantic"], payload["tensors"]
    if not isinstance(semantic, dict) or not isinstance(tensors, dict):
        raise ValueError("Organism latent corpus payload drifted.")
    if set(tensors) != set(TENSOR_KEYS) | {"coarse_center", "coarse_scale", "fine_center", "fine_scale"}:
        raise ValueError("Organism latent corpus tensor inventory drifted.")
    stored = semantic.get("semantic_sha256")
    without = dict(semantic); without.pop("semantic_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(without)).hexdigest():
        raise ValueError("Organism latent corpus semantic hash failed.")
    if semantic.get("format") != CORPUS_FORMAT or semantic.get("vae_source_sha256") != VAE_SOURCE_SHA256 or semantic.get("vae_manifest_sha256") != VAE_MANIFEST_SHA256 or semantic.get("vae_checkpoint_sha256") != VAE_CHECKPOINT_SHA256:
        raise ValueError("Organism latent corpus authority drifted.")
    expected_shapes = {
        "coarse_mean": (45, 32, 12, 12), "coarse_log_variance": (45, 32, 12, 12),
        "fine_mean": (45, 16, 24, 24), "fine_log_variance": (45, 16, 24, 24),
        "condition": (45, 192), "family": (45,), "subtype": (45,), "role": (45,),
        "genes": (45, 16), "style": (45, 8), "coarse_center": (1, 32, 1, 1),
        "coarse_scale": (1, 32, 1, 1), "fine_center": (1, 16, 1, 1), "fine_scale": (1, 16, 1, 1),
    }
    for key, shape in expected_shapes.items():
        value = tensors.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape or not value.is_contiguous() or (value.dtype not in (torch.float32, torch.int64)) or (value.dtype.is_floating_point and not bool(torch.isfinite(value).all())):
            raise ValueError(f"Organism latent corpus tensor {key} drifted.")
        if semantic.get("tensor_hashes", {}).get(key) != _tensor_hash(value):
            raise ValueError(f"Organism latent corpus tensor {key} hash failed.")
    if semantic.get("sample_count") != 45 or semantic.get("family_census") != [11, 10, 9, 8, 7] or len(semantic.get("sample_ids", [])) != 45 or len(set(semantic["sample_ids"])) != 45:
        raise ValueError("Organism latent corpus census drifted.")
    return {"semantic": semantic, "tensors": tensors}
