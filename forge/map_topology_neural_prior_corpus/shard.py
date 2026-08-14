from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import time
from typing import Any, Final
import zipfile

import numpy as np
import torch

from ..map_topology_neural.artifacts import deterministic_npz_bytes
from ..map_topology_neural.corpus import FROZEN_CORPUS_MANIFEST_FILE_SHA256, FROZEN_CORPUS_SHA256
from ..map_topology_neural.hashing import named_arrays_sha256
from ..map_topology_neural_prior.dataset import FrozenLatentDataset
from .contract import (
    MAX_MANIFEST_BYTES,
    MAX_SHARD_BYTES,
    SHARD_FORMAT,
    authority,
    canonical_json_bytes,
    corpus_source_sha256,
    sha256_bytes,
    sha256_file,
    source_manifest,
)


ARRAY_NAME: Final[str] = "latents.npz"
MANIFEST_NAME: Final[str] = "manifest.json"
ARRAY_NAMES: Final[tuple[str, ...]] = (
    "tokens", "valid_mask", "point_conditions", "global_conditions",
    "theme_index", "sample_index",
)


def _canonical_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    if result.dtype.hasobject or not result.dtype.isnative:
        raise ValueError("Latent shard arrays must be native non-object dtypes.")
    return result


def _arrays_from_batch(batch: dict[str, torch.Tensor], refs: tuple[Any, ...]) -> dict[str, np.ndarray]:
    arrays = {
        "tokens": batch["targets"].numpy().astype(np.uint16, copy=False),
        "valid_mask": batch["valid_mask"][:, 0].numpy().astype(np.uint8, copy=False),
        "point_conditions": batch["point_conditions"].numpy().astype(np.uint8, copy=False),
        "global_conditions": batch["global_conditions"].numpy().astype(np.float32, copy=False),
        "theme_index": batch["theme_index"].numpy().astype(np.uint8, copy=False),
        "sample_index": np.asarray([ref.sample_index for ref in refs], dtype=np.uint16),
    }
    return {name: _canonical_array(arrays[name]) for name in ARRAY_NAMES}


def _array_contract(arrays: dict[str, np.ndarray], count: int) -> None:
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Latent shard array census drifted.")
    tokens = arrays["tokens"]
    if tokens.dtype != np.uint16 or tokens.ndim != 3 or tokens.shape[0] != count or not np.all(tokens < 512):
        raise ValueError("Latent shard token array is malformed.")
    shape = tokens.shape
    if arrays["valid_mask"].dtype != np.uint8 or arrays["valid_mask"].shape != shape or not np.all(arrays["valid_mask"] == 1):
        raise ValueError("Latent shard valid mask is malformed.")
    if arrays["point_conditions"].dtype != np.uint8 or arrays["point_conditions"].shape != (count, 4, shape[1], shape[2]):
        raise ValueError("Latent shard point conditioning is malformed.")
    if not np.all(arrays["point_conditions"] <= 1):
        raise ValueError("Latent shard point conditioning exceeds its binary domain.")
    if arrays["global_conditions"].dtype != np.float32 or arrays["global_conditions"].shape != (count, 14) or not np.isfinite(arrays["global_conditions"]).all():
        raise ValueError("Latent shard global conditioning is malformed.")
    if arrays["theme_index"].dtype != np.uint8 or arrays["theme_index"].shape != (count,) or not np.all(arrays["theme_index"] < 6):
        raise ValueError("Latent shard theme indices are malformed.")
    if arrays["sample_index"].dtype != np.uint16 or arrays["sample_index"].shape != (count,):
        raise ValueError("Latent shard sample indices are malformed.")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_SHARD_BYTES:
        raise ValueError("Latent shard array artifact is missing or oversized.")
    payload = path.read_bytes()
    expected_members = {f"{name}.npy" for name in ARRAY_NAMES}
    with zipfile.ZipFile(BytesIO(payload), "r") as archive:
        infos = archive.infolist()
        if len(infos) != len(expected_members) or {info.filename for info in infos} != expected_members:
            raise ValueError("Latent shard ZIP member census drifted.")
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError("Latent shard ZIP has duplicate members.")
        for info in infos:
            if info.filename.startswith(("/", "\\")) or ".." in Path(info.filename).parts:
                raise ValueError("Latent shard ZIP path is unsafe.")
            if info.file_size > MAX_SHARD_BYTES or info.compress_size > MAX_SHARD_BYTES:
                raise ValueError("Latent shard ZIP member exceeds its bound.")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: _canonical_array(archive[name]) for name in ARRAY_NAMES}
    if deterministic_npz_bytes(arrays) != payload:
        raise ValueError("Latent shard NPZ encoding is not canonical.")
    return arrays


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("Latent shard manifest is missing or oversized.")
    encoded = path.read_bytes()
    manifest = json.loads(encoded)
    if not isinstance(manifest, dict) or encoded != canonical_json_bytes(manifest):
        raise ValueError("Latent shard manifest is not canonical JSON.")
    stored = manifest.pop("manifest_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(manifest)):
        raise ValueError("Latent shard manifest self-hash failed.")
    manifest["manifest_sha256"] = stored
    return manifest


def shard_refs(dataset: FrozenLatentDataset, shard_id: str) -> tuple[Any, ...]:
    refs = tuple(sorted((ref for ref in dataset.dataset.refs if ref.shard_id == shard_id), key=lambda ref: ref.sample_index))
    if not refs or len({ref.shape for ref in refs}) != 1 or len({ref.theme for ref in refs}) != 1:
        raise ValueError("Latent source shard must be nonempty, homogeneous shape, and single-theme.")
    return refs


def _expected_manifest(
    shard_id: str,
    refs: tuple[Any, ...],
    arrays: dict[str, np.ndarray],
    artifact: bytes,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": SHARD_FORMAT,
        "status": "passed",
        "source_sha256": corpus_source_sha256(),
        "source_manifest": source_manifest(),
        "authority": authority(),
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
        "shard_id": shard_id,
        "sample_count": len(refs),
        "shape": list(arrays["tokens"].shape[1:]),
        "theme": refs[0].theme,
        "source_refs": [ref.identity_payload() for ref in refs],
        "arrays": {
            "file": ARRAY_NAME,
            "bytes": len(artifact),
            "sha256": sha256_bytes(artifact),
            "semantic_sha256": named_arrays_sha256(arrays),
            "members": {
                name: {"dtype": arrays[name].dtype.str, "shape": list(arrays[name].shape)}
                for name in ARRAY_NAMES
            },
        },
        "gates": {
            "codec_frozen": True,
            "source_refs_exact": True,
            "tokens_in_vocab": True,
            "conditions_exact": True,
            "runtime_integration_disabled": True,
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def build_shard(corpus_root: Path, destination: Path, shard_id: str) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Latent shard destination already exists.")
    dataset = FrozenLatentDataset(corpus_root)
    refs = shard_refs(dataset, shard_id)
    batch, _ = dataset.encode(refs)
    arrays = _arrays_from_batch(batch, refs)
    _array_contract(arrays, len(refs))
    artifact = deterministic_npz_bytes(arrays)
    if len(artifact) > MAX_SHARD_BYTES:
        raise ValueError("Latent shard artifact exceeds its byte bound.")
    manifest = _expected_manifest(shard_id, refs, arrays, artifact)
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / ARRAY_NAME).write_bytes(artifact)
        (staging / MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest))
        validate_shard(corpus_root, staging, replay_source=False)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()
    return manifest


def validate_shard(corpus_root: Path, destination: Path, *, replay_source: bool) -> dict[str, Any]:
    destination = Path(destination).resolve()
    manifest = _read_manifest(destination / MANIFEST_NAME)
    if manifest.get("format") != SHARD_FORMAT or manifest.get("status") != "passed":
        raise ValueError("Latent shard format/status failed.")
    if manifest.get("source_sha256") != corpus_source_sha256() or manifest.get("source_manifest") != source_manifest():
        raise ValueError("Latent shard source drifted.")
    if manifest.get("authority") != authority() or manifest.get("corpus_sha256") != FROZEN_CORPUS_SHA256 or manifest.get("corpus_manifest_file_sha256") != FROZEN_CORPUS_MANIFEST_FILE_SHA256:
        raise ValueError("Latent shard authority drifted.")
    if not isinstance(manifest.get("gates"), dict) or set(manifest["gates"]) != {"codec_frozen", "source_refs_exact", "tokens_in_vocab", "conditions_exact", "runtime_integration_disabled"} or not all(value is True for value in manifest["gates"].values()):
        raise ValueError("Latent shard gates drifted.")
    arrays_path = destination / manifest["arrays"]["file"]
    if manifest["arrays"]["bytes"] != arrays_path.stat().st_size or manifest["arrays"]["sha256"] != sha256_file(arrays_path):
        raise ValueError("Latent shard artifact identity failed.")
    arrays = _load_arrays(arrays_path)
    _array_contract(arrays, int(manifest["sample_count"]))
    if manifest["arrays"]["semantic_sha256"] != named_arrays_sha256(arrays):
        raise ValueError("Latent shard semantic hash failed.")
    if manifest["arrays"]["members"] != {
        name: {"dtype": arrays[name].dtype.str, "shape": list(arrays[name].shape)} for name in ARRAY_NAMES
    }:
        raise ValueError("Latent shard member metadata drifted.")
    if replay_source:
        dataset = FrozenLatentDataset(corpus_root)
        refs = shard_refs(dataset, str(manifest["shard_id"]))
        batch, _ = dataset.encode(refs)
        expected = _arrays_from_batch(batch, refs)
        if manifest["source_refs"] != [ref.identity_payload() for ref in refs] or any(not np.array_equal(arrays[name], expected[name]) for name in ARRAY_NAMES):
            raise ValueError("Latent shard exact source replay failed.")
        if manifest != _expected_manifest(str(manifest["shard_id"]), refs, expected, deterministic_npz_bytes(expected)):
            raise ValueError("Latent shard canonical manifest replay failed.")
    return manifest

