from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from ..world_frame_vae import WorldFrameVAERuntime
from .contract import CORPUS_FORMAT, canonical, corpus_source_sha256
from .data import encode_cellular_episodes

DISK_FLOOR = 100 * 1024**3
ARRAY_NAMES = (
    "previous",
    "current",
    "target",
    "previous_control",
    "control",
    "previous_action",
    "action",
    "state",
    "actor_state",
    "target_actor_state",
    "actor_field",
    "target_actor_field",
    "current_frame",
    "target_frame",
    "current_tick",
    "target_tick",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"nullvector-cellular-temporal-shard-v7\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0" + str(value.shape).encode() + b"\0")
        digest.update(memoryview(value))
    return digest.hexdigest()


def _validate_episode(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(episode) != set(ARRAY_NAMES):
        raise ValueError("encoded cellular shard members drifted")
    arrays = {name: np.ascontiguousarray(episode[name]) for name in ARRAY_NAMES}
    pairs = len(arrays["current"])
    if not 1 <= pairs <= 2000 or any(len(value) != pairs for value in arrays.values()):
        raise ValueError("encoded cellular shard pair count drifted")
    if arrays["current"].shape[1:] != (48, 32, 32) or arrays["previous"].shape != arrays["current"].shape or arrays["target"].shape != arrays["current"].shape:
        raise ValueError("encoded cellular latent shape drifted")
    if arrays["actor_state"].shape[1:] != (128,) or arrays["target_actor_state"].shape != arrays["actor_state"].shape:
        raise ValueError("encoded cellular actor state shape drifted")
    if arrays["actor_field"].shape[1:] != (8, 32, 32) or arrays["target_actor_field"].shape != arrays["actor_field"].shape:
        raise ValueError("encoded cellular actor field shape drifted")
    if arrays["current_frame"].shape[1:] != (256, 256, 3) or arrays["target_frame"].shape != arrays["current_frame"].shape:
        raise ValueError("encoded cellular frame shape drifted")
    if any(not np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.floating)):
        raise ValueError("encoded cellular shard contains nonfinite values")
    return arrays


def write_encoded_corpus(destination: Path, episodes, sources, *, vae_checkpoint_sha256: str, vae_ema_sha256: str) -> dict:
    destination = Path(destination)
    episodes = tuple(episodes)
    sources = tuple(sources)
    if destination.exists() or not episodes or len(episodes) != len(sources) or len(episodes) > 512:
        raise ValueError("encoded cellular corpus publication contract drifted")
    if len(vae_checkpoint_sha256) != 64 or len(vae_ema_sha256) != 64:
        raise ValueError("encoded cellular VAE provenance drifted")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < DISK_FLOOR:
        raise RuntimeError("encoded cellular corpus reached disk floor")
    staging = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    shard_root = staging / "shards"
    shard_root.mkdir(parents=True)
    records = []
    try:
        for index, (episode, source) in enumerate(zip(episodes, sources)):
            arrays = _validate_episode(episode)
            session = str(source.get("session_id", ""))
            if not session or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in session):
                raise ValueError("encoded cellular source identity drifted")
            path = shard_root / f"{index:04d}-{session}.npz"
            np.savez_compressed(path, **arrays)
            record = {
                "index": index,
                "session_id": session,
                "source_manifest_sha256": source["manifest_sha256"],
                "source_arrays_sha256": source["arrays_sha256"],
                "pairs": len(arrays["current"]),
                "semantic_sha256": _array_digest(arrays),
                "artifact": {"path": path.relative_to(staging).as_posix(), "bytes": path.stat().st_size, "sha256": _file_sha256(path)},
                "shapes": {name: list(value.shape) for name, value in arrays.items()},
                "dtypes": {name: str(value.dtype) for name, value in arrays.items()},
            }
            records.append(record)
        manifest = {
            "format": CORPUS_FORMAT,
            "corpus_source_sha256": corpus_source_sha256(),
            "vae_checkpoint_sha256": vae_checkpoint_sha256,
            "vae_ema_sha256": vae_ema_sha256,
            "worlds": len(records),
            "pairs": sum(record["pairs"] for record in records),
            "shards": records,
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_bytes(canonical(manifest))
        os.replace(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_encoded_corpus(root: Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    provided = manifest.pop("manifest_sha256", None)
    if manifest.get("format") != CORPUS_FORMAT or manifest.get("corpus_source_sha256") != corpus_source_sha256():
        raise ValueError("encoded cellular corpus provenance drifted")
    if provided != hashlib.sha256(canonical(manifest)).hexdigest():
        raise ValueError("encoded cellular corpus manifest drifted")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or manifest.get("worlds") != len(shards) or not 1 <= len(shards) <= 512:
        raise ValueError("encoded cellular corpus inventory drifted")
    pairs = 0
    seen = set()
    for index, record in enumerate(shards):
        if record.get("index") != index or record.get("session_id") in seen:
            raise ValueError("encoded cellular shard identity drifted")
        seen.add(record["session_id"])
        relative = record.get("artifact", {}).get("path", "")
        expected = f"shards/{index:04d}-{record['session_id']}.npz"
        if relative != expected:
            raise ValueError("encoded cellular shard path drifted")
        path = root / Path(relative)
        if not path.is_file() or path.stat().st_size != record["artifact"]["bytes"] or _file_sha256(path) != record["artifact"]["sha256"]:
            raise ValueError("encoded cellular shard artifact drifted")
        with np.load(path, allow_pickle=False) as archive:
            arrays = _validate_episode({name: archive[name] for name in archive.files})
        if _array_digest(arrays) != record["semantic_sha256"]:
            raise ValueError("encoded cellular shard semantic replay drifted")
        if any(list(value.shape) != record["shapes"].get(name) or str(value.dtype) != record["dtypes"].get(name) for name, value in arrays.items()):
            raise ValueError("encoded cellular shard tensor contract drifted")
        pairs += len(arrays["current"])
    if pairs != manifest.get("pairs"):
        raise ValueError("encoded cellular corpus pair total drifted")
    manifest["manifest_sha256"] = provided
    return manifest


def load_encoded_corpus(root: Path):
    manifest = validate_encoded_corpus(root)
    episodes = []
    for record in manifest["shards"]:
        with np.load(Path(root) / record["artifact"]["path"], allow_pickle=False) as archive:
            episodes.append({name: archive[name].copy() for name in ARRAY_NAMES})
    return tuple(episodes), manifest


def build_encoded_corpus(destination: Path, trajectories, vae_checkpoint: Path, *, device="cuda") -> dict:
    vae_checkpoint = Path(vae_checkpoint)
    payload = torch.load(vae_checkpoint, map_location="cpu", weights_only=True)
    ema_sha256 = payload.get("ema_sha256")
    if not isinstance(ema_sha256, str) or len(ema_sha256) != 64:
        raise ValueError("world VAE checkpoint lacks EMA provenance")
    runtime = WorldFrameVAERuntime.from_checkpoint(vae_checkpoint, device=device)
    episodes, sources, _ = encode_cellular_episodes(trajectories, runtime)
    return write_encoded_corpus(destination, episodes, sources, vae_checkpoint_sha256=_file_sha256(vae_checkpoint), vae_ema_sha256=ema_sha256)
