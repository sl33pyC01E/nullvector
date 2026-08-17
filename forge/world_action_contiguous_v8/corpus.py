from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from ..action_teacher_v2 import validate_trajectory
from ..safety import require_disk_floor
from ..world_frame_vae import WorldFrameVAERuntime
from .contract import DEFAULT_OUTPUT, FORMAT, SOURCE_NAMES, SOURCE_ROOT, VAE_CHECKPOINT, VAE_CHECKPOINT_SHA256, canonical, file_sha256, source_sha256

RAW_NAMES = ("frame", "state", "actor_state", "actor_field", "control", "action", "tick")


def _encode(runtime, frames, batch=8):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch):
            value = torch.from_numpy(frames[start:start + batch]).permute(0, 3, 1, 2).float().div_(255).to(runtime.device)
            rows.append(runtime.model.encode(value)[0].float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(rows))


def build(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024**3)
    if file_sha256(VAE_CHECKPOINT) != VAE_CHECKPOINT_SHA256:
        raise ValueError("contiguous corpus encoder drifted")
    runtime = WorldFrameVAERuntime.from_checkpoint(VAE_CHECKPOINT, device="cuda")
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    shard_root = staging / "shards"
    shard_root.mkdir(parents=True)
    records = []
    try:
        for index, name in enumerate(SOURCE_NAMES):
            root = SOURCE_ROOT / name
            source = validate_trajectory(root)
            with np.load(root / source["artifact"]["path"], allow_pickle=False) as archive:
                raw = {member: archive[member].copy() for member in RAW_NAMES}
            if len(raw["frame"]) < 32 or not np.all(np.diff(raw["tick"]) >= 0):
                raise ValueError("teacher sequence is not contiguous")
            latent = _encode(runtime, raw["frame"])
            path = shard_root / f"{index:04d}-{name}.npz"
            np.savez_compressed(path, latent=latent)
            records.append({
                "index": index,
                "session_id": name,
                "frames": len(latent),
                "source": {
                    "path": root.relative_to(SOURCE_ROOT.parent).as_posix(),
                    "manifest_sha256": source["manifest_sha256"],
                    "arrays_sha256": source["arrays_sha256"],
                },
                "latent": {"path": path.relative_to(staging).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path), "shape": list(latent.shape), "dtype": str(latent.dtype)},
            })
        payload = {"format": FORMAT, "source_sha256": source_sha256(), "vae_checkpoint_sha256": VAE_CHECKPOINT_SHA256, "worlds": len(records), "frames": sum(row["frames"] for row in records), "records": records}
        payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
        (staging / "manifest.json").write_bytes(canonical(payload))
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate(output)


def validate(output=DEFAULT_OUTPUT):
    root = Path(output).resolve()
    raw = (root / "manifest.json").read_bytes()
    payload = json.loads(raw)
    if raw != canonical(payload) or payload.get("format") != FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("vae_checkpoint_sha256") != VAE_CHECKPOINT_SHA256:
        raise ValueError("contiguous action corpus manifest drifted")
    expected = hashlib.sha256(canonical({key: value for key, value in payload.items() if key != "manifest_sha256"})).hexdigest()
    if payload.get("manifest_sha256") != expected or payload.get("worlds") != len(SOURCE_NAMES) or len(payload.get("records", ())) != len(SOURCE_NAMES):
        raise ValueError("contiguous action corpus inventory drifted")
    total = 0
    for index, record in enumerate(payload["records"]):
        if record.get("index") != index or record.get("session_id") != SOURCE_NAMES[index]:
            raise ValueError("contiguous action corpus order drifted")
        source_root = SOURCE_ROOT.parent / record["source"]["path"]
        source = validate_trajectory(source_root)
        if source["manifest_sha256"] != record["source"]["manifest_sha256"] or source["arrays_sha256"] != record["source"]["arrays_sha256"]:
            raise ValueError("contiguous action source drifted")
        artifact = root / record["latent"]["path"]
        if artifact.stat().st_size != record["latent"]["bytes"] or file_sha256(artifact) != record["latent"]["sha256"]:
            raise ValueError("contiguous latent artifact drifted")
        with np.load(artifact, allow_pickle=False) as archive:
            if archive.files != ["latent"]:
                raise ValueError("contiguous latent member drifted")
            latent = archive["latent"]
        if list(latent.shape) != record["latent"]["shape"] or str(latent.dtype) != record["latent"]["dtype"] or latent.shape != (record["frames"], 48, 32, 32) or not np.isfinite(latent).all():
            raise ValueError("contiguous latent tensor drifted")
        total += len(latent)
    if total != payload.get("frames"):
        raise ValueError("contiguous action frame total drifted")
    return {"passed": True, "worlds": payload["worlds"], "frames": total, "manifest_sha256": payload["manifest_sha256"]}


def load(output=DEFAULT_OUTPUT):
    validate(output)
    root = Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    sequences = []
    for record in manifest["records"]:
        with np.load(root / record["latent"]["path"], allow_pickle=False) as archive:
            latent = archive["latent"].copy()
        source_root = SOURCE_ROOT.parent / record["source"]["path"]
        source = validate_trajectory(source_root)
        with np.load(source_root / source["artifact"]["path"], allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in RAW_NAMES}
        values["latent"] = latent
        sequences.append(values)
    return tuple(sequences), manifest
