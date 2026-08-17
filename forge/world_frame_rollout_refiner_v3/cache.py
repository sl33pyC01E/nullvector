from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from ..safety import require_disk_floor
from ..world_action_natural_v10 import load
from ..world_frame_rollout_decoder_v2.contract import NATURAL_CORPUS
from ..world_frame_rollout_decoder_v2.corpus import load_corpus
from ..world_frame_vae.contract import ModelConfig
from ..world_frame_vae.model import WorldFrameVAE
from .contract import CACHE_FORMAT, DECODER, DECODER_SHA256, DEFAULT_CACHE, ROLLOUT_CORPUS, ROLLOUT_CORPUS_SHA256, canonical, file_sha256, source_sha256


def build_cache(output: Path = DEFAULT_CACHE):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if file_sha256(DECODER) != DECODER_SHA256:
        raise ValueError("rollout refiner decoder drifted")
    rollout, manifest = load_corpus(ROLLOUT_CORPUS)
    if manifest["manifest_sha256"] != ROLLOUT_CORPUS_SHA256:
        raise ValueError("rollout refiner corpus drifted")
    sequences, natural = load(NATURAL_CORPUS)
    release = torch.load(DECODER, map_location="cpu", weights_only=True)
    device = torch.device("cuda:0")
    decoder = WorldFrameVAE(ModelConfig(**release["model_config"]))
    decoder.load_state_dict(release["state"])
    decoder.to(device).eval()
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    shards = staging / "shards"
    shards.mkdir(parents=True)
    records = []
    try:
        with torch.inference_mode():
            for world, row in enumerate(rollout):
                decoded = []
                for start in range(0, len(row["candidate"]), 8):
                    latent = torch.from_numpy(row["candidate"][start:start + 8]).float().to(device)
                    decoded.append(decoder.decode(latent).float().cpu())
                base = np.clip(torch.cat(decoded).permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
                target = sequences[world]["frame"][row["target"]]
                artifact = shards / f"world-{world}.npz"
                np.savez_compressed(artifact, base=base, target=target)
                records.append({"world": world, "rows": len(base), "path": artifact.relative_to(staging).as_posix(), "bytes": artifact.stat().st_size, "sha256": file_sha256(artifact)})
        payload = {"format": CACHE_FORMAT, "source_sha256": source_sha256(), "decoder_sha256": DECODER_SHA256, "rollout_corpus_sha256": ROLLOUT_CORPUS_SHA256, "natural_corpus_sha256": natural["manifest_sha256"], "worlds": 6, "rows": sum(record["rows"] for record in records), "records": records}
        payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
        (staging / "manifest.json").write_bytes(canonical(payload))
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_cache(output)


def validate_cache(root: Path = DEFAULT_CACHE):
    root = Path(root).resolve()
    raw = (root / "manifest.json").read_bytes()
    payload = json.loads(raw)
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if raw != canonical(payload) or payload.get("format") != CACHE_FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("decoder_sha256") != DECODER_SHA256 or payload.get("rollout_corpus_sha256") != ROLLOUT_CORPUS_SHA256 or payload.get("manifest_sha256") != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ValueError("rollout refiner cache manifest drifted")
    total = 0
    for world, record in enumerate(payload.get("records", ())):
        artifact = root / record["path"]
        if record.get("world") != world or artifact.stat().st_size != record["bytes"] or file_sha256(artifact) != record["sha256"]:
            raise ValueError("rollout refiner cache artifact drifted")
        with np.load(artifact, allow_pickle=False) as archive:
            if archive.files != ["base", "target"] or archive["base"].shape != (record["rows"], 256, 256, 3) or archive["target"].shape != archive["base"].shape or archive["base"].dtype != np.uint8 or archive["target"].dtype != np.uint8:
                raise ValueError("rollout refiner cache tensors drifted")
        total += record["rows"]
    if total != payload.get("rows") or len(payload.get("records", ())) != 6:
        raise ValueError("rollout refiner cache inventory drifted")
    return {"passed": True, "worlds": 6, "rows": total, "manifest_sha256": payload["manifest_sha256"]}


def load_cache(root: Path = DEFAULT_CACHE):
    validate_cache(root)
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    rows = []
    for record in manifest["records"]:
        with np.load(root / record["path"], allow_pickle=False) as archive:
            rows.append({name: archive[name].copy() for name in archive.files})
    return tuple(rows), manifest
