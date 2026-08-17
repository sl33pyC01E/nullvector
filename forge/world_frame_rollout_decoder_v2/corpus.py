from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..recurrent_world_student_v6.training import _normalizers
from ..safety import require_disk_floor
from ..world_action_natural_v10 import load
from ..world_latent_dit.contract import ModelConfig
from .contract import CORPUS_FORMAT, DEFAULT_CORPUS, HORIZONS, NATURAL_CORPUS, NATURAL_CORPUS_SHA256, RECURRENT, RECURRENT_SHA256, canonical, file_sha256, source_sha256


@torch.inference_mode()
def _roll(model, sequence, starts, horizon, norms, device, bias, ramp):
    lm, ls, am, ass = norms
    previous = torch.from_numpy(sequence["latent"][starts - 1]).to(device)
    current = torch.from_numpy(sequence["latent"][starts]).to(device)
    previous_actor = torch.from_numpy(sequence["actor_state"][starts - 1]).to(device)
    actor = torch.from_numpy(sequence["actor_state"][starts]).to(device)
    for offset in range(horizon):
        indices = starts + offset
        action = torch.from_numpy(sequence["action"][indices + 1].astype(np.int64)).to(device)
        control = torch.from_numpy(sequence["control"][indices + 1]).to(device)
        state = torch.from_numpy(sequence["state"][indices]).to(device)
        visibility = torch.from_numpy(sequence["visibility"][indices]).to(device)
        memory = torch.from_numpy(sequence["memory"][indices]).to(device)
        cn, pn = (current - lm) / ls, (previous - lm) / ls
        delta, logits = model.gated_action(cn, pn, action, control, state, actor, visibility, memory)
        applied = bias * min(offset / ramp, 1.0) if ramp else bias
        next_latent = (cn + torch.sigmoid(logits + applied) * delta) * ls + lm
        an, pan = (actor - am) / ass, (previous_actor - am) / ass
        result = model.actor(an, pan, action, control, state, visibility, memory)
        next_actor = (an + 0.9 * (result.gate >= 0.7) * (result.state - an)) * ass + am
        previous, current = current, next_latent
        previous_actor, actor = actor, next_actor
    return current.float().cpu().numpy()


def build_corpus(output: Path = DEFAULT_CORPUS):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024**3)
    if file_sha256(RECURRENT) != RECURRENT_SHA256:
        raise ValueError("rollout decoder recurrent parent drifted")
    sequences, natural = load(NATURAL_CORPUS)
    if natural["manifest_sha256"] != NATURAL_CORPUS_SHA256:
        raise ValueError("rollout decoder natural corpus drifted")
    payload = torch.load(RECURRENT, map_location="cpu", weights_only=True)
    device = torch.device("cuda:0")
    model = PerceptionRecurrentWorldStudent(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state"])
    model.to(device).eval()
    norms = _normalizers(payload, device)
    bias = float(payload["inference"]["gate_logit_bias_max"])
    ramp = int(payload["inference"]["gate_logit_bias_ramp_steps"])
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    shard_root = staging / "shards"
    shard_root.mkdir(parents=True)
    records = []
    try:
        for world, sequence in enumerate(sequences):
            starts = np.arange(1, len(sequence["latent"]) - max(HORIZONS) - 1, dtype=np.int32)
            horizons = np.asarray([HORIZONS[index % len(HORIZONS)] for index in range(len(starts))], dtype=np.int16)
            candidate = np.empty((len(starts), 48, 32, 32), dtype=np.float16)
            for horizon in HORIZONS:
                positions = np.flatnonzero(horizons == horizon)
                for begin in range(0, len(positions), 32):
                    chosen = positions[begin:begin + 32]
                    candidate[chosen] = _roll(model, sequence, starts[chosen], horizon, norms, device, bias, ramp).astype(np.float16)
            targets = starts + horizons.astype(np.int32)
            artifact = shard_root / f"world-{world}.npz"
            np.savez_compressed(artifact, candidate=candidate, start=starts, horizon=horizons, target=targets)
            records.append({"world": world, "rows": len(starts), "path": artifact.relative_to(staging).as_posix(), "bytes": artifact.stat().st_size, "sha256": file_sha256(artifact)})
        manifest = {"format": CORPUS_FORMAT, "source_sha256": source_sha256(), "natural_corpus_sha256": NATURAL_CORPUS_SHA256, "recurrent_sha256": RECURRENT_SHA256, "horizons": list(HORIZONS), "worlds": len(records), "rows": sum(row["rows"] for row in records), "records": records}
        manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_bytes(canonical(manifest))
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_corpus(output)


def validate_corpus(root: Path = DEFAULT_CORPUS):
    root = Path(root).resolve()
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if raw != canonical(manifest) or manifest.get("format") != CORPUS_FORMAT or manifest.get("source_sha256") != source_sha256() or manifest.get("natural_corpus_sha256") != NATURAL_CORPUS_SHA256 or manifest.get("recurrent_sha256") != RECURRENT_SHA256 or manifest.get("manifest_sha256") != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ValueError("rollout decoder corpus manifest drifted")
    total = 0
    for world, record in enumerate(manifest.get("records", ())):
        artifact = root / record["path"]
        if record.get("world") != world or artifact.stat().st_size != record["bytes"] or file_sha256(artifact) != record["sha256"]:
            raise ValueError("rollout decoder corpus artifact drifted")
        with np.load(artifact, allow_pickle=False) as archive:
            if archive.files != ["candidate", "start", "horizon", "target"]:
                raise ValueError("rollout decoder corpus members drifted")
            candidate, start, horizon, target = (archive[name] for name in archive.files)
            if candidate.shape != (record["rows"], 48, 32, 32) or candidate.dtype != np.float16 or start.dtype != np.int32 or horizon.dtype != np.int16 or target.dtype != np.int32 or not np.isfinite(candidate).all() or not np.array_equal(target, start + horizon):
                raise ValueError("rollout decoder corpus tensors drifted")
        total += record["rows"]
    if total != manifest.get("rows") or len(manifest.get("records", ())) != 6:
        raise ValueError("rollout decoder corpus inventory drifted")
    return {"passed": True, "rows": total, "worlds": 6, "manifest_sha256": manifest["manifest_sha256"]}


def load_corpus(root: Path = DEFAULT_CORPUS):
    validate_corpus(root)
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    rows = []
    for record in manifest["records"]:
        with np.load(root / record["path"], allow_pickle=False) as archive:
            rows.append({name: archive[name].copy() for name in archive.files})
    return tuple(rows), manifest
