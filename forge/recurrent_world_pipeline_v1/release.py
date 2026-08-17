from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..world_action_natural_v10 import load
from .contract import DECODER, DECODER_SHA256, DEFAULT_RELEASE, FORMAT, NATURAL_CORPUS, NATURAL_CORPUS_SHA256, RECURRENT, RECURRENT_SHA256, canonical, file_sha256, source_sha256
from .runtime import RecurrentWorldPipeline


def build_release(output: Path = DEFAULT_RELEASE):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sequences, manifest = load(NATURAL_CORPUS)
    if manifest["manifest_sha256"] != NATURAL_CORPUS_SHA256:
        raise ValueError("pipeline benchmark corpus drifted")
    runtime = RecurrentWorldPipeline.load("cuda")
    sequence = sequences[5]
    start, steps = 64, 64
    runtime.initialize(sequence["latent"][start - 1], sequence["latent"][start], sequence["actor_state"][start - 1], sequence["actor_state"][start])
    for offset in range(4):
        index = start + offset
        runtime.step(sequence["action"][index + 1:index + 2], sequence["control"][index + 1:index + 2], sequence["state"][index:index + 1], sequence["visibility"][index:index + 1], sequence["memory"][index:index + 1])
    runtime.initialize(sequence["latent"][start - 1], sequence["latent"][start], sequence["actor_state"][start - 1], sequence["actor_state"][start])
    torch.cuda.reset_peak_memory_stats(runtime.device)
    torch.cuda.synchronize(runtime.device)
    began = time.perf_counter()
    checksum = hashlib.sha256()
    for offset in range(steps):
        index = start + offset
        frame = runtime.step(sequence["action"][index + 1:index + 2], sequence["control"][index + 1:index + 2], sequence["state"][index:index + 1], sequence["visibility"][index:index + 1], sequence["memory"][index:index + 1])
        checksum.update(np.ascontiguousarray(frame).tobytes())
    torch.cuda.synchronize(runtime.device)
    elapsed = time.perf_counter() - began
    recurrent_bytes, decoder_bytes = Path(RECURRENT).stat().st_size, Path(DECODER).stat().st_size
    payload = {"format": FORMAT, "status": "ready", "source_sha256": source_sha256(), "components": {"recurrent": {"path": str(RECURRENT), "sha256": RECURRENT_SHA256, "bytes": recurrent_bytes}, "decoder": {"path": str(DECODER), "sha256": DECODER_SHA256, "bytes": decoder_bytes}}, "natural_corpus_sha256": NATURAL_CORPUS_SHA256, "parameters": runtime.parameter_count, "artifact_bytes": recurrent_bytes + decoder_bytes, "benchmark": {"device": str(runtime.device), "steps": steps, "seconds": elapsed, "frames_per_second": steps / elapsed, "milliseconds_per_frame": elapsed * 1000 / steps, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime.device)), "output_sha256": checksum.hexdigest()}, "capabilities": {"action_conditioned_recurrent_state": True, "continuous_vae_raster": True, "actor_state_recurrence": True, "visibility_and_memory_conditioning": True, "foreground_aware_rollout_decode": True}, "limitations": ["Long rollouts remain softer and less state-accurate than teacher frames.", "This pipeline predicts the learned world view; deterministic physics remains authoritative in the playable scaffold."]}
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    output.write_bytes(canonical(payload))
    return validate_release(output)


def validate_release(path: Path = DEFAULT_RELEASE):
    path = Path(path).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw)
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if raw != canonical(payload) or payload.get("format") != FORMAT or payload.get("status") != "ready" or payload.get("source_sha256") != source_sha256() or payload.get("manifest_sha256") != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ValueError("recurrent world pipeline release drifted")
    if file_sha256(RECURRENT) != RECURRENT_SHA256 or file_sha256(DECODER) != DECODER_SHA256 or payload["components"]["recurrent"]["sha256"] != RECURRENT_SHA256 or payload["components"]["decoder"]["sha256"] != DECODER_SHA256:
        raise ValueError("recurrent world pipeline components drifted")
    if payload["benchmark"]["peak_reserved_bytes"] >= 12 * 1024**3 or payload["benchmark"]["frames_per_second"] <= 0:
        raise ValueError("recurrent world pipeline benchmark failed")
    return payload
