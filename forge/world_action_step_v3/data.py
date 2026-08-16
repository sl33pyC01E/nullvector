from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ..action_teacher_v1 import validate_trajectory


def align_causal_step(latent: np.ndarray, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pair pre-command state with the next command and its post-command frame."""
    if len(latent) != len(raw["frame"]) or len(latent) < 2:
        raise ValueError("causal action episode cannot form a transition")
    return {
        "current": latent[:-1],
        "target": latent[1:],
        "control": raw["control"][1:],
        "action": raw["action"][1:],
        "state": raw["state"][:-1],
        "current_frame": raw["frame"][:-1],
        "target_frame": raw["frame"][1:],
        "current_tick": raw["tick"][:-1],
        "target_tick": raw["tick"][1:],
    }


def encode_episodes(paths, vae_runtime):
    episodes = []
    sources = []
    for path in map(Path, paths):
        manifest = validate_trajectory(path)
        with np.load(path / manifest["artifact"]["path"], allow_pickle=False) as archive:
            raw = {name: archive[name].copy() for name in ("frame", "control", "action", "state", "tick")}
        tensors = []
        with torch.inference_mode():
            for start in range(0, len(raw["frame"]), 8):
                frame = torch.from_numpy(raw["frame"][start : start + 8]).permute(0, 3, 1, 2).float().div_(255).to(vae_runtime.device)
                mean, _ = vae_runtime.model.encode(frame)
                tensors.append(mean.float().cpu().numpy())
        episode = align_causal_step(np.concatenate(tensors), raw)
        episodes.append(episode)
        sources.append(
            {
                "session_id": manifest["session_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "arrays_sha256": manifest["arrays_sha256"],
                "pairs": len(episode["current"]),
            }
        )
    digest = hashlib.sha256(b"nullvector-causal-action-step-corpus-v3\0")
    for source, episode in zip(sources, episodes):
        digest.update(source["manifest_sha256"].encode() + b"\0" + source["arrays_sha256"].encode() + b"\0")
        for name in ("current", "target", "control", "action", "state", "current_tick", "target_tick"):
            value = np.ascontiguousarray(episode[name])
            digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0" + str(value.shape).encode() + b"\0")
            digest.update(memoryview(value))
    return tuple(episodes), tuple(sources), digest.hexdigest()
