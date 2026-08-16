from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ..action_teacher_v2 import validate_trajectory
from .contract import CORPUS_FORMAT


def temporal_action_and_settle_mask(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.ndim != 1:
        raise ValueError("cellular temporal action sequence must be one-dimensional")
    active = actions != 0
    keep = active.copy()
    keep[1:] |= active[:-1]
    return keep


def align_temporal_cellular(latent: np.ndarray, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Align two visual observations and current cellular truth to the next state.

    The action/control stored at target frame ``t+1`` caused the transition from
    current frame ``t``.  The hidden actor truth therefore comes from ``t`` and
    its supervised next value comes from ``t+1``.  Frame ``t-1`` supplies motion
    context so velocity and appendage phase are observable.
    """
    required = {"frame", "state", "actor_state", "actor_field", "control", "action", "tick"}
    if set(raw) < required:
        raise ValueError("cellular temporal episode members drifted")
    count = len(latent)
    if count < 3 or any(len(raw[name]) != count for name in required):
        raise ValueError("cellular temporal episode cannot form a three-frame transition")
    aligned = {
        "previous": latent[:-2],
        "current": latent[1:-1],
        "target": latent[2:],
        "previous_control": raw["control"][1:-1],
        "control": raw["control"][2:],
        "previous_action": raw["action"][1:-1],
        "action": raw["action"][2:],
        "state": raw["state"][1:-1],
        "actor_state": raw["actor_state"][1:-1],
        "target_actor_state": raw["actor_state"][2:],
        "actor_field": raw["actor_field"][1:-1],
        "target_actor_field": raw["actor_field"][2:],
        "current_frame": raw["frame"][1:-1],
        "target_frame": raw["frame"][2:],
        "current_tick": raw["tick"][1:-1],
        "target_tick": raw["tick"][2:],
    }
    keep = temporal_action_and_settle_mask(aligned["action"])
    return {name: np.ascontiguousarray(value[keep]) for name, value in aligned.items()}


def _encode_frames(frames: np.ndarray, vae_runtime) -> np.ndarray:
    rows = []
    with torch.inference_mode():
        for start in range(0, len(frames), 8):
            value = torch.from_numpy(frames[start : start + 8]).permute(0, 3, 1, 2).float().div_(255).to(vae_runtime.device)
            mean, _ = vae_runtime.model.encode(value)
            rows.append(mean.float().cpu().numpy())
    return np.concatenate(rows)


def encode_cellular_episodes(paths, vae_runtime):
    episodes = []
    sources = []
    digest = hashlib.sha256((CORPUS_FORMAT + "\0").encode())
    for path in map(Path, paths):
        manifest = validate_trajectory(path)
        with np.load(path / manifest["artifact"]["path"], allow_pickle=False) as archive:
            names = ("frame", "state", "actor_state", "actor_field", "control", "action", "tick")
            raw = {name: archive[name].copy() for name in names}
        episode = align_temporal_cellular(_encode_frames(raw["frame"], vae_runtime), raw)
        source = {
            "session_id": manifest["session_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "arrays_sha256": manifest["arrays_sha256"],
            "source_frames": manifest["frames"],
            "pairs": len(episode["current"]),
        }
        episodes.append(episode)
        sources.append(source)
        digest.update(source["manifest_sha256"].encode() + b"\0" + source["arrays_sha256"].encode() + b"\0")
        for name, value in sorted(episode.items()):
            digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0" + str(value.shape).encode() + b"\0")
            digest.update(memoryview(value))
    if not episodes:
        raise ValueError("cellular temporal corpus is empty")
    return tuple(episodes), tuple(sources), digest.hexdigest()
