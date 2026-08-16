from __future__ import annotations

import hashlib

import numpy as np

from ..world_action_step_v3.data import encode_episodes


def causal_action_and_settle_mask(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.ndim != 1 or len(actions) < 2:
        raise ValueError("spatial action sequence must be one-dimensional")
    active = actions != 0
    keep = active.copy()
    keep[1:] |= active[:-1]
    return keep


def encode_spatial_episodes(paths, vae_runtime):
    episodes, sources, upstream_sha = encode_episodes(paths, vae_runtime)
    filtered = []
    filtered_sources = []
    digest = hashlib.sha256(b"nullvector-spatial-action-corpus-v4\0" + upstream_sha.encode() + b"\0")
    for episode, source in zip(episodes, sources):
        keep = causal_action_and_settle_mask(episode["action"])
        item = {name: value[keep] for name, value in episode.items()}
        filtered.append(item)
        record = {**source, "source_pairs": int(len(keep)), "pairs": int(keep.sum()), "filter": "command-or-post-command-settle-v1"}
        filtered_sources.append(record)
        digest.update(record["manifest_sha256"].encode() + b"\0" + np.packbits(keep).tobytes() + b"\0")
        for name in ("current", "target", "control", "action", "state", "current_tick", "target_tick"):
            value = np.ascontiguousarray(item[name])
            digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0" + str(value.shape).encode() + b"\0")
            digest.update(memoryview(value))
    return tuple(filtered), tuple(filtered_sources), digest.hexdigest()
