from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from ..whole_viewport_latent_v1.data import load_corpus, split_episodes


NUMERIC_FIELDS = (
    "spatial", "organisms", "organism_mask", "state", "actor_state", "actor_field",
    "visibility", "memory", "control", "action", "timeline", "timeline_event", "counterfactual",
)


def episode_pairs(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Causal rows: state[t] + command[t+1] -> complete state/frame[t+1]."""
    result = {f"current_{name}": episode[name][:-1] for name in NUMERIC_FIELDS if name not in ("action", "control")}
    result.update({f"target_{name}": episode[name][1:] for name in NUMERIC_FIELDS if name not in ("action", "control")})
    result["action"] = episode["action"][1:]
    result["control"] = episode["control"][1:]
    result["current_frame"] = episode["frame"][:-1]
    result["target_frame"] = episode["frame"][1:]
    return result


def concatenate_pairs(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    paired = [episode_pairs(episode) for episode in episodes]
    return {name: np.concatenate([episode[name] for episode in paired]) for name in paired[0]}


def sequence_starts(episodes: list[dict[str, np.ndarray]], length: int) -> np.ndarray:
    starts, cursor = [], 0
    for episode in episodes:
        count = len(episode["frame"]) - 1
        starts.extend(range(cursor, cursor + max(0, count - length + 1)))
        cursor += count
    return np.asarray(starts, dtype=np.int64)


@torch.inference_mode()
def attach_latents(data: dict[str, np.ndarray], decoder, device: torch.device, batch_size: int = 16) -> dict[str, np.ndarray]:
    result = dict(data)
    for prefix in ("current", "target"):
        encoded = []
        frames = data[f"{prefix}_frame"]
        for start in range(0, len(frames), batch_size):
            image = torch.as_tensor(frames[start:start + batch_size], device=device).permute(0, 3, 1, 2).float() / 255
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                encoded.append(decoder.encode(image)[0].half().cpu().numpy())
        result[f"{prefix}_latent"] = np.concatenate(encoded)
    return result


def corpus_identity(manifests: list[dict]) -> str:
    payload = json.dumps([item["manifest_sha256"] for item in manifests], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_aligned(corpus: Path, decoder, device: torch.device):
    episodes, manifests = load_corpus(corpus)
    train_episodes, validation_episodes, holdout = split_episodes(episodes)
    train = attach_latents(concatenate_pairs(train_episodes), decoder, device)
    validation = attach_latents(concatenate_pairs(validation_episodes), decoder, device)
    return train, validation, sequence_starts(train_episodes, 4), sequence_starts(validation_episodes, 4), manifests, holdout
