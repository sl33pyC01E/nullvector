from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ..neural_city_layout_v1.teacher import _condition, render_teacher_city
from ..neural_world_state_v1.contract import CONDITION_NAMES, WorldStateModelConfig
from ..neural_world_state_v1.data import build_corpus
from ..neural_world_state_v1.model import build_model as build_codec
from ..world_action_cellular_v7.corpus import load_encoded_corpus
from .contract import CORPUS, WORLD_STATE


def _city_templates(episode: int, family: int) -> tuple[np.ndarray, ...]:
    base = _condition(0x434F4E54455854 + episode * 7919)
    return tuple(render_teacher_city(episode, replace(base, family=family, building_target=target))[::2, ::2] for target in range(1, 13))


@torch.inference_mode()
def build_aligned_context(*, device: torch.device) -> tuple[tuple[dict[str, np.ndarray], ...], str]:
    episodes, manifest = load_encoded_corpus(CORPUS); payload = torch.load(WORLD_STATE, map_location="cpu", weights_only=True); codec = build_codec(WorldStateModelConfig(**payload["model_config"])); codec.load_state_dict(payload["state"], strict=True); codec.to(device).eval(); base = build_corpus(128, seed=0x434F4E5445585442)
    aligned = []
    for episode_index, episode in enumerate(episodes):
        states = episode["state"].astype(np.float32); count = len(states); terrain = np.repeat(base.terrain[episode_index][None], count, 0); continuous = np.repeat(base.continuous[episode_index][None].astype(np.float32), count, 0); family = int(np.asarray(states[:, 1:6]).mean(0).argmax()); templates = _city_templates(episode_index, family); city = np.stack([templates[min(11, max(0, int(round(1 + row[0] * 11))))] for row in states]); condition = np.zeros((count, len(CONDITION_NAMES)), np.float32); condition[:, episode_index] = 1; condition[:, 6:11] = states[:, 1:6]
        season_index = states[:, 28:32].argmax(1); angle = season_index * (np.pi / 2); condition[:, -4] = np.sin(angle); condition[:, -3] = np.cos(angle); condition[:, -2] = states[:, 0]; condition[:, -1] = np.clip(states[:, 9] + states[:, 10], 0, 1)
        continuous[:, 3] *= (.2 + .8 * np.clip((states[:, 20] + states[:, 21]) * .5, 0, 1))[:, None, None]; continuous[:, 4] *= (.2 + .8 * np.clip(states[:, 14], 0, 1))[:, None, None]; continuous[:, 5] *= (.2 + .8 * np.clip(states[:, 12], 0, 1))[:, None, None]; continuous[:, 6] *= (.2 + .8 * np.clip(np.maximum.reduce((states[:, 15], states[:, 16], states[:, 18])), 0, 1))[:, None, None]
        rows = []
        for start in range(0, count, 64):
            end = start + 64; t = torch.from_numpy(terrain[start:end]).long().to(device); c = torch.from_numpy(city[start:end]).long().to(device); x = torch.from_numpy(continuous[start:end]).float().to(device); q = torch.from_numpy(condition[start:end]).float().to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"): spatial, global_state, _, _ = codec.encode(t, c, x, q, sample=False)
            rows.append(torch.cat((global_state.float(), spatial.float().mean((2, 3))), 1).cpu().numpy())
        aligned.append({"context": np.concatenate(rows).astype(np.float32), **episode})
    return tuple(aligned), manifest["manifest_sha256"]
