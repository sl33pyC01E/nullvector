from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT
from ..neural_city_layout_v1.teacher import _condition, render_teacher_city
from .contract import CONDITION_NAMES, CONTINUOUS_NAMES, GRID_SIZE


THEMES = ("arena", "rooms", "caves", "archipelago", "garden", "anomaly")
MAP_ROOT = PROJECT_ROOT / "outputs/neural_world_synthesis_v1/build_004/topology_maps"


@dataclass(frozen=True, slots=True)
class WorldStateCorpus:
    terrain: np.ndarray
    city: np.ndarray
    continuous: np.ndarray
    condition: np.ndarray
    sha256: str


def _maps() -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for theme in THEMES:
        roots = tuple(MAP_ROOT.glob(f"{theme}-*/arrays"))
        if len(roots) != 1: raise ValueError(f"World-state map authority drifted: {theme}")
        root = roots[0]; result[theme] = {name: np.load(root / f"{name}.npy", allow_pickle=False) for name in ("terrain", "elevation", "walkability", "nav_cost")}
    return result


def _smooth_noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); small = rng.random((8, 8), dtype=np.float32)
    expanded = np.repeat(np.repeat(small, 4, 0), 4, 1)
    return (expanded + np.roll(expanded, 2, 0) + np.roll(expanded, 2, 1) + np.roll(expanded, (2, 2), (0, 1))) * .25


def build_corpus(count: int, *, seed: int) -> WorldStateCorpus:
    if not 128 <= count <= 100_000: raise ValueError("World-state corpus size is outside contract.")
    maps = _maps(); terrain = np.empty((count, GRID_SIZE, GRID_SIZE), np.uint8); city = np.empty_like(terrain); continuous = np.empty((count, len(CONTINUOUS_NAMES), GRID_SIZE, GRID_SIZE), np.float16); condition = np.zeros((count, len(CONDITION_NAMES)), np.float32)
    for index in range(count):
        item_seed = int((seed + index * 0x9E3779B97F4A7C15) & ((1 << 63) - 1)); theme_index = index % len(THEMES); theme = THEMES[theme_index]; base = maps[theme]; rng = np.random.default_rng(item_seed); shift = (int(rng.integers(GRID_SIZE)), int(rng.integers(GRID_SIZE)))
        terrain[index] = np.roll(base["terrain"], shift, (0, 1)); elevation = np.roll(base["elevation"].astype(np.float32) / 5, shift, (0, 1)); walkability = np.roll(base["walkability"].astype(np.float32), shift, (0, 1)); nav = np.clip(np.roll(base["nav_cost"].astype(np.float32) / 1.4, shift, (0, 1)), 0, 1)
        city_condition = _condition(item_seed ^ 0x43495459); city[index] = render_teacher_city(item_seed, city_condition)[::2, ::2]
        noise_a = _smooth_noise(item_seed ^ 0x42494F4D415353); noise_b = _smooth_noise(item_seed ^ 0x4D494E4552414C); noise_c = _smooth_noise(item_seed ^ 0x4D4F495354555245)
        t = terrain[index]; biomass = np.clip((np.isin(t, (2, 4, 7)) * .55 + walkability * .18 + noise_a * .38) * (1 - .22 * elevation), 0, 1); mineral = np.clip(elevation * .55 + np.isin(t, (1, 3, 5)) * .30 + noise_b * .25, 0, 1); moisture = np.clip(np.isin(t, (2, 4)) * .52 + noise_c * .48, 0, 1); energy = np.clip((theme == "anomaly") * .48 + (theme == "arena") * .12 + np.abs(noise_a - noise_b) * .55, 0, 1)
        continuous[index] = np.stack((elevation, walkability, nav, biomass, mineral, moisture, energy)).astype(np.float16)
        condition[index, theme_index] = 1; condition[index, 6 + city_condition.family] = 1; season = float(rng.uniform(0, np.pi * 2)); condition[index, -4:] = (np.sin(season), np.cos(season), min(1, city_condition.building_target / 12), float(rng.uniform(0, 1)))
    digest = hashlib.sha256(b"nullvector-world-state-corpus-v1\0")
    for value in (terrain, city, continuous, condition): digest.update(str(value.dtype).encode() + str(value.shape).encode() + memoryview(np.ascontiguousarray(value)))
    return WorldStateCorpus(terrain, city, continuous, condition, digest.hexdigest())
