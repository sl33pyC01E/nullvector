from __future__ import annotations

import math
import numpy as np

from ..nature_sim_v2.climate import SEASONS
from ..nature_world_scale_v1.atlas import BIOMES
from ..qud_society_v1.architecture import PURPOSES
from .contract import GLOBAL_FEATURES, PATCH_SIZE, STATE_CHANNELS, WORLD_SIZE

EVENTS = (None, "drought", "spore_bloom", "mineral_upwelling", "phase_storm")


def _pool2(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.shape[-2:] != (WORLD_SIZE, WORLD_SIZE):
        raise ValueError("macro state requires a 64x64 authority world")
    prefix = value.shape[:-2]
    return value.reshape(*prefix, PATCH_SIZE, 2, PATCH_SIZE, 2).mean((-3, -1), dtype=np.float32)


def extract_patch_state(world, society) -> np.ndarray:
    if world.size != WORLD_SIZE:
        raise ValueError("macro state world size drifted")
    full = np.zeros((len(STATE_CHANNELS), WORLD_SIZE, WORLD_SIZE), np.float32)
    full[:10] = np.asarray(world.fields, np.float32)
    energy_sum = np.zeros((WORLD_SIZE, WORLD_SIZE), np.float32)
    health_sum = np.zeros_like(energy_sum)
    organism_count = np.zeros_like(energy_sum)
    for entity in world.organisms.values():
        if not entity.alive:
            continue
        y, x = world._cell(entity.position)
        full[10 + entity.family, y, x] += 1.0
        energy_sum[y, x] += float(entity.energy)
        health_sum[y, x] += float(entity.body.systems()["integrity"])
        organism_count[y, x] += 1.0
        if entity.colony_id is not None:
            full[17, y, x] += 1.0
    occupied = organism_count > 0
    full[15, occupied] = energy_sum[occupied] / organism_count[occupied]
    full[16, occupied] = health_sum[occupied] / organism_count[occupied]
    for settlement in society.settlements.values():
        for building in settlement.buildings:
            channel = 18 + PURPOSES.index(building.purpose)
            for x, y, material in building.cells:
                if material != "wall":
                    full[channel, y % WORLD_SIZE, x % WORLD_SIZE] = 1.0
        for x, y in settlement.roads:
            full[27, y % WORLD_SIZE, x % WORLD_SIZE] = 1.0
    full[28] = world.materials.structure_id > 0
    full[29] = np.clip(world.materials.mass, 0, 1)
    full[30] = np.clip(world.materials.damage, 0, 1)
    full[31] = np.clip(world.materials.temperature / 2.0, 0, 1)
    pooled = _pool2(full)
    pooled[10:15] = np.clip(pooled[10:15] * 4.0 / max(1, world.max_population / 24), 0, 1)
    pooled[17] = np.clip(pooled[17] * 2.0, 0, 1)
    return np.ascontiguousarray(np.clip(pooled, 0, 1), dtype=np.float32)


def extract_global_state(world, society) -> np.ndarray:
    row = np.zeros(GLOBAL_FEATURES, np.float32)
    climate = world.climate.current
    row[SEASONS.index(climate.season)] = 1.0
    row[6:11] = (climate.light, climate.rainfall, climate.heat, climate.phase_flux, climate.toxin)
    row[11 + EVENTS.index(climate.event)] = 1.0
    angle = math.tau * ((world.time / 180.0) % 1.0)
    row[16:18] = ((math.sin(angle) + 1) * .5, (math.cos(angle) + 1) * .5)
    counts = np.bincount([item.family for item in world.organisms.values() if item.alive], minlength=5)
    row[18:23] = np.clip(counts / max(1, world.max_population), 0, 1)
    row[23:27] = np.clip(np.asarray((world.births, world.deaths, world.predation_events, world.mutation_count), np.float32) / 128.0, 0, 1)
    row[27:30] = np.clip((len(world.colonies) / 24, len(society.factions) / 12, len(society.settlements) / 24), 0, 1)
    settlements = tuple(society.settlements.values())
    factions = tuple(society.factions.values())
    if settlements:
        row[30:34] = np.clip(np.mean([(s.wealth / 8, s.food / 8, s.power / 8, len(s.buildings) / 24) for s in settlements], axis=0), 0, 1)
    if factions:
        row[34:36] = np.clip((np.mean([f.knowledge / 4 for f in factions]), np.mean([f.cohesion for f in factions])), 0, 1)
    biome = world.biome or BIOMES[world.seed % len(BIOMES)]
    row[36 + BIOMES.index(biome)] = 1.0
    if not np.isfinite(row).all() or np.any((row < 0) | (row > 1)):
        raise FloatingPointError("macro global state invalid")
    return row
