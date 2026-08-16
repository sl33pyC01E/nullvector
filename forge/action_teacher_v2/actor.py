from __future__ import annotations

import math

import numpy as np

from ..creature_stage_developmental.contract import FAMILIES, TISSUES, TRAITS
from ..living_body_substrate.contract import ORGAN_SYSTEM, SYSTEMS
from ..nature_sim_v2.contract import ECO_TRAITS, INTENTS, LIFE_STAGES, RESOURCE_NAMES
from .contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE

ACTOR_FIELD_NAMES = ("occupancy", "health", "fluid", "scar", "connected", "neural", "vital", "locomotor")
ACTOR_FEATURE_NAMES = tuple(
    [f"family_{name}" for name in FAMILIES]
    + [f"stage_{name}" for name in LIFE_STAGES]
    + [f"intent_{name}" for name in INTENTS]
    + [f"family_mix_{name}" for name in FAMILIES]
    + [f"development_{name}" for name in TRAITS]
    + [f"ecology_{name}" for name in ECO_TRAITS]
    + [f"diet_{name}" for name in RESOURCE_NAMES]
    + [f"system_{name}" for name in SYSTEMS]
    + [
        "energy", "reserve", "age_normalized", "reproduction_cooldown", "gestation", "decomposition", "body_energy",
        "alive", "incapacitated", "dead", "velocity_x", "velocity_y", "heading_sin", "heading_cos", "cell_count",
        "alive_ratio", "connected_ratio", "detached_ratio", "leak_amount", "polyp_count", "biomass_count",
    ]
    + [f"consumed_{name}" for name in RESOURCE_NAMES[:4]]
    + ["neural_contact_ratio", "neural_muscle_mean", "neural_muscle_std", "neural_muscle_max", "appendage_count", "component_count", "generation", "parent_count"]
    + [f"tissue_health_{name}" for name in TISSUES]
    + ["fluid_ratio", "scar_mean", "health_min", "health_std"]
)
if len(ACTOR_FEATURE_NAMES) != ACTOR_FEATURES:
    raise AssertionError(f"actor feature contract has {len(ACTOR_FEATURE_NAMES)} values")


def _one_hot(index: int, count: int) -> np.ndarray:
    row = np.zeros(count, dtype=np.float32)
    if 0 <= index < count:
        row[index] = 1
    return row


def extract_actor_features(world, entity_id: int) -> np.ndarray:
    entity = world.organisms.get(int(entity_id))
    if entity is None:
        return np.zeros(ACTOR_FEATURES, dtype=np.float32)
    body = entity.body
    snapshot = body.snapshot()
    developed = body.organism
    developmental = entity.genome.developmental
    alive = body.alive_mask
    cell_count = max(1, developed.cell_count)
    connected = body._connected_to_core()
    consumed = np.asarray(entity.consumed, dtype=np.float32)
    if consumed.shape != (len(RESOURCE_NAMES),):
        raise ValueError("actor consumed-resource vector drifted")
    muscles = np.asarray(entity.neural_muscles, dtype=np.float32)
    contacts = np.asarray(entity.neural_contacts, dtype=np.bool_)
    scalars = np.asarray(
        (
            entity.energy, entity.reserve, min(1.0, entity.age / 400), min(1.0, entity.reproduction_cooldown / 64), min(1.0, entity.gestation_remaining / 64), entity.decomposition, body.energy,
            float(entity.alive), float(snapshot.incapacitated), float(snapshot.dead), np.clip(entity.velocity[0] / 12, -1, 1), np.clip(entity.velocity[1] / 12, -1, 1), math.sin(entity.heading), math.cos(entity.heading), min(1.0, cell_count / 1024),
            snapshot.alive_cells / cell_count, snapshot.connected_cells / cell_count, snapshot.detached_cells / cell_count, min(1.0, snapshot.leak_amount / 8), min(1.0, snapshot.polyp_count / 16), min(1.0, snapshot.biomass_count / 16),
        ), dtype=np.float32,
    )
    tissue_health = np.zeros(len(TISSUES), dtype=np.float32)
    for index in range(len(TISSUES)):
        selected = developed.tissue == index
        tissue_health[index] = float(body.health[selected].mean()) if selected.any() else 0.0
    fluid_ratio = float(body.fluid.sum() / max(float(body.fluid_capacity.sum()), 1e-6))
    trailing = np.asarray((fluid_ratio, float(body.scar.mean()), float(body.health.min()), float(body.health.std())), dtype=np.float32)
    neural = np.asarray((float(contacts.mean()) if contacts.size else 0, float(muscles.mean()) if muscles.size else 0, float(muscles.std()) if muscles.size else 0, float(muscles.max()) if muscles.size else 0, min(1.0, len(developmental.appendages) / 32), min(1.0, len(developmental.components) / 32), min(1.0, developmental.generation / 128), min(1.0, len(entity.parent_ids) / 4)), dtype=np.float32)
    row = np.concatenate((
        _one_hot(entity.family, len(FAMILIES)), _one_hot(LIFE_STAGES.index(entity.stage), len(LIFE_STAGES)), _one_hot(INTENTS.index(entity.intent), len(INTENTS)),
        np.asarray(developmental.family_mix, dtype=np.float32), np.asarray(developmental.traits, dtype=np.float32), np.asarray(entity.genome.eco_traits, dtype=np.float32), np.asarray(entity.genome.diet, dtype=np.float32), np.asarray([snapshot.systems[name] for name in SYSTEMS], dtype=np.float32),
        scalars, np.clip(consumed[:4] / 32, 0, 1), neural, tissue_health, trailing,
    )).astype(np.float32)
    if row.shape != (ACTOR_FEATURES,) or not np.isfinite(row).all():
        raise ValueError("actor feature extraction drifted")
    return row


def extract_actor_field(world, entity_id: int) -> np.ndarray:
    entity = world.organisms.get(int(entity_id))
    field = np.zeros(ACTOR_FIELD_SHAPE, dtype=np.float32)
    if entity is None:
        return field.astype(np.float16)
    body = entity.body
    organism = body.organism
    xy = np.rint(organism.cell_xy.astype(np.float32) * 0.6 + 15.5).astype(np.int32)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < 32) & (xy[:, 1] >= 0) & (xy[:, 1] < 32)
    connected = body._connected_to_core()
    capacity = np.maximum(body.fluid_capacity, 1e-6)
    vital = np.asarray([ORGAN_SYSTEM.get(organ, "") in ("neural", "circulation", "respiration", "digestion") for organ in body.organ], dtype=np.float32)
    neural = np.asarray([ORGAN_SYSTEM.get(organ, "") == "neural" for organ in body.organ], dtype=np.float32)
    values = np.stack((np.ones(organism.cell_count), body.health, body.fluid / capacity, body.scar, connected.astype(np.float32), neural, vital, body._appendage_mask.astype(np.float32))).astype(np.float32)
    counts = np.zeros((32, 32), dtype=np.float32)
    for index in np.flatnonzero(valid):
        x, y = xy[index]
        field[0, y, x] = 1
        field[1:5, y, x] += values[1:5, index]
        field[5:, y, x] = np.maximum(field[5:, y, x], values[5:, index])
        counts[y, x] += 1
    occupied = counts > 0
    field[1:5, occupied] /= counts[occupied]
    return field.astype(np.float16)
