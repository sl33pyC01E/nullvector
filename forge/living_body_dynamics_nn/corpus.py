from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from ..creature_stage_developmental.contract import TISSUES
from ..creature_stage_developmental.development import develop
from ..creature_stage_morphology_v2.genomes import morphology_review_genomes
from ..creature_stage_neural_grasper_v1.feeding import (
    FeedingState,
    FoodClump,
    absorb_food,
    feeder_status,
    metabolize_reserve,
)
from ..living_body_substrate import LivingBody
from ..living_body_substrate.contract import ORGAN_SYSTEM, SYSTEMS
from .contract import (
    ACTION_KINDS,
    ACTION_SLICE,
    ACTIVITY_INDEX,
    APPENDAGE_INDEX,
    CELL_STATE_SLICE,
    CONNECTED_INDEX,
    CONTACT_INDEX,
    CONSUMED_INDEX,
    DELTA_INDEX,
    DIGESTIVE_INDEX,
    ENERGY_INDEX,
    FAMILY_NUTRITION_INDEX,
    FEEDER_INDEX,
    FEATURES,
    FOOD_MASS_INDEX,
    FULLNESS_INDEX,
    LOCAL_ACTION_SLICE,
    NUTRIENT_DENSITY_INDEX,
    POSITION_SLICE,
    RESERVE_INDEX,
    SIDE_INDEX,
)


SYSTEM_INDEX = {"": 0, "neural": 1, "circulation": 2, "respiration": 3, "digestion": 4, "senses": 5}


def _action_field(body: LivingBody, kind: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]]:
    points = body.organism.cell_xy.astype(np.float32)
    field = np.zeros((body.organism.cell_count, 3), dtype=np.float32)
    if kind not in (1, 2, 3):
        return field, {"kind": "idle"}
    center = points[int(rng.integers(0, len(points)))].copy()
    if kind == 1:
        radius = float(rng.uniform(2.2, 6.5))
        amount = float(rng.uniform(.18, .72))
        distance = np.linalg.norm(points - center, axis=1)
        weight = np.clip(1 - distance / radius, 0, 1)
        field[:, 0] = weight * amount
        return field, {"kind": "impact", "center": center.tolist(), "radius": radius, "amount": amount}
    if kind == 2:
        angle = float(rng.uniform(0, math.tau))
        direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32)
        start = center - direction * 9
        end = center + direction * 9
        width = float(rng.uniform(.45, 1.35))
        delta = end - start
        t = np.clip(((points - start) @ delta) / max(float(delta @ delta), 1e-6), 0, 1)
        distance = np.linalg.norm(points - (start[None] + t[:, None] * delta[None]), axis=1)
        field[:, 1] = (distance <= width).astype(np.float32)
        return field, {"kind": "cut", "start": start.tolist(), "end": end.tolist(), "width": width}
    radius = float(rng.uniform(2.5, 7.0))
    amount = float(rng.uniform(.08, .38))
    distance = np.linalg.norm(points - center, axis=1)
    field[:, 2] = np.clip(1 - distance / radius, 0, 1) * amount
    return field, {"kind": "heal", "center": center.tolist(), "radius": radius, "amount": amount}


def _apply(body: LivingBody, descriptor: dict[str, object]) -> None:
    kind = descriptor["kind"]
    if kind == "impact":
        body.impact(tuple(descriptor["center"]), float(descriptor["radius"]), float(descriptor["amount"]))
    elif kind == "cut":
        body.cut(tuple(descriptor["start"]), tuple(descriptor["end"]), width=float(descriptor["width"]))
    elif kind == "heal":
        body.heal(tuple(descriptor["center"]), float(descriptor["radius"]), float(descriptor["amount"]))


def encode_transition_features(
    body: LivingBody,
    *,
    action_kind: int,
    local_action: np.ndarray,
    feeding: FeedingState,
    clump: FoodClump,
    contact: bool,
    activity: float,
    delta: float,
) -> np.ndarray:
    """Encode one authoritative pixel-body state for training or runtime."""
    if not 0 <= action_kind < len(ACTION_KINDS):
        raise ValueError("living dynamics action kind drifted")
    local_action = np.asarray(local_action, dtype=np.float32)
    if local_action.shape != (body.organism.cell_count, 3) or not np.isfinite(local_action).all():
        raise ValueError("living dynamics local action field drifted")
    status = feeder_status(body)
    organism = body.organism
    family = body.family
    count = organism.cell_count
    features = np.zeros((count, FEATURES), dtype=np.float32)
    features[:, family] = 1
    features[np.arange(count), 5 + organism.tissue.astype(np.int64)] = 1
    for cell, organ in enumerate(body.organ):
        features[cell, 20 + SYSTEM_INDEX[ORGAN_SYSTEM.get(organ, "")]] = 1
    points = organism.cell_xy.astype(np.float32)
    midpoint = (points.min(0) + points.max(0)) * .5
    features[:, POSITION_SLICE] = (points - midpoint[None]) / 32
    features[:, APPENDAGE_INDEX] = (organism.appendage_index >= 0).astype(np.float32)
    features[:, SIDE_INDEX] = organism.side.astype(np.float32)
    features[:, CELL_STATE_SLICE] = np.stack((body.health, body.fluid, body.scar), axis=1)
    features[:, CONNECTED_INDEX] = body._connected_to_core().astype(np.float32)
    features[:, FEEDER_INDEX] = status.feeder_mask.astype(np.float32)
    features[:, DIGESTIVE_INDEX] = status.digestive_mask.astype(np.float32)
    features[:, ACTION_SLICE.start + action_kind] = 1
    features[:, LOCAL_ACTION_SLICE] = local_action
    features[:, CONTACT_INDEX] = float(contact)
    features[:, FOOD_MASS_INDEX] = min(clump.mass / 3.0, 1.0)
    features[:, NUTRIENT_DENSITY_INDEX] = min(clump.nutrient_density / 6.0, 1.0)
    features[:, FAMILY_NUTRITION_INDEX] = clump.nutrition_by_family[family]
    features[:, RESERVE_INDEX] = feeding.reserve / feeding.reserve_capacity
    features[:, FULLNESS_INDEX] = feeding.fullness_seconds / feeding.fullness_capacity_seconds
    features[:, CONSUMED_INDEX] = min(feeding.consumed_mass / 4.0, 1.0)
    features[:, ENERGY_INDEX] = min(body.energy / 4.0, 1.0)
    features[:, ACTIVITY_INDEX] = activity
    features[:, DELTA_INDEX] = delta
    return features


class BodyTransitionCorpus(Dataset[dict[str, torch.Tensor]]):
    """Paired causal interventions, including the complete feeding transition.

    Contact is supplied by the physical pixel collision layer.  Route integrity,
    absorption, digestive conversion, reserve/fullness, and energy release are
    learned targets.  Precondition scenarios explicitly ablate feeder cells,
    digestive cells, or connectivity so a model cannot pass by memorizing a
    healthy-body average.
    """

    def __init__(self, repeats: int = 4) -> None:
        if not 1 <= repeats <= 16:
            raise ValueError("living transition repeat count drifted")
        self.organisms = tuple(develop(genome) for genome in morphology_review_genomes())
        self.rows = tuple(
            (identity, scenario, action, repeat)
            for identity in range(30)
            for scenario in range(5)
            for action in range(len(ACTION_KINDS))
            for repeat in range(repeats)
        )
        digest = hashlib.sha256(b"nullvector-living-body-transition-corpus-v3-conserved-feeding\0")
        for organism in self.organisms:
            digest.update(organism.identity_sha256.encode("ascii") + b"\0")
        digest.update(np.asarray(self.rows, dtype="<i2").tobytes())
        self.semantic_sha256 = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        identity, scenario, action_kind, repeat = self.rows[index]
        rng = np.random.default_rng(0x424F44594E4E ^ (index * 0x9E3779B1))
        body = LivingBody(self.organisms[identity], seed=index)
        self._precondition(body, scenario, repeat, rng)
        before_connected = body._connected_to_core().astype(np.float32)
        action, descriptor = _action_field(body, action_kind, rng)
        before_health = body.health.copy()
        before_fluid = body.fluid.copy()
        before_scar = body.scar.copy()
        before_energy = float(body.energy)
        status_before = feeder_status(body)
        delta = float(rng.uniform(.06, .24))
        activity = float(rng.uniform(0, 1))
        reserve_capacity = 4.0
        fullness_capacity = 240.0
        reserve = float(rng.uniform(0, reserve_capacity * (1.0 if repeat % 4 != 3 else .12)))
        if repeat % 4 == 3 and action_kind in (4, 6):
            reserve = reserve_capacity
        feeding = FeedingState(
            reserve=reserve,
            reserve_capacity=reserve_capacity,
            fullness_seconds=float(rng.uniform(0, fullness_capacity)),
            fullness_capacity_seconds=fullness_capacity,
            consumed_mass=float(rng.uniform(0, 2.5)),
        )
        initial_reserve = feeding.reserve
        initial_fullness = feeding.fullness_seconds
        initial_consumed = feeding.consumed_mass
        initial_mass = float(rng.uniform(.08, 3.0))
        density = float(rng.uniform(.2, 6.0))
        family_profile = rng.uniform(.05, 1.0, size=5).astype(np.float64)
        if repeat % 4 == 2 and action_kind in (4, 6):
            family_profile[body.family] = 0.0
        feeder_points = body.organism.cell_xy[status_before.feeder_mask & body.alive_mask].astype(np.float64)
        # Contact examples are anchored to a literal live feeder pixel.  The
        # centroid can fall in empty space when a feeder has several lobes.
        anchor = feeder_points[int(rng.integers(0, len(feeder_points)))] if feeder_points.size else np.zeros(2, dtype=np.float64)
        requested_contact = bool(repeat % 4 != 1)
        offset = rng.normal(0, .18, size=2) if requested_contact else np.asarray((8.0, 8.0))
        clump = FoodClump(
            position=anchor + offset,
            velocity=np.zeros(2),
            mass=initial_mass,
            radius=float(rng.uniform(.22, .72)),
            nutrient_density=density,
            nutrition_by_family=tuple(float(value) for value in family_profile),
        )
        distance = float(np.min(np.linalg.norm(feeder_points - clump.position, axis=1))) if feeder_points.size else math.inf
        contact = distance <= 1.15 + clump.radius
        _apply(body, descriptor)
        absorbed = 0.0
        nutrition = 0.0
        contacted = False
        route_intact = feeder_status(body).route_intact
        if action_kind in (4, 6):
            result = absorb_food(body, feeding, clump, body_position=np.zeros(2), delta=delta)
            absorbed = result.absorbed_mass
            nutrition = result.nutrition
            contacted = result.contacted
            route_intact = result.route_intact
        released = 0.0
        if action_kind in (5, 6):
            released = metabolize_reserve(body, feeding, delta=delta, activity=activity)
        target_snapshot = body.tick(.1)

        # Encode the pre-action state.  The local intervention is supplied as a
        # field, while feeding/contact context comes from the physical layer.
        organism = body.organism
        family = body.family
        current_health, current_fluid, current_scar, current_energy = body.health, body.fluid, body.scar, body.energy
        body.health, body.fluid, body.scar, body.energy = before_health, before_fluid, before_scar, before_energy
        features = encode_transition_features(
            body, action_kind=action_kind, local_action=action, feeding=FeedingState(
                initial_reserve, reserve_capacity, initial_fullness, fullness_capacity, initial_consumed,
            ), clump=FoodClump(
                position=clump.position.copy(), velocity=clump.velocity.copy(), mass=initial_mass,
                radius=clump.radius, nutrient_density=density,
                nutrition_by_family=tuple(float(value) for value in family_profile),
            ), contact=contact, activity=activity, delta=delta,
        )
        body.health, body.fluid, body.scar, body.energy = current_health, current_fluid, current_scar, current_energy
        target = np.stack((body.health, body.fluid, body.scar), axis=1).astype(np.float32)
        systems = np.asarray([target_snapshot.systems[name] for name in SYSTEMS], dtype=np.float32)
        feeding_target = np.asarray(
            (
                absorbed,
                nutrition / 4.0,
                feeding.reserve / reserve_capacity,
                feeding.fullness_seconds / fullness_capacity,
                body.energy / 4.0,
                released / .01,
                float(contacted),
                float(route_intact),
                clump.mass / initial_mass,
            ),
            dtype=np.float32,
        )
        return {
            "features": torch.from_numpy(features),
            "edges": torch.from_numpy(organism_edges(body)).long(),
            "target": torch.from_numpy(target),
            "systems": torch.from_numpy(systems),
            "feeding_target": torch.from_numpy(feeding_target),
            "family": torch.tensor(family, dtype=torch.long),
            "identity": torch.tensor(identity, dtype=torch.long),
            "action_kind": torch.tensor(action_kind, dtype=torch.long),
            "scenario": torch.tensor(scenario, dtype=torch.long),
        }

    @staticmethod
    def _precondition(body: LivingBody, scenario: int, repeat: int, rng: np.random.Generator) -> None:
        if scenario == 0:
            return
        if scenario == 1:
            for _ in range(1 + repeat % 2):
                _, descriptor = _action_field(body, 1, rng)
                _apply(body, descriptor)
                body.tick(.1)
            return
        status = feeder_status(body)
        if scenario == 2:
            mask = status.feeder_mask
            body.health[mask] *= np.float32(0.0 if repeat % 2 == 0 else .28)
            body.fluid[mask] *= np.float32(.25)
            return
        if scenario == 3:
            mask = status.digestive_mask
            body.health[mask] *= np.float32(0.0 if repeat % 2 == 0 else .22)
            body.fluid[mask] *= np.float32(.20)
            return
        # A structural cut supplies disconnected-but-living route examples.
        points = body.organism.cell_xy.astype(np.float32)
        center = points.mean(0)
        horizontal = bool(repeat % 2)
        direction = np.asarray((1.0, 0.0) if horizontal else (0.0, 1.0), dtype=np.float32)
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float32)
        offset = float(rng.uniform(-2.0, 2.0))
        start = center + normal * offset - direction * 32
        end = center + normal * offset + direction * 32
        body.cut(tuple(start), tuple(end), width=float(rng.uniform(.45, .9)))
        body.tick(.1)


def organism_edges(body: LivingBody) -> np.ndarray:
    edges = body.adjacency.astype(np.int64)
    reverse = edges[:, ::-1]
    return np.concatenate((edges, reverse), axis=0).T


def collate_graphs(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    offset = 0
    edges = []
    graph_index = []
    for graph, row in enumerate(rows):
        edges.append(row["edges"] + offset)
        graph_index.append(torch.full((len(row["features"]),), graph, dtype=torch.long))
        offset += len(row["features"])
    return {
        "features": torch.cat([row["features"] for row in rows]),
        "edges": torch.cat(edges, dim=1),
        "graph_index": torch.cat(graph_index),
        "target": torch.cat([row["target"] for row in rows]),
        "systems": torch.stack([row["systems"] for row in rows]),
        "feeding_target": torch.stack([row["feeding_target"] for row in rows]),
        "family": torch.stack([row["family"] for row in rows]),
        "identity": torch.stack([row["identity"] for row in rows]),
        "action_kind": torch.stack([row["action_kind"] for row in rows]),
        "scenario": torch.stack([row["scenario"] for row in rows]),
    }
