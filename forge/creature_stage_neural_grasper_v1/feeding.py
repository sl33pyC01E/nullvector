from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..creature_stage_developmental.contract import TISSUES
from ..living_body_substrate.contract import ORGAN_SYSTEM
from ..living_body_substrate.state import LivingBody


FEEDER_KINDS = ("mouth", "root_feeder", "transmuter_aperture", "fuel_port")


@dataclass(slots=True)
class FoodClump:
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    radius: float
    nutrient_density: float
    nutrition_by_family: tuple[float, float, float, float, float]
    material: str = "biomass"

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.position.shape != (2,) or self.velocity.shape != (2,) or not np.isfinite(self.position).all() or not np.isfinite(self.velocity).all():
            raise ValueError("food clump kinematics drifted")
        if not math.isfinite(self.mass) or not 0 <= self.mass <= 1_000 or not math.isfinite(self.radius) or not .05 <= self.radius <= 32:
            raise ValueError("food clump geometry drifted")
        if not math.isfinite(self.nutrient_density) or not 0 <= self.nutrient_density <= 8:
            raise ValueError("food clump nutrition drifted")
        if len(self.nutrition_by_family) != 5 or not all(math.isfinite(value) and 0 <= value <= 1 for value in self.nutrition_by_family):
            raise ValueError("food clump diet profile drifted")


@dataclass(slots=True)
class FeedingState:
    reserve: float = 1.5
    reserve_capacity: float = 4.0
    fullness_seconds: float = 90.0
    fullness_capacity_seconds: float = 240.0
    consumed_mass: float = 0.0

    def __post_init__(self) -> None:
        values = (self.reserve, self.reserve_capacity, self.fullness_seconds, self.fullness_capacity_seconds, self.consumed_mass)
        if not all(math.isfinite(value) and value >= 0 for value in values) or self.reserve > self.reserve_capacity or self.fullness_seconds > self.fullness_capacity_seconds:
            raise ValueError("feeding reserve drifted")


@dataclass(frozen=True, slots=True)
class FeederStatus:
    kind: str
    feeder_mask: np.ndarray
    digestive_mask: np.ndarray
    live_feeder_cells: int
    live_digestive_cells: int
    route_intact: bool
    capacity: float


@dataclass(frozen=True, slots=True)
class IntakeResult:
    contacted: bool
    route_intact: bool
    absorbed_mass: float
    nutrition: float
    reserve: float
    fullness_seconds: float


def _boundary_mask(body: LivingBody) -> np.ndarray:
    degree = np.zeros(body.organism.cell_count, dtype=np.uint8)
    np.add.at(degree, body.adjacency[:, 0], 1)
    np.add.at(degree, body.adjacency[:, 1], 1)
    return degree < 8


def _component_mask(body: LivingBody, predicate) -> np.ndarray:
    selected = np.asarray([bool(predicate(component)) for component in body.organism.genome.components], dtype=np.bool_)
    return selected[body.component_owner]


def _surface_aperture(body: LivingBody, predicate, boundary: np.ndarray) -> np.ndarray:
    component_indices = [
        index for index, component in enumerate(body.organism.genome.components)
        if predicate(component)
    ]
    result = np.zeros(body.organism.cell_count, dtype=np.bool_)
    surface = np.flatnonzero(boundary)
    if not component_indices or surface.size == 0:
        return result
    anchors = np.asarray(
        [body.organism.genome.components[index].anchor for index in component_indices],
        dtype=np.float32,
    )
    distance = np.linalg.norm(
        body.organism.cell_xy[surface, None].astype(np.float32) - anchors[None], axis=2,
    ).min(axis=1)
    count = min(5, surface.size)
    result[surface[np.argsort(distance, kind="stable")[:count]]] = True
    return result


def _feeder_mask(body: LivingBody) -> tuple[str, np.ndarray]:
    boundary = _boundary_mask(body)
    literal = _component_mask(body, lambda component: component.kind == "mouth" or component.organ == "jaw") & boundary
    if literal.any():
        return "mouth", literal
    family = body.family
    if family == 0:
        head = _component_mask(body, lambda component: component.kind == "head") & boundary
        if head.any():
            points = body.organism.cell_xy
            median_y = float(np.median(points[head, 1]))
            mask = head & (points[:, 1] >= median_y) & (np.abs(points[:, 0]) <= 2)
            return "mouth", mask if mask.any() else head
    if family == 2:
        root_indices = {
            index for index, appendage in enumerate(body.organism.genome.appendages)
            if appendage.kind == "root"
        }
        root = np.isin(body.organism.appendage_index, tuple(root_indices)) & boundary
        if root.any():
            cutoff = float(np.quantile(body.organism.cell_xy[root, 1], .60))
            selected = root & (body.organism.cell_xy[:, 1] >= cutoff)
            return "root_feeder", selected if selected.any() else root
    if family == 3:
        aperture = _component_mask(body, lambda component: component.organ == "transmuter") & boundary
        if not aperture.any():
            aperture = _surface_aperture(body, lambda component: component.organ == "transmuter", boundary)
        if aperture.any():
            return "transmuter_aperture", aperture
    if family == 4:
        port = _component_mask(body, lambda component: component.organ == "battery") & boundary
        if port.any():
            return "fuel_port", port
    raise ValueError("developed organism has no feasible feeder aperture")


def _digestive_mask(body: LivingBody) -> np.ndarray:
    return np.asarray([ORGAN_SYSTEM.get(organ, "") == "digestion" for organ in body.organ], dtype=np.bool_)


def _reachable(body: LivingBody, starts: np.ndarray, goals: np.ndarray, alive: np.ndarray) -> bool:
    start_indices = np.flatnonzero(starts & alive)
    if start_indices.size == 0 or not np.any(goals & alive):
        return False
    if np.any(goals[start_indices]):
        return True
    neighbors: list[list[int]] = [[] for _ in range(body.organism.cell_count)]
    for left_raw, right_raw in body.adjacency:
        left, right = int(left_raw), int(right_raw)
        neighbors[left].append(right)
        neighbors[right].append(left)
    seen = np.zeros(body.organism.cell_count, dtype=np.bool_)
    stack = [int(value) for value in start_indices]
    seen[start_indices] = True
    while stack:
        current = stack.pop()
        for neighbor in neighbors[current]:
            if not alive[neighbor] or seen[neighbor]:
                continue
            if goals[neighbor]:
                return True
            seen[neighbor] = True
            stack.append(neighbor)
    return False


def feeder_status(body: LivingBody) -> FeederStatus:
    kind, feeder = _feeder_mask(body)
    digestive = _digestive_mask(body)
    alive = body.alive_mask
    route = _reachable(body, feeder, digestive, alive)
    feeder_health = float(body.health[feeder].mean()) if feeder.any() else 0.0
    digestive_health = float(body.health[digestive].mean()) if digestive.any() else 0.0
    capacity = min(feeder_health, digestive_health, body.systems()["digestion"]) if route else 0.0
    return FeederStatus(
        kind, feeder.copy(), digestive.copy(), int(np.count_nonzero(feeder & alive)),
        int(np.count_nonzero(digestive & alive)), route, float(np.clip(capacity, 0, 1)),
    )


def physical_feeder_anchor(body: LivingBody) -> np.ndarray:
    status = feeder_status(body)
    points = body.organism.cell_xy[status.feeder_mask & body.alive_mask]
    if points.size == 0:
        raise ValueError("living feeder has no anchor cells")
    return np.clip(points.astype(np.float32).mean(axis=0) / 24.0, -1, 1).astype(np.float32)


def absorb_food(
    body: LivingBody,
    feeding: FeedingState,
    clump: FoodClump,
    *,
    body_position: np.ndarray,
    delta: float,
    contact_field: float = 1.15,
    intake_rate: float = .45,
) -> IntakeResult:
    if not math.isfinite(delta) or not .001 <= delta <= 1 or not .25 <= contact_field <= 3 or not .01 <= intake_rate <= 4:
        raise ValueError("feeding step drifted")
    origin = np.asarray(body_position, dtype=np.float64)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("feeding body position drifted")
    status = feeder_status(body)
    feeder_points = body.organism.cell_xy[status.feeder_mask & body.alive_mask].astype(np.float64) + origin
    distance = float(np.min(np.linalg.norm(feeder_points - clump.position, axis=1))) if feeder_points.size else math.inf
    contacted = distance <= contact_field + clump.radius
    family_nutrition = float(clump.nutrition_by_family[body.family])
    available_capacity = max(0.0, feeding.reserve_capacity - feeding.reserve)
    can_absorb = contacted and status.route_intact and not body.dead and family_nutrition > 0 and clump.mass > 0 and available_capacity > 0
    absorbed = min(clump.mass, intake_rate * delta * (.30 + .70 * status.capacity), available_capacity / max(clump.nutrient_density * family_nutrition, 1e-8)) if can_absorb else 0.0
    nutrition = absorbed * clump.nutrient_density * family_nutrition
    clump.mass = max(0.0, clump.mass - absorbed)
    feeding.reserve = min(feeding.reserve_capacity, feeding.reserve + nutrition)
    feeding.fullness_seconds = min(feeding.fullness_capacity_seconds, feeding.fullness_seconds + nutrition * 42.0)
    feeding.consumed_mass += absorbed
    return IntakeResult(contacted, status.route_intact, float(absorbed), float(nutrition), float(feeding.reserve), float(feeding.fullness_seconds))


def metabolize_reserve(body: LivingBody, feeding: FeedingState, *, delta: float, activity: float = 0.0) -> float:
    if not math.isfinite(delta) or not .001 <= delta <= 1 or not math.isfinite(activity) or not 0 <= activity <= 1:
        raise ValueError("feeding metabolism drifted")
    status = feeder_status(body)
    feeding.fullness_seconds = max(0.0, feeding.fullness_seconds - delta)
    requested = delta * (.0015 + .0035 * activity)
    released = min(feeding.reserve, requested * (.20 + .80 * status.capacity)) if status.route_intact else 0.0
    feeding.reserve -= released
    body.energy = min(4.0, body.energy + released)
    return float(released)
