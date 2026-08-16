from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..creature_stage_developmental.contract import TISSUES
from ..creature_stage_developmental.development import DevelopedOrganism
from .contract import FAMILY_NAMES, FLUID_TISSUES, FORMAT, ORGAN_SYSTEM, POLYP_FAMILIES, SYSTEMS


@dataclass(frozen=True, slots=True)
class BodySnapshot:
    tick: int
    systems: dict[str, float]
    energy: float
    alive_cells: int
    connected_cells: int
    detached_cells: int
    leak_amount: float
    polyp_count: int
    biomass_count: int
    incapacitated: bool
    dead: bool
    semantic_sha256: str


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b - a
    denominator = max(float(np.dot(delta, delta)), 1e-8)
    t = np.clip(((points - a) @ delta) / denominator, 0.0, 1.0)
    projection = a[None] + t[:, None] * delta[None]
    return np.linalg.norm(points - projection, axis=1)


class LivingBody:
    """Deterministic causal body scaffold consumed by gameplay and future NNs.

    No global hit points exist. Health belongs to cells, organs are component
    fields over those cells, locomotion belongs to connected appendages, and
    fluids can leave damaged tissue into a top-down diffusion puddle.
    """

    def __init__(self, organism: DevelopedOrganism, *, seed: int = 0) -> None:
        self.organism = organism
        self.seed = int(seed)
        self.family = int(np.argmax(organism.genome.family_mix))
        self.health = np.ones(organism.cell_count, dtype=np.float32)
        self.scar = np.zeros(organism.cell_count, dtype=np.float32)
        self.fluid = np.asarray(
            [FLUID_TISSUES.get(TISSUES[int(tissue)], (.12, .5))[0] for tissue in organism.tissue],
            dtype=np.float32,
        )
        self.fluid_capacity = self.fluid.copy()
        self.component_owner = organism.component_weights.argmax(1).astype(np.int16)
        self.organ = tuple(organism.genome.components[int(owner)].organ for owner in self.component_owner)
        self.adjacency = self._build_adjacency()
        self._edge_left = self.adjacency[:, 0]
        self._edge_right = self.adjacency[:, 1]
        neighbors: list[list[int]] = [[] for _ in range(organism.cell_count)]
        for left, right in self.adjacency:
            neighbors[int(left)].append(int(right))
            neighbors[int(right)].append(int(left))
        self._neighbors = tuple(tuple(row) for row in neighbors)
        authority = organism.component_weights.max(1).astype(np.float32)
        self._system_indices = {
            system: np.asarray([ORGAN_SYSTEM.get(organ, "") == system for organ in self.organ], dtype=np.bool_)
            for system in ("neural", "circulation", "respiration", "digestion", "senses")
        }
        self._system_weights = {system: authority[indices] for system, indices in self._system_indices.items()}
        self._appendage_mask = organism.appendage_index >= 0
        self.main_seed_cell = self._core_seed()
        self.separation_age = np.zeros(organism.cell_count, dtype=np.float32)
        self.external_puddle: list[dict[str, object]] = []
        self.polyps: list[dict[str, object]] = []
        self.biomass: list[dict[str, object]] = []
        self.energy = 1.0
        self.tick_index = 0
        self.incapacitated = False
        self.dead = False
        self._last_connected = np.ones(organism.cell_count, dtype=np.bool_)
        self._connectivity_alive: np.ndarray | None = None
        self._connectivity_connected: np.ndarray | None = None
        self._systems_health: np.ndarray | None = None
        self._systems_fluid: np.ndarray | None = None
        self._systems_cache: dict[str, float] | None = None
        self._assert_healthy_spawn()

    def _build_adjacency(self) -> np.ndarray:
        lookup = {(int(x), int(y)): index for index, (x, y) in enumerate(self.organism.cell_xy)}
        edges: list[tuple[int, int]] = []
        for index, (x, y) in enumerate(self.organism.cell_xy):
            for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
                neighbor = lookup.get((int(x + dx), int(y + dy)))
                if neighbor is not None:
                    edges.append((index, neighbor))
        if not edges:
            raise ValueError("living body has no cellular connections")
        return np.asarray(edges, dtype=np.int32)

    def _core_seed(self) -> int:
        core = np.flatnonzero(self.component_owner == 0)
        if core.size == 0:
            raise ValueError("living body lacks a primary component")
        center = np.asarray(self.organism.genome.components[0].anchor, dtype=np.float32)
        return int(core[np.argmin(np.linalg.norm(self.organism.cell_xy[core] - center, axis=1))])

    def _assert_healthy_spawn(self) -> None:
        snapshot = self.snapshot()
        if snapshot.dead or snapshot.incapacitated:
            raise ValueError("healthy body spawned incapacitated")
        if snapshot.systems["integrity"] < .999 or snapshot.connected_cells < int(.90 * self.organism.cell_count):
            raise ValueError("healthy body spawned structurally invalid")

    @property
    def alive_mask(self) -> np.ndarray:
        return self.health > .08

    def _connected_to_core(self) -> np.ndarray:
        alive = self.alive_mask
        if self._connectivity_alive is not None and np.array_equal(alive, self._connectivity_alive):
            assert self._connectivity_connected is not None
            return self._connectivity_connected
        connected = np.zeros(self.organism.cell_count, dtype=np.bool_)
        if not alive[self.main_seed_cell]:
            self._connectivity_alive = alive.copy()
            self._connectivity_connected = connected
            return connected
        stack = [self.main_seed_cell]
        connected[self.main_seed_cell] = True
        while stack:
            current = stack.pop()
            for neighbor in self._neighbors[current]:
                if alive[neighbor] and not connected[neighbor]:
                    connected[neighbor] = True
                    stack.append(neighbor)
        self._connectivity_alive = alive.copy()
        self._connectivity_connected = connected
        return connected

    def cut(self, start: tuple[float, float], end: tuple[float, float], *, width: float = .7) -> int:
        if not math.isfinite(width) or not .1 <= width <= 5:
            raise ValueError("cut width drifted")
        distance = _point_segment_distance(
            self.organism.cell_xy.astype(np.float32),
            np.asarray(start, dtype=np.float32),
            np.asarray(end, dtype=np.float32),
        )
        selected = (distance <= width) & self.alive_mask
        self.health[selected] = 0.0
        self._emit_leaks(np.flatnonzero(selected), impulse=.34)
        return int(selected.sum())

    def impact(self, center: tuple[float, float], radius: float, damage: float) -> int:
        if not math.isfinite(radius) or not .25 <= radius <= 20:
            raise ValueError("impact radius drifted")
        if not math.isfinite(damage) or not 0 < damage <= 1:
            raise ValueError("impact damage drifted")
        distance = np.linalg.norm(self.organism.cell_xy - np.asarray(center, dtype=np.float32), axis=1)
        weight = np.clip(1 - distance / radius, 0, 1)
        selected = weight > 0
        previous = self.health.copy()
        self.health = np.clip(self.health - weight * damage, 0, 1).astype(np.float32)
        newly_open = np.flatnonzero((previous >= .55) & (self.health < .55))
        self._emit_leaks(newly_open, impulse=float(damage))
        return int(selected.sum())

    def heal(self, center: tuple[float, float], radius: float, amount: float) -> int:
        if self.dead or self.energy <= 0:
            return 0
        distance = np.linalg.norm(self.organism.cell_xy - np.asarray(center, dtype=np.float32), axis=1)
        weight = np.clip(1 - distance / max(radius, .25), 0, 1)
        connected = self._connected_to_core()
        selected = (weight > 0) & connected & (self.health > 0)
        affordable = min(float(amount), self.energy * .45)
        previous = self.health.copy()
        self.health[selected] = np.clip(self.health[selected] + weight[selected] * affordable, 0, 1)
        self.scar[selected] = np.maximum(self.scar[selected], (self.health[selected] - previous[selected]) * .55)
        self.energy = max(0.0, self.energy - float((self.health - previous).sum()) * .0025)
        return int(selected.sum())

    def _emit_leaks(self, indices: np.ndarray, impulse: float) -> None:
        for index_raw in indices:
            index = int(index_raw)
            if self.fluid[index] <= 0:
                continue
            tissue = TISSUES[int(self.organism.tissue[index])]
            _, viscosity = FLUID_TISSUES.get(tissue, (.12, .5))
            amount = min(float(self.fluid[index]), .08 + impulse * .18)
            self.fluid[index] -= amount
            self.external_puddle.append({
                "xy": self.organism.cell_xy[index].astype(np.float32).copy(),
                "amount": amount,
                "radius": .45,
                "viscosity": viscosity,
                "tissue": tissue,
            })

    def _system_capacity(self, system: str, connected: np.ndarray) -> float:
        indices = self._system_indices[system]
        if not indices.any():
            # A family without a mammalian organ uses distributed tissue or a
            # different named equivalent; absence is not spontaneous failure.
            return 1.0
        weights = self._system_weights[system]
        values = self.health[indices] * connected[indices]
        capacity = float((values * weights).sum() / max(float(weights.sum()), 1e-6))
        if system == "circulation":
            fluid_ratio = float(self.fluid[indices].sum() / max(float(self.fluid_capacity[indices].sum()), 1e-6))
            capacity *= .35 + .65 * fluid_ratio
        return float(np.clip(capacity, 0, 1))

    def systems(self) -> dict[str, float]:
        if (
            self._systems_cache is not None
            and self._systems_health is not None
            and self._systems_fluid is not None
            and np.array_equal(self.health, self._systems_health)
            and np.array_equal(self.fluid, self._systems_fluid)
        ):
            return self._systems_cache.copy()
        connected = self._connected_to_core()
        self._last_connected = connected
        integrity = float(self.health.mean())
        values = {"integrity": integrity}
        for system in ("neural", "circulation", "respiration", "digestion", "senses"):
            values[system] = self._system_capacity(system, connected)
        appendage = self._appendage_mask
        if appendage.any():
            values["locomotion"] = float((self.health[appendage] * connected[appendage]).mean())
        else:
            values["locomotion"] = values["integrity"]
        result = {key: round(float(np.clip(values[key], 0, 1)), 8) for key in SYSTEMS}
        self._systems_health = self.health.copy()
        self._systems_fluid = self.fluid.copy()
        self._systems_cache = result
        return result.copy()

    def _diffuse_fluid(self, delta: float) -> None:
        transfer = np.zeros_like(self.fluid)
        alive = self.alive_mask
        valid = alive[self._edge_left] & alive[self._edge_right]
        left = self._edge_left[valid]
        right = self._edge_right[valid]
        flow = (self.fluid[left] - self.fluid[right]) * np.float32(.08 * delta)
        np.add.at(transfer, left, -flow)
        np.add.at(transfer, right, flow)
        self.fluid = np.clip(self.fluid + transfer, 0, self.fluid_capacity).astype(np.float32)
        for puddle in self.external_puddle:
            viscosity = float(puddle["viscosity"])
            puddle["radius"] = float(puddle["radius"]) + delta * (1.35 - viscosity) * .42
            puddle["amount"] = float(puddle["amount"]) * math.exp(-delta * (.018 + viscosity * .012))
        self.external_puddle = [item for item in self.external_puddle if float(item["amount"]) > 1e-4]

    def _resolve_detached(self, delta: float) -> None:
        connected = self._connected_to_core()
        detached = self.alive_mask & ~connected
        self.separation_age[connected | ~self.alive_mask] = 0
        self.separation_age[detached] += delta
        grace = (3.0, 4.5, 7.5, 10.0, 8.5)[self.family]
        ready = detached & (self.separation_age >= grace)
        if not ready.any():
            return
        # Resolve one connected detached component at a time so a limb becomes
        # one polyp/biomass object rather than exploding into per-cell debris.
        unresolved = set(int(index) for index in np.flatnonzero(ready))
        while unresolved:
            seed = next(iter(unresolved))
            component = []
            stack = [seed]
            unresolved.remove(seed)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in self._neighbors[current]:
                    if neighbor in unresolved:
                        unresolved.remove(neighbor)
                        stack.append(neighbor)
            points = self.organism.cell_xy[component].astype(np.float32)
            record = {
                "cell_count": len(component),
                "centroid": points.mean(0).tolist(),
                "source_family": FAMILY_NAMES[self.family],
                "source_tick": self.tick_index,
            }
            if self.family in POLYP_FAMILIES and len(component) >= 3:
                record["viability"] = round(float(np.mean(self.health[component])) * min(1.0, len(component) / 12), 8)
                self.polyps.append(record)
            else:
                record["nutrient"] = round(float(np.sum(self.health[component])) * .62, 8)
                self.biomass.append(record)
            self.health[component] = 0
            self.fluid[component] = 0
            self.separation_age[component] = 0

    def tick(self, delta: float = .1) -> BodySnapshot:
        if not math.isfinite(delta) or not .001 <= delta <= 1:
            raise ValueError("living body tick delta drifted")
        self.tick_index += 1
        self._diffuse_fluid(delta)
        self._resolve_detached(delta)
        systems = self.systems()
        metabolic = systems["circulation"] * systems["respiration"] * systems["digestion"]
        self.energy = max(0.0, self.energy - delta * (.0008 + (1 - metabolic) * .008))
        # Healing is slow, local, energy-limited, and leaves scar tissue.
        if not self.dead and self.energy > .08 and systems["circulation"] > .25:
            connected = self._last_connected
            wounded = connected & (self.health > 0) & (self.health < .96)
            amount = min(self.energy * .004 * delta, .003 * delta)
            self.health[wounded] = np.minimum(1, self.health[wounded] + amount)
            self.scar[wounded] = np.minimum(1, self.scar[wounded] + amount * .32)
            self.energy -= float(wounded.sum()) * amount * .003
        systems = self.systems()
        self.incapacitated = systems["neural"] < .16 or systems["locomotion"] < .06
        self.dead = systems["neural"] < .035 or systems["integrity"] < .10 or (
            systems["circulation"] < .04 and systems["respiration"] < .04
        )
        return self.snapshot()

    def snapshot(self) -> BodySnapshot:
        systems = self.systems()
        connected = self._last_connected
        alive = self.alive_mask
        payload = b"".join((
            FORMAT.encode("ascii"),
            np.asarray(self.tick_index, dtype="<i8").tobytes(),
            self.health.astype("<f4").tobytes(),
            self.scar.astype("<f4").tobytes(),
            self.fluid.astype("<f4").tobytes(),
            _canonical_systems(systems),
        ))
        return BodySnapshot(
            tick=self.tick_index,
            systems=systems,
            energy=round(float(self.energy), 8),
            alive_cells=int(alive.sum()),
            connected_cells=int((alive & connected).sum()),
            detached_cells=int((alive & ~connected).sum()),
            leak_amount=round(sum(float(item["amount"]) for item in self.external_puddle), 8),
            polyp_count=len(self.polyps),
            biomass_count=len(self.biomass),
            incapacitated=bool(self.incapacitated),
            dead=bool(self.dead),
            semantic_sha256=hashlib.sha256(payload).hexdigest(),
        )


def _canonical_systems(systems: dict[str, float]) -> bytes:
    return b"".join(name.encode("ascii") + b"=" + f"{systems[name]:.8f}".encode("ascii") + b"\0" for name in SYSTEMS)
