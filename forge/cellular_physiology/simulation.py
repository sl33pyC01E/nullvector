from __future__ import annotations

from collections import deque
import math
from typing import Mapping

import numpy as np

from .contract import DEPENDENCIES, SYSTEM_NAMES


class PhysiologyState:
    """Deterministic reference for damage cascades across connected organ systems."""

    def __init__(self, anatomy: Mapping[str, np.ndarray], physiology: Mapping[str, np.ndarray]) -> None:
        self.cell_count = len(anatomy["position_xy"])
        self.health = anatomy["max_health"].astype(np.float32).copy()
        self.max_health = self.health.copy()
        self.fluid = anatomy["fluid_initial"].astype(np.float32).copy()
        self.fluid_reference = self.fluid.copy()
        self.alive = np.ones(self.cell_count, dtype=bool)
        self.bond_ab = anatomy["bond_ab"].astype(np.int32).copy()
        self.bond_alive = np.ones(len(self.bond_ab), dtype=bool)
        self.roles = physiology["system_role"].astype(np.uint8).copy()
        self.weights = physiology["system_weight"].astype(np.float32).copy()
        self.oxygen = 1.0
        self.nutrients = 1.0
        self.energy = 1.0
        self.age = 0.0
        self._adjacency: list[list[tuple[int, int]]] = [[] for _ in range(self.cell_count)]
        for bond_index, (a_raw, b_raw) in enumerate(self.bond_ab):
            a, b = int(a_raw), int(b_raw)
            self._adjacency[a].append((b, bond_index)); self._adjacency[b].append((a, bond_index))
        for values in self._adjacency:
            values.sort()

    def kill_cells(self, indices: np.ndarray | list[int]) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.cell_count):
            raise IndexError("Physiology cell damage is outside the body")
        self.alive[indices] = False; self.health[indices] = 0.0
        if len(self.bond_ab):
            self.bond_alive[np.isin(self.bond_ab[:, 0], indices) | np.isin(self.bond_ab[:, 1], indices)] = False

    def break_bonds(self, indices: np.ndarray | list[int]) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self.bond_ab)):
            raise IndexError("Physiology bond damage is outside the graph")
        self.bond_alive[indices] = False

    def _reachable_from(self, starts: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
        reachable = np.zeros(self.cell_count, dtype=bool)
        permitted = self.alive if allowed is None else self.alive & np.asarray(allowed, dtype=bool)
        queue: deque[int] = deque()
        for index in map(int, starts):
            if permitted[index] and not reachable[index]:
                reachable[index] = True; queue.append(index)
        while queue:
            current = queue.popleft()
            for neighbor, bond_index in self._adjacency[current]:
                if self.bond_alive[bond_index] and permitted[neighbor] and not reachable[neighbor]:
                    reachable[neighbor] = True; queue.append(neighbor)
        return reachable

    def network_delivery(self) -> dict[str, np.ndarray]:
        """Return widest-path delivery, restricted to each declared system graph."""
        result: dict[str, np.ndarray] = {}
        viability = np.divide(self.health, self.max_health, out=np.zeros_like(self.health), where=self.max_health > 0)
        perfusion = np.divide(
            self.fluid, self.fluid_reference,
            out=np.ones_like(self.fluid), where=self.fluid_reference > 1e-8,
        )
        perfusion = np.clip(perfusion, 0.0, 1.0)
        for system_id, name in enumerate(SYSTEM_NAMES):
            members = self.weights[system_id] > 0
            cores = (self.roles[system_id] == 1) & self.alive & members
            node_viability = np.minimum(viability, perfusion) if name == "circulation" else viability
            signal = np.zeros(self.cell_count, dtype=np.float32)
            queue: deque[int] = deque()
            for index in map(int, np.flatnonzero(cores)):
                signal[index] = np.float32(max(0.0, min(1.0, float(node_viability[index])))); queue.append(index)
            while queue:
                current = queue.popleft(); current_signal = float(signal[current])
                for neighbor, bond_index in self._adjacency[current]:
                    if not self.bond_alive[bond_index] or not self.alive[neighbor] or not members[neighbor]:
                        continue
                    candidate = min(current_signal, max(0.0, min(1.0, float(node_viability[neighbor]))))
                    if candidate <= float(signal[neighbor]) + 1e-7:
                        continue
                    signal[neighbor] = np.float32(candidate); queue.append(neighbor)
            result[name] = signal
        return result

    def _capacities_from_delivery(self, delivery: Mapping[str, np.ndarray]) -> dict[str, float]:
        raw: dict[str, float] = {}
        for system_id, name in enumerate(SYSTEM_NAMES):
            weights = self.weights[system_id]; members = weights > 0; cores = self.roles[system_id] == 1
            core_total = float(weights[cores].sum()); total = float(weights[members].sum())
            core_fraction = float(weights[cores & self.alive].sum()) / max(core_total, 1e-8)
            survival = float(weights[members & self.alive].sum()) / max(total, 1e-8)
            connected = float((weights * delivery[name]).sum()) / max(total, 1e-8)
            raw[name] = max(0.0, min(1.0, core_fraction ** 1.55 * (0.25 * survival + 0.75 * connected)))
        effective: dict[str, float] = {}
        for name in SYSTEM_NAMES:
            dependency = 1.0
            for required in DEPENDENCIES[name]:
                dependency *= max(0.0, effective[required]) ** 0.45
            effective[name] = max(0.0, min(1.0, raw[name] * dependency))
        return effective

    def delivery_fields(self) -> dict[str, np.ndarray]:
        """Diffuse service from valid system paths into still-bonded living tissue."""
        network = self.network_delivery(); capacity = self._capacities_from_delivery(network); result: dict[str, np.ndarray] = {}
        viability = np.divide(self.health, self.max_health, out=np.zeros_like(self.health), where=self.max_health > 0)
        for name in SYSTEM_NAMES:
            signal = np.ascontiguousarray(network[name] * np.float32(capacity[name]), dtype=np.float32)
            queue: deque[int] = deque(map(int, np.flatnonzero(signal > 0)))
            while queue:
                current = queue.popleft(); current_signal = float(signal[current])
                for neighbor, bond_index in self._adjacency[current]:
                    if not self.bond_alive[bond_index] or not self.alive[neighbor]:
                        continue
                    candidate = current_signal * 0.94 * max(0.0, min(1.0, float(viability[neighbor])))
                    if candidate <= float(signal[neighbor]) + 1e-5:
                        continue
                    signal[neighbor] = np.float32(candidate); queue.append(neighbor)
            result[name] = signal
        return result

    def capacities(self) -> dict[str, float]:
        return self._capacities_from_delivery(self.network_delivery())

    def step(self, dt: float, *, nutrient_input: float = 0.0) -> dict[str, float]:
        if not math.isfinite(dt) or not 0 < dt <= 0.1 or not math.isfinite(nutrient_input) or nutrient_input < 0:
            raise ValueError("Physiology step inputs are invalid")
        network = self.network_delivery(); capacity = self._capacities_from_delivery(network); delivery = self.delivery_fields(); self.nutrients += nutrient_input
        oxygen_gain = 0.65 * capacity["respiration"] * capacity["circulation"] * dt
        oxygen_use = (0.08 + 0.12 * capacity["locomotion"]) * dt
        self.oxygen = max(0.0, min(1.0, self.oxygen + oxygen_gain - oxygen_use))
        digestion = min(self.nutrients, 0.45 * capacity["digestion"] * dt)
        self.nutrients -= digestion
        produced = digestion * 2.6 * capacity["circulation"] * min(1.0, self.oxygen * 2.0)
        self.energy = max(0.0, min(4.0, self.energy + produced - 0.055 * dt))
        if self.oxygen < 0.14:
            neural_cells = self.roles[SYSTEM_NAMES.index("neural")] > 0
            damage = (0.14 - self.oxygen) * 0.7 * dt
            self.health[self.alive & neural_cells] -= np.float32(damage)
        local_repair = delivery["immune"] * delivery["circulation"] * delivery["digestion"]
        heal_rate = 0.025 * min(1.0, self.energy)
        wounded = self.alive & (self.health < self.max_health)
        if bool(wounded.any()) and heal_rate > 0:
            amount = np.minimum(self.max_health[wounded] - self.health[wounded], heal_rate * dt * local_repair[wounded])
            self.health[wounded] += amount; self.energy = max(0.0, self.energy - float(amount.sum()) * 0.03)
        newly_dead = self.alive & (self.health <= 0)
        if bool(newly_dead.any()):
            self.kill_cells(np.flatnonzero(newly_dead))
        self.age += dt
        return {**capacity, "oxygen": self.oxygen, "nutrients": self.nutrients, "energy": self.energy}
