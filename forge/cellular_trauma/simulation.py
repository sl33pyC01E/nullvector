from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from ..cellular_physiology.simulation import PhysiologyState


@dataclass(frozen=True, slots=True)
class FragmentEvent:
    cells: tuple[int, ...]
    fate: str
    age_seconds: float


class TraumaState:
    """Reference wound/fragment model over immutable cellular anatomy.

    Positions and rigid-body dynamics remain a native-runtime concern.  This
    class is the deterministic authority for wound exposure, clotting, scar
    deposition, organ-capacity cascades, bounded reconnection, and detached
    component fate.
    """

    def __init__(self, anatomy: Mapping[str, np.ndarray], physiology: Mapping[str, np.ndarray], trauma: Mapping[str, np.ndarray], profile: Mapping[str, Any]) -> None:
        self.anatomy = anatomy; self.physiology = physiology; self.trauma = trauma; self.profile = dict(profile)
        self.cell_count = len(anatomy["position_xy"]); self.bond_ab = anatomy["bond_ab"].astype(np.int64).copy()
        self.health = anatomy["max_health"].astype(np.float32).copy(); self.max_health = self.health.copy(); self.alive = np.ones(self.cell_count, dtype=bool)
        self.fluid = anatomy["fluid_initial"].astype(np.float32).copy(); self.fluid_capacity = anatomy["fluid_capacity"].astype(np.float32).copy()
        self.bond_alive = np.ones(len(self.bond_ab), dtype=bool); self.bond_wound_age = np.zeros(len(self.bond_ab), dtype=np.float32); self.wound_age = np.zeros(self.cell_count, dtype=np.float32); self.clot = np.zeros(self.cell_count, dtype=np.float32); self.scar = np.zeros(self.cell_count, dtype=np.float32)
        self.energy = float(anatomy["energy_initial"].sum()); self.fluid_lost = 0.0; self.time = 0.0
        self._component_age: dict[tuple[int, ...], float] = {}; self._terminal_fates: dict[tuple[int, ...], str] = {}; self.fragment_events: list[FragmentEvent] = []
        self._incident: list[list[int]] = [[] for _ in range(self.cell_count)]
        for bond_index, (a, b) in enumerate(self.bond_ab): self._incident[int(a)].append(bond_index); self._incident[int(b)].append(bond_index)

    def _physiology_state(self) -> PhysiologyState:
        state = PhysiologyState(self.anatomy, self.physiology); state.health = self.health.copy(); state.max_health = self.max_health.copy(); state.alive = self.alive.copy(); state.bond_alive = self.bond_alive.copy(); state.energy = self.energy
        state.fluid = self.fluid.copy(); state.fluid_reference = self.anatomy["fluid_initial"].astype(np.float32).copy()
        return state

    def capacities(self) -> dict[str, float]: return self._physiology_state().capacities()

    def damage_cells(self, indices: Iterable[int], amount: float) -> None:
        selected = np.asarray(sorted(set(map(int, indices))), dtype=np.int64)
        if selected.size == 0: return
        if selected.min() < 0 or selected.max() >= self.cell_count or amount < 0: raise ValueError("Invalid cellular trauma damage")
        self.health[selected] -= np.float32(amount); killed = selected[self.health[selected] <= 0]
        self.health[killed] = 0.0; self.alive[killed] = False; self.wound_age[selected] = np.maximum(self.wound_age[selected], np.float32(1e-6))
        killed_set = set(map(int, killed))
        for bond_index, (a, b) in enumerate(self.bond_ab):
            if int(a) in killed_set or int(b) in killed_set:
                self.bond_alive[bond_index] = False; self.bond_wound_age[bond_index] = max(float(self.bond_wound_age[bond_index]), 1e-6)

    def cut_bonds(self, indices: Iterable[int]) -> None:
        for raw in sorted(set(map(int, indices))):
            if not 0 <= raw < len(self.bond_alive): raise ValueError("Invalid cellular trauma bond")
            if self.bond_alive[raw]:
                self.bond_alive[raw] = False; self.bond_wound_age[raw] = np.float32(1e-6); a, b = map(int, self.bond_ab[raw]); self.wound_age[[a, b]] = np.maximum(self.wound_age[[a, b]], np.float32(1e-6))

    def _open_cells(self) -> np.ndarray:
        open_mask = (~self.alive).copy()
        for bond_index, (a_raw, b_raw) in enumerate(self.bond_ab):
            a, b = int(a_raw), int(b_raw)
            if not self.bond_alive[bond_index] or not self.alive[a] or not self.alive[b]: open_mask[a] = True; open_mask[b] = True
        return open_mask

    def components(self) -> tuple[tuple[int, ...], ...]:
        adjacency: list[list[int]] = [[] for _ in range(self.cell_count)]
        for bond_index, (a_raw, b_raw) in enumerate(self.bond_ab):
            a, b = int(a_raw), int(b_raw)
            if self.bond_alive[bond_index] and self.alive[a] and self.alive[b]: adjacency[a].append(b); adjacency[b].append(a)
        unseen = set(map(int, np.flatnonzero(self.alive))); result: list[tuple[int, ...]] = []
        while unseen:
            seed = min(unseen); queue = deque([seed]); unseen.remove(seed); component = []
            while queue:
                current = queue.popleft(); component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in unseen: unseen.remove(neighbor); queue.append(neighbor)
            result.append(tuple(sorted(component)))
        return tuple(sorted(result, key=lambda item: (-len(item), item)))

    def main_component(self) -> tuple[int, ...]:
        components = self.components()
        if not components: return ()
        heart = set(map(int, np.flatnonzero(self.physiology["system_role"][0] == 1)))
        candidates = [component for component in components if heart.intersection(component)]
        return max(candidates or list(components), key=lambda item: (len(item), tuple(-value for value in item)))

    def magnetic_force(self, bond_index: int, separation_cells: float) -> float:
        if not 0 <= bond_index < len(self.bond_alive) or self.bond_alive[bond_index] or separation_cells < 0: return 0.0
        a, b = map(int, self.bond_ab[bond_index]); age = float(self.bond_wound_age[bond_index]); window = float(self.profile["reconnect_window_seconds"]); radius = float(self.profile["magnetic_radius_cells"])
        if not self.alive[a] or not self.alive[b] or age > window or separation_cells > radius: return 0.0
        time_gain = max(0.0, 1.0 - age / max(window, 1e-6)); distance_gain = max(0.0, 1.0 - separation_cells / max(radius, 1e-6))
        return float(self.trauma["bond_magnetic_weight"][bond_index]) * time_gain * distance_gain

    def attempt_reconnect(self, bond_index: int, separation_cells: float) -> bool:
        force = self.magnetic_force(bond_index, separation_cells)
        if force <= 0.0 or separation_cells > 0.72 or self.energy < 0.05: return False
        a, b = map(int, self.bond_ab[bond_index]); repair = float(self.trauma["bond_repair_weight"][bond_index])
        if repair < 0.05: return False
        self.bond_alive[bond_index] = True; self.bond_wound_age[bond_index] = 0.0; self.clot[[a, b]] = np.maximum(self.clot[[a, b]], np.float32(0.72)); self.scar[[a, b]] = np.clip(self.scar[[a, b]] + self.trauma["scar_bias"][[a, b]] * np.float32(0.28), 0.0, 1.0); self.energy -= 0.05
        return True

    def _advance_components(self, dt: float) -> None:
        main = set(self.main_component()); observed: dict[tuple[int, ...], float] = {}
        for component in self.components():
            if main.intersection(component): continue
            age = self._component_age.get(component, 0.0) + dt; observed[component] = age
            if component in self._terminal_fates or age < float(self.profile["reconnect_window_seconds"]): continue
            minimum = int(self.profile["polyp_min_cells"]); desired = str(self.profile["detached_fate"])
            fate = desired if desired != "biomass" and len(component) >= minimum else "biomass"
            self._terminal_fates[component] = fate; self.fragment_events.append(FragmentEvent(component, fate, age))
        self._component_age = observed
        for component, fate in tuple(self._terminal_fates.items()):
            if fate != "biomass": continue
            living = [index for index in component if self.alive[index]]
            if not living: continue
            decay = np.float32((0.08 if self.profile["family"] == "humanoid" else 0.10) * dt)
            self.health[living] -= decay; dead = np.asarray(living, dtype=np.int64)[self.health[living] <= 0]; self.health[dead] = 0.0; self.alive[dead] = False

    def step(self, dt: float) -> None:
        if not 0 < dt <= 0.25: raise ValueError("Trauma timestep outside stable range")
        self.time += dt; broken_live = ~self.bond_alive
        for bond_index, (a, b) in enumerate(self.bond_ab):
            if broken_live[bond_index] and self.alive[int(a)] and self.alive[int(b)]: self.bond_wound_age[bond_index] += np.float32(dt)
        capacity = self.capacities(); open_mask = self._open_cells(); alive_open = open_mask & self.alive
        self.wound_age[alive_open] += np.float32(dt); circulation = capacity["circulation"]; immune = capacity["immune"]
        clot_gain = self.trauma["clotting_weight"] * np.float32(float(self.profile["clot_rate"]) * circulation * (0.18 + 0.82 * immune) * dt)
        self.clot[alive_open] = np.clip(self.clot[alive_open] + clot_gain[alive_open], 0.0, 1.0)
        pressure = self.fluid / np.maximum(self.fluid_capacity, np.float32(1e-6)); leak = np.minimum(self.fluid, pressure * alive_open.astype(np.float32) * (1.0 - self.clot) * np.float32(0.42 * dt))
        self.fluid -= leak; self.fluid_lost += float(leak.sum())
        wounded = self.alive & (self.health < self.max_health); heal_rate = self.trauma["regrowth_weight"] * np.float32(0.035 * immune * circulation * dt)
        healing = np.minimum(self.max_health - self.health, heal_rate); affordable = min(1.0, self.energy / max(1e-6, float(healing.sum()) * 0.35)); healing *= np.float32(affordable); self.health[wounded] += healing[wounded]; self.energy = max(0.0, self.energy - float(healing.sum()) * 0.35)
        actively_healing = alive_open & (healing > 0); self.scar[actively_healing] = np.clip(self.scar[actively_healing] + self.trauma["scar_bias"][actively_healing] * healing[actively_healing] * np.float32(0.16), 0.0, 1.0)
        sealed = alive_open & (self.clot >= 0.995) & (self.health >= self.max_health * 0.985); self.wound_age[sealed] = 0.0
        self._advance_components(dt)

    def fragment_fates(self) -> dict[tuple[int, ...], str]: return dict(self._terminal_fates)

    def snapshot(self) -> dict[str, object]:
        capacity = self.capacities(); return {"time": round(self.time, 7), "alive_cells": int(self.alive.sum()), "intact_bonds": int(self.bond_alive.sum()), "fluid_fraction": float(self.fluid.sum() / max(1e-6, self.fluid_capacity.sum())), "fluid_lost": self.fluid_lost, "mean_clot": float(self.clot.mean()), "mean_scar": float(self.scar.mean()), "maximum_bond_wound_age": float(self.bond_wound_age.max(initial=0.0)), "component_count": len(self.components()), "fragment_fates": {",".join(map(str, key)): value for key, value in sorted(self._terminal_fates.items())}, "capacities": capacity}
