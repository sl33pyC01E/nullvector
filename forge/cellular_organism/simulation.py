from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np

from .contract import CellFlag, SimulationDefaults, TissueType
from .orientation import ORIENTATION_CONTRACT_SHA256, SurfacePuddleField


@dataclass(slots=True)
class ReproductionEvent:
    parent_generation: int
    child_generation: int
    child_seed: int
    mutation_count: int
    child_genome: dict[str, object]
    transferred_energy: float


class OrganismState:
    """Deterministic CPU reference for the native pixel-cell simulation contract."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        genome: Mapping[str, object],
        *,
        defaults: SimulationDefaults = SimulationDefaults(),
    ) -> None:
        self.defaults = defaults
        self.genome = dict(genome)
        self.position = arrays["position_xy"].astype(np.float32) * float(defaults.cell_pixel_scale)
        self.velocity = np.zeros_like(self.position)
        self.max_health = arrays["max_health"].astype(np.float32).copy()
        self.health = self.max_health.copy()
        self.fluid_capacity = arrays["fluid_capacity"].astype(np.float32).copy()
        self.fluid = arrays["fluid_initial"].astype(np.float32).copy()
        self.nutrient = arrays["nutrient_initial"].astype(np.float32).copy()
        self.energy = arrays["energy_initial"].astype(np.float32).copy()
        self.mass = arrays["mass"].astype(np.float32).copy()
        self.stiffness = arrays["stiffness"].astype(np.float32).copy()
        self.tissue = arrays["tissue"].astype(np.uint8).copy()
        self.organ_id = arrays["organ_id"].astype(np.uint16).copy()
        self.flags = arrays["cell_flags"].astype(np.uint8).copy()
        self.bond_ab = arrays["bond_ab"].astype(np.int32).copy()
        self.bond_rest = arrays["bond_rest"].astype(np.float32).copy() * float(defaults.cell_pixel_scale)
        self.bond_strength = arrays["bond_strength"].astype(np.float32).copy()
        self.bond_conductance = arrays["bond_conductance"].astype(np.float32).copy()
        self.bond_alive = np.ones(len(self.bond_ab), dtype=bool)
        self.alive = np.ones(len(self.position), dtype=bool)
        self.age_seconds = 0.0
        self.food_consumed = 0.0
        self.fluid_lost = 0.0
        self.damage_taken = 0.0
        self.birth_count = 0
        self._initial_fluid = float(self.fluid.sum())
        surface_extent = max(256, int(np.ceil(float(self.position.max(initial=0)))) + 24)
        self.surface_fluid = SurfacePuddleField(surface_extent, surface_extent)

    @property
    def cell_count(self) -> int:
        return int(len(self.position))

    def apply_damage(
        self,
        center_xy: tuple[float, float],
        *,
        radius: float,
        damage: float,
        impulse: float,
    ) -> dict[str, int | float]:
        if radius <= 0 or damage < 0 or impulse < 0:
            raise ValueError("Damage radius must be positive and damage/impulse non-negative.")
        delta = self.position - np.asarray(center_xy, dtype=np.float32)[None]
        distance = np.linalg.norm(delta, axis=1)
        falloff = np.clip(1.0 - distance / radius, 0.0, 1.0).astype(np.float32)
        affected = self.alive & (falloff > 0)
        before_alive = int(self.alive.sum())
        applied = damage * falloff
        self.health[affected] -= applied[affected]
        self.damage_taken += float(applied[affected].sum())
        direction = delta / np.maximum(distance[:, None], 0.2)
        self.velocity[affected] += direction[affected] * (impulse * falloff[affected, None] / self.mass[affected, None])
        killed = affected & (self.health <= 0)
        self.alive[killed] = False
        self.health[killed] = 0
        endpoints_affected = falloff[self.bond_ab].max(axis=1)
        fracture = self.bond_alive & (
            (~self.alive[self.bond_ab[:, 0]])
            | (~self.alive[self.bond_ab[:, 1]])
            | (impulse * endpoints_affected * self.defaults.fracture_impulse_scale > self.bond_strength)
        )
        broken = int(fracture.sum())
        self.bond_alive[fracture] = False
        return {
            "affected_cells": int(affected.sum()),
            "killed_cells": before_alive - int(self.alive.sum()),
            "broken_bonds": broken,
            "damage_applied": float(applied[affected].sum()),
        }

    def feed(self, amount: float) -> float:
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Food amount must be finite and positive.")
        mouths = self.alive & ((self.flags & int(CellFlag.MOUTH)) != 0)
        digestive = self.alive & (self.tissue == int(TissueType.DIGESTIVE))
        recipients = mouths | digestive
        if not bool(recipients.any()):
            return 0.0
        each = amount / float(recipients.sum())
        self.nutrient[recipients] += each
        self.food_consumed += amount
        return amount

    def _fluid_step(self, dt: float) -> None:
        active_bonds = self.bond_alive & self.alive[self.bond_ab[:, 0]] & self.alive[self.bond_ab[:, 1]]
        if bool(active_bonds.any()):
            indices = np.where(active_bonds)[0]
            a = self.bond_ab[indices, 0]
            b = self.bond_ab[indices, 1]
            pressure_a = self.fluid[a] / np.maximum(self.fluid_capacity[a], 1e-6)
            pressure_b = self.fluid[b] / np.maximum(self.fluid_capacity[b], 1e-6)
            transfer = (pressure_a - pressure_b) * self.bond_conductance[indices] * self.defaults.fluid_diffusion_rate * dt
            transfer = np.clip(transfer, -self.fluid[b], self.fluid[a])
            delta = np.zeros_like(self.fluid)
            np.add.at(delta, a, -transfer)
            np.add.at(delta, b, transfer)
            self.fluid += delta
        incident = np.zeros(self.cell_count, dtype=np.int32)
        open_bonds = np.zeros(self.cell_count, dtype=np.int32)
        for index, (a, b) in enumerate(self.bond_ab):
            incident[a] += 1
            incident[b] += 1
            if not self.bond_alive[index]:
                open_bonds[a] += 1
                open_bonds[b] += 1
        exposure = open_bonds.astype(np.float32) / np.maximum(incident, 1)
        exposure += np.clip(1.0 - self.health / np.maximum(self.max_health, 1e-6), 0.0, 1.0)
        exposure += (~self.alive).astype(np.float32) * 2.0
        leaked = np.minimum(self.fluid, exposure * self.defaults.leak_rate * dt)
        self.fluid -= leaked
        self.fluid_lost += float(leaked.sum())
        self.surface_fluid.deposit(self.position, leaked)
        self.surface_fluid.step(dt)

    def _metabolism_step(self, dt: float) -> None:
        alive_count = max(1, int(self.alive.sum()))
        rate = float(self.genome["metabolic_rate"]) / alive_count
        digestive = self.alive & (self.tissue == int(TissueType.DIGESTIVE))
        conversion = np.minimum(self.nutrient, float(self.genome["digestion_efficiency"]) * dt)
        conversion *= digestive.astype(np.float32)
        self.nutrient -= conversion
        self.energy += conversion * 8.0
        photosynthetic = self.alive & ((self.flags & int(CellFlag.PHOTOSYNTHETIC)) != 0)
        self.energy[photosynthetic] += 0.015 * dt
        consumed = np.minimum(self.energy, rate * dt)
        consumed *= self.alive.astype(np.float32)
        self.energy -= consumed
        starving = self.alive & (self.energy <= 1e-5)
        self.health[starving] -= self.defaults.starvation_damage_rate * dt
        self.alive &= self.health > 0
        vascular = self.alive & (self.tissue == int(TissueType.VASCULAR))
        healing = vascular & (self.energy > self.defaults.regeneration_energy_cost * dt) & (self.health < self.max_health)
        amount = np.minimum(
            self.max_health - self.health,
            float(self.genome["tissue_regeneration_rate"]) * dt,
        )
        self.health[healing] += amount[healing]
        self.energy[healing] -= amount[healing] * self.defaults.regeneration_energy_cost

    def _physics_step(self, dt: float) -> None:
        active = self.bond_alive & self.alive[self.bond_ab[:, 0]] & self.alive[self.bond_ab[:, 1]]
        if bool(active.any()):
            indices = np.where(active)[0]
            a = self.bond_ab[indices, 0]
            b = self.bond_ab[indices, 1]
            delta = self.position[b] - self.position[a]
            length = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(length[:, None], 1e-4)
            strain = length / np.maximum(self.bond_rest[indices], 1e-4)
            fracture = strain > (1.25 + self.bond_strength[indices] * 0.22)
            if bool(fracture.any()):
                self.bond_alive[indices[fracture]] = False
            force = (length - self.bond_rest[indices]) * self.bond_strength[indices] * 8.0
            force[fracture] = 0.0
            impulse = direction * force[:, None] * dt
            np.add.at(self.velocity, a, impulse / self.mass[a, None])
            np.add.at(self.velocity, b, -impulse / self.mass[b, None])
        dead = ~self.alive
        if bool(dead.any()) and bool(self.alive.any()):
            center = self.position[self.alive].mean(axis=0)
            toward = center[None] - self.position[dead]
            distance = np.maximum(np.linalg.norm(toward, axis=1), 18.0)
            magnetic = np.minimum(0.0018, 0.055 / distance) * dt
            self.velocity[dead] += toward * magnetic[:, None]
        self.velocity *= self.defaults.linear_damping
        self.position += self.velocity * dt

    def step(self, dt: float, *, gravity: bool = False) -> dict[str, object]:
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Simulation dt must be finite in (0,0.1].")
        if gravity:
            raise ValueError("Uniform screen gravity is incompatible with the top-down surface contract.")
        sub_dt = dt / self.defaults.substeps
        for _ in range(self.defaults.substeps):
            self._physics_step(sub_dt)
            self._fluid_step(sub_dt)
            self._metabolism_step(sub_dt)
        self.age_seconds += dt
        return self.status()

    def can_reproduce(self) -> bool:
        essential = (
            self.alive & (self.tissue == int(TissueType.NEURAL))
        ).any() and (
            self.alive & ((self.flags & int(CellFlag.CIRCULATORY_CORE)) != 0)
        ).any() and (
            self.alive & ((self.flags & int(CellFlag.REPRODUCTIVE)) != 0)
        ).any()
        healthy = float(self.alive.sum()) / self.cell_count >= 0.75
        return bool(essential and healthy and float(self.energy.sum()) >= float(self.genome["reproduction_energy_threshold"]))

    def reproduce(self, mutation_seed: int) -> ReproductionEvent:
        if not self.can_reproduce():
            raise ValueError("Organism lacks the energy, health, or essential organs to reproduce.")
        transferred = float(self.energy.sum()) * float(self.genome["offspring_energy_fraction"])
        self.energy *= np.float32(1.0 - float(self.genome["offspring_energy_fraction"]))
        child = dict(self.genome)
        child_generation = int(self.genome.get("generation", 0)) + 1
        child["generation"] = child_generation
        child["genome_seed"] = int(mutation_seed)
        mutation_count = 0
        mutable = (
            "metabolic_rate",
            "digestion_efficiency",
            "fluid_regeneration_rate",
            "tissue_regeneration_rate",
            "reproduction_energy_threshold",
            "mutation_rate",
        )
        for trait in mutable:
            digest = hashlib.sha256(f"{mutation_seed}:{trait}".encode("ascii")).digest()
            unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
            if unit < float(self.genome["mutation_rate"]):
                signed = int.from_bytes(digest[8:16], "big") / float((1 << 64) - 1) * 2.0 - 1.0
                child[trait] = round(float(child[trait]) * (1.0 + signed * float(self.genome["mutation_scale"])), 7)
                mutation_count += 1
        self.birth_count += 1
        return ReproductionEvent(
            parent_generation=int(self.genome.get("generation", 0)),
            child_generation=child_generation,
            child_seed=int(mutation_seed),
            mutation_count=mutation_count,
            child_genome=child,
            transferred_energy=transferred,
        )

    def status(self) -> dict[str, object]:
        return {
            "age_seconds": self.age_seconds,
            "alive_cells": int(self.alive.sum()),
            "dead_cells": self.cell_count - int(self.alive.sum()),
            "intact_bonds": int(self.bond_alive.sum()),
            "broken_bonds": len(self.bond_alive) - int(self.bond_alive.sum()),
            "health_fraction": float(self.health.sum() / np.maximum(self.max_health.sum(), 1e-6)),
            "fluid_fraction": float(self.fluid.sum() / np.maximum(self.fluid_capacity.sum(), 1e-6)),
            "fluid_lost": self.fluid_lost,
            "surface_fluid": self.surface_fluid.total,
            "surface_spread_radius": self.surface_fluid.rms_radius(),
            "orientation_contract_sha256": ORIENTATION_CONTRACT_SHA256,
            "nutrients": float(self.nutrient.sum()),
            "energy": float(self.energy.sum()),
            "food_consumed": self.food_consumed,
            "damage_taken": self.damage_taken,
            "birth_count": self.birth_count,
            "can_reproduce": self.can_reproduce(),
        }
