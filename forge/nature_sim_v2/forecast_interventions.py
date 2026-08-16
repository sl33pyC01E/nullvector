from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True, slots=True)
class ForecastIntervention:
    intervention_id: str
    label: str
    description: str
    costs: tuple[tuple[str, float], ...]
    tuned_for: tuple[str, ...]


INTERVENTIONS = (
    ForecastIntervention("seed_ark", "Seed ark", "Grow a wet trophic nursery from stored biomass.", (("biomass", 1.2), ("water", .8)), ("birth", "death", "quiet")),
    ForecastIntervention("climate_condenser", "Climate condenser", "Cool, oxygenate, and hydrate a threatened district.", (("metal", .8), ("water", .7)), ("climate", "death")),
    ForecastIntervention("phase_anchor", "Phase anchor", "Ground mutation pressure into a physical crystal lattice.", (("crystal", 1.0), ("metal", .45)), ("mutation", "discovery")),
    ForecastIntervention("predator_ward", "Predator ward", "Broadcast a repellant field and open an escape corridor.", (("biomass", .65), ("knowledge", .5)), ("predation", "migration")),
    ForecastIntervention("habitat_knot", "Habitat knot", "Assemble a conglomerated refuge and seed its margins.", (("rock", 1.4), ("metal", .65), ("biomass", .35)), ("colony", "construction", "migration")),
)


def intervention_offers(event: str, *, epoch: int = 0) -> tuple[ForecastIntervention, ...]:
    """Return one forecast-tuned option and two stable off-axis alternatives."""
    preferred = [item for item in INTERVENTIONS if event in item.tuned_for]
    primary = preferred[0] if preferred else INTERVENTIONS[0]
    remaining = [item for item in INTERVENTIONS if item != primary]
    digest = hashlib.sha256(f"{event}:{int(epoch)}:nullvector-intervention".encode()).digest()
    start = int.from_bytes(digest[:2], "little") % len(remaining)
    alternatives = tuple(remaining[(start + index) % len(remaining)] for index in range(2))
    return (primary,) + alternatives


def _disk(world, center: np.ndarray, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[:world.size, :world.size]
    dx = (xx - float(center[0]) + world.size * .5) % world.size - world.size * .5
    dy = (yy - float(center[1]) + world.size * .5) % world.size - world.size * .5
    return np.hypot(dx, dy) <= radius


def _spend(adventure, costs: tuple[tuple[str, float], ...]) -> None:
    missing = [f"{amount:g} {name.upper()}" for name, amount in costs if adventure.inventory.get(name, 0.0) + 1e-9 < amount]
    if missing:
        raise ValueError("INTERVENTION NEEDS // " + " + ".join(missing))
    for name, amount in costs:
        adventure.inventory[name] -= amount


def apply_intervention(world, adventure, entity, intervention: ForecastIntervention, *, forecast_event: str) -> str:
    _spend(adventure, intervention.costs)
    center = np.asarray(entity.position, np.float64)
    region = _disk(world, center, 5.5)
    affected = 0
    if intervention.intervention_id == "seed_ark":
        world.fields[0, region] = np.clip(world.fields[0, region] + .20, 0, 1)
        world.fields[1, region] = np.clip(world.fields[1, region] + .10, 0, 1)
        world.fields[8, region] = np.clip(world.fields[8, region] + .34, 0, 1)
        world.fields[9, region] = np.clip(world.fields[9, region] + .17, 0, 1)
        for offset in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            world.materials.deposit("biomass", tuple((center + offset) % world.size), .12, 1.35)
        affected = int(region.sum())
    elif intervention.intervention_id == "climate_condenser":
        world.fields[0, region] = np.clip(world.fields[0, region] + .30, 0, 1)
        world.fields[5, region] = np.clip(world.fields[5, region] + .24, 0, 1)
        world.fields[6, region] = np.clip(world.fields[6, region] - .20, 0, 1)
        world.fields[7, region] = np.clip(world.fields[7, region] - .25, 0, 1)
        world.materials.deposit("water", tuple(center), .55, 3.4)
        affected = int(region.sum())
    elif intervention.intervention_id == "phase_anchor":
        world.fields[4, region] = np.clip(world.fields[4, region] - .34, 0, 1)
        world.fields[7, region] = np.clip(world.fields[7, region] - .18, 0, 1)
        world.fields[2, region] = np.clip(world.fields[2, region] + .12, 0, 1)
        for angle in np.linspace(0, np.pi * 2, 7)[:-1]:
            point=(center+np.asarray((np.cos(angle),np.sin(angle)))*3.2)%world.size
            world.materials.deposit("crystal", tuple(point), .09, .85)
        affected = int(region.sum())
    elif intervention.intervention_id == "predator_ward":
        for other in world.organisms.values():
            if not other.alive or other.entity_id == entity.entity_id:
                continue
            delta = world._delta(center, other.position);distance = float(np.linalg.norm(delta))
            if distance <= 7 and other.genome.trait("aggression") > .48:
                outward = delta / max(distance, .25);other.velocity += outward * .9;other.intent = "flee";affected += 1
        world.fields[7, region] = np.clip(world.fields[7, region] + .035, 0, .16)
    elif intervention.intervention_id == "habitat_knot":
        origin = (np.rint(center + np.asarray((5, 0))).astype(int)) % world.size
        mask = np.zeros((world.size, world.size), np.bool_)
        for oy in range(-2, 3):
            for ox in range(-2, 3):
                if abs(ox) == 2 or abs(oy) == 2:
                    mask[(origin[1] + oy) % world.size, (origin[0] + ox) % world.size] = True
        world.materials.add_structure(mask, structure_id=1_000_000 + world.tick_index * 17 + entity.entity_id, material="rock")
        margin = _disk(world, origin.astype(float), 4.2)
        world.fields[8, margin] = np.clip(world.fields[8, margin] + .16, 0, 1)
        affected = int(mask.sum())
    else:
        raise ValueError("unknown forecast intervention")
    fit = forecast_event in intervention.tuned_for
    if fit:
        adventure.inventory["knowledge"] += .18
    adventure.score += 20 + (15 if fit else 0)
    world.events.append({"tick": world.tick_index, "type": "neural_intervention", "intervention": intervention.intervention_id, "forecast": forecast_event, "matched": fit, "affected": affected})
    return f"{'FORECAST MATCH' if fit else 'COUNTERFACTUAL'} // {intervention.label.upper()} // {affected} CELLS AFFECTED"
