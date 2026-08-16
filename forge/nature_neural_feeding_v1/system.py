from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from ..creature_stage_neural_grasper_v1.constraint import GraspBody, GraspConstraint, solve_grasp
from ..creature_stage_neural_grasper_v1.feeding import FoodClump, FeedingState, IntakeResult, absorb_food, metabolize_reserve
from ..creature_stage_neural_grasper_v1.runtime import NeuralGrasperRuntime
from ..creature_stage_manipulation_v1.articulation import ArticulatedBody
from ..creature_stage_neural_grasper_v1.feeding import feeder_status
from .contract import CONTROLLER, FORMAT, assert_runtime


CELLS_PER_WORLD = 12.0
MATERIAL_PROFILES = {
    "flora": (0.40, 1.00, 0.08, 0.08, 0.00),
    "biomass": (0.90, 0.78, 0.00, 0.18, 0.03),
    "mineral": (0.06, 0.00, 0.15, 0.42, 1.00),
    "charge": (0.00, 0.00, 0.04, 0.66, 1.00),
    "phase": (0.03, 0.00, 0.08, 1.00, 0.24),
}


@dataclass(slots=True)
class WorldClump:
    clump_id: int
    food: FoodClump
    cohesion: float
    source: str
    height: float = 0.0
    vertical_velocity: float = 0.0
    angle: float = 0.0
    angular_velocity: float = 0.0
    impact_mode: str = "thud"
    impacts: int = 0


@dataclass(slots=True)
class EntityFeeding:
    feeding: FeedingState
    articulation: ArticulatedBody
    constraint: GraspConstraint
    target_id: int | None = None
    identity: str = ""
    grasp_appendage: int | None = None


class NatureNeuralFeedingSystem:
    """Neural grasper + physical feeder bridge for the persistent ecology.

    World coordinates remain top-down. Anatomy, reach, feeder contact, and
    constraints are solved in cell coordinates and projected back exactly.
    """

    def __init__(self, *, seed: int = 0x464545444552, device: str = "cpu") -> None:
        assert_runtime()
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.controller = NeuralGrasperRuntime.from_checkpoint(CONTROLLER, device=device)
        self.clumps: dict[int, WorldClump] = {}
        self.entities: dict[int, EntityFeeding] = {}
        self.next_clump_id = 1
        self.absorbed_mass = 0.0
        self.throws = self.grasps = 0

    def add_clump(self, position, *, material: str, mass: float = 1.0, velocity=(0.0, 0.0), cohesion: float = .7, source: str = "environment", impact_mode: str | None = None) -> int:
        if material not in MATERIAL_PROFILES:
            raise ValueError("unknown neural feeding material")
        mode = {"phase": "bounce", "charge": "bounce", "mineral": "roll"}.get(material, "thud") if impact_mode is None else impact_mode
        if mode not in {"bounce", "roll", "thud"}:
            raise ValueError("unknown neural feeding impact mode")
        clump_id = self.next_clump_id; self.next_clump_id += 1
        radius = float(np.clip(.22 + math.sqrt(max(mass, 0)) * .16, .25, 1.2))
        food = FoodClump(np.asarray(position, np.float64), np.asarray(velocity, np.float64), float(mass), radius, 1.0, MATERIAL_PROFILES[material], material)
        self.clumps[clump_id] = WorldClump(clump_id, food, float(cohesion), source, impact_mode=mode)
        return clump_id

    def throw_clump(self, clump_id: int, horizontal_velocity, *, height: float = .46, vertical_velocity: float = 3.1) -> None:
        """Launch a persistent material clump into the 2.5D ballistic layer."""
        if clump_id not in self.clumps:
            raise ValueError("unknown neural feeding clump")
        velocity = np.asarray(horizontal_velocity, np.float64)
        if velocity.shape != (2,) or not np.isfinite(velocity).all() or float(np.linalg.norm(velocity)) > 18:
            raise ValueError("neural feeding throw velocity drifted")
        if not math.isfinite(height) or not 0 <= height <= 8 or not math.isfinite(vertical_velocity) or not 0 <= vertical_velocity <= 12:
            raise ValueError("neural feeding throw elevation drifted")
        clump = self.clumps[clump_id]
        clump.food.velocity = velocity.copy()
        clump.height = float(height)
        clump.vertical_velocity = float(vertical_velocity)
        clump.angular_velocity = float(velocity[0]) / max(clump.food.radius, .1) * .15
        self.throws += 1

    def _integrate_clump(self, clump: WorldClump, world, delta: float) -> None:
        clump.food.position = (clump.food.position + clump.food.velocity * delta) % world.size
        clump.angle = math.fmod(clump.angle + clump.angular_velocity * delta, math.tau)
        airborne = clump.height > 0 or clump.vertical_velocity > 0
        if airborne:
            clump.height += clump.vertical_velocity * delta
            clump.vertical_velocity -= 4.5 * delta
            clump.food.velocity *= math.exp(-delta * .10)
            if clump.height <= 0:
                clump.height = 0.0; clump.impacts += 1
                impact_speed = max(0.0, -clump.vertical_velocity)
                if clump.impact_mode == "bounce" and impact_speed > .42:
                    clump.vertical_velocity = impact_speed * .58
                    clump.food.velocity *= .88
                    clump.angular_velocity += float(clump.food.velocity[0]) * .12
                elif clump.impact_mode == "roll":
                    clump.vertical_velocity = 0.0; clump.food.velocity *= .92
                    clump.angular_velocity = float(clump.food.velocity[0]) / max(clump.food.radius, .1)
                else:
                    clump.vertical_velocity = 0.0; clump.food.velocity *= .28; clump.angular_velocity *= .18
        else:
            drag = {"bounce": 1.0, "roll": .34, "thud": 3.4}[clump.impact_mode]
            clump.food.velocity *= math.exp(-delta * drag)
            if clump.impact_mode == "roll":
                no_slip = float(clump.food.velocity[0]) / max(clump.food.radius, .1)
                clump.angular_velocity += (no_slip - clump.angular_velocity) * min(1.0, delta * 8.0)
            else:
                clump.angular_velocity *= math.exp(-delta * (1.2 if clump.impact_mode == "bounce" else 5.0))

    def seed_from_fields(self, world, *, per_material: int = 8) -> None:
        if self.clumps:
            return
        mapping = ((8, "flora"), (9, "biomass"), (2, "mineral"), (3, "charge"), (4, "phase"))
        for resource, material in mapping:
            flat = world.fields[resource].reshape(-1)
            candidates = np.argpartition(flat, -min(per_material * 4, flat.size))[-min(per_material * 4, flat.size):]
            candidates = sorted(map(int, candidates), key=lambda index: (-float(flat[index]), index))
            used: list[np.ndarray] = []
            for index in candidates:
                position = np.asarray((index % world.size + .5, index // world.size + .5), np.float64)
                if any(np.linalg.norm(world._delta(position, prior)) < 3 for prior in used):
                    continue
                self.add_clump(position, material=material, mass=.65 + float(flat[index]), source="seed")
                used.append(position)
                if len(used) >= per_material:
                    break

    def _entity(self, entity) -> EntityFeeding:
        identity = entity.body.organism.identity_sha256
        state = self.entities.get(entity.entity_id)
        if state is None or state.identity != identity:
            initial_fullness = float(np.clip((entity.energy - .25) * 160.0, 0, 90))
            feeding = FeedingState(reserve=float(np.clip(entity.reserve * 4, 0, 4)), fullness_seconds=initial_fullness)
            articulation = ArticulatedBody.from_organism(entity.body.organism)
            state = EntityFeeding(feeding, articulation, GraspConstraint(), identity=str(identity))
            self.entities[entity.entity_id] = state
        return state

    def _nutrition(self, entity, clump: WorldClump) -> float:
        return float(clump.food.nutrition_by_family[entity.family])

    def _nearest(self, world, entity, *, max_distance: float) -> tuple[WorldClump | None, np.ndarray | None]:
        options = []
        for clump in self.clumps.values():
            if clump.food.mass <= 1e-6 or self._nutrition(entity, clump) <= .03:
                continue
            delta = world._delta(entity.position, clump.food.position)
            distance = float(np.linalg.norm(delta))
            if distance <= max_distance:
                options.append((distance, clump.clump_id, clump, delta))
        if not options:
            return None, None
        _, _, clump, delta = min(options, key=lambda item: (item[0], item[1]))
        return clump, delta

    def forage_direction(self, world, entity) -> np.ndarray | None:
        state = self._entity(entity)
        if state.feeding.fullness_seconds > 24 and entity.energy > .72:
            return None
        clump, delta = self._nearest(world, entity, max_distance=7 + 9 * entity.genome.trait("perception"))
        if clump is None:
            return None
        state.target_id = clump.clump_id
        # Hold a body-sized standoff instead of walking the chassis over the
        # clump. The learned appendage controller owns the final cell-scale
        # reach, grasp, and feeder transfer.
        maximum_reach = max(state.articulation.length(index) for index in range(len(state.articulation.chain_ids)))
        standoff = float(np.clip(maximum_reach / CELLS_PER_WORLD * .78, .45, 1.15))
        if float(np.linalg.norm(delta)) <= standoff:
            return np.zeros(2, dtype=np.float64)
        return delta

    def step_environment(self, world, delta: float) -> None:
        self.seed_from_fields(world)
        held = {
            int(state.target_id) for state in self.entities.values()
            if state.constraint.attached and state.target_id is not None
        }
        for clump_id in sorted(tuple(self.clumps)):
            clump = self.clumps[clump_id]
            # A held clump belongs to the hand constraint.  Advancing it as an
            # independent ballistic body before the creature update produced
            # the old one-frame trailing/float artifact.
            if clump_id not in held:
                self._integrate_clump(clump, world, delta)
            if clump.food.mass <= 1e-5:
                del self.clumps[clump_id]
        # Bounded conversion of field abundance into tangible matter prevents
        # precision-starvation without flooding the active physics set.
        if world.tick_index % 240 == 0 and len(self.clumps) < 72:
            for resource, material in ((8, "flora"), (2, "mineral"), (3, "charge"), (4, "phase")):
                index = int(np.argmax(world.fields[resource]))
                y, x = divmod(index, world.size)
                if world.fields[resource, y, x] > .12:
                    self.add_clump((x + .5, y + .5), material=material, mass=.55, source="regeneration")

    def step_entity(self, world, entity, delta: float) -> dict[str, float | bool | int]:
        state = self._entity(entity)
        released = metabolize_reserve(entity.body, state.feeding, delta=delta, activity=min(1.0, float(np.linalg.norm(entity.velocity))))
        entity.energy = min(1.2, entity.energy + released * 12.0)
        entity.reserve = state.feeding.reserve / state.feeding.reserve_capacity
        if state.constraint.attached and state.target_id in self.clumps:
            clump = self.clumps[state.target_id]
            world_delta = world._delta(entity.position, clump.food.position)
        else:
            clump, world_delta = self._nearest(world, entity, max_distance=2.8)
        if clump is None or world_delta is None:
            state.constraint.attached = False; state.target_id = None; state.grasp_appendage = None
            return {"contact": False, "absorbed": 0.0, "attached": False, "target": -1}
        # Another organism may consume the last measurable fraction earlier in
        # this same world tick. Retire that remainder before constructing a
        # positive-mass physics body; ghost clumps never reach the solver.
        if not np.isfinite(clump.food.mass) or clump.food.mass <= 1e-5:
            self.clumps.pop(clump.clump_id, None)
            state.constraint.attached = False; state.target_id = None; state.grasp_appendage = None
            return {"contact": False, "absorbed": 0.0, "attached": False, "target": -1}
        state.target_id = clump.clump_id
        local_target = world_delta * CELLS_PER_WORLD
        distance_cells = float(np.linalg.norm(local_target))
        direction = local_target / max(distance_cells, 1e-8)
        command = self.controller.plan(
            entity.body.organism, target_type="material", goal="consume", direction=direction,
            distance=min(1.25, distance_cells / 24), mass=min(1.0, clump.food.mass / 4),
            cohesion=min(1.0, clump.cohesion), mobility=1.0, attached=state.constraint.attached,
        )
        appendage = min(command.appendage, len(state.articulation.chain_ids) - 1)
        if entity.family == 2:
            preferred = tuple(range(len(state.articulation.chain_ids)))
        else:
            preferred = tuple(
                index for index, gene in enumerate(entity.body.organism.genome.appendages)
                if gene.kind not in {"leg", "root", "wheel"}
            ) or tuple(range(len(state.articulation.chain_ids)))
        if appendage not in preferred:
            appendage = min(
                preferred,
                key=lambda index: (float(np.linalg.norm(state.articulation.endpoint(index) - local_target)), index),
            )
        if not state.articulation.feasible(appendage, local_target):
            feasible = [index for index in preferred if state.articulation.feasible(index, local_target)]
            if feasible:
                appendage = min(feasible, key=lambda index: (float(np.linalg.norm(state.articulation.endpoint(index) - local_target)), index))
        if state.constraint.attached and state.grasp_appendage is not None:
            appendage = state.grasp_appendage
        if state.constraint.attached:
            status = feeder_status(entity.body)
            feeder_cells = entity.body.organism.cell_xy[status.feeder_mask & entity.body.alive_mask].astype(np.float64)
            feasible_feeders = [point for point in feeder_cells if state.articulation.feasible(appendage, point, contact_radius=1.8)]
            desired = min(feasible_feeders, key=lambda point: float(np.linalg.norm(point - state.articulation.endpoint(appendage)))) if feasible_feeders else np.asarray(command.reach, np.float64) * 24
        else:
            desired = np.asarray(command.reach, np.float64) * 24
        limb_cells = state.articulation._skinned_cell_ids(appendage)
        capacity = float(np.mean(entity.body.health[limb_cells])) if limb_cells.size else 0.0
        capacity = float(np.clip(capacity * (.18 + .82 * entity.body.systems()["neural"]), 0, 1))
        effector = state.articulation.solve(
            appendage, desired, min(1.0, delta * (10 + 8 * command.force)),
            delta=delta, actuation=capacity, load=clump.food.mass if state.constraint.attached else 0.0,
        )
        body = GraspBody(np.zeros(2), entity.velocity.copy() * CELLS_PER_WORLD, max(1.0, entity.body.organism.cell_count * .08))
        target = GraspBody(local_target.copy(), clump.food.velocity.copy() * CELLS_PER_WORLD, clump.food.mass)
        was_attached = state.constraint.attached
        # Consume-mode grasp is a closed hand/contact patch. Ordinary carrying
        # cannot repeatedly tear and reattach the same clump; explicit pull/
        # cut actions retain the lower material cohesion path elsewhere.
        can_hold = capacity >= .12 and clump.food.mass <= .10 + capacity * 2.2
        result = solve_grasp(
            body, target, effector=effector,
            engage=(command.engage or state.constraint.attached) and can_hold,
            force=command.force * capacity, brace=command.brace, cohesion=100.0,
            state=state.constraint, delta=delta,
        )
        entity.velocity = body.velocity / CELLS_PER_WORLD
        if state.constraint.attached:
            # Convert the exact cell-space hand pose back into the persistent
            # toroidal world.  Position and velocity are both inherited from
            # the hand; no secondary spring is allowed to trail behind it.
            clump.food.position = (entity.position + effector / CELLS_PER_WORLD) % world.size
            clump.food.velocity = entity.velocity + state.articulation.endpoint_velocity(appendage) / CELLS_PER_WORLD
            clump.height = 0.0
            clump.vertical_velocity = 0.0
            target.position[:] = effector
            target.velocity[:] = clump.food.velocity * CELLS_PER_WORLD
        else:
            clump.food.velocity = target.velocity / CELLS_PER_WORLD
        if state.constraint.attached and not was_attached:
            self.grasps += 1
            state.grasp_appendage = appendage
        elif was_attached and not state.constraint.attached:
            state.grasp_appendage = None
        local_food = FoodClump(target.position, target.velocity, clump.food.mass, clump.food.radius * CELLS_PER_WORLD, clump.food.nutrient_density, clump.food.nutrition_by_family, clump.food.material)
        intake = absorb_food(entity.body, state.feeding, local_food, body_position=np.zeros(2), delta=delta, contact_field=1.8, intake_rate=.55) if state.constraint.attached else IntakeResult(False, False, 0.0, 0.0, state.feeding.reserve, state.feeding.fullness_seconds)
        clump.food.mass = local_food.mass
        self.absorbed_mass += intake.absorbed_mass
        if intake.absorbed_mass > 0:
            entity.consumed[8 if clump.food.material == "flora" else 9 if clump.food.material == "biomass" else 2 if clump.food.material == "mineral" else 3 if clump.food.material == "charge" else 4] += intake.absorbed_mass
        return {"contact": intake.contacted, "absorbed": intake.absorbed_mass, "attached": bool(result["attached"]), "target": clump.clump_id}

    def on_predation(self, world, predator, prey, damage: float) -> None:
        mass = max(.035, damage * prey.body.organism.cell_count * .08)
        material = "phase" if prey.family == 3 else "mineral" if prey.family == 4 else "flora" if prey.family == 2 else "biomass"
        jitter = self.rng.normal(0, .06, 2)
        self.add_clump((prey.position + jitter) % world.size, material=material, mass=mass, source=f"injury:{prey.entity_id}")

    def on_death(self, world, entity) -> None:
        material = "phase" if entity.family == 3 else "mineral" if entity.family == 4 else "flora" if entity.family == 2 else "biomass"
        self.add_clump(entity.position.copy(), material=material, mass=max(.4, entity.body.organism.cell_count * .015), cohesion=.4, source=f"death:{entity.entity_id}")

    def semantic_sha256(self) -> str:
        payload = {
            "format": FORMAT, "next": self.next_clump_id, "absorbed": round(self.absorbed_mass, 8),
            "clumps": [(item.clump_id, item.food.material, tuple(np.round(item.food.position, 8)), tuple(np.round(item.food.velocity, 8)), round(item.food.mass, 8), item.source, round(item.height, 8), round(item.vertical_velocity, 8), round(item.angle, 8), round(item.angular_velocity, 8), item.impact_mode, item.impacts) for item in sorted(self.clumps.values(), key=lambda value: value.clump_id)],
            "entities": [(key, round(value.feeding.reserve, 8), round(value.feeding.fullness_seconds, 8), value.target_id, value.constraint.attached, value.grasp_appendage) for key, value in sorted(self.entities.items())],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "format": FORMAT, "seed": self.seed, "rng": self.rng.bit_generator.state,
            "next_clump_id": self.next_clump_id, "absorbed_mass": self.absorbed_mass,
            "throws": self.throws, "grasps": self.grasps,
            "clumps": [{
                "id": item.clump_id, "position": item.food.position.tolist(), "velocity": item.food.velocity.tolist(),
                "mass": item.food.mass, "radius": item.food.radius, "density": item.food.nutrient_density,
                "nutrition": list(item.food.nutrition_by_family), "material": item.food.material,
                "cohesion": item.cohesion, "source": item.source, "height": item.height,
                "vertical_velocity": item.vertical_velocity, "angle": item.angle,
                "angular_velocity": item.angular_velocity, "impact_mode": item.impact_mode, "impacts": item.impacts,
            } for item in sorted(self.clumps.values(), key=lambda value: value.clump_id)],
            "entities": [{
                "id": entity_id, "reserve": state.feeding.reserve, "capacity": state.feeding.reserve_capacity,
                "fullness": state.feeding.fullness_seconds, "fullness_capacity": state.feeding.fullness_capacity_seconds,
                "consumed": state.feeding.consumed_mass, "posed_nodes": state.articulation.nodes.tolist(),
                "node_velocities": state.articulation.velocities.tolist(),
                "reach_targets": state.articulation.targets.tolist(),
                "reach_target_velocities": state.articulation.target_velocities.tolist(),
                "component_offsets": state.articulation.component_offsets.tolist(),
                "component_velocities": state.articulation.component_velocities.tolist(),
                "limb_attached": state.articulation.attached.tolist(),
                "detached_age": state.articulation.detached_age.tolist(),
                "articulation_elapsed": state.articulation.elapsed,
                "attached": state.constraint.attached, "target_kind": state.constraint.target_kind,
                "constraint_target": state.constraint.target_id, "strain": state.constraint.strain,
                "target_id": state.target_id, "identity": state.identity, "grasp_appendage": state.grasp_appendage,
            } for entity_id, state in sorted(self.entities.items())],
            "semantic_sha256": self.semantic_sha256(),
        }

    def restore(self, payload: dict[str, object], world=None) -> None:
        if payload.get("format") != FORMAT:
            raise ValueError("neural feeding save format drifted")
        self.rng.bit_generator.state = payload["rng"]
        self.next_clump_id = int(payload["next_clump_id"])
        self.absorbed_mass = float(payload["absorbed_mass"]); self.throws = int(payload["throws"]); self.grasps = int(payload["grasps"])
        self.clumps.clear(); self.entities.clear()
        for raw in payload["clumps"]:
            food = FoodClump(np.asarray(raw["position"]), np.asarray(raw["velocity"]), float(raw["mass"]), float(raw["radius"]), float(raw["density"]), tuple(raw["nutrition"]), str(raw["material"]))
            item = WorldClump(int(raw["id"]), food, float(raw["cohesion"]), str(raw["source"]), float(raw.get("height", 0)), float(raw.get("vertical_velocity", 0)), float(raw.get("angle", 0)), float(raw.get("angular_velocity", 0)), str(raw.get("impact_mode", {"phase": "bounce", "charge": "bounce", "mineral": "roll"}.get(str(raw["material"]), "thud"))), int(raw.get("impacts", 0))); self.clumps[item.clump_id] = item
        for raw in payload["entities"]:
            if world is None or int(raw["id"]) not in world.organisms:
                raise ValueError("neural feeding restore requires matching world anatomy")
            feeding = FeedingState(float(raw["reserve"]), float(raw["capacity"]), float(raw["fullness"]), float(raw["fullness_capacity"]), float(raw["consumed"]))
            constraint = GraspConstraint(bool(raw["attached"]), str(raw["target_kind"]), int(raw["constraint_target"]), float(raw["strain"]))
            articulation = ArticulatedBody.from_organism(world.organisms[int(raw["id"])].body.organism)
            posed_nodes = np.asarray(raw["posed_nodes"], np.float32)
            if posed_nodes.shape != articulation.nodes.shape or not np.isfinite(posed_nodes).all():
                raise ValueError("neural feeding posed anatomy drifted")
            articulation.nodes[:] = posed_nodes
            velocities = np.asarray(raw.get("node_velocities", np.zeros_like(articulation.velocities)), np.float32)
            if velocities.shape != articulation.velocities.shape or not np.isfinite(velocities).all():
                raise ValueError("neural feeding articulation velocity drifted")
            articulation.velocities[:] = velocities
            reach_targets = np.asarray(raw.get("reach_targets", articulation.targets), np.float32)
            reach_velocities = np.asarray(raw.get("reach_target_velocities", articulation.target_velocities), np.float32)
            component_offsets = np.asarray(raw.get("component_offsets", articulation.component_offsets), np.float32)
            component_velocities = np.asarray(raw.get("component_velocities", articulation.component_velocities), np.float32)
            limb_attached = np.asarray(raw.get("limb_attached", articulation.attached), np.bool_)
            detached_age = np.asarray(raw.get("detached_age", articulation.detached_age), np.float32)
            if reach_targets.shape != articulation.targets.shape or reach_velocities.shape != articulation.target_velocities.shape:
                raise ValueError("neural feeding reach governor drifted")
            if component_offsets.shape != articulation.component_offsets.shape or component_velocities.shape != articulation.component_velocities.shape:
                raise ValueError("neural feeding component posture drifted")
            if limb_attached.shape != articulation.attached.shape or detached_age.shape != articulation.detached_age.shape:
                raise ValueError("neural feeding limb state drifted")
            if not np.isfinite(reach_targets).all() or not np.isfinite(reach_velocities).all() or not np.isfinite(component_offsets).all() or not np.isfinite(component_velocities).all() or not np.isfinite(detached_age).all():
                raise ValueError("neural feeding articulation state became non-finite")
            articulation.targets[:] = reach_targets
            articulation.target_velocities[:] = reach_velocities
            articulation.component_offsets[:] = component_offsets
            articulation.component_velocities[:] = component_velocities
            component_count = len(articulation.organism.genome.components)
            articulation.nodes[:component_count, :2] = articulation.organism.skeleton_nodes[:component_count, :2] + component_offsets
            articulation.attached[:] = limb_attached
            articulation.detached_age[:] = detached_age
            articulation.elapsed = float(raw.get("articulation_elapsed", 0.0))
            self.entities[int(raw["id"])] = EntityFeeding(feeding, articulation, constraint, raw["target_id"], str(raw["identity"]), raw.get("grasp_appendage"))
        if payload.get("semantic_sha256") != self.semantic_sha256():
            raise ValueError("neural feeding save semantic hash drifted")
