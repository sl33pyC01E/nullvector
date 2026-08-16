from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..creature_stage_developmental.development import DevelopedOrganism
from ..creature_stage_neural_grasper_v1.constraint import GraspBody, GraspConstraint, solve_grasp
from ..creature_stage_neural_grasper_v1.feeding import FoodClump, FeedingState, IntakeResult, absorb_food, feeder_status
from ..creature_stage_neural_grasper_v1.runtime import NeuralGrasperRuntime
from ..living_body_substrate import LivingBody
from .articulation import ArticulatedBody
from .contract import CONTROLLER, assert_controller


@dataclass(frozen=True, slots=True)
class ManipulationStep:
    appendage: int
    attached: bool
    thrown: bool
    torn: bool
    target_distance: float
    feeder_contact: bool
    absorbed_mass: float
    reserve: float
    fullness_seconds: float


class NeuralManipulationArena:
    """Small, source-bound closed loop around the learned grasper controller.

    Coordinates are body-cell units. The neural model chooses the appendage and
    command; this arena only integrates the effector, constraint, target mass,
    recoil, contact ingestion, and drag.
    """

    def __init__(self, organism: DevelopedOrganism, *, device: str = "cpu") -> None:
        assert_controller()
        self.organism = organism
        self.living = LivingBody(organism)
        self.feeding = FeedingState()
        self.controller = NeuralGrasperRuntime.from_checkpoint(CONTROLLER, device=device)
        mass = max(1.0, organism.cell_count * .08)
        self.body = GraspBody(np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64), mass)
        self.articulation = ArticulatedBody.from_organism(organism)
        if not 1 <= len(self.articulation.chain_ids) <= 8:
            raise ValueError("manipulation arena appendage census drifted")
        self.constraint = GraspConstraint()
        self.grasp_appendage: int | None = None
        self.held_target: int | None = None
        self.targets: dict[int, FoodClump] = {}
        self.cohesion: dict[int, float] = {}
        self.next_target_id = 1

    def add_clump(self, clump: FoodClump, *, cohesion: float = .6) -> int:
        if not math.isfinite(cohesion) or not .01 <= cohesion <= 4:
            raise ValueError("manipulation target cohesion drifted")
        target_id = self.next_target_id
        self.next_target_id += 1
        self.targets[target_id] = clump
        self.cohesion[target_id] = float(cohesion)
        return target_id

    def _target_features(self, target: FoodClump) -> tuple[np.ndarray, float]:
        delta = target.position - self.body.position
        distance_cells = float(np.linalg.norm(delta))
        direction = delta / max(distance_cells, 1e-8)
        return direction, min(1.25, distance_cells / 24.0)

    def _feeder_target(self, appendage: int) -> np.ndarray:
        status = feeder_status(self.living)
        candidates = self.organism.cell_xy[status.feeder_mask & self.living.alive_mask].astype(np.float64)
        feasible = [point for point in candidates if self.articulation.feasible(appendage, point, contact_radius=1.8)]
        if not feasible:
            return np.asarray(self.organism.genome.appendages[appendage].endpoint, np.float64)
        endpoint = self.articulation.endpoint(appendage)
        return min(feasible, key=lambda point: float(np.linalg.norm(point - endpoint)))

    def step(self, target_id: int, *, goal: str, delta: float = .05, throw_strength: float = .85) -> ManipulationStep:
        if target_id not in self.targets or not math.isfinite(delta) or not .005 <= delta <= .25:
            raise ValueError("manipulation step drifted")
        target = self.targets[target_id]
        if target.mass <= 1e-8:
            self.constraint.attached = False
            self.held_target = None
            intake = IntakeResult(False, False, 0.0, 0.0, self.feeding.reserve, self.feeding.fullness_seconds)
            return ManipulationStep(0, False, False, False, 0.0, False, 0.0, intake.reserve, intake.fullness_seconds)
        direction, distance = self._target_features(target)
        attached = self.constraint.attached and self.held_target == target_id
        command = self.controller.plan(
            self.organism, target_type="material", goal=goal, direction=direction,
            distance=distance, mass=min(1.0, target.mass / 4.0),
            cohesion=min(1.0, self.cohesion[target_id]), mobility=1.0,
            throw=throw_strength if goal == "throw" else 0.0, attached=attached,
        )
        predicted_appendage = min(command.appendage, len(self.articulation.chain_ids) - 1)
        local_target = target.position - self.body.position
        if not self.articulation.feasible(predicted_appendage, local_target):
            feasible = [index for index in range(len(self.articulation.chain_ids)) if self.articulation.feasible(index, local_target)]
            if feasible:
                predicted_appendage = min(feasible, key=lambda index: (float(np.linalg.norm(self.articulation.endpoint(index) - local_target)), index))
        appendage = self.grasp_appendage if self.constraint.attached and self.grasp_appendage is not None else predicted_appendage
        desired_local = self._feeder_target(appendage) if self.constraint.attached and goal == "consume" else np.asarray(command.reach, dtype=np.float64) * 24.0
        desired = self.body.position + desired_local
        response = min(1.0, delta * (10.0 + 8.0 * command.force))
        effector = self.articulation.solve(appendage, desired - self.body.position, response) + self.body.position
        target_body = GraspBody(target.position, target.velocity, target.mass)
        release = np.asarray(command.throw_impulse, dtype=np.float64) * (6.0 * throw_strength) if command.release and goal == "throw" else None
        result = solve_grasp(
            self.body, target_body, effector=effector,
            engage=(command.engage or self.constraint.attached) and not command.release, force=command.force,
            brace=command.brace, cohesion=self.cohesion[target_id], state=self.constraint,
            delta=delta, release_impulse=release,
        )
        if result["attached"]:
            self.held_target = target_id
            self.grasp_appendage = appendage
        elif self.held_target == target_id:
            self.held_target = None
            self.grasp_appendage = None
        self.body.position += self.body.velocity * delta
        target.position += target.velocity * delta
        self.body.velocity *= math.exp(-delta * 3.2)
        target.velocity *= math.exp(-delta * (1.2 + .4 / max(target.mass, .1)))
        intake = absorb_food(
            self.living, self.feeding, target, body_position=self.body.position,
            delta=delta, contact_field=1.80, intake_rate=.55,
        )
        if intake.absorbed_mass > 0 and target.mass <= 1e-8:
            self.constraint.attached = False
            self.held_target = None
        return ManipulationStep(
            appendage, bool(result["attached"]), bool(result["thrown"]), bool(result["torn"]),
            float(np.linalg.norm(target.position - self.body.position)), intake.contacted,
            intake.absorbed_mass, intake.reserve, intake.fullness_seconds,
        )
