from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..creature_stage_developmental.development import DevelopedOrganism
from ..creature_stage_neural_grasper_v1.constraint import GraspBody, GraspConstraint, solve_grasp
from ..creature_stage_neural_grasper_v1.feeding import FoodClump, FeedingState, IntakeResult, absorb_food, feeder_status
from ..creature_stage_neural_grasper_v1.runtime import NeuralGrasperRuntime
from ..creature_stage_neural_limb_pose_v1.runtime import NeuralLimbPoseDriver
from ..living_body_substrate import LivingBody
from .articulation import ArticulatedBody
from .contract import CONTROLLER, LIMB_POSE_CONTROLLER, assert_controller, assert_limb_pose_controller


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
    actuation: float = 1.0
    detached: bool = False


@dataclass(slots=True)
class TargetKinetics:
    height: float = 0.0
    vertical_velocity: float = 0.0
    angle: float = 0.0
    angular_velocity: float = 0.0
    impact_mode: str = "thud"
    impacts: int = 0


class NeuralManipulationArena:
    """Small, source-bound closed loop around the learned grasper controller.

    Coordinates are body-cell units. The neural model chooses the appendage and
    command; this arena only integrates the effector, constraint, target mass,
    recoil, contact ingestion, and drag.
    """

    def __init__(self, organism: DevelopedOrganism, *, device: str = "cpu", neural_limb_pose: bool = True) -> None:
        assert_controller()
        self.organism = organism
        self.living = LivingBody(organism)
        self.feeding = FeedingState()
        self.controller = NeuralGrasperRuntime.from_checkpoint(CONTROLLER, device=device)
        mass = max(1.0, organism.cell_count * .08)
        self.body = GraspBody(np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64), mass)
        self.articulation = ArticulatedBody.from_organism(organism)
        if neural_limb_pose:
            assert_limb_pose_controller()
            self.articulation.pose_driver = NeuralLimbPoseDriver.from_checkpoint(LIMB_POSE_CONTROLLER, device=device)
        if int(np.argmax(organism.genome.family_mix)) == 0:
            self.articulation.require_peer_limbs({"arm"}, {"leg"})
        if not 1 <= len(self.articulation.chain_ids) <= 8:
            raise ValueError("manipulation arena appendage census drifted")
        self.constraint = GraspConstraint()
        self.grasp_appendage: int | None = None
        self.held_target: int | None = None
        self.targets: dict[int, FoodClump] = {}
        self.target_kinetics: dict[int, TargetKinetics] = {}
        self.cohesion: dict[int, float] = {}
        self.next_target_id = 1

    def ground_plane_y(self) -> float:
        contacts = [
            float(gene.endpoint[1]) for gene in self.organism.genome.appendages
            if gene.kind in {"leg", "root", "wheel"}
        ]
        # This is the same skeletal contact authority used by the accepted
        # grounded locomotion rollout. Raster cells follow the constrained
        # skeleton; they do not redefine or push the contact plane.
        local = max(contacts, default=float(self.organism.skeleton_nodes[:, 1].max() + 3.0))
        return float(self.body.position[1] + local)

    @property
    def family(self) -> int:
        return int(np.argmax(self.organism.genome.family_mix))

    def acquisition_strategy(self) -> str:
        return ("kneel_grasp", "ground_bite", "root_siphon", "phase_tractor", "suspension_tool")[self.family]

    def pose_for_acquisition(self, amount: float, *, delta: float) -> None:
        """Apply a vertical-only 2.5D acquisition posture for this chassis."""
        amount = float(np.clip(amount, 0, 1))
        if self.family in (0, 4):
            capacities = [self.appendage_capacity(index) for index in self.grasper_indices()]
            posture_capacity = max(capacities, default=0.0)
            amount *= float(np.clip((posture_capacity - .08) / .62, 0.0, 1.0))
        offsets = np.zeros_like(self.articulation.component_offsets)
        for index, component in enumerate(self.organism.genome.components):
            if self.family == 0:  # translate the chassis; grounded legs supply the fold
                drop = 4.0
            elif self.family == 1:  # articulated neck fold into a ground bite
                drop = 1.25
            elif self.family == 4:  # compress wheel/track suspension under a tool hardpoint
                drop = 2.0
            else:
                drop = 0.0
            offsets[index, 1] = drop * amount
        self.articulation.pose_components(offsets, delta=delta, response=.92)
        # Kneeling is a contact-constrained action, not a vertical squash.
        # Re-solve authored feet/wheels against their ground anchors after the
        # chassis moves, matching the grounded locomotion vocabulary.
        for appendage, gene in enumerate(self.organism.genome.appendages):
            if gene.kind not in {"leg", "wheel"} or not self.articulation.attached[appendage]:
                continue
            self.articulation.solve(
                appendage,
                np.asarray(gene.endpoint, np.float64),
                .72,
                delta=delta,
                actuation=self.appendage_capacity(appendage),
            )

    def posed_feeder_points(self) -> np.ndarray:
        status = feeder_status(self.living)
        return self.articulation.cells(floor_y=self.ground_plane_y() - self.body.position[1])[status.feeder_mask & self.living.alive_mask].astype(np.float64) + self.body.position

    def grasper_indices(self) -> tuple[int, ...]:
        # Plant roots are both locomotors and literal feeder/manipulators.
        if int(np.argmax(self.organism.genome.family_mix)) == 2:
            return tuple(range(len(self.organism.genome.appendages)))
        non_locomotor = tuple(
            index for index, gene in enumerate(self.organism.genome.appendages)
            if gene.kind not in {"leg", "root", "wheel"}
        )
        return non_locomotor or tuple(range(len(self.organism.genome.appendages)))

    def appendage_capacity(self, appendage: int) -> float:
        if not self.articulation.attached[appendage]:
            return 0.0
        cells = self.articulation._skinned_cell_ids(appendage)
        if cells.size == 0:
            return 0.0
        connected = self.living._connected_to_core()[cells]
        structural = float(np.mean(self.living.health[cells] * connected))
        neural = float(self.living.systems()["neural"])
        return float(np.clip(structural * (.18 + .82 * neural), 0, 1))

    def damage_appendage(self, appendage: int, *, remaining_health: float = .22) -> None:
        if not 0 <= remaining_health <= 1:
            raise ValueError("appendage damage drifted")
        cells = self.articulation._skinned_cell_ids(appendage)
        self.living.health[cells] = np.minimum(self.living.health[cells], np.float32(remaining_health))
        wounded = cells[self.living.fluid[cells] > 0]
        self.living._emit_leaks(wounded[:: max(1, len(wounded) // 6)], impulse=.45)

    def sever_appendage(self, appendage: int, *, impulse: tuple[float, float] = (.6, -1.2)) -> None:
        """Break the root bridge while preserving the detached limb's cells."""
        cells = self.articulation._skinned_cell_ids(appendage)
        if cells.size == 0:
            raise ValueError("cannot sever an empty appendage")
        root = self.articulation.root(appendage)
        distance = np.linalg.norm(self.organism.cell_xy[cells] - root, axis=1)
        bridge = cells[distance <= max(1.35, float(np.quantile(distance, .16)))]
        self.living.health[bridge] = 0
        self.living._emit_leaks(bridge, impulse=.9)
        dropped_target = self.held_target if self.grasp_appendage == appendage else None
        if dropped_target is not None and dropped_target in self.targets:
            target = self.targets[dropped_target]
            kinetics = self.target_kinetics[dropped_target]
            plane = self.ground_plane_y()
            kinetics.height = max(.0, plane - float(target.position[1]))
            kinetics.vertical_velocity = 0.0
            target.position[1] = plane
            target.velocity[:] = self.articulation.endpoint_velocity(appendage) + self.body.velocity
            target.velocity[1] = 0.0
        self.articulation.sever(appendage, impulse=np.asarray(impulse, np.float32))
        if self.grasp_appendage == appendage:
            self.constraint.attached = False
            self.held_target = None
            self.grasp_appendage = None

    def _step_detached_limbs(self, delta: float) -> None:
        for appendage in np.flatnonzero(~self.articulation.attached):
            cells = self.articulation._skinned_cell_ids(int(appendage))
            residual = float(np.mean(self.living.health[cells])) if cells.size else 0.0
            self.articulation.step_detached(
                int(appendage), delta=delta, ground_y=self.ground_plane_y(), residual=residual,
            )

    def add_clump(self, clump: FoodClump, *, cohesion: float = .6, impact_mode: str | None = None) -> int:
        if not math.isfinite(cohesion) or not .01 <= cohesion <= 4:
            raise ValueError("manipulation target cohesion drifted")
        default_mode = {"phase": "bounce", "charge": "bounce", "mineral": "roll"}.get(clump.material, "thud")
        mode = default_mode if impact_mode is None else impact_mode
        if mode not in {"bounce", "roll", "thud"}:
            raise ValueError("manipulation impact mode drifted")
        target_id = self.next_target_id
        self.next_target_id += 1
        self.targets[target_id] = clump
        self.target_kinetics[target_id] = TargetKinetics(impact_mode=mode)
        self.cohesion[target_id] = float(cohesion)
        return target_id

    def integrate_free_target(self, target_id: int, delta: float) -> None:
        """Advance planar travel plus independent 2.5D elevation and impact."""
        if target_id not in self.targets or not math.isfinite(delta) or not .001 <= delta <= .25:
            raise ValueError("free target integration drifted")
        target = self.targets[target_id]
        kinetics = self.target_kinetics[target_id]
        target.position += target.velocity * delta
        kinetics.angle = math.fmod(kinetics.angle + kinetics.angular_velocity * delta, math.tau)
        airborne = kinetics.height > 0 or kinetics.vertical_velocity > 0
        if airborne:
            kinetics.height += kinetics.vertical_velocity * delta
            kinetics.vertical_velocity -= 14.0 * delta
            target.velocity *= math.exp(-delta * .10)
            if kinetics.height <= 0:
                kinetics.height = 0.0
                kinetics.impacts += 1
                impact_speed = max(0.0, -kinetics.vertical_velocity)
                if kinetics.impact_mode == "bounce" and impact_speed > 2.2:
                    kinetics.vertical_velocity = impact_speed * .58
                    target.velocity *= .88
                    kinetics.angular_velocity += float(target.velocity[0]) * .12
                elif kinetics.impact_mode == "roll":
                    kinetics.vertical_velocity = 0.0
                    target.velocity *= .92
                    kinetics.angular_velocity = float(target.velocity[0]) / max(target.radius, .1)
                else:
                    kinetics.vertical_velocity = 0.0
                    target.velocity *= .28
                    kinetics.angular_velocity *= .18
        else:
            drag = {"bounce": 1.0, "roll": .34, "thud": 3.4}[kinetics.impact_mode]
            target.velocity *= math.exp(-delta * drag)
            if kinetics.impact_mode == "roll":
                no_slip = float(target.velocity[0]) / max(target.radius, .1)
                kinetics.angular_velocity += (no_slip - kinetics.angular_velocity) * min(1.0, delta * 8.0)
            else:
                kinetics.angular_velocity *= math.exp(-delta * (1.2 if kinetics.impact_mode == "bounce" else 5.0))

    def _target_features(self, target: FoodClump) -> tuple[np.ndarray, float]:
        delta = target.position - self.body.position
        distance_cells = float(np.linalg.norm(delta))
        direction = delta / max(distance_cells, 1e-8)
        return direction, min(1.25, distance_cells / 24.0)

    def _feeder_target(self, appendage: int) -> np.ndarray:
        status = feeder_status(self.living)
        candidates = self.articulation.cells()[status.feeder_mask & self.living.alive_mask].astype(np.float64)
        feasible = [point for point in candidates if self.articulation.feasible(appendage, point, contact_radius=1.8)]
        if not feasible:
            return np.asarray(self.organism.genome.appendages[appendage].endpoint, np.float64)
        endpoint = self.articulation.endpoint(appendage)
        return min(feasible, key=lambda point: float(np.linalg.norm(point - endpoint)))

    def _absorb_at_posed_feeder(self, target_id: int, delta: float, *, intake_rate: float | None = None) -> IntakeResult:
        """Reuse physiology while making contact against the posed cell skin."""
        target = self.targets[target_id]
        kinetics = self.target_kinetics[target_id]
        visual_position = target.position.copy()
        if not (self.constraint.attached and self.held_target == target_id):
            visual_position[1] -= kinetics.height
        posed = self.posed_feeder_points()
        contacted = bool(
            posed.size and float(np.min(np.linalg.norm(posed - visual_position, axis=1)))
            <= 1.80 + target.radius
        )
        if self.acquisition_strategy() in {"kneel_grasp", "suspension_tool"}:
            contacted = contacted and self.constraint.attached and self.held_target == target_id
        if not contacted:
            return IntakeResult(False, feeder_status(self.living).route_intact, 0.0, 0.0, self.feeding.reserve, self.feeding.fullness_seconds)
        original = target.position.copy()
        # absorb_food remains the single metabolic/route authority. Move only
        # its contact probe onto a live static feeder cell, then restore the
        # physical clump position immediately.
        static = self.organism.cell_xy[feeder_status(self.living).feeder_mask & self.living.alive_mask]
        target.position[:] = static[0].astype(np.float64) + self.body.position
        try:
            return absorb_food(
                self.living, self.feeding, target, body_position=self.body.position,
                delta=delta, contact_field=1.80,
                # Collection remains readable instead of deleting a clump on
                # first contact. Roots and phase fields metabolize gradually;
                # mouths and machine tools retain a faster physical intake.
                intake_rate=(.28, .24, .14, .22, .22)[self.family] if intake_rate is None else intake_rate,
            )
        finally:
            target.position[:] = original

    def step_family_acquisition(self, target_id: int, *, delta: float = .05) -> ManipulationStep:
        """Family-specific ground collection on top of shared cell physiology."""
        strategy = self.acquisition_strategy()
        self.pose_for_acquisition(1.0 if strategy in {"kneel_grasp", "ground_bite", "suspension_tool"} else 0.0, delta=delta)
        target = self.targets[target_id]
        kinetics = self.target_kinetics[target_id]
        if strategy in {"kneel_grasp", "suspension_tool"}:
            return self.step(target_id, goal="consume", delta=delta)
        if strategy == "phase_tractor":
            feeders = self.posed_feeder_points()
            destination = feeders.mean(axis=0)
            desired_height = max(0.0, self.ground_plane_y() - float(destination[1]))
            target.velocity[0] += (float(destination[0]) - float(target.position[0])) * delta * 3.0
            target.velocity[0] *= math.exp(-delta * 4.0)
            target.position[0] += target.velocity[0] * delta
            kinetics.vertical_velocity += (desired_height - kinetics.height) * delta * 5.0
            kinetics.vertical_velocity *= math.exp(-delta * 4.0)
            kinetics.height = max(0.0, kinetics.height + kinetics.vertical_velocity * delta)
        intake = self._absorb_at_posed_feeder(target_id, delta)
        return ManipulationStep(
            0, False, False, False,
            float(np.linalg.norm(target.position - self.body.position)), intake.contacted,
            intake.absorbed_mass, intake.reserve, intake.fullness_seconds,
        )

    def step(self, target_id: int, *, goal: str, delta: float = .05, throw_strength: float = .85) -> ManipulationStep:
        if target_id not in self.targets or not math.isfinite(delta) or not .005 <= delta <= .25:
            raise ValueError("manipulation step drifted")
        target = self.targets[target_id]
        self._step_detached_limbs(delta)
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
        preferred = self.grasper_indices()
        available = [index for index in preferred if self.articulation.attached[index]]
        if predicted_appendage not in available and available:
            predicted_appendage = min(
                available,
                key=lambda index: (float(np.linalg.norm(self.articulation.endpoint(index) - local_target)), index),
            )
        if not self.articulation.feasible(predicted_appendage, local_target):
            feasible = [index for index in available if self.articulation.feasible(index, local_target)]
            if feasible:
                predicted_appendage = min(feasible, key=lambda index: (float(np.linalg.norm(self.articulation.endpoint(index) - local_target)), index))
        appendage = self.grasp_appendage if self.constraint.attached and self.grasp_appendage is not None else predicted_appendage
        capacity = self.appendage_capacity(appendage)
        if self.constraint.attached and capacity < .08:
            self.constraint.attached = False
            self.held_target = None
            self.grasp_appendage = None
        desired_local = self._feeder_target(appendage) if self.constraint.attached and goal == "consume" else np.asarray(command.reach, dtype=np.float64) * 24.0
        if capacity < .45:
            rest_endpoint = np.asarray(self.organism.genome.appendages[appendage].endpoint, dtype=np.float64)
            useful_motion = float(np.clip(capacity / .45, 0, 1)) ** 1.7
            desired_local = rest_endpoint + (desired_local - rest_endpoint) * useful_motion
        desired = self.body.position + desired_local
        response = min(1.0, delta * (10.0 + 8.0 * command.force))
        effector = self.articulation.solve(
            appendage, desired - self.body.position, response, delta=delta,
            actuation=capacity, load=target.mass if self.constraint.attached else 0.0,
        ) + self.body.position
        target_body = GraspBody(target.position, target.velocity, target.mass)
        release = np.asarray(command.throw_impulse, dtype=np.float64) * (17.5 * throw_strength) if command.release and goal == "throw" else None
        if release is not None:
            magnitude = float(np.linalg.norm(release))
            if magnitude > 11.3:
                release *= 11.3 / magnitude
        effective_force = float(command.force * capacity)
        can_hold = capacity >= .12 and target.mass <= .10 + capacity * 2.2
        result = solve_grasp(
            self.body, target_body, effector=effector,
            engage=(command.engage or self.constraint.attached) and not command.release and can_hold,
            force=effective_force,
            # A closed feeding grip follows the hand through its curl without
            # repeatedly tearing off during normal joint inertia. Explicit
            # pull/cut actions still use the material's authored cohesion.
            brace=command.brace, cohesion=100.0 if goal == "consume" else self.cohesion[target_id], state=self.constraint,
            delta=delta, release_impulse=release,
        )
        if result["attached"]:
            self.held_target = target_id
            self.grasp_appendage = appendage
            kinetics = self.target_kinetics[target_id]
            # While held, the target already occupies the hand's authored
            # screen-space cell position. Elevation becomes independent only
            # at release; adding both would visually lift food above the hand.
            kinetics.height = 0.0
            kinetics.vertical_velocity = 0.0
            # A grasp is a positional hand constraint. The payload inherits the
            # endpoint velocity and cannot lag behind a faster arm.
            target.position[:] = effector
            target.velocity[:] = self.body.velocity + self.articulation.endpoint_velocity(appendage)
        elif self.held_target == target_id:
            self.held_target = None
            self.grasp_appendage = None
        if result["thrown"]:
            kinetics = self.target_kinetics[target_id]
            plane = self.ground_plane_y()
            kinetics.height = max(.5, plane - float(target.position[1]))
            target.position[1] = plane
            target.velocity[1] = 0.0
            kinetics.vertical_velocity = 11.5 + min(4.0, abs(float(command.throw_impulse[1])) * 4.0)
        self.body.position += self.body.velocity * delta
        self.body.velocity *= math.exp(-delta * 3.2)
        if not result["attached"]:
            self.integrate_free_target(target_id, delta)
        intake = self._absorb_at_posed_feeder(target_id, delta, intake_rate=None if goal == "consume" else .55)
        if intake.absorbed_mass > 0 and target.mass <= 1e-8:
            self.constraint.attached = False
            self.held_target = None
        return ManipulationStep(
            appendage, bool(result["attached"]), bool(result["thrown"]), bool(result["torn"]),
            float(np.linalg.norm(target.position - self.body.position)), intake.contacted,
            intake.absorbed_mass, intake.reserve, intake.fullness_seconds,
            capacity, not bool(self.articulation.attached[appendage]),
        )
