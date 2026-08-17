from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np

from ..creature_stage_developmental.dynamics import _actuator_forces
from ..creature_stage_developmental.motion import pose, skin_cells
from ..creature_stage_grounded_locomotion.contract import GroundedLocomotionConfig
from ..creature_stage_grounded_locomotion.physics import (
    GroundedCycle, GroundedFrame, _edge_strain, _inverse_mass, _project_edges_and_contacts,
    _terminal_nodes, _traction, _vertical_axis_degrees, dominant_family, locomotor_modes,
    primary_mode,
)


class FeedbackPolicy(Protocol):
    def predict(self, organism, nodes_local: np.ndarray, node_velocity: np.ndarray,
                previous_contact: np.ndarray, phase: float, body_velocity: float
                ) -> tuple[np.ndarray, np.ndarray, float]: ...


def simulate_feedback_cycle(organism, policy: FeedbackPolicy,
                            config: GroundedLocomotionConfig | None = None) -> GroundedCycle:
    """Execute neural decisions causally through the accepted grounded PBD solver."""
    config = config or GroundedLocomotionConfig()
    modes = locomotor_modes(organism); primary = primary_mode(organism, modes)
    terminals = _terminal_nodes(organism)
    endpoints = [float(organism.genome.appendages[i].endpoint[1]) for i, mode in enumerate(modes)
                 if mode in {"step", "drag", "wheel"}]
    ground_y = max(endpoints, default=float(organism.skeleton_nodes[:, 1].max() + 3))
    positions = organism.skeleton_nodes[:, :2].astype(np.float32, copy=True)
    velocity = np.zeros_like(positions); inverse_mass = _inverse_mass(organism)
    rest_lengths = np.linalg.norm(
        organism.skeleton_nodes[organism.skeleton_edges[:, 1], :2]
        - organism.skeleton_nodes[organism.skeleton_edges[:, 0], :2], axis=1,
    ).astype(np.float32)
    anchors = np.full((len(modes), 2), np.nan, np.float32)
    previous_active = np.zeros(len(modes), np.bool_)
    body_x = body_velocity = maximum_strain = maximum_slip = traction_work = vertical_axis = 0.0
    recorded: list[GroundedFrame] = []; seam_reference = None; start_world_x = None
    anomaly_fraction = float(organism.genome.family_mix[3]); components = len(organism.genome.components)

    for cycle_index in range(config.settle_cycles + 1):
        for frame_index in range(config.frame_count):
            phase = frame_index / config.frame_count
            authored = pose(organism, phase)
            local = positions.copy(); local[:, 0] -= body_x
            activation, active, neural_drive = policy.predict(
                organism, local, velocity, previous_active, phase, body_velocity,
            )
            activation = np.asarray(activation, np.float32); active = np.asarray(active, np.bool_)
            if activation.shape != (len(organism.muscles),) or active.shape != (len(modes),):
                raise ValueError("grounded feedback policy output drifted")
            active &= np.asarray([mode in {"step", "drag", "wheel"} for mode in modes])
            for index in range(len(modes)):
                if active[index] and not previous_active[index]:
                    terminal = int(terminals[index]); anchors[index] = (positions[terminal, 0], ground_y)
                elif not active[index]:
                    anchors[index] = np.nan
            reaction = np.zeros((len(modes), 2), np.float32)
            for index in np.flatnonzero(active):
                terminal = int(terminals[index])
                if modes[index] == "wheel":
                    gene = organism.genome.appendages[index]
                    local_phase = (phase + gene.phase) % 1.0
                    stance_u = local_phase / config.wheel_stance_fraction
                    contact_local_x = float(gene.endpoint[0] - 4.4 * stance_u)
                    desired = float(anchors[index, 0] - contact_local_x)
                else:
                    desired = float(anchors[index, 0] - authored.nodes[terminal, 0])
                reaction[index, 0] = np.clip((desired - body_x) * _traction(modes[index], config), -.42, .42)
            contact_drive = float(reaction[active, 0].mean()) if bool(active.any()) else 0.0
            # Contacts are the authority. The auxiliary neural drive head is
            # trained and reported as an interpretable intention signal, but
            # cannot inject ungrounded displacement into the world solver.
            body_velocity = body_velocity * config.body_damping + contact_drive
            body_velocity += config.float_drive * anomaly_fraction * anomaly_fraction
            body_velocity = float(np.clip(body_velocity, -config.maximum_body_speed, config.maximum_body_speed))
            body_x += body_velocity; traction_work += abs(contact_drive * body_velocity)
            target_world = authored.nodes[:, :2].astype(np.float32, copy=True); target_world[:, 0] += body_x
            previous_positions = positions.copy()
            for _ in range(config.substeps):
                direct = np.full((len(positions), 1), .075, np.float32); direct[:components] = .115
                acceleration = (target_world - positions) * direct
                acceleration += _actuator_forces(organism, positions, activation) * .72
                if primary != "float": acceleration[:, 1] += config.gravity
                velocity = velocity * config.node_damping + acceleration / config.substeps
                positions += velocity / config.substeps
                _project_edges_and_contacts(organism, positions, rest_lengths, inverse_mass,
                                            target_world[0], terminals, active, anchors, config)
            velocity = velocity * .30 + (positions - previous_positions) * .70
            if not (np.isfinite(positions).all() and np.isfinite(velocity).all() and math.isfinite(body_x)):
                raise FloatingPointError("grounded feedback physics became non-finite")
            for index in np.flatnonzero(active):
                maximum_slip = max(maximum_slip, float(np.linalg.norm(positions[int(terminals[index])] - anchors[index])))
            maximum_strain = max(maximum_strain, _edge_strain(organism, positions, rest_lengths))
            nodes_local = positions.copy(); nodes_local[:, 0] -= body_x
            vertical_axis = max(vertical_axis, _vertical_axis_degrees(organism, nodes_local))
            if cycle_index == config.settle_cycles - 1 and frame_index == 0: seam_reference = nodes_local.copy()
            if cycle_index == config.settle_cycles:
                if start_world_x is None: start_world_x = body_x
                cells = skin_cells(organism, positions); cells[:, 0] -= body_x
                recorded.append(GroundedFrame(phase, positions.copy(), nodes_local, cells, velocity.copy(),
                    body_x, body_velocity, active.copy(), anchors.copy(), reaction.copy(), activation.copy()))
            previous_active = active
    if seam_reference is None or start_world_x is None or len(recorded) != config.frame_count:
        raise RuntimeError("grounded feedback cycle incomplete")
    distance = float(recorded[-1].body_world_x - start_world_x)
    seam = float(np.max(np.abs(recorded[0].nodes_local - seam_reference)))
    digest = hashlib.sha256(b"nullvector-grounded-feedback-cycle-v2\0" + organism.identity_sha256.encode("ascii"))
    for frame in recorded:
        digest.update(np.ascontiguousarray(frame.nodes_local).tobytes())
        digest.update(np.ascontiguousarray(frame.contact_active).tobytes())
        digest.update(np.ascontiguousarray(frame.muscle_activation).tobytes())
    return GroundedCycle(organism.identity_sha256, modes, primary, tuple(recorded), float(ground_y), distance,
        distance / max(config.frame_count - 1, 1), seam, maximum_strain, maximum_slip,
        float(traction_work), float(vertical_axis), digest.hexdigest())
