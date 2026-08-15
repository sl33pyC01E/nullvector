from __future__ import annotations

import hashlib
import math

import numpy as np

from ..creature_stage_developmental.development import DevelopedOrganism
from ..creature_stage_developmental.dynamics import _actuator_forces
from ..creature_stage_developmental.motion import pose, skin_cells
from ..creature_stage_grounded_locomotion.contract import GroundedLocomotionConfig
from ..creature_stage_grounded_locomotion.physics import (
    GroundedCycle, GroundedFrame, _edge_strain, _inverse_mass,
    _project_edges_and_contacts, _terminal_nodes, _traction,
    _vertical_axis_degrees, dominant_family, locomotor_modes, primary_mode,
)


def simulate_controlled_cycle(organism: DevelopedOrganism, contact_schedule: np.ndarray,
                              muscle_schedule: np.ndarray,
                              config: GroundedLocomotionConfig | None = None) -> GroundedCycle:
    """Execute a neural cyclic policy through the authoritative PBD solver."""

    config = config or GroundedLocomotionConfig(); modes = locomotor_modes(organism); primary = primary_mode(organism, modes)
    appendages, muscles = len(modes), len(organism.muscles)
    if contact_schedule.shape != (config.frame_count, appendages) or contact_schedule.dtype != np.bool_:
        raise ValueError("neural grounded contact schedule drifted")
    if muscle_schedule.shape != (config.frame_count, muscles) or muscle_schedule.dtype != np.float32:
        raise ValueError("neural grounded muscle schedule drifted")
    if not np.isfinite(muscle_schedule).all() or np.any((muscle_schedule < 0) | (muscle_schedule > 1)):
        raise ValueError("neural grounded muscle activation drifted")
    allowed = np.asarray([mode != "passive" for mode in modes], dtype=np.bool_)
    contact_schedule = contact_schedule & allowed[None]
    terminals = _terminal_nodes(organism)
    endpoints = [float(organism.genome.appendages[index].endpoint[1]) for index, mode in enumerate(modes) if mode != "passive"]
    ground_y = max(endpoints, default=float(organism.skeleton_nodes[:, 1].max() + 3.0))
    positions = organism.skeleton_nodes[:, :2].astype(np.float32, copy=True); velocity = np.zeros_like(positions)
    inverse_mass = _inverse_mass(organism)
    rest_lengths = np.linalg.norm(organism.skeleton_nodes[organism.skeleton_edges[:, 1], :2] - organism.skeleton_nodes[organism.skeleton_edges[:, 0], :2], axis=1).astype(np.float32)
    anchors = np.full((appendages, 2), np.nan, dtype=np.float32); previous_active = np.zeros(appendages, dtype=np.bool_)
    body_x = 0.0; body_velocity = 0.0; maximum_strain = 0.0; maximum_slip = 0.0; traction_work = 0.0; vertical_axis = 0.0
    recorded: list[GroundedFrame] = []; seam_reference = None; start_world_x = None
    anomaly_fraction = float(organism.genome.family_mix[3]); component_count = len(organism.genome.components)
    for cycle_index in range(config.settle_cycles + 1):
        for frame_index in range(config.frame_count):
            phase = frame_index / config.frame_count; authored = pose(organism, phase)
            active = contact_schedule[frame_index]
            for appendage_index in range(appendages):
                if active[appendage_index] and not previous_active[appendage_index]:
                    terminal = int(terminals[appendage_index]); anchors[appendage_index] = (positions[terminal, 0], ground_y)
                elif not active[appendage_index]:
                    anchors[appendage_index] = np.nan
            reaction = np.zeros((appendages, 2), dtype=np.float32)
            for appendage_index in np.flatnonzero(active):
                terminal = int(terminals[appendage_index])
                if modes[appendage_index] == "wheel":
                    gene = organism.genome.appendages[appendage_index]
                    local_phase = (phase + gene.phase) % 1.0
                    stance_u = local_phase / config.wheel_stance_fraction
                    desired_body_x = float(anchors[appendage_index, 0] - (gene.endpoint[0] - 4.4 * stance_u))
                else:
                    desired_body_x = float(anchors[appendage_index, 0] - authored.nodes[terminal, 0])
                reaction[appendage_index, 0] = np.clip((desired_body_x - body_x) * _traction(modes[appendage_index], config), -.42, .42)
            active_reaction = reaction[active, 0]; contact_drive = float(active_reaction.mean()) if active_reaction.size else 0.0
            field_drive = config.float_drive * anomaly_fraction * anomaly_fraction
            body_velocity = float(np.clip(body_velocity * config.body_damping + contact_drive + field_drive, -config.maximum_body_speed, config.maximum_body_speed))
            body_x += body_velocity; traction_work += abs(contact_drive * body_velocity)
            target_world = authored.nodes[:, :2].astype(np.float32, copy=True); target_world[:, 0] += body_x
            previous_positions = positions.copy(); activation = muscle_schedule[frame_index]
            for _ in range(config.substeps):
                drive = np.full((len(positions), 1), .075, dtype=np.float32); drive[:component_count] = .115
                acceleration = (target_world - positions) * drive
                acceleration += _actuator_forces(organism, positions, activation) * .72
                if primary != "float":
                    acceleration[:, 1] += config.gravity
                velocity = velocity * config.node_damping + acceleration / config.substeps
                positions += velocity / config.substeps
                _project_edges_and_contacts(organism, positions, rest_lengths, inverse_mass, target_world[0], terminals, active, anchors, config)
            velocity = velocity * .30 + (positions - previous_positions) * .70
            if not (np.isfinite(positions).all() and np.isfinite(velocity).all() and math.isfinite(body_x) and math.isfinite(body_velocity)):
                raise FloatingPointError("neural grounded physics became non-finite")
            for appendage_index in np.flatnonzero(active):
                maximum_slip = max(maximum_slip, float(np.linalg.norm(positions[int(terminals[appendage_index])] - anchors[appendage_index])))
            maximum_strain = max(maximum_strain, _edge_strain(organism, positions, rest_lengths))
            nodes_local = positions.copy(); nodes_local[:, 0] -= body_x
            vertical_axis = max(vertical_axis, _vertical_axis_degrees(organism, nodes_local))
            if cycle_index == config.settle_cycles - 1 and frame_index == 0:
                seam_reference = nodes_local.copy()
            if cycle_index == config.settle_cycles:
                if start_world_x is None:
                    start_world_x = body_x
                cells_local = skin_cells(organism, positions); cells_local[:, 0] -= body_x
                recorded.append(GroundedFrame(phase, positions.copy(), nodes_local, cells_local, velocity.copy(), body_x, body_velocity, active.copy(), anchors.copy(), reaction.copy(), activation.copy()))
            previous_active = active.copy()
    if seam_reference is None or start_world_x is None or len(recorded) != config.frame_count:
        raise RuntimeError("neural grounded physics failed to publish")
    distance = float(recorded[-1].body_world_x - start_world_x); seam = float(np.max(np.abs(recorded[0].nodes_local - seam_reference)))
    digest = hashlib.sha256(b"nullvector-neural-grounded-controlled-cycle-v1\0" + organism.identity_sha256.encode("ascii"))
    digest.update(contact_schedule.tobytes() + muscle_schedule.tobytes())
    for frame in recorded:
        digest.update(np.ascontiguousarray(frame.cells_local).tobytes())
    return GroundedCycle(organism.identity_sha256, modes, primary, tuple(recorded), float(ground_y), distance,
                         distance / max(config.frame_count - 1, 1), seam, maximum_strain, maximum_slip,
                         float(traction_work), float(vertical_axis), digest.hexdigest())
