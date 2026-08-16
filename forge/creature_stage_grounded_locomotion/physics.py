from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..creature_stage_developmental.contract import TRAITS
from ..creature_stage_developmental.development import DevelopedOrganism
from ..creature_stage_developmental.dynamics import _actuator_forces
from ..creature_stage_developmental.motion import pose, skin_cells
from .contract import GroundedLocomotionConfig


@dataclass(frozen=True, slots=True)
class GroundedFrame:
    phase: float
    nodes_world: np.ndarray
    nodes_local: np.ndarray
    cells_local: np.ndarray
    node_velocity: np.ndarray
    body_world_x: float
    body_velocity_x: float
    contact_active: np.ndarray
    contact_anchor_world: np.ndarray
    contact_force: np.ndarray
    muscle_activation: np.ndarray


@dataclass(frozen=True, slots=True)
class GroundedCycle:
    organism_identity_sha256: str
    modes: tuple[str, ...]
    primary_mode: str
    frames: tuple[GroundedFrame, ...]
    ground_y: float
    distance_px: float
    average_speed_px_per_frame: float
    loop_seam_max_abs: float
    maximum_edge_strain: float
    maximum_contact_slip_px: float
    traction_work: float
    vertical_axis_max_degrees: float
    identity_sha256: str


def dominant_family(organism: DevelopedOrganism) -> int:
    return int(np.argmax(np.asarray(organism.genome.family_mix, dtype=np.float32)))


def locomotor_mode(organism: DevelopedOrganism, appendage_index: int) -> str:
    """Classify locomotion by the inherited component, not by family label."""

    kind = organism.genome.appendages[appendage_index].kind
    if kind == "leg":
        return "step"
    if kind == "root":
        return "drag"
    if kind == "wheel":
        return "wheel"
    return "passive"


def locomotor_modes(organism: DevelopedOrganism) -> tuple[str, ...]:
    return tuple(locomotor_mode(organism, index) for index in range(len(organism.genome.appendages)))


def primary_mode(organism: DevelopedOrganism, modes: tuple[str, ...] | None = None) -> str:
    modes = locomotor_modes(organism) if modes is None else modes
    if "wheel" in modes:
        return "wheel"
    if "step" in modes:
        return "step"
    if "drag" in modes:
        return "drag"
    if dominant_family(organism) == 3:
        return "float"
    return "passive"


def _terminal_nodes(organism: DevelopedOrganism) -> np.ndarray:
    terminals = np.full(len(organism.genome.appendages), -1, dtype=np.int16)
    for appendage_index in range(len(terminals)):
        edge_ids = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
        if edge_ids.size < 2:
            raise ValueError("grounded appendage lacks an articulated terminal")
        terminals[appendage_index] = organism.skeleton_edges[int(edge_ids[-1]), 1]
    return terminals


def _inverse_mass(organism: DevelopedOrganism) -> np.ndarray:
    count = len(organism.skeleton_nodes)
    component_count = len(organism.genome.components)
    inverse = np.ones(count, dtype=np.float32)
    radii = np.maximum(organism.skeleton_nodes[:component_count, 2], .5)
    inverse[:component_count] = 1.0 / np.maximum(np.square(radii) * .18, 1.0)
    inverse[0] *= .32
    return inverse


def _contact_schedule(
    organism: DevelopedOrganism,
    modes: tuple[str, ...],
    phase: float,
    config: GroundedLocomotionConfig,
) -> np.ndarray:
    result = np.zeros(len(modes), dtype=np.bool_)
    for index, mode in enumerate(modes):
        local_phase = (phase + organism.genome.appendages[index].phase) % 1.0
        if mode == "step":
            result[index] = local_phase < config.step_stance_fraction
        elif mode == "drag":
            result[index] = local_phase < config.drag_stance_fraction
        elif mode == "wheel":
            result[index] = local_phase < config.wheel_stance_fraction
    return result


def _traction(mode: str, config: GroundedLocomotionConfig) -> float:
    return {
        "step": config.step_traction,
        "drag": config.drag_traction,
        "wheel": config.wheel_traction,
    }.get(mode, 0.0)


def _project_edges_and_contacts(
    organism: DevelopedOrganism,
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    inverse_mass: np.ndarray,
    target_root: np.ndarray,
    terminals: np.ndarray,
    contact_active: np.ndarray,
    anchors: np.ndarray,
    config: GroundedLocomotionConfig,
) -> None:
    for _ in range(config.edge_iterations):
        for edge_index, (left_raw, right_raw) in enumerate(organism.skeleton_edges):
            left, right = int(left_raw), int(right_raw)
            delta = positions[right] - positions[left]
            distance = max(float(np.linalg.norm(delta)), 1e-6)
            total = float(inverse_mass[left] + inverse_mass[right])
            if total <= 1e-8:
                continue
            correction = delta * ((distance - float(rest_lengths[edge_index])) / distance)
            positions[left] += correction * (inverse_mass[left] / total)
            positions[right] -= correction * (inverse_mass[right] / total)
        # This is the 2.5D vertical lock: it stabilizes the chassis origin but
        # never rotates or mirrors the organism to face its travel direction.
        positions[0] += (target_root - positions[0]) * .42
        if dominant_family(organism) == 1:
            # A low quadruped has a short top-to-bottom axis, so ordinary
            # inertial lag can look like a large sprite rotation even when it
            # is only a pixel. Keep organ/chassis nodes on the vertical 2.5D
            # axis; articulated legs retain their full lateral dynamics.
            component_count = len(organism.genome.components)
            positions[:component_count, 0] = positions[0, 0]
        for appendage_index in np.flatnonzero(contact_active):
            terminal = int(terminals[appendage_index])
            positions[terminal] = (
                positions[terminal] * config.contact_compliance
                + anchors[appendage_index] * (1.0 - config.contact_compliance)
            )


def _edge_strain(organism: DevelopedOrganism, positions: np.ndarray, rest_lengths: np.ndarray) -> float:
    current = np.linalg.norm(
        positions[organism.skeleton_edges[:, 1]]
        - positions[organism.skeleton_edges[:, 0]],
        axis=1,
    )
    return float(np.max(np.abs(current - rest_lengths) / np.maximum(rest_lengths, 1e-6)))


def _vertical_axis_degrees(organism: DevelopedOrganism, nodes_local: np.ndarray) -> float:
    components = len(organism.genome.components)
    top = int(np.argmin(organism.skeleton_nodes[:components, 1]))
    bottom = int(np.argmax(organism.skeleton_nodes[:components, 1]))
    delta = nodes_local[top] - nodes_local[bottom]
    if abs(float(delta[1])) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(float(delta[0]), -float(delta[1]))))


def simulate_grounded_cycle(
    organism: DevelopedOrganism,
    config: GroundedLocomotionConfig | None = None,
) -> GroundedCycle:
    config = config or GroundedLocomotionConfig()
    modes = locomotor_modes(organism)
    primary = primary_mode(organism, modes)
    terminals = _terminal_nodes(organism)
    contact_endpoints = [
        float(organism.genome.appendages[index].endpoint[1])
        for index, mode in enumerate(modes)
        if mode in {"step", "drag", "wheel"}
    ]
    ground_y = max(contact_endpoints, default=float(organism.skeleton_nodes[:, 1].max() + 3.0))
    positions = organism.skeleton_nodes[:, :2].astype(np.float32, copy=True)
    velocity = np.zeros_like(positions)
    inverse_mass = _inverse_mass(organism)
    rest_lengths = np.linalg.norm(
        organism.skeleton_nodes[organism.skeleton_edges[:, 1], :2]
        - organism.skeleton_nodes[organism.skeleton_edges[:, 0], :2],
        axis=1,
    ).astype(np.float32)
    anchors = np.full((len(modes), 2), np.nan, dtype=np.float32)
    previous_active = np.zeros(len(modes), dtype=np.bool_)
    body_x = 0.0
    body_velocity = 0.0
    maximum_strain = 0.0
    maximum_slip = 0.0
    traction_work = 0.0
    vertical_axis = 0.0
    recorded: list[GroundedFrame] = []
    seam_reference: np.ndarray | None = None
    start_world_x: float | None = None
    anomaly_fraction = float(organism.genome.family_mix[3])
    component_count = len(organism.genome.components)

    for cycle_index in range(config.settle_cycles + 1):
        for frame_index in range(config.frame_count):
            phase = frame_index / config.frame_count
            authored = pose(organism, phase)
            active = _contact_schedule(organism, modes, phase, config)
            for appendage_index in range(len(modes)):
                if active[appendage_index] and not previous_active[appendage_index]:
                    terminal = int(terminals[appendage_index])
                    anchors[appendage_index] = (positions[terminal, 0], ground_y)
                elif not active[appendage_index]:
                    anchors[appendage_index] = np.nan

            reaction = np.zeros((len(modes), 2), dtype=np.float32)
            for appendage_index in np.flatnonzero(active):
                terminal = int(terminals[appendage_index])
                if modes[appendage_index] == "wheel":
                    gene = organism.genome.appendages[appendage_index]
                    local_phase = (phase + gene.phase) % 1.0
                    stance_u = local_phase / config.wheel_stance_fraction
                    # A wheel's instantaneous ground patch travels backward
                    # relative to its axle while remaining fixed in the world.
                    contact_local_x = float(gene.endpoint[0] - 4.4 * stance_u)
                    desired_body_x = float(anchors[appendage_index, 0] - contact_local_x)
                else:
                    desired_body_x = float(anchors[appendage_index, 0] - authored.nodes[terminal, 0])
                error = desired_body_x - body_x
                reaction[appendage_index, 0] = np.clip(
                    error * _traction(modes[appendage_index], config), -.42, .42
                )
            active_reaction = reaction[active, 0]
            contact_drive = float(active_reaction.mean()) if active_reaction.size else 0.0
            # Native anomalies propel through a phase field.  Grafted legs or
            # wheels remain real contacts and add their ground reaction.
            field_drive = config.float_drive * anomaly_fraction * anomaly_fraction
            body_velocity = body_velocity * config.body_damping + contact_drive + field_drive
            body_velocity = float(np.clip(body_velocity, -config.maximum_body_speed, config.maximum_body_speed))
            body_x += body_velocity
            traction_work += abs(contact_drive * body_velocity)

            target_world = authored.nodes[:, :2].astype(np.float32, copy=True)
            target_world[:, 0] += body_x
            previous_positions = positions.copy()
            for _ in range(config.substeps):
                drive = np.full((len(positions), 1), .075, dtype=np.float32)
                drive[:component_count] = .115
                acceleration = (target_world - positions) * drive
                acceleration += _actuator_forces(
                    organism, positions, authored.muscle_activation
                ) * .72
                if primary != "float":
                    acceleration[:, 1] += config.gravity
                velocity = velocity * config.node_damping + acceleration / config.substeps
                positions += velocity / config.substeps
                _project_edges_and_contacts(
                    organism, positions, rest_lengths, inverse_mass,
                    target_world[0], terminals, active, anchors, config,
                )
            velocity = velocity * .30 + (positions - previous_positions) * .70
            if not (
                np.isfinite(positions).all() and np.isfinite(velocity).all()
                and math.isfinite(body_x) and math.isfinite(body_velocity)
            ):
                raise FloatingPointError("grounded locomotion became non-finite")

            for appendage_index in np.flatnonzero(active):
                terminal = int(terminals[appendage_index])
                maximum_slip = max(
                    maximum_slip,
                    float(np.linalg.norm(positions[terminal] - anchors[appendage_index])),
                )
            maximum_strain = max(maximum_strain, _edge_strain(organism, positions, rest_lengths))
            nodes_local = positions.copy()
            nodes_local[:, 0] -= body_x
            vertical_axis = max(vertical_axis, _vertical_axis_degrees(organism, nodes_local))
            if cycle_index == config.settle_cycles - 1 and frame_index == 0:
                seam_reference = nodes_local.copy()
            if cycle_index == config.settle_cycles:
                if start_world_x is None:
                    start_world_x = body_x
                cells_local = skin_cells(organism, positions)
                cells_local[:, 0] -= body_x
                recorded.append(
                    GroundedFrame(
                        phase=phase,
                        nodes_world=positions.copy(),
                        nodes_local=nodes_local,
                        cells_local=cells_local,
                        node_velocity=velocity.copy(),
                        body_world_x=body_x,
                        body_velocity_x=body_velocity,
                        contact_active=active.copy(),
                        contact_anchor_world=anchors.copy(),
                        contact_force=reaction.copy(),
                        muscle_activation=authored.muscle_activation.copy(),
                    )
                )
            previous_active = active

    if seam_reference is None or start_world_x is None or len(recorded) != config.frame_count:
        raise RuntimeError("grounded locomotion failed to publish a complete cycle")
    distance = float(recorded[-1].body_world_x - start_world_x)
    seam = float(np.max(np.abs(recorded[0].nodes_local - seam_reference)))
    digest = hashlib.sha256(b"nullvector-grounded-cycle-v1\0")
    digest.update(organism.identity_sha256.encode("ascii"))
    for frame in recorded:
        for array in (
            frame.nodes_local, frame.cells_local, frame.node_velocity,
            frame.contact_active, frame.contact_anchor_world, frame.contact_force,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(np.asarray(
            [frame.body_world_x, frame.body_velocity_x], dtype="<f8"
        ).tobytes())
    return GroundedCycle(
        organism_identity_sha256=organism.identity_sha256,
        modes=modes,
        primary_mode=primary,
        frames=tuple(recorded),
        ground_y=float(ground_y),
        distance_px=distance,
        average_speed_px_per_frame=distance / max(config.frame_count - 1, 1),
        loop_seam_max_abs=seam,
        maximum_edge_strain=maximum_strain,
        maximum_contact_slip_px=maximum_slip,
        traction_work=float(traction_work),
        vertical_axis_max_degrees=float(vertical_axis),
        identity_sha256=digest.hexdigest(),
    )
