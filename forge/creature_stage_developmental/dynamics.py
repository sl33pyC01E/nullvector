from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .development import DevelopedOrganism
from .motion import pose, skin_cells


@dataclass(frozen=True, slots=True)
class DynamicFrame:
    phase: float
    nodes: np.ndarray
    cells: np.ndarray
    velocities: np.ndarray
    muscle_activation: np.ndarray
    planted_contacts: np.ndarray


@dataclass(frozen=True, slots=True)
class DynamicCycle:
    frames: tuple[DynamicFrame, ...]
    loop_seam_max_abs: float
    maximum_edge_strain: float


def _terminal_nodes(organism: DevelopedOrganism) -> np.ndarray:
    result = np.full(len(organism.genome.appendages), -1, dtype=np.int16)
    for appendage_index in range(len(result)):
        edge_ids = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
        if edge_ids.size == 0:
            raise ValueError("developmental dynamics appendage lacks a terminal")
        result[appendage_index] = organism.skeleton_edges[int(edge_ids[-1]), 1]
    return result


def _inverse_mass(organism: DevelopedOrganism) -> np.ndarray:
    count = len(organism.skeleton_nodes)
    component_count = len(organism.genome.components)
    value = np.ones(count, dtype=np.float32)
    # Chassis and organ chunks carry more mass than distal limb segments.
    radii = np.maximum(organism.skeleton_nodes[:component_count, 2], .5)
    value[:component_count] = 1.0 / np.maximum(np.square(radii) * .16, 1.0)
    value[0] = 0.0  # vertical-lock reference; translated by its authored target
    return value


def _actuator_forces(organism: DevelopedOrganism, nodes: np.ndarray, activation: np.ndarray) -> np.ndarray:
    forces = np.zeros_like(nodes[:, :2], dtype=np.float32)
    for muscle_index, muscle in enumerate(organism.muscles):
        origin = int(muscle[0])
        insertion = int(muscle[1])
        delta = nodes[insertion, :2] - nodes[origin, :2]
        length = max(float(np.linalg.norm(delta)), 1e-6)
        normal = np.asarray([-delta[1], delta[0]], dtype=np.float32) / length
        # Antagonistic channels apply opposing moments around the same joint.
        force = normal * float(muscle[3]) * float(activation[muscle_index]) * .085
        forces[insertion] += force
        forces[origin] -= force * .35
    return forces


def _project_lengths(
    positions: np.ndarray,
    edges: np.ndarray,
    rest_lengths: np.ndarray,
    inverse_mass: np.ndarray,
    anchor: np.ndarray,
    terminal_nodes: np.ndarray,
    planted: np.ndarray,
    target_nodes: np.ndarray,
) -> None:
    for _ in range(5):
        for edge_index, (left_raw, right_raw) in enumerate(edges):
            left = int(left_raw)
            right = int(right_raw)
            delta = positions[right, :2] - positions[left, :2]
            distance = max(float(np.linalg.norm(delta)), 1e-6)
            total_weight = float(inverse_mass[left] + inverse_mass[right])
            if total_weight <= 1e-8:
                continue
            correction = delta * ((distance - float(rest_lengths[edge_index])) / distance)
            positions[left, :2] += correction * (inverse_mass[left] / total_weight)
            positions[right, :2] -= correction * (inverse_mass[right] / total_weight)
        positions[0, :2] = anchor
        for appendage_index in np.flatnonzero(planted):
            terminal = int(terminal_nodes[appendage_index])
            positions[terminal, :2] = positions[terminal, :2] * .18 + target_nodes[terminal, :2] * .82


def _edge_strain(organism: DevelopedOrganism, positions: np.ndarray, rest_lengths: np.ndarray) -> float:
    current = np.linalg.norm(
        positions[organism.skeleton_edges[:, 1], :2] - positions[organism.skeleton_edges[:, 0], :2],
        axis=1,
    )
    return float(np.max(np.abs(current - rest_lengths) / np.maximum(rest_lengths, 1e-6)))


def simulate_cycle(
    organism: DevelopedOrganism,
    *,
    frame_count: int = 72,
    settle_cycles: int = 24,
) -> DynamicCycle:
    """Damped periodic skeleton rollout driven by joint muscle channels.

    The authored gait supplies desired contacts and a weak posture reference.
    Actual node state is recurrent: inertia, antagonistic joint moments, edge
    constraints, and planted contacts determine each next frame.  Repeating the
    cyclic forcing reaches a deterministic limit cycle before publication.
    """

    if type(frame_count) is not int or not 24 <= frame_count <= 240:
        raise ValueError("developmental dynamics frame count drifted")
    if type(settle_cycles) is not int or not 4 <= settle_cycles <= 64:
        raise ValueError("developmental dynamics settling contract drifted")
    targets = tuple(pose(organism, frame / frame_count) for frame in range(frame_count))
    positions = organism.skeleton_nodes.astype(np.float32, copy=True)
    velocity = np.zeros_like(positions[:, :2], dtype=np.float32)
    inverse_mass = _inverse_mass(organism)
    terminal_nodes = _terminal_nodes(organism)
    rest_lengths = np.linalg.norm(
        organism.skeleton_nodes[organism.skeleton_edges[:, 1], :2]
        - organism.skeleton_nodes[organism.skeleton_edges[:, 0], :2],
        axis=1,
    ).astype(np.float32)
    component_count = len(organism.genome.components)
    recorded: list[DynamicFrame] = []
    seam_reference: np.ndarray | None = None
    maximum_strain = 0.0
    for cycle in range(settle_cycles + 1):
        for frame_index, target in enumerate(targets):
            previous = positions[:, :2].copy()
            node_activation = np.zeros(len(positions), dtype=np.float32)
            node_counts = np.zeros(len(positions), dtype=np.float32)
            for muscle_index, muscle in enumerate(organism.muscles):
                value = float(target.muscle_activation[muscle_index])
                for node in (int(muscle[0]), int(muscle[1])):
                    node_activation[node] += value
                    node_counts[node] += 1.0
            node_activation /= np.maximum(node_counts, 1.0)
            drive = .055 + node_activation[:, None] * .075
            drive[:component_count] += .055
            acceleration = (target.nodes[:, :2] - positions[:, :2]) * drive
            acceleration += _actuator_forces(organism, positions, target.muscle_activation)
            velocity = velocity * .72 + acceleration
            positions[:, :2] += velocity
            _project_lengths(
                positions,
                organism.skeleton_edges,
                rest_lengths,
                inverse_mass,
                target.nodes[0, :2],
                terminal_nodes,
                target.planted_contacts,
                target.nodes,
            )
            velocity = velocity * .35 + (positions[:, :2] - previous) * .65
            if not np.isfinite(positions).all() or not np.isfinite(velocity).all():
                raise FloatingPointError("developmental dynamics became non-finite")
            maximum_strain = max(maximum_strain, _edge_strain(organism, positions, rest_lengths))
            if cycle == settle_cycles - 1 and frame_index == 0:
                seam_reference = positions[:, :2].copy()
            if cycle == settle_cycles:
                recorded.append(
                    DynamicFrame(
                        phase=frame_index / frame_count,
                        nodes=positions.copy(),
                        cells=skin_cells(organism, positions),
                        velocities=velocity.copy(),
                        muscle_activation=target.muscle_activation.copy(),
                        planted_contacts=target.planted_contacts.copy(),
                    )
                )
    if seam_reference is None or len(recorded) != frame_count:
        raise RuntimeError("developmental dynamics failed to record its limit cycle")
    # Compare identical phase-zero samples from successive settled cycles.
    seam = float(np.max(np.abs(recorded[0].nodes[:, :2] - seam_reference)))
    if not math.isfinite(seam) or not math.isfinite(maximum_strain):
        raise FloatingPointError("developmental dynamics diagnostics became non-finite")
    return DynamicCycle(tuple(recorded), seam, maximum_strain)
