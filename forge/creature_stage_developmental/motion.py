from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contract import FAMILIES, TRAITS
from .development import DevelopedOrganism


@dataclass(frozen=True, slots=True)
class MotionPose:
    phase: float
    nodes: np.ndarray
    cells: np.ndarray
    muscle_activation: np.ndarray
    planted_contacts: np.ndarray


def _smooth(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _dominant_family(organism: DevelopedOrganism) -> int:
    return int(np.argmax(np.asarray(organism.genome.family_mix, dtype=np.float32)))


def _body_pose(organism: DevelopedOrganism, phase: float) -> np.ndarray:
    rest = organism.skeleton_nodes.copy()
    component_count = len(organism.genome.components)
    family = _dominant_family(organism)
    theta = math.tau * phase
    # Humanoid and plantlike are approved authorities.  Keep their established
    # driver byte-for-byte while the other three families receive dedicated
    # mechanics below.
    if family in (0, 2):
        sway = (0.65, 0.28)[family // 2]
        bob = (0.72, 0.34)[family // 2]
        for index in range(component_count):
            y = float(rest[index, 1])
            leverage = np.clip(abs(y) / 15.0, .18, 1.0)
            rest[index, 0] += math.sin(theta + index * .37) * sway * leverage
            rest[index, 1] -= abs(math.sin(theta + index * .21)) * bob * (0.55 + leverage * .45)
        return rest

    component_ids = {component.component_id: index for index, component in enumerate(organism.genome.components)}
    if family == 1:
        # A short quadruped chassis advances through diagonal foot pairs.  The
        # spine flex is deliberately much slower and smaller than limb travel,
        # preventing the old whole-body expansion/contraction look.
        chassis_shift = math.sin(theta * 2.0) * .16
        vertical_load = abs(math.sin(theta * 2.0)) * .42
        rest[:component_count, 0] += chassis_shift
        rest[:component_count, 1] -= vertical_load
        for name in ("neck", "head", "muzzle", "eyes"):
            if name in component_ids:
                rest[component_ids[name], 0] += math.sin(theta * 2.0 + .42) * .05
        if "haunch" in component_ids:
            rest[component_ids["haunch"], 0] -= math.sin(theta * 2.0) * .08
        return rest

    if family == 3:
        # Anomaly cores hover as one rounded mass while their components phase
        # at integer harmonics.  The motion is aperiodic-looking but exactly
        # loop closed and is intentionally unrelated to a bipedal sway.
        for index in range(component_count):
            radial = .22 + abs(float(rest[index, 1])) / 55.0
            rest[index, 0] += math.sin(theta * 2.0 + index * 1.1) * radial
            rest[index, 1] += math.cos(theta * 3.0 + index * .8) * radial * .78
        return rest

    if family == 4:
        # Rigid machines translate almost as a single chassis.  The mast and
        # hardpoints are stabilized rather than inheriting organic body sway.
        rest[:component_count, 0] += math.sin(theta) * .12
        rest[:component_count, 1] -= abs(math.sin(theta * 2.0)) * .10
        if "mast" in component_ids:
            rest[component_ids["mast"], 0] -= math.sin(theta) * .06
        return rest

    raise ValueError("developmental motion family drifted")


def _target(organism: DevelopedOrganism, appendage_index: int, phase: float) -> tuple[np.ndarray, bool]:
    gene = organism.genome.appendages[appendage_index]
    endpoint = np.asarray(gene.endpoint, dtype=np.float32)
    p = (phase + gene.phase) % 1.0
    family = _dominant_family(organism)
    theta = math.tau * p
    if family == 1 and gene.kind == "tail":
        # The centered dorsal grasper remains vertically aligned.  Its small
        # axial pulse is distinct from leg travel and never becomes contact.
        endpoint[1] += math.cos(theta) * .45
        return endpoint, False
    if family == 3 and gene.kind == "tendril":
        # Outer and middle fibers can briefly support the body.  Inner fibers
        # remain sensory/propulsive streamers, so six tendrils never collapse
        # into a six-legged animal gait.
        support = gene.root_offset[1] <= 3.1
        if support and p < .56:
            u = p / .56
            endpoint[0] += 2.6 * (.5 - u) + math.sin(math.tau * u) * .38
            return endpoint, True
        if support:
            u = _smooth((p - .56) / .44)
            endpoint[0] += 2.6 * (-.5 + u) + math.sin(math.tau * u) * .55
            endpoint[1] -= math.sin(math.pi * u) * 2.0
        else:
            endpoint[0] += math.sin(theta) * 2.35 + math.sin(theta * 2.0) * .55
            endpoint[1] += math.cos(theta) * 1.30
        return endpoint, False
    if family == 4 and gene.kind == "wheel":
        # Wheel/suspension terminals trace a shallow rolling path without the
        # high swing arc of an organic leg.
        endpoint[0] += math.sin(theta) * 1.40
        endpoint[1] += math.cos(theta) * .22
        return endpoint, True
    if family == 4 and gene.kind == "hardpoint":
        endpoint[0] += math.sin(theta) * .22
        endpoint[1] += math.cos(theta) * .08
        return endpoint, False
    locomotor = gene.kind in {"leg", "root", "wheel"} or (family == 3 and gene.kind == "tendril")
    if not locomotor:
        gain = {"arm": 2.2, "tail": 3.2, "frond": 2.8, "tendril": 3.5, "hardpoint": .55}.get(gene.kind, 1.2)
        endpoint[0] += math.sin(theta) * gain
        endpoint[1] -= abs(math.sin(theta)) * gain * .34
        return endpoint, False
    stance_fraction = (0.60, 0.50, 0.74, 0.54, 0.68)[family]
    stride = (6.2, 7.2, 3.1, 5.2, 4.4)[family]
    lift = (3.8, 4.4, 2.0, 4.1, 2.8)[family]
    if p < stance_fraction:
        u = p / stance_fraction
        endpoint[0] += stride * (.5 - u)
        return endpoint, True
    u = _smooth((p - stance_fraction) / (1.0 - stance_fraction))
    endpoint[0] += stride * (-.5 + u)
    endpoint[1] -= math.sin(math.pi * u) * lift
    return endpoint, False


def _fabrik(chain: np.ndarray, root: np.ndarray, target: np.ndarray, bend: int) -> np.ndarray:
    result = chain.astype(np.float32, copy=True)
    result[0] = root
    lengths = np.linalg.norm(chain[1:] - chain[:-1], axis=1)
    total = float(lengths.sum())
    distance = float(np.linalg.norm(target - root))
    if distance >= total - 1e-6:
        direction = (target - root) / max(distance, 1e-6)
        for index, length in enumerate(lengths, 1):
            result[index] = result[index - 1] + direction * length
        return result
    # Seed the solve with a signed bend so knees/elbows do not flip at mid-stride.
    direction = target - root
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-6)
    for index in range(1, len(result) - 1):
        t = index / (len(result) - 1)
        result[index] += normal * math.sin(math.pi * t) * bend * min(2.5, total * .16)
    for _ in range(7):
        result[-1] = target
        for index in range(len(result) - 2, -1, -1):
            delta = result[index] - result[index + 1]
            result[index] = result[index + 1] + delta / max(float(np.linalg.norm(delta)), 1e-6) * lengths[index]
        result[0] = root
        for index in range(1, len(result)):
            delta = result[index] - result[index - 1]
            result[index] = result[index - 1] + delta / max(float(np.linalg.norm(delta)), 1e-6) * lengths[index - 1]
    return result


def skin_cells(organism: DevelopedOrganism, posed_nodes: np.ndarray) -> np.ndarray:
    rest_nodes = organism.skeleton_nodes[:, :2]
    points = organism.cell_xy.astype(np.float32)
    distance = np.linalg.norm(points[:, None, :] - rest_nodes[None, :, :], axis=2)
    nearest = np.argpartition(distance, kth=min(2, distance.shape[1] - 1), axis=1)[:, :3]
    selected = np.take_along_axis(distance, nearest, axis=1)
    weights = np.exp(-selected * .72)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    delta = posed_nodes[:, :2] - rest_nodes
    selected_delta = delta[nearest]
    return points + (selected_delta * weights[:, :, None]).sum(axis=1)


def pose(organism: DevelopedOrganism, phase: float) -> MotionPose:
    if not 0.0 <= phase < 1.0 or not math.isfinite(phase):
        raise ValueError("developmental motion phase drifted")
    nodes = _body_pose(organism, phase)
    planted = np.zeros(len(organism.genome.appendages), dtype=np.bool_)
    for appendage_index, gene in enumerate(organism.genome.appendages):
        edge_ids = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
        if edge_ids.size < 2:
            raise ValueError("developmental appendage lacks a root and segment chain")
        edges = organism.skeleton_edges[edge_ids]
        root_node = int(edges[0, 1])
        chain_ids = [root_node] + [int(edge[1]) for edge in edges[1:]]
        root_component = int(edges[0, 0])
        root = nodes[root_component, :2] + np.asarray(gene.root_offset, dtype=np.float32)
        target, is_planted = _target(organism, appendage_index, phase)
        solved = _fabrik(organism.skeleton_nodes[chain_ids, :2], root, target, gene.bend)
        nodes[chain_ids, :2] = solved
        planted[appendage_index] = is_planted
    activations = np.zeros(len(organism.muscles), dtype=np.float32)
    for index, muscle in enumerate(organism.muscles):
        appendage = int(muscle[2])
        joint = int(muscle[6])
        offset = organism.genome.appendages[appendage].phase
        joint_lag = joint * .085 * (1.0 if organism.genome.appendages[appendage].side >= 0 else -1.0)
        wave = .5 + .5 * math.sin(math.tau * (phase + offset + joint_lag) + (math.pi if muscle[3] > 0 else 0.0))
        activations[index] = wave * float(muscle[4])
    cells = skin_cells(organism, nodes)
    return MotionPose(phase, nodes, cells, activations, planted)
