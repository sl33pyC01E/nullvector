from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .contract import DevelopmentalGenome, FAMILIES, TISSUES, TRAITS


@dataclass(frozen=True, slots=True)
class DevelopedOrganism:
    genome: DevelopmentalGenome
    cell_xy: np.ndarray
    tissue: np.ndarray
    component_weights: np.ndarray
    trait_fields: np.ndarray
    skeleton_nodes: np.ndarray
    skeleton_edges: np.ndarray
    skeleton_edge_appendage: np.ndarray
    skeleton_edge_side: np.ndarray
    muscles: np.ndarray
    appendage_index: np.ndarray
    side: np.ndarray
    identity_sha256: str

    @property
    def cell_count(self) -> int:
        return int(self.cell_xy.shape[0])


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denominator = max(float(np.dot(ab, ab)), 1e-8)
    t = np.clip(((points - a) @ ab) / denominator, 0.0, 1.0)
    projection = a[None] + t[:, None] * ab[None]
    return np.linalg.norm(points - projection, axis=1)


def _appendage_nodes(genome: DevelopmentalGenome, component_lookup: dict[str, int], component_nodes: list[list[float]]) -> tuple[list[list[float]], list[list[int]], list[list[float]], list[int], list[int]]:
    nodes = [row[:] for row in component_nodes]
    edges: list[list[int]] = []
    muscles: list[list[float]] = []
    edge_appendage: list[int] = []
    edge_side: list[int] = []
    # Parent component connections are part of the load-bearing soma.
    for index, component in enumerate(genome.components):
        if component.parent is not None:
            edges.append([component_lookup[component.parent], index])
            edge_appendage.append(-1)
            edge_side.append(component.side)
    for appendage_index, appendage in enumerate(genome.appendages):
        root_component_index = component_lookup[appendage.root_component]
        anchor = np.asarray(nodes[root_component_index][:2], dtype=np.float32) + np.asarray(appendage.root_offset, dtype=np.float32)
        endpoint = np.asarray(appendage.endpoint, dtype=np.float32)
        root_node = len(nodes)
        nodes.append([float(anchor[0]), float(anchor[1]), .75, .55])
        edges.append([root_component_index, root_node])
        edge_appendage.append(appendage_index)
        edge_side.append(appendage.side)
        previous = root_node
        chain_nodes: list[int] = []
        for segment in range(1, appendage.segments + 1):
            t = segment / appendage.segments
            arc = math.sin(math.pi * t) * min(3.0, np.linalg.norm(endpoint - anchor) * .18) * appendage.bend
            direction = endpoint - anchor
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
            normal /= max(float(np.linalg.norm(normal)), 1e-6)
            point = anchor * (1.0 - t) + endpoint * t + normal * arc
            node_index = len(nodes)
            nodes.append([float(point[0]), float(point[1]), .9, .65])
            edges.append([previous, node_index])
            edge_appendage.append(appendage_index)
            edge_side.append(appendage.side)
            chain_nodes.append(node_index)
            previous = node_index
        if chain_nodes:
            # Every articulated joint receives its own antagonistic pair.  Each
            # actuator spans the joint (node before -> node after) and records a
            # joint ordinal for the controller.  This is deliberately richer
            # than one muscle pair spanning an entire appendage.
            strength = np.clip(genome.traits[TRAITS.index("muscle_strength")] + appendage.trait_delta[TRAITS.index("muscle_strength")], 0.0, 1.0)
            joint_nodes = [root_component_index, root_node] + chain_nodes
            for joint in range(appendage.segments):
                origin = joint_nodes[joint]
                insertion = joint_nodes[joint + 2]
                local_strength = float(strength * (1.0 - .08 * joint))
                muscles.append([origin, insertion, float(appendage_index), -1.0, local_strength, float(appendage.side), float(joint)])
                muscles.append([origin, insertion, float(appendage_index), 1.0, local_strength, float(appendage.side), float(joint)])
    return nodes, edges, muscles, edge_appendage, edge_side


def develop(genome: DevelopmentalGenome) -> DevelopedOrganism:
    component_lookup = {component.component_id: index for index, component in enumerate(genome.components)}
    component_nodes = [[component.anchor[0], component.anchor[1], max(component.radius), 1.0] for component in genome.components]
    nodes, edges, muscles, edge_appendage, edge_side = _appendage_nodes(genome, component_lookup, component_nodes)
    nodes_array = np.asarray(nodes, dtype=np.float32)
    edges_array = np.asarray(edges, dtype=np.int16).reshape(-1, 2)
    muscles_array = np.asarray(muscles, dtype=np.float32).reshape(-1, 7)
    minimum = np.floor(nodes_array[:, :2].min(axis=0) - 6).astype(np.int16)
    maximum = np.ceil(nodes_array[:, :2].max(axis=0) + 6).astype(np.int16)
    if np.any(maximum - minimum > 64):
        raise ValueError("developmental organism escaped 64-cell review extent")
    yy, xx = np.mgrid[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1]
    points = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1).astype(np.float32)
    dominant_family = int(np.argmax(genome.family_mix))
    component_raw = np.zeros((points.shape[0], len(genome.components)), dtype=np.float32)
    occupancy = np.zeros(points.shape[0], dtype=np.float32)
    for index, component in enumerate(genome.components):
        delta = (points - np.asarray(component.anchor, dtype=np.float32)) / np.asarray(component.radius, dtype=np.float32)
        # Machine components grow from rectilinear superellipse fields; organic
        # and anomalous components keep smooth radial development.
        radius_squared = (
            np.square(np.max(np.abs(delta), axis=1))
            if dominant_family == 4
            else np.square(delta).sum(axis=1)
        )
        field = np.exp(-radius_squared * 1.65)
        component_raw[:, index] = field
        occupancy = np.maximum(occupancy, np.exp(-radius_squared * 1.15))
    edge_distance = np.full(points.shape[0], 999.0, dtype=np.float32)
    edge_owner = np.full(points.shape[0], -1, dtype=np.int16)
    edge_side_cell = np.zeros(points.shape[0], dtype=np.int8)
    for edge_index, edge in enumerate(edges_array):
        distance = _point_segment_distance(points, nodes_array[edge[0], :2], nodes_array[edge[1], :2])
        update = distance < edge_distance
        edge_distance[update] = distance[update]
        edge_owner[update] = edge_appendage[edge_index]
        edge_side_cell[update] = edge_side[edge_index]
        tube_radius = 1.35
        appendage_index = edge_appendage[edge_index]
        if appendage_index >= 0:
            kind = genome.appendages[appendage_index].kind
            if dominant_family == 1 and kind == "leg":
                tube_radius = 1.02
            elif dominant_family == 0 and kind == "arm":
                tube_radius = 1.18
            elif dominant_family == 1 and kind == "tail":
                tube_radius = .82
            elif dominant_family == 3 and kind == "tendril":
                tube_radius = .76
            elif dominant_family == 4 and kind == "hardpoint":
                tube_radius = .90
        occupancy = np.maximum(occupancy, np.exp(-np.square(distance / tube_radius) * 1.3))
    active = occupancy >= .20
    points = points[active]
    component_raw = component_raw[active]
    edge_distance = edge_distance[active]
    edge_owner = edge_owner[active]
    edge_side_cell = edge_side_cell[active]
    # Diffusion is explicit: every component source contributes a Gaussian field;
    # normalized overlaps interpolate local traits and component identity.
    component_total = component_raw.sum(axis=1, keepdims=True)
    # Cells grown only around an appendage edge can sit outside every Gaussian
    # component source.  They still need a developmental authority: inherit the
    # nearest component instead of becoming an all-zero, traitless seam.
    unsupported = component_total[:, 0] <= 1e-8
    if np.any(unsupported):
        anchors = np.asarray([component.anchor for component in genome.components], dtype=np.float32)
        nearest = np.argmin(
            np.linalg.norm(points[unsupported, None, :] - anchors[None, :, :], axis=2),
            axis=1,
        )
        rows = np.flatnonzero(unsupported)
        component_raw[rows, nearest] = 1.0
        component_total = component_raw.sum(axis=1, keepdims=True)
    component_weights = component_raw / component_total
    base_traits = np.asarray(genome.traits, dtype=np.float32)
    deltas = np.asarray([component.trait_delta for component in genome.components], dtype=np.float32)
    trait_fields = np.clip(base_traits[None] + component_weights @ deltas, 0.0, 1.0)
    boundary = np.zeros(points.shape[0], dtype=bool)
    point_set = {(int(x), int(y)) for x, y in points}
    for index, (x, y) in enumerate(points.astype(np.int16)):
        boundary[index] = any((int(x + dx), int(y + dy)) not in point_set for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)))
    tissue = np.full(points.shape[0], TISSUES.index("skin"), dtype=np.uint8)
    if dominant_family == 2:
        tissue[:] = TISSUES.index("root")
    elif dominant_family == 3:
        tissue[:] = TISSUES.index("phase")
    elif dominant_family == 4:
        tissue[:] = TISSUES.index("machine")
    # Bone is the internal constraint graph; muscle forms a sheath around it.
    bone_threshold = .34 + (1.0 - trait_fields[:, TRAITS.index("bone_density")]) * .32
    muscle_outer = 1.2 + trait_fields[:, TRAITS.index("muscle_density")] * 1.25
    bone = edge_distance <= bone_threshold
    muscle = (edge_distance > bone_threshold) & (edge_distance <= muscle_outer)
    tissue[muscle] = TISSUES.index("muscle")
    tissue[bone] = TISSUES.index("bone")
    # Organ/component signals overwrite interior support but never the boundary.
    organ_tissue = {
        "brain": "neural", "phase_brain": "neural", "processor": "neural", "meristem": "neural",
        "heart": "vascular", "vascular": "vascular", "coolant_pump": "vascular",
        "lung": "respiratory", "gut": "digestive", "transmuter": "digestive",
        "eye": "sensor", "photoreceptor": "sensor", "singularity": "sensor", "optic": "sensor",
        "bulb": "storage", "battery": "storage", "orbital": "phase", "jaw": "weapon",
    }
    for component_index, component in enumerate(genome.components):
        if component.organ == "none" or component.organ not in organ_tissue:
            continue
        local = component_raw[:, component_index]
        target_tissue = organ_tissue[component.organ]
        # Eyes/photoreceptors must meet the exterior; protected organs remain
        # internal.  A 0.52 authority threshold preserves small neural clusters
        # that would otherwise be completely overwritten by their support bone.
        selected = local >= .45
        if target_tissue != "sensor":
            selected &= ~boundary
        tissue[selected] = TISSUES.index(target_tissue)
    # Armor is itself a diffusing component and is constrained to the surface.
    for component_index, component in enumerate(genome.components):
        if component.kind == "armor":
            tissue[(component_weights[:, component_index] >= .25) & boundary] = TISSUES.index("armor")
    edge_appendage_array = np.asarray(edge_appendage, dtype=np.int16)
    edge_side_array = np.asarray(edge_side, dtype=np.int8)
    payload = b"".join((points.astype("<i2").tobytes(), tissue.tobytes(), component_weights.astype("<f4").tobytes(), trait_fields.astype("<f4").tobytes(), nodes_array.astype("<f4").tobytes(), edges_array.astype("<i2").tobytes(), edge_appendage_array.astype("<i2").tobytes(), edge_side_array.tobytes(), muscles_array.astype("<f4").tobytes()))
    return DevelopedOrganism(
        genome=genome,
        cell_xy=points.astype(np.int16),
        tissue=tissue,
        component_weights=component_weights.astype(np.float32),
        trait_fields=trait_fields.astype(np.float32),
        skeleton_nodes=nodes_array,
        skeleton_edges=edges_array,
        skeleton_edge_appendage=edge_appendage_array,
        skeleton_edge_side=edge_side_array,
        muscles=muscles_array,
        appendage_index=edge_owner.astype(np.int16),
        side=edge_side_cell.astype(np.int8),
        identity_sha256=hashlib.sha256(payload).hexdigest(),
    )
