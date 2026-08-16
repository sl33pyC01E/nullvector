from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..creature_stage_developmental.development import DevelopedOrganism
@dataclass(frozen=True, slots=True)
class LimbGeometry:
    kind: str
    segments: int
    length: float
    cell_count: int


@dataclass(slots=True)
class ArticulatedBody:
    """Cell-skinned appendage chains with fixed chassis roots and bone lengths."""

    organism: DevelopedOrganism
    nodes: np.ndarray
    chain_ids: tuple[np.ndarray, ...]
    root_nodes: tuple[int, ...]
    velocities: np.ndarray

    @classmethod
    def from_organism(cls, organism: DevelopedOrganism) -> "ArticulatedBody":
        chains: list[np.ndarray] = []
        roots: list[int] = []
        for appendage_index in range(len(organism.genome.appendages)):
            edge_ids = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
            if edge_ids.size < 2:
                raise ValueError("grasper appendage has no articulated chain")
            edges = organism.skeleton_edges[edge_ids]
            roots.append(int(edges[0, 0]))
            chains.append(np.asarray([int(edges[0, 1]), *[int(edge[1]) for edge in edges[1:]]], np.int16))
        return cls(
            organism,
            organism.skeleton_nodes.copy(),
            tuple(chains),
            tuple(roots),
            np.zeros_like(organism.skeleton_nodes[:, :2], dtype=np.float32),
        )

    def endpoint(self, appendage: int) -> np.ndarray:
        return self.nodes[self.chain_ids[appendage][-1], :2].astype(np.float64)

    def solve(self, appendage: int, target: np.ndarray, response: float) -> np.ndarray:
        """Advance one inertial, length-constrained limb toward a hand target.

        This is the same physical vocabulary used by grounded locomotors:
        recurrent node velocity, damped muscle drive and iterative bone-length
        projection.  A grasp target attracts the terminal hand; it does not
        overwrite a pose or teleport intermediate joints.
        """
        gene = self.organism.genome.appendages[appendage]
        chain_ids = self.chain_ids[appendage]
        root = self.nodes[self.root_nodes[appendage], :2].astype(np.float32) + np.asarray(gene.root_offset, np.float32)
        rest = self.organism.skeleton_nodes[chain_ids, :2].astype(np.float32)
        positions = self.nodes[chain_ids, :2].astype(np.float32, copy=True)
        velocity = self.velocities[chain_ids].astype(np.float32, copy=True)
        lengths = np.linalg.norm(rest[1:] - rest[:-1], axis=1).astype(np.float32)
        desired = np.asarray(target, np.float32)
        reach = float(lengths.sum())
        delta = desired - root
        distance = float(np.linalg.norm(delta))
        if distance > reach and distance > 1e-6:
            desired = root + delta * (reach / distance)
        gain = float(np.clip(response, 0.0, 1.0))
        for _ in range(3):
            previous = positions.copy()
            # Terminal muscle drive; weak distributed posture tone prevents the
            # chain folding into a numerically limp knot while retaining sway.
            velocity[-1] += (desired - positions[-1]) * (.10 + .18 * gain)
            velocity[1:-1] += (rest[1:-1] - positions[1:-1]) * .012
            velocity[1:] *= .70
            positions[1:] += velocity[1:] * (.42 + .28 * gain)
            positions[0] = root
            for _constraint in range(8):
                positions[0] = root
                for segment, length in enumerate(lengths):
                    left = segment
                    right = segment + 1
                    edge = positions[right] - positions[left]
                    current = max(float(np.linalg.norm(edge)), 1e-7)
                    correction = edge * ((current - float(length)) / current)
                    if left == 0:
                        positions[right] -= correction
                    else:
                        positions[left] += correction * .5
                        positions[right] -= correction * .5
                positions[0] = root
            velocity = velocity * .30 + (positions - previous) * .70
            velocity[0] = 0
        self.nodes[chain_ids, :2] = positions
        self.velocities[chain_ids] = velocity
        return self.endpoint(appendage)

    def cells(self) -> np.ndarray:
        points = self.organism.cell_xy.astype(np.float32, copy=True)
        for appendage, chain_ids in enumerate(self.chain_ids):
            cell_ids = self._skinned_cell_ids(appendage)
            if cell_ids.size == 0:
                continue
            rest = self.organism.skeleton_nodes[chain_ids, :2].astype(np.float32)
            posed = self.nodes[chain_ids, :2].astype(np.float32)
            cells = points[cell_ids]
            rest_vectors = rest[1:] - rest[:-1]
            denominators = np.maximum(np.square(rest_vectors).sum(axis=1), 1e-8)
            relative = cells[:, None, :] - rest[:-1][None, :, :]
            along = np.clip((relative * rest_vectors[None]).sum(axis=2) / denominators[None], 0, 1)
            projections = rest[:-1][None] + along[:, :, None] * rest_vectors[None]
            distance = np.linalg.norm(cells[:, None] - projections, axis=2)
            segment = distance.argmin(axis=1)
            row = np.arange(cell_ids.size)
            chosen_along = along[row, segment]
            rest_direction = rest_vectors[segment]
            rest_normal = np.stack((-rest_direction[:, 1], rest_direction[:, 0]), axis=1)
            rest_normal /= np.maximum(np.linalg.norm(rest_normal, axis=1, keepdims=True), 1e-8)
            lateral = ((cells - projections[row, segment]) * rest_normal).sum(axis=1)
            posed_vector = posed[segment + 1] - posed[segment]
            posed_normal = np.stack((-posed_vector[:, 1], posed_vector[:, 0]), axis=1)
            posed_normal /= np.maximum(np.linalg.norm(posed_normal, axis=1, keepdims=True), 1e-8)
            points[cell_ids] = posed[segment] + chosen_along[:, None] * posed_vector + lateral[:, None] * posed_normal
        return points

    def _skinned_cell_ids(self, appendage: int) -> np.ndarray:
        """Return limb cells without reclassifying load-bearing torso cells.

        The same selection is used for rendering and limb-proportion gates, so
        a grasper cannot appear muscular merely because its root crosses the
        chassis.  Distal arms and legs retain the developmental tube thickness.
        """
        cell_ids = np.flatnonzero(self.organism.appendage_index == appendage)
        if cell_ids.size == 0:
            return cell_ids
        gene = self.organism.genome.appendages[appendage]
        root_component = next(component for component in self.organism.genome.components if component.component_id == gene.root_component)
        points = self.organism.cell_xy.astype(np.float32, copy=False)
        root_delta = (points[cell_ids] - np.asarray(root_component.anchor, np.float32)) / np.asarray(root_component.radius, np.float32)
        core_distance = np.max(np.abs(root_delta), axis=1) if int(np.argmax(self.organism.genome.family_mix)) == 4 else np.linalg.norm(root_delta, axis=1)
        return cell_ids[core_distance >= .92]

    def geometry(self, appendage: int) -> LimbGeometry:
        gene = self.organism.genome.appendages[appendage]
        return LimbGeometry(gene.kind, gene.segments, self.length(appendage), int(self._skinned_cell_ids(appendage).size))

    def require_peer_limbs(self, grasp_kinds: set[str], locomotor_kinds: set[str]) -> None:
        """Fail closed when ordinary graspers stop reading as locomotor peers.

        Specialized tails, tendrils, roots and hardpoints intentionally opt out.
        For arm-and-leg chassis, reach, articulated joints and visible cell mass
        remain comparable; only the terminal contact role differs.
        """
        graspers = [self.geometry(index) for index, gene in enumerate(self.organism.genome.appendages) if gene.kind in grasp_kinds]
        locomotors = [self.geometry(index) for index, gene in enumerate(self.organism.genome.appendages) if gene.kind in locomotor_kinds]
        if not graspers or not locomotors:
            return
        grasp_length = float(np.mean([limb.length for limb in graspers]))
        locomotor_length = float(np.mean([limb.length for limb in locomotors]))
        grasp_cells = float(np.mean([limb.cell_count for limb in graspers]))
        locomotor_cells = float(np.mean([limb.cell_count for limb in locomotors]))
        if {limb.segments for limb in graspers} != {limb.segments for limb in locomotors}:
            raise ValueError("grasper and locomotor joint vocabularies diverged")
        if not .80 <= grasp_length / locomotor_length <= 1.20:
            raise ValueError("grasper and locomotor reaches diverged")
        if not .70 <= grasp_cells / max(locomotor_cells, 1.0) <= 1.30:
            raise ValueError("grasper and locomotor cell masses diverged")

    def chain(self, appendage: int) -> np.ndarray:
        return self.nodes[self.chain_ids[appendage], :2].copy()

    def root(self, appendage: int) -> np.ndarray:
        gene = self.organism.genome.appendages[appendage]
        return self.nodes[self.root_nodes[appendage], :2].astype(np.float64) + np.asarray(gene.root_offset, np.float64)

    def length(self, appendage: int) -> float:
        chain = self.organism.skeleton_nodes[self.chain_ids[appendage], :2]
        return float(np.linalg.norm(chain[1:] - chain[:-1], axis=1).sum())

    def feasible(self, appendage: int, target: np.ndarray, contact_radius: float = 1.25) -> bool:
        return float(np.linalg.norm(np.asarray(target) - self.root(appendage))) <= self.length(appendage) + contact_radius

    def max_length_error(self) -> float:
        maximum = 0.0
        for chain_ids in self.chain_ids:
            rest = self.organism.skeleton_nodes[chain_ids, :2]
            posed = self.nodes[chain_ids, :2]
            error = np.abs(np.linalg.norm(rest[1:] - rest[:-1], axis=1) - np.linalg.norm(posed[1:] - posed[:-1], axis=1))
            maximum = max(maximum, float(error.max(initial=0)))
        return maximum
