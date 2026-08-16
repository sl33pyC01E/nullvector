from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..creature_stage_developmental.development import DevelopedOrganism
from ..creature_stage_developmental.motion import _fabrik, skin_cells


@dataclass(slots=True)
class ArticulatedBody:
    """Cell-skinned appendage chains with fixed chassis roots and bone lengths."""

    organism: DevelopedOrganism
    nodes: np.ndarray
    chain_ids: tuple[np.ndarray, ...]
    root_nodes: tuple[int, ...]

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
        return cls(organism, organism.skeleton_nodes.copy(), tuple(chains), tuple(roots))

    def endpoint(self, appendage: int) -> np.ndarray:
        return self.nodes[self.chain_ids[appendage][-1], :2].astype(np.float64)

    def solve(self, appendage: int, target: np.ndarray, response: float) -> np.ndarray:
        gene = self.organism.genome.appendages[appendage]
        chain_ids = self.chain_ids[appendage]
        root = self.nodes[self.root_nodes[appendage], :2] + np.asarray(gene.root_offset, np.float32)
        rest_chain = self.organism.skeleton_nodes[chain_ids, :2]
        solved = _fabrik(rest_chain, root, np.asarray(target, np.float32), gene.bend)
        blended_endpoint = self.nodes[chain_ids[-1], :2] + (solved[-1] - self.nodes[chain_ids[-1], :2]) * float(np.clip(response, 0, 1))
        self.nodes[chain_ids, :2] = _fabrik(rest_chain, root, blended_endpoint, gene.bend)
        return self.endpoint(appendage)

    def cells(self) -> np.ndarray:
        return skin_cells(self.organism, self.nodes)

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
