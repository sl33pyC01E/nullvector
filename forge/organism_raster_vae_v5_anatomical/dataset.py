from __future__ import annotations

import math

import numpy as np
import torch

from ..creature_stage_developmental.contract import APPENDAGE_KINDS, TRAITS
from ..creature_stage_developmental.motion import pose
from ..organism_raster_vae_v4_graph.dataset import GraphTokenCorpus
from .contract import (
    MAX_APPENDAGES,
    MAX_JOINTS,
    MAX_ORGANS,
    MAX_TOKENS,
    ORGAN_VOCAB,
    TOKEN_APPENDAGE,
    TOKEN_FEATURES,
    TOKEN_JOINT,
    TOKEN_ORGAN,
)


def _raster_positions(organism, phase_index: int) -> tuple[np.ndarray, np.ndarray]:
    motion = pose(organism, phase_index / 16)
    low = organism.cell_xy.min(0).astype(np.float32)
    high = organism.cell_xy.max(0).astype(np.float32)
    midpoint = (low + high) * .5
    offset = np.asarray((23.5, 23.5), dtype=np.float32) - midpoint
    return motion.cells.astype(np.float32) + offset, motion.nodes[:, :2].astype(np.float32) + offset


def _write_common(token: np.ndarray, token_type: int, family: int, side: float) -> None:
    token[token_type] = 1.0
    token[3 + family] = 1.0
    token[8] = side


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b - a
    denominator = max(float(np.dot(delta, delta)), 1e-8)
    t = np.clip(((points - a) @ delta) / denominator, 0.0, 1.0)
    projection = a[None] + t[:, None] * delta[None]
    return np.linalg.norm(points - projection, axis=1)


class AnatomicalGraphCorpus(GraphTokenCorpus):
    """Held-out motion corpus with explicit cell-to-anatomy authority.

    The raster target stays identical to v3/v4. Only conditioning and authority
    targets become richer, so improvements cannot be attributed to a changed
    target renderer.
    """

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(index)
        identity, phase_index = self.rows[index]
        organism = self.organisms[identity]
        phase = phase_index / 16
        family = int(np.argmax(organism.genome.family_mix))
        cell_xy, nodes_xy = _raster_positions(organism, phase_index)

        tokens = np.zeros((MAX_TOKENS, TOKEN_FEATURES), dtype=np.float32)
        token_mask = np.zeros(MAX_TOKENS, dtype=np.bool_)
        token_group = np.full(MAX_TOKENS, -1, dtype=np.int8)

        # Appendage tokens occupy stable slots 0..7.
        for appendage_index, gene in enumerate(organism.genome.appendages):
            if appendage_index >= MAX_APPENDAGES:
                raise ValueError("appendage census exceeded anatomical token contract")
            token = tokens[appendage_index]
            _write_common(token, TOKEN_APPENDAGE, family, float(gene.side))
            token[9 + APPENDAGE_KINDS.index(gene.kind)] = 1.0
            token[17] = gene.segments / 4.0
            token[18] = math.sin(math.tau * gene.phase)
            token[19] = math.cos(math.tau * gene.phase)
            token[20:22] = np.asarray(gene.root_offset, dtype=np.float32) / 24
            token[22:24] = np.asarray(gene.endpoint, dtype=np.float32) / 24
            delta = np.asarray(gene.endpoint, dtype=np.float32) - np.asarray(gene.root_offset, dtype=np.float32)
            token[24:26] = delta / 24
            token[26] = float(np.linalg.norm(delta) / 32)
            token[27] = gene.bend
            token[28] = float(gene.paired_with is not None)
            mode = {"leg": 0, "root": 1, "wheel": 2}.get(gene.kind, 3)
            token[29 + mode] = 1.0
            local_phase = (phase + gene.phase) % 1
            stance = {"leg": .58, "root": .76, "wheel": .56}.get(gene.kind, 0)
            token[33] = float(stance > 0 and local_phase < stance)
            token[34:49] = np.asarray(organism.genome.traits, dtype=np.float32)
            token_mask[appendage_index] = True
            token_group[appendage_index] = TOKEN_APPENDAGE

        # Joint tokens occupy appendage-major stable slots. A joint is the
        # anatomical edge downstream from the authored appendage root.
        joint_slot_for_edge: dict[int, int] = {}
        for appendage_index, gene in enumerate(organism.genome.appendages):
            edge_ids = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
            # The first edge connects soma to appendage root. Remaining edges
            # are the articulated chain and own the visible limb cells.
            articulated = edge_ids[1:] if edge_ids.size > 1 else edge_ids
            for joint_ordinal, edge_id_raw in enumerate(articulated):
                joint_linear = appendage_index * 4 + joint_ordinal
                if joint_linear >= MAX_JOINTS:
                    raise ValueError("joint census exceeded anatomical token contract")
                slot = MAX_APPENDAGES + joint_linear
                edge_id = int(edge_id_raw)
                left, right = organism.skeleton_edges[edge_id]
                a, b = nodes_xy[int(left)], nodes_xy[int(right)]
                token = tokens[slot]
                _write_common(token, TOKEN_JOINT, family, float(gene.side))
                token[9 + APPENDAGE_KINDS.index(gene.kind)] = 1.0
                token[17] = joint_ordinal / 4
                token[18] = math.sin(math.tau * (phase + gene.phase))
                token[19] = math.cos(math.tau * (phase + gene.phase))
                token[20:22] = a / 24 - 1
                token[22:24] = b / 24 - 1
                delta = b - a
                token[24:26] = delta / 24
                token[26] = float(np.linalg.norm(delta) / 12)
                token[27] = gene.bend
                token[29 + {"leg": 0, "root": 1, "wheel": 2}.get(gene.kind, 3)] = 1.0
                token[33] = tokens[appendage_index, 33]
                muscle_rows = organism.muscles[
                    (organism.muscles[:, 2] == appendage_index)
                    & (organism.muscles[:, 6] == joint_ordinal)
                ]
                if muscle_rows.size:
                    token[34] = float(muscle_rows[:, 4].mean())
                    token[35] = float(muscle_rows[:, 4].max())
                token[36:51] = np.asarray(organism.genome.traits, dtype=np.float32)
                token_mask[slot] = True
                token_group[slot] = TOKEN_JOINT
                joint_slot_for_edge[edge_id] = slot

        # Organ tokens occupy stable component-order slots at the tail.
        organ_slot_for_component: dict[int, int] = {}
        organ_count = 0
        for component_index, component in enumerate(organism.genome.components):
            if component.organ == "none":
                continue
            if component.organ not in ORGAN_VOCAB:
                raise ValueError(f"unregistered organ {component.organ!r}")
            if organ_count >= MAX_ORGANS:
                raise ValueError("organ census exceeded anatomical token contract")
            slot = MAX_APPENDAGES + MAX_JOINTS + organ_count
            token = tokens[slot]
            _write_common(token, TOKEN_ORGAN, family, float(component.side))
            token[9 + ORGAN_VOCAB.index(component.organ)] = 1.0
            token[28] = float(component.parent is not None)
            token[29:31] = np.asarray(component.anchor, dtype=np.float32) / 24
            token[31:33] = np.asarray(component.radius, dtype=np.float32) / 12
            token[33:48] = np.clip(
                np.asarray(organism.genome.traits, dtype=np.float32)
                + np.asarray(component.trait_delta, dtype=np.float32),
                0,
                1,
            )
            token[48] = float(component.kind in {"head", "sensor_crown", "mouth"})
            token[49] = float(component.kind in {"soma", "pelvis", "circulator", "respirator", "gut"})
            token_mask[slot] = True
            token_group[slot] = TOKEN_ORGAN
            organ_slot_for_component[component_index] = slot
            organ_count += 1

        appendage_owner = np.full((48, 48), -1, dtype=np.int64)
        joint_owner = np.full((48, 48), -1, dtype=np.int64)
        organ_owner = np.full((48, 48), -1, dtype=np.int64)
        priority = np.full((48, 48), -1, dtype=np.int16)

        edge_distances = np.full((organism.cell_count, len(organism.skeleton_edges)), 999.0, dtype=np.float32)
        for edge_index, (left, right) in enumerate(organism.skeleton_edges):
            edge_distances[:, edge_index] = _point_segment_distance(
                cell_xy, nodes_xy[int(left)], nodes_xy[int(right)]
            )
        nearest_edge = edge_distances.argmin(1)
        component_owner = organism.component_weights.argmax(1)
        for cell_index, (xf, yf) in enumerate(cell_xy):
            x = int(np.clip(round(float(xf)), 0, 47))
            y = int(np.clip(round(float(yf)), 0, 47))
            appendage = int(organism.appendage_index[cell_index])
            score = 2 if appendage >= 0 else 1
            if score >= priority[y, x]:
                if appendage >= 0:
                    appendage_owner[y, x] = appendage
                edge_id = int(nearest_edge[cell_index])
                if edge_id in joint_slot_for_edge and edge_distances[cell_index, edge_id] <= 2.1:
                    joint_owner[y, x] = joint_slot_for_edge[edge_id]
                component = int(component_owner[cell_index])
                if component in organ_slot_for_component and organism.component_weights[cell_index, component] >= .35:
                    organ_owner[y, x] = organ_slot_for_component[component]
                priority[y, x] = score

        result["tokens"] = torch.from_numpy(tokens)
        result["token_mask"] = torch.from_numpy(token_mask)
        result["token_group"] = torch.from_numpy(token_group)
        result["appendage_owner"] = torch.from_numpy(appendage_owner)
        result["joint_owner"] = torch.from_numpy(joint_owner)
        result["organ_owner"] = torch.from_numpy(organ_owner)
        # v4's smaller target is removed to prevent ambiguous supervision.
        result.pop("token_owner", None)
        return result
