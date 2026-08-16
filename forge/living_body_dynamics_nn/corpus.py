from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from ..creature_stage_developmental.contract import TISSUES
from ..creature_stage_developmental.development import develop
from ..creature_stage_morphology_v2.genomes import morphology_review_genomes
from ..living_body_substrate import LivingBody
from ..living_body_substrate.contract import ORGAN_SYSTEM, SYSTEMS
from .contract import FEATURES


SYSTEM_INDEX = {"": 0, "neural": 1, "circulation": 2, "respiration": 3, "digestion": 4, "senses": 5}


def _action_field(body: LivingBody, kind: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]]:
    points = body.organism.cell_xy.astype(np.float32)
    field = np.zeros((body.organism.cell_count, 3), dtype=np.float32)
    if kind == 0:
        return field, {"kind": "idle"}
    center = points[int(rng.integers(0, len(points)))].copy()
    if kind == 1:
        radius = float(rng.uniform(2.2, 6.5))
        amount = float(rng.uniform(.18, .72))
        distance = np.linalg.norm(points - center, axis=1)
        weight = np.clip(1 - distance / radius, 0, 1)
        field[:, 0] = weight * amount
        return field, {"kind": "impact", "center": center.tolist(), "radius": radius, "amount": amount}
    if kind == 2:
        angle = float(rng.uniform(0, math.tau))
        direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32)
        start = center - direction * 9
        end = center + direction * 9
        width = float(rng.uniform(.45, 1.35))
        delta = end - start
        t = np.clip(((points - start) @ delta) / max(float(delta @ delta), 1e-6), 0, 1)
        distance = np.linalg.norm(points - (start[None] + t[:, None] * delta[None]), axis=1)
        field[:, 1] = (distance <= width).astype(np.float32)
        return field, {"kind": "cut", "start": start.tolist(), "end": end.tolist(), "width": width}
    radius = float(rng.uniform(2.5, 7.0))
    amount = float(rng.uniform(.08, .38))
    distance = np.linalg.norm(points - center, axis=1)
    field[:, 2] = np.clip(1 - distance / radius, 0, 1) * amount
    return field, {"kind": "heal", "center": center.tolist(), "radius": radius, "amount": amount}


def _apply(body: LivingBody, descriptor: dict[str, object]) -> None:
    kind = descriptor["kind"]
    if kind == "impact":
        body.impact(tuple(descriptor["center"]), float(descriptor["radius"]), float(descriptor["amount"]))
    elif kind == "cut":
        body.cut(tuple(descriptor["start"]), tuple(descriptor["end"]), width=float(descriptor["width"]))
    elif kind == "heal":
        body.heal(tuple(descriptor["center"]), float(descriptor["radius"]), float(descriptor["amount"]))


class BodyTransitionCorpus(Dataset[dict[str, torch.Tensor]]):
    """Deterministic paired interventions covering every family and identity."""

    def __init__(self, repeats: int = 4) -> None:
        if not 1 <= repeats <= 16:
            raise ValueError("living transition repeat count drifted")
        self.organisms = tuple(develop(genome) for genome in morphology_review_genomes())
        self.rows = tuple((identity, history, action, repeat) for identity in range(30) for history in range(3) for action in range(4) for repeat in range(repeats))
        digest = hashlib.sha256(b"nullvector-living-body-transition-corpus-v1\0")
        for organism in self.organisms:
            digest.update(organism.identity_sha256.encode("ascii") + b"\0")
        digest.update(np.asarray(self.rows, dtype="<i2").tobytes())
        self.semantic_sha256 = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        identity, history, action_kind, repeat = self.rows[index]
        rng = np.random.default_rng(0x424F44594E4E ^ (index * 0x9E3779B1))
        body = LivingBody(self.organisms[identity], seed=index)
        for previous in range(history):
            _, descriptor = _action_field(body, 1 + ((action_kind + previous + repeat) % 3), rng)
            _apply(body, descriptor)
            body.tick(.1)
        before_connected = body._connected_to_core().astype(np.float32)
        action, descriptor = _action_field(body, action_kind, rng)
        before_health = body.health.copy()
        before_fluid = body.fluid.copy()
        before_scar = body.scar.copy()
        before_energy = float(body.energy)
        _apply(body, descriptor)
        target_snapshot = body.tick(.1)

        organism = body.organism
        family = body.family
        count = organism.cell_count
        features = np.zeros((count, FEATURES), dtype=np.float32)
        features[:, family] = 1
        features[np.arange(count), 5 + organism.tissue.astype(np.int64)] = 1
        for cell, organ in enumerate(body.organ):
            features[cell, 20 + SYSTEM_INDEX[ORGAN_SYSTEM.get(organ, "")]] = 1
        points = organism.cell_xy.astype(np.float32)
        midpoint = (points.min(0) + points.max(0)) * .5
        features[:, 26:28] = (points - midpoint[None]) / 32
        features[:, 28] = (organism.appendage_index >= 0).astype(np.float32)
        features[:, 29] = organism.side.astype(np.float32)
        features[:, 30] = before_health
        features[:, 31] = before_fluid
        features[:, 32] = before_scar
        features[:, 33] = before_connected
        features[:, 34:37] = action
        features[:, 37] = before_energy
        features[:, 38] = history / 3
        target = np.stack((body.health, body.fluid, body.scar), axis=1).astype(np.float32)
        systems = np.asarray([target_snapshot.systems[name] for name in SYSTEMS], dtype=np.float32)
        return {
            "features": torch.from_numpy(features),
            "edges": torch.from_numpy(organism_edges(body)).long(),
            "target": torch.from_numpy(target),
            "systems": torch.from_numpy(systems),
            "family": torch.tensor(family, dtype=torch.long),
            "identity": torch.tensor(identity, dtype=torch.long),
            "action_kind": torch.tensor(action_kind, dtype=torch.long),
        }


def organism_edges(body: LivingBody) -> np.ndarray:
    edges = body.adjacency.astype(np.int64)
    reverse = edges[:, ::-1]
    return np.concatenate((edges, reverse), axis=0).T


def collate_graphs(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    offset = 0
    edges = []
    graph_index = []
    for graph, row in enumerate(rows):
        edges.append(row["edges"] + offset)
        graph_index.append(torch.full((len(row["features"]),), graph, dtype=torch.long))
        offset += len(row["features"])
    return {
        "features": torch.cat([row["features"] for row in rows]),
        "edges": torch.cat(edges, dim=1),
        "graph_index": torch.cat(graph_index),
        "target": torch.cat([row["target"] for row in rows]),
        "systems": torch.stack([row["systems"] for row in rows]),
        "family": torch.stack([row["family"] for row in rows]),
        "identity": torch.stack([row["identity"] for row in rows]),
        "action_kind": torch.stack([row["action_kind"] for row in rows]),
    }
