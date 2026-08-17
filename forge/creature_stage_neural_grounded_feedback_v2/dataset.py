from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from ..creature_stage_developmental.development import DevelopedOrganism, develop
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_grounded_locomotion.physics import GroundedCycle, _terminal_nodes, simulate_grounded_cycle
from ..creature_stage_neural_grounded_cyclic.curriculum import _curriculum_data
from .contract import (
    APPENDAGE_KINDS, GLOBAL_FEATURES, MAX_APPENDAGES, MAX_MUSCLES,
    MUSCLE_FEATURES, OWNER_FEATURES,
)


@dataclass(slots=True)
class FeedbackCorpus:
    owner_state: torch.Tensor
    global_state: torch.Tensor
    owner_mask: torch.Tensor
    muscle_meta: torch.Tensor
    muscle_owner: torch.Tensor
    muscle_mask: torch.Tensor
    muscle_target: torch.Tensor
    contact_target: torch.Tensor
    body_target: torch.Tensor
    identity: torch.Tensor
    frame: torch.Tensor
    semantic_sha256: str
    organisms: tuple[DevelopedOrganism, ...]
    cycles: tuple[GroundedCycle, ...]

    @property
    def samples(self) -> int:
        return int(self.identity.numel())

    def batch(self, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        names = (
            "owner_state", "global_state", "owner_mask", "muscle_meta", "muscle_owner",
            "muscle_mask", "muscle_target", "contact_target", "body_target",
        )
        return {name: getattr(self, name)[indices].to(device) for name in names}


def owner_metadata(organism: DevelopedOrganism) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((MAX_APPENDAGES, 16), np.float32); mask = np.zeros(MAX_APPENDAGES, np.bool_)
    if len(organism.genome.appendages) > MAX_APPENDAGES:
        raise ValueError("grounded feedback appendage census drifted")
    for index, gene in enumerate(organism.genome.appendages):
        if gene.kind not in APPENDAGE_KINDS:
            raise ValueError("grounded feedback appendage kind drifted")
        values[index, APPENDAGE_KINDS.index(gene.kind)] = 1
        values[index, 8:] = (
            float(gene.side), float(gene.segments) / 5,
            math.sin(math.tau * gene.phase), math.cos(math.tau * gene.phase),
            float(gene.root_offset[0]) / 24, float(gene.root_offset[1]) / 24,
            float(gene.endpoint[0]) / 24, float(gene.endpoint[1]) / 24,
        )
        mask[index] = True
    return values, mask


def muscle_metadata(organism: DevelopedOrganism) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((MAX_MUSCLES, MUSCLE_FEATURES), np.float32)
    owner = np.zeros(MAX_MUSCLES, np.int64); mask = np.zeros(MAX_MUSCLES, np.bool_)
    if len(organism.muscles) > MAX_MUSCLES:
        raise ValueError("grounded feedback muscle census drifted")
    for index, muscle in enumerate(organism.muscles):
        appendage = int(muscle[2]); joint = float(muscle[6]); gene = organism.genome.appendages[appendage]
        owner[index] = appendage
        values[index] = (
            float(muscle[3]), float(muscle[4]), float(muscle[5]), joint / 5,
            math.sin(math.tau * gene.phase), math.cos(math.tau * gene.phase),
            math.sin(math.tau * joint / 5), math.cos(math.tau * joint / 5),
        )
        mask[index] = True
    return values, owner, mask


def encode_live(
    organism: DevelopedOrganism,
    nodes_local: np.ndarray,
    node_velocity: np.ndarray,
    previous_contact: np.ndarray,
    phase: float,
    body_velocity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes_local = np.asarray(nodes_local, np.float32); node_velocity = np.asarray(node_velocity, np.float32)
    previous_contact = np.asarray(previous_contact, np.bool_)
    appendages = len(organism.genome.appendages)
    if nodes_local.shape != organism.skeleton_nodes[:, :2].shape or node_velocity.shape != nodes_local.shape:
        raise ValueError("grounded feedback live node state drifted")
    if previous_contact.shape != (appendages,) or not np.isfinite(nodes_local).all() or not np.isfinite(node_velocity).all():
        raise ValueError("grounded feedback live contact state drifted")
    base, owner_mask = owner_metadata(organism)
    owners = np.zeros((MAX_APPENDAGES, OWNER_FEATURES), np.float32); owners[:, :16] = base
    terminals = _terminal_nodes(organism)
    for appendage in range(appendages):
        terminal = int(terminals[appendage])
        owners[appendage, 16:20] = (
            float(nodes_local[terminal, 0]) / 24, float(nodes_local[terminal, 1]) / 24,
            float(node_velocity[terminal, 0]) / 2, float(node_velocity[terminal, 1]) / 2,
        )
        owners[appendage, 20:23] = (
            float(previous_contact[appendage]), math.sin(math.tau * phase), math.cos(math.tau * phase),
        )
    global_state = np.zeros(GLOBAL_FEATURES, np.float32)
    family = int(np.argmax(np.asarray(organism.genome.family_mix, np.float32))); global_state[family] = 1
    traits = np.asarray(organism.genome.traits, np.float32)
    if traits.shape != (15,):
        raise ValueError("grounded feedback trait vocabulary drifted")
    global_state[5:20] = traits
    global_state[20:23] = (math.sin(math.tau * phase), math.cos(math.tau * phase), float(body_velocity) / .55)
    muscle_meta, muscle_owner, muscle_mask = muscle_metadata(organism)
    return owners, global_state, owner_mask, muscle_meta, muscle_owner, muscle_mask


def _teacher(split: str, variants_per_family: int) -> tuple[tuple[DevelopedOrganism, ...], tuple[GroundedCycle, ...]]:
    if split == "train":
        _genomes, organisms, cycles = _curriculum_data(variants_per_family)
        return tuple(organisms), tuple(cycles)
    if split == "validation":
        organisms = tuple(develop(genome) for genome in review_genomes())
        return organisms, tuple(simulate_grounded_cycle(organism) for organism in organisms)
    raise ValueError("grounded feedback split drifted")


def build_corpus(*, split: str, variants_per_family: int = 2) -> FeedbackCorpus:
    organisms, cycles = _teacher(split, variants_per_family)
    rows = []
    for identity, (organism, cycle) in enumerate(zip(organisms, cycles, strict=True)):
        for frame in range(72):
            previous = cycle.frames[(frame - 1) % 72]
            target = cycle.frames[frame]
            encoded = encode_live(
                organism, previous.nodes_local, previous.node_velocity,
                previous.contact_active, frame / 72, previous.body_velocity_x,
            )
            muscle_target = np.zeros(MAX_MUSCLES, np.float32)
            muscle_target[:len(organism.muscles)] = target.muscle_activation
            contact_target = np.zeros(MAX_APPENDAGES, np.float32)
            contact_target[:len(organism.genome.appendages)] = target.contact_active
            rows.append((*encoded, muscle_target, contact_target, np.float32(target.body_velocity_x / .55), identity, frame))
    arrays = [np.ascontiguousarray(np.stack([row[index] for row in rows])) for index in range(11)]
    arrays[2] = arrays[2].astype(np.bool_); arrays[4] = arrays[4].astype(np.int64); arrays[5] = arrays[5].astype(np.bool_)
    arrays[9] = arrays[9].astype(np.int64); arrays[10] = arrays[10].astype(np.int64)
    digest = hashlib.sha256(b"nullvector-neural-grounded-feedback-corpus-v2\0" + split.encode("ascii"))
    for value in arrays:
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(memoryview(value))
    return FeedbackCorpus(*(torch.from_numpy(value) for value in arrays), digest.hexdigest(), organisms, cycles)
