from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from ..creature_stage_developmental.development import DevelopedOrganism, develop
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_manipulation_v1.articulation import ArticulatedBody, curved_muscle_pose
from .contract import APPENDAGE_KINDS, CONTEXT_FEATURES, MAX_NODES, NODE_FEATURES


@dataclass(slots=True)
class PoseCorpus:
    nodes: torch.Tensor
    context: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    scale: torch.Tensor
    identity: torch.Tensor
    appendage: torch.Tensor
    semantic_sha256: str

    @property
    def samples(self) -> int:
        return int(self.identity.numel())

    def batch(self, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name)[indices].to(device) for name in ("nodes", "context", "mask", "target", "scale")}


def bend_sign_for(organism: DevelopedOrganism, appendage: int, body: ArticulatedBody | None = None) -> float:
    body = body or ArticulatedBody.from_organism(organism)
    chain = body.chain_ids[appendage]
    rest = organism.skeleton_nodes[chain, :2].astype(np.float32)
    chord = rest[-1] - rest[0]
    normal = np.asarray((-chord[1], chord[0]), np.float32)
    sign = float(np.sign(np.mean((rest[1:-1] - rest[0]) @ normal))) if len(rest) > 2 else 0.0
    if sign == 0:
        sign = -1.0 if organism.genome.appendages[appendage].root_offset[0] < 0 else 1.0
    return sign


def encode_case(
    organism: DevelopedOrganism,
    appendage: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    root: np.ndarray,
    target: np.ndarray,
    lengths: np.ndarray,
    *,
    response: float,
    actuation: float,
    load: float,
    bend_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    positions = np.asarray(positions, np.float32); velocities = np.asarray(velocities, np.float32)
    root = np.asarray(root, np.float32); target = np.asarray(target, np.float32); lengths = np.asarray(lengths, np.float32)
    count = len(lengths) + 1
    if not 2 <= count <= MAX_NODES or positions.shape != (count, 2) or velocities.shape != (count, 2):
        raise ValueError("limb pose chain shape drifted")
    reach = max(float(lengths.sum()), 1e-6)
    nodes = np.zeros((MAX_NODES, NODE_FEATURES), np.float32)
    mask = np.zeros(MAX_NODES, np.bool_); mask[:count] = True
    cumulative = np.concatenate((np.zeros(1, np.float32), np.cumsum(lengths))) / reach
    segment = np.concatenate((lengths, np.zeros(1, np.float32))) / reach
    nodes[:count, 0:2] = (positions - root) / reach
    nodes[:count, 2:4] = velocities / 10.0
    nodes[:count, 4] = cumulative
    nodes[:count, 5] = segment
    nodes[:count, 6] = 1.0
    nodes[count - 1, 7] = 1.0
    context = np.zeros(CONTEXT_FEATURES, np.float32)
    context[0:2] = np.clip((target - root) / reach, -1, 1)
    context[2:5] = (float(response), float(actuation), min(float(load) / 8.0, 1.0))
    family = int(np.argmax(np.asarray(organism.genome.family_mix, np.float32)))
    context[5 + family] = 1.0
    kind = organism.genome.appendages[appendage].kind
    if kind not in APPENDAGE_KINDS:
        raise ValueError("limb pose appendage kind drifted")
    context[10 + APPENDAGE_KINDS.index(kind)] = 1.0
    context[18:21] = (float(organism.genome.appendages[appendage].side), float(bend_sign), (count - 2) / (MAX_NODES - 2))
    return nodes, context, mask, reach


def _case(
    organism: DevelopedOrganism,
    identity: int,
    appendage: int,
    case: int,
    body: ArticulatedBody,
    sign: float,
    augment_geometry: bool,
):
    seed = 0x4C494D42504F5345 ^ identity * 0x9E3779B1 ^ appendage * 0x85EBCA77 ^ case * 0xC2B2AE3D
    rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
    chain = body.chain_ids[appendage]
    rest = organism.skeleton_nodes[chain, :2].astype(np.float32)
    gene = organism.genome.appendages[appendage]
    root = organism.skeleton_nodes[body.root_nodes[appendage], :2].astype(np.float32) + np.asarray(gene.root_offset, np.float32)
    rest_vectors = rest[1:] - rest[:-1]
    lengths = np.linalg.norm(rest_vectors, axis=1).astype(np.float32)
    if augment_geometry:
        directions = rest_vectors / np.maximum(lengths[:, None], 1e-6)
        lengths = lengths * rng.uniform(.62, 1.42, len(lengths)).astype(np.float32)
        rest = np.concatenate((root[None], root[None] + np.cumsum(directions * lengths[:, None], axis=0)), axis=0)
    reach = float(lengths.sum())
    angle = float(rng.uniform(-math.pi, math.pi)); distance = float(rng.uniform(.08, 1.0) * reach)
    target = root + np.asarray((math.cos(angle), math.sin(angle)), np.float32) * distance
    teacher = curved_muscle_pose(root, target, lengths, sign)
    mixture = float(rng.uniform(0, 1))
    positions = rest * (1 - mixture) + teacher * mixture + rng.normal(0, .04, rest.shape).astype(np.float32)
    positions[0] = root
    velocities = rng.normal(0, .22, rest.shape).astype(np.float32); velocities[0] = 0
    response = float(rng.uniform(.05, 1)); actuation = float(rng.uniform(.08, 1)); load = float(rng.uniform(0, 6))
    nodes, context, mask, scale = encode_case(
        organism, appendage, positions, velocities, root, target, lengths,
        response=response, actuation=actuation, load=load, bend_sign=sign,
    )
    target_pose = np.zeros((MAX_NODES, 2), np.float32)
    target_pose[:len(chain)] = np.clip((teacher - root) / scale, -1, 1)
    return nodes, context, mask, target_pose, np.float32(scale), identity, appendage


def build_corpus(*, split: str, cases_per_appendage: int = 640) -> PoseCorpus:
    genomes = review_genomes()
    if split == "train":
        identities = tuple(range(len(genomes)))
        case_offset = 0
    elif split == "validation":
        # Hold out target/state samples rather than known chassis. Geometry is
        # already broadened continuously during training, while all ten
        # supported review chassis must be represented in both quality axes.
        identities = tuple(range(len(genomes)))
        case_offset = 1_000_000
    else:
        raise ValueError("limb pose split drifted")
    organisms = tuple(develop(genomes[index]) for index in identities)
    rows = []
    for identity, organism in zip(identities, organisms, strict=True):
        body = ArticulatedBody.from_organism(organism)
        for appendage in range(len(organism.genome.appendages)):
            sign = bend_sign_for(organism, appendage, body)
            rows.extend(
                _case(organism, identity, appendage, case_offset + case, body, sign, split == "train")
                for case in range(cases_per_appendage)
            )
    arrays = [np.ascontiguousarray(np.stack([row[index] for row in rows])) for index in range(7)]
    arrays[2] = arrays[2].astype(np.bool_); arrays[4] = arrays[4].astype(np.float32)
    arrays[5] = arrays[5].astype(np.int64); arrays[6] = arrays[6].astype(np.int64)
    digest = hashlib.sha256(b"nullvector-neural-limb-pose-corpus-v1\0" + split.encode("ascii"))
    for value in arrays:
        digest.update(value.dtype.str.encode("ascii") + np.asarray(value.shape, dtype="<i8").tobytes() + memoryview(value))
    return PoseCorpus(*(torch.from_numpy(value) for value in arrays), digest.hexdigest())
