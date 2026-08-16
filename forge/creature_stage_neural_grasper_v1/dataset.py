from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from ..creature_stage_neural_grounded_components.dataset import ComponentCurriculumTeacher, ComponentSentinelTeacher
from ..creature_stage_neural_grounded_controller.dataset import owner_metadata
from .contract import GLOBAL_FEATURES, GOALS, MAX_APPENDAGES, OWNER_FEATURES, TARGET_FEATURES, TARGET_TYPES


KIND_GRIP = {"arm": 1.0, "tendril": .96, "tail": .82, "root": .76, "frond": .58, "hardpoint": .46, "leg": .30, "wheel": .10}


@dataclass(slots=True)
class GrasperCorpus:
    owner_meta: torch.Tensor
    owner_mask: torch.Tensor
    target: torch.Tensor
    global_state: torch.Tensor
    appendage_target: torch.Tensor
    engage_target: torch.Tensor
    reach_target: torch.Tensor
    force_target: torch.Tensor
    type_target: torch.Tensor
    brace_target: torch.Tensor
    release_target: torch.Tensor
    throw_target: torch.Tensor
    identity: torch.Tensor
    family: torch.Tensor
    semantic_sha256: str

    def batch(self, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name)[indices].to(device) for name in (
            "owner_meta", "owner_mask", "target", "global_state", "appendage_target", "engage_target",
            "reach_target", "force_target", "type_target", "brace_target", "identity", "family",
            "release_target", "throw_target",
        )}


def _case(organism, identity: int, case: int):
    rng = np.random.default_rng(0x4752415350 + identity * 65537 + case * 8191)
    owner, mask = owner_metadata(organism)
    target_type = 0 if case % 19 == 0 else 1 + (case % 3)
    goal = case % len(GOALS)
    angle = float(rng.uniform(-math.pi, math.pi))
    direction = np.asarray((math.cos(angle), math.sin(angle)), np.float32)
    distance = float(rng.uniform(.05, 1.25))
    mass = float(rng.beta(2.0, 3.0))
    cohesion = float(rng.beta(2.4, 2.0))
    mobility = float(rng.uniform(0, 1))
    hostility = float(rng.uniform(0, 1) if target_type == 1 else 0)
    throw = float(rng.uniform(.25, 1) if goal == GOALS.index("throw") else 0)
    attached = float(goal == GOALS.index("throw") or (target_type != 0 and case % 7 == 0))
    target = np.zeros(TARGET_FEATURES, np.float32)
    target[target_type] = 1
    target[4:6] = direction
    target[6:12] = (distance, mass, cohesion, mobility, hostility, throw)
    target[12 + goal] = 1
    target[17] = attached
    family = int(np.argmax(np.asarray(organism.genome.family_mix, dtype=np.float32)))
    global_state = np.zeros(GLOBAL_FEATURES, np.float32)
    global_state[family] = 1
    traits = np.asarray(organism.genome.traits, np.float32)
    global_state[5:] = (float(np.mean(traits[3:8])), float(np.mean(traits[8:13])), float(np.mean(traits[13:18])))

    scores = np.full(MAX_APPENDAGES, -1e6, np.float32)
    capacities = np.zeros(MAX_APPENDAGES, np.float32)
    for index, appendage in enumerate(organism.genome.appendages):
        endpoint = np.asarray(appendage.endpoint, np.float32)
        endpoint_norm = endpoint / max(float(np.linalg.norm(endpoint)), 1e-6)
        alignment = float(np.dot(endpoint_norm, direction))
        reach = min(1.25, float(np.linalg.norm(endpoint)) / 20.0 + .20)
        capacity = np.clip(KIND_GRIP[appendage.kind] * (.45 + .11 * appendage.segments) * (.75 + .25 * global_state[5]), 0, 1)
        capacities[index] = capacity
        same_side = 1 - abs(float(appendage.side) - float(np.sign(direction[0]))) * .12
        scores[index] = 1.8 * alignment + 1.2 * reach + 1.4 * capacity + same_side - .25 * abs(distance - reach)
    chosen = int(np.argmax(scores)) if bool(mask.any()) else 0
    capacity = float(capacities[chosen])
    release = float(goal == GOALS.index("throw") and attached and target_type != 0)
    engage = float(not release and target_type != 0 and distance <= 1.08 + .30 * capacity and mass <= .78 + .38 * capacity)
    if goal == 3 and cohesion > capacity + .28:
        engage *= .35
    reach_target = direction * min(distance, .72 + .42 * capacity) if engage else np.zeros(2, np.float32)
    force = engage * np.clip(.18 + .58 * mass + .24 * cohesion + .20 * hostility + .18 * (goal == 3), 0, 1)
    brace = max(engage, release) * np.clip(.12 + mass * .70 + cohesion * .20 - mobility * .16, 0, 1)
    throw_target = direction * throw * np.clip(.35 + .65 * capacity - .30 * mass, .12, 1.0) if release else np.zeros(2, np.float32)
    return owner, mask, target, global_state, chosen, engage, reach_target, force, target_type, brace, release, throw_target, family


def build_corpus(*, split: str, cases_per_identity: int = 192) -> GrasperCorpus:
    if not 32 <= cases_per_identity <= 1024:
        raise ValueError("grasper case census drifted")
    if split == "train":
        teacher = ComponentCurriculumTeacher(); identities = range(len(teacher.organisms))
    elif split == "validation":
        teacher = ComponentSentinelTeacher(); identities = teacher.split_indices("validation")
    else:
        raise ValueError("grasper split drifted")
    rows = [_case(teacher.organisms[identity], identity, case) for identity in identities for case in range(cases_per_identity)]
    arrays = [np.stack([row[index] for row in rows]) for index in range(13)]
    dtypes = (np.float32, np.bool_, np.float32, np.float32, np.int64, np.float32, np.float32, np.float32, np.int64, np.float32, np.float32, np.float32, np.int64)
    arrays = [np.ascontiguousarray(value, dtype=dtype) for value, dtype in zip(arrays, dtypes)]
    identity = np.repeat(np.asarray(tuple(identities), np.int64), cases_per_identity)
    digest = hashlib.sha256(b"nullvector-neural-grasper-corpus-v1\0" + split.encode())
    for value in (*arrays, identity):
        digest.update(value.dtype.str.encode() + np.asarray(value.shape, dtype="<i8").tobytes() + memoryview(value))
    return GrasperCorpus(*(torch.from_numpy(value) for value in arrays[:12]), torch.from_numpy(identity), torch.from_numpy(arrays[12]), digest.hexdigest())
