from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from ..creature_stage_developmental.development import DevelopedOrganism
from ..creature_stage_neural_grounded_components.dataset import ComponentCurriculumTeacher, ComponentSentinelTeacher
from ..creature_stage_neural_grounded_components.training import load_model as load_parent_model
from .contract import (
    APPENDAGE_KINDS, MAX_APPENDAGES, MAX_MUSCLES, MUSCLE_META_FEATURES,
    OWNER_META_FEATURES, PARENT,
)


POOLED_FEATURES = 61 + 4 + 16 + 4


def owner_metadata(organism: DevelopedOrganism) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((MAX_APPENDAGES, OWNER_META_FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_APPENDAGES, dtype=np.bool_)
    if len(organism.genome.appendages) > MAX_APPENDAGES:
        raise ValueError("controller appendage census exceeded")
    for index, appendage in enumerate(organism.genome.appendages):
        if appendage.kind not in APPENDAGE_KINDS:
            raise ValueError(f"controller appendage kind drifted: {appendage.kind}")
        values[index, APPENDAGE_KINDS.index(appendage.kind)] = 1
        values[index, 8:] = (
            float(appendage.side), float(appendage.segments) / 5.0,
            math.sin(math.tau * appendage.phase), math.cos(math.tau * appendage.phase),
            float(appendage.root_offset[0]) / 24.0, float(appendage.root_offset[1]) / 24.0,
            float(appendage.endpoint[0]) / 24.0, float(appendage.endpoint[1]) / 24.0,
        )
        mask[index] = True
    return values, mask


def muscle_metadata(organism: DevelopedOrganism) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((MAX_MUSCLES, MUSCLE_META_FEATURES), dtype=np.float32)
    owner = np.zeros(MAX_MUSCLES, dtype=np.int64)
    mask = np.zeros(MAX_MUSCLES, dtype=np.bool_)
    if len(organism.muscles) > MAX_MUSCLES:
        raise ValueError("controller muscle census exceeded")
    for index, muscle in enumerate(organism.muscles):
        appendage = int(muscle[2]); joint = float(muscle[6]); gene = organism.genome.appendages[appendage]
        owner[index] = appendage
        values[index] = (
            float(muscle[3]), float(muscle[4]), float(muscle[5]), joint / 5.0,
            math.sin(math.tau * gene.phase), math.cos(math.tau * gene.phase),
            math.sin(math.tau * joint / 5.0), math.cos(math.tau * joint / 5.0),
        )
        mask[index] = True
    return values, owner, mask


@dataclass(slots=True)
class ControllerCorpus:
    owner_input: Tensor
    global_input: Tensor
    owner_meta: Tensor
    owner_mask: Tensor
    muscle_meta: Tensor
    muscle_owner: Tensor
    muscle_mask: Tensor
    muscle_target: Tensor
    contact_target: Tensor
    body_target: Tensor
    identity: Tensor
    frame: Tensor
    family: Tensor
    semantic_sha256: str
    organisms: tuple[DevelopedOrganism, ...]
    teacher: ComponentCurriculumTeacher | ComponentSentinelTeacher

    @property
    def samples(self) -> int:
        return int(self.identity.numel())

    def batch(self, indices: Tensor, device: torch.device) -> dict[str, Tensor]:
        names = (
            "owner_input", "global_input", "owner_meta", "owner_mask", "muscle_meta",
            "muscle_owner", "muscle_mask", "muscle_target", "contact_target", "body_target",
            "identity", "frame", "family",
        )
        return {name: getattr(self, name)[indices].to(device) for name in names}


def _rows(teacher: ComponentCurriculumTeacher | ComponentSentinelTeacher,
          coordinates: Sequence[tuple[int, int]], device: torch.device) -> dict[str, Tensor]:
    records = [teacher.sample(identity, frame) for identity, frame in coordinates]
    result = {name: torch.from_numpy(np.stack([record[name] for record in records]).copy()).to(device) for name in ("static", "state", "dynamic", "mask", "adjacency", "controls")}
    result["owner"] = torch.from_numpy(np.stack([teacher.arrays["appendage_owner"][identity] for identity, _ in coordinates]).astype(np.int64)).to(device)
    for name in ("family", "morphotype", "motion"):
        result[name] = torch.tensor([int(record[name]) for record in records], dtype=torch.long, device=device)
    result["phase"] = torch.tensor([float(record["phase"]) for record in records], device=device)
    return result


def _pool(value: Tensor, owner: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    one_hot = torch.nn.functional.one_hot(owner.clamp_min(0), MAX_APPENDAGES).to(value.dtype)
    one_hot = one_hot * ((owner >= 0) & mask)[:, :, None]
    counts = one_hot.sum(1)
    pooled = torch.bmm(one_hot.transpose(1, 2), value) / counts.clamp_min(1)[:, :, None]
    active = mask[:, :, None].to(value.dtype)
    global_value = (value * active).sum(1) / active.sum(1).clamp_min(1)
    return pooled, global_value


def build_corpus(*, split: str, device: str | torch.device = "cuda", batch_size: int = 10) -> ControllerCorpus:
    target_device = torch.device(device)
    if split == "train":
        teacher: ComponentCurriculumTeacher | ComponentSentinelTeacher = ComponentCurriculumTeacher()
        identities = tuple(range(len(teacher.organisms)))
    elif split == "validation":
        teacher = ComponentSentinelTeacher()
        identities = teacher.split_indices("validation")
    else:
        raise ValueError("controller corpus split drifted")
    parent, payload = load_parent_model(PARENT, ema=False, device=target_device)
    coordinates = tuple((identity, frame) for identity in identities for frame in range(72))
    owner_inputs, global_inputs = [], []
    parent.eval()
    with torch.inference_mode():
        for start in range(0, len(coordinates), batch_size):
            batch_coordinates = coordinates[start:start + batch_size]
            row = _rows(teacher, batch_coordinates, target_device)
            result = parent(
                row["static"], row["state"], row["dynamic"], row["owner"], row["mask"], row["adjacency"],
                row["family"], row["morphotype"], row["motion"], row["phase"], row["controls"],
            )
            pooled, global_value = _pool(torch.cat((row["static"].float(), row["state"].float(), row["dynamic"].float(), result.cells.float()), dim=-1), row["owner"], row["mask"])
            owner_inputs.append(pooled.cpu().to(torch.float16)); global_inputs.append(global_value.cpu().to(torch.float16))
    owner_meta, owner_mask, muscle_meta, muscle_owner, muscle_mask = [], [], [], [], []
    muscle_target, contact_target, body_target, family = [], [], [], []
    for identity, frame in coordinates:
        organism = teacher.organisms[identity]
        om, omask = owner_metadata(organism); mm, mo, mmask = muscle_metadata(organism)
        owner_meta.append(om); owner_mask.append(omask); muscle_meta.append(mm); muscle_owner.append(mo); muscle_mask.append(mmask)
        muscle_target.append(teacher.arrays["muscle_activation"][identity, frame])
        contact_target.append(teacher.arrays["contact_active"][identity, frame])
        body_target.append(teacher.arrays["body_velocity_x"][identity, frame] / .55)
        family.append(int(teacher.arrays["family"][identity]))
    digest = hashlib.sha256(b"nullvector-grounded-controller-corpus-v1\0")
    digest.update(split.encode("ascii") + b"\0" + teacher.semantic_sha256.encode("ascii") + payload["model_state_sha256"].encode("ascii"))
    for value in (torch.cat(owner_inputs).numpy(), torch.cat(global_inputs).numpy(), np.stack(owner_meta), np.stack(muscle_target), np.stack(contact_target)):
        digest.update(memoryview(np.ascontiguousarray(value)))
    identity_values = np.asarray([identity for identity, _ in coordinates], dtype=np.int64)
    frame_values = np.asarray([frame for _, frame in coordinates], dtype=np.int64)
    return ControllerCorpus(
        torch.cat(owner_inputs), torch.cat(global_inputs), torch.from_numpy(np.stack(owner_meta)),
        torch.from_numpy(np.stack(owner_mask)), torch.from_numpy(np.stack(muscle_meta)),
        torch.from_numpy(np.stack(muscle_owner)), torch.from_numpy(np.stack(muscle_mask)),
        torch.from_numpy(np.stack(muscle_target).astype(np.float32)),
        torch.from_numpy(np.stack(contact_target).astype(np.float32)),
        torch.from_numpy(np.asarray(body_target, dtype=np.float32)),
        torch.from_numpy(identity_values), torch.from_numpy(frame_values),
        torch.from_numpy(np.asarray(family, dtype=np.int64)), digest.hexdigest(), tuple(teacher.organisms), teacher,
    )
