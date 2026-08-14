from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..map_decorator_ml.dataset import TeacherSample, collate_teacher_samples
from ..map_decorator_production.training import CorpusSampleRef
from ..map_decorator_production_v4.proposal import ProposalAuthority, ProposalFields, build_proposal_fields
from ..maps.model import THEMES


@dataclass(slots=True)
class ProposalTeacherSample:
    teacher: TeacherSample
    proposals: ProposalFields

    def __post_init__(self) -> None:
        if self.teacher.shape != self.proposals.shape:
            raise ValueError("Teacher and proposal shapes differ.")


def full_sample(authority: ProposalAuthority, ref: CorpusSampleRef) -> ProposalTeacherSample:
    teacher, proposals = authority.sample_and_proposals(ref)
    return ProposalTeacherSample(teacher, proposals)


def crop_proposals(full: ProposalFields, crop: TeacherSample) -> ProposalFields:
    if crop.crop is None:
        raise ValueError("V4 proposal crop requires a cropped teacher sample.")
    ys, xs = crop.crop.slices(full.shape)
    decal = np.ascontiguousarray(full.decal[:, ys, xs], dtype=bool)
    prop = np.ascontiguousarray(full.prop[:, ys, xs], dtype=bool)
    from ..map_decorator.hashing import named_arrays_sha256

    return ProposalFields(
        decal=decal,
        prop=prop,
        map_seed=full.map_seed,
        theme=full.theme,
        channel_manifest_sha256=full.channel_manifest_sha256,
        fields_sha256=named_arrays_sha256({"decal": decal, "prop": prop}),
    )


def proposals_for_teacher(
    authority: ProposalAuthority,
    ref: CorpusSampleRef,
    teacher: TeacherSample,
) -> ProposalFields:
    original = authority.authority.corpus.sample(ref)
    full = build_proposal_fields(
        map_seed=authority.map_seed(ref),
        theme=THEMES[original.theme_index],
        shape=original.shape,
        legal_masks=original.legal_masks,
        hard_empty=original.hard_empty,
    )
    return full if teacher.crop is None else crop_proposals(full, teacher)


def collate_proposal_samples(samples: list[ProposalTeacherSample]) -> dict[str, object]:
    if not samples:
        raise ValueError("V4 proposal collation requires at least one sample.")
    batch = collate_teacher_samples([sample.teacher for sample in samples])
    height, width = batch["valid_cells"].shape[-2:]  # type: ignore[union-attr]
    proposal_batch: dict[str, list[torch.Tensor]] = {"decal": [], "prop": []}
    for sample in samples:
        for head in ("decal", "prop"):
            array = getattr(sample.proposals, head)
            padded = torch.zeros((array.shape[0], height, width), dtype=torch.bool)
            padded[:, : array.shape[1], : array.shape[2]] = torch.from_numpy(array.copy())
            proposal_batch[head].append(padded)
    batch["proposals"] = {
        head: torch.stack(values, dim=0) for head, values in proposal_batch.items()
    }
    return batch
