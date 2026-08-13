from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np
from torch import Tensor
from torch.utils.data import default_collate

from ..morphology.constants import FAMILIES, ROLE_NAMES, SUBTYPE_NAMES
from ..multifield_data import MorphologyCorpusDataset
from .checkpoint import LoadedMultiFieldCheckpoint


CONDITION_GRID_FORMAT = "nullvector-multifield-condition-grid-v1"
GRID_MODES = ("fixed", "stratified", "exhaustive")


@dataclass(frozen=True, slots=True)
class ConditionRecord:
    ordinal: int
    grid_mode: str
    source_index: int
    variation: int
    sample_seed: int
    morphology: int
    subtype: int
    role: int

    @property
    def sample_id(self) -> str:
        return (
            f"{self.ordinal:04d}_f{self.morphology}_s{self.subtype:02d}_"
            f"r{self.role}_v{self.variation:02d}"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "sample_id": self.sample_id,
            "grid_mode": self.grid_mode,
            "source_index": self.source_index,
            "variation": self.variation,
            "sample_seed": self.sample_seed,
            "morphology_id": self.morphology,
            "morphology_name": FAMILIES[self.morphology],
            "subtype_id": self.subtype,
            "subtype_name": SUBTYPE_NAMES[self.subtype],
            "role_id": self.role,
            "role_name": ROLE_NAMES[self.role],
        }


def _derived_sample_seed(
    base_seed: int,
    *,
    grid_mode: str,
    source_index: int,
    morphology: int,
    subtype: int,
    role: int,
    variation: int,
) -> int:
    digest = hashlib.sha256()
    digest.update(b"nullvector-multifield-sample-seed-v1\0")
    digest.update(int(base_seed).to_bytes(8, "little", signed=False))
    digest.update(grid_mode.encode("ascii"))
    digest.update(b"\0")
    for value in (source_index, morphology, subtype, role, variation):
        digest.update(int(value).to_bytes(8, "little", signed=False))
    # Stay inside signed int64 so manifests, NumPy, and Torch agree everywhere.
    return int.from_bytes(digest.digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


def _pick_source(
    bundle: LoadedMultiFieldCheckpoint,
    morphology: int,
    subtype: int,
    role: int,
    *,
    variation: int,
    base_seed: int,
) -> int:
    corpus = bundle.corpus
    candidates = bundle.validation_indices[
        (corpus.morphologies[bundle.validation_indices] == morphology)
        & (corpus.subtypes[bundle.validation_indices] == subtype)
        & (corpus.roles[bundle.validation_indices] == role)
    ]
    if len(candidates) == 0:
        raise ValueError(
            "Validation split has no specimen for condition "
            f"family={morphology}, subtype={subtype}, role={role}."
        )
    digest = hashlib.sha256()
    digest.update(b"nullvector-condition-source-v1\0")
    for value in (base_seed, morphology, subtype, role, variation):
        digest.update(int(value).to_bytes(8, "little", signed=False))
    offset = int.from_bytes(digest.digest()[:8], "little") % len(candidates)
    return int(candidates[offset])


def _condition_axes(mode: str) -> list[tuple[int, int, int]]:
    if mode == "stratified":
        # Five family rows by eight role columns.  Cycling the four family-local
        # subtypes across columns covers all twenty subtypes exactly twice.
        return [
            (family, family * 4 + (role % 4), role)
            for family in range(len(FAMILIES))
            for role in range(len(ROLE_NAMES))
        ]
    if mode == "exhaustive":
        # Every legal family-local subtype crossed with every role: 20 x 8.
        return [
            (subtype // 4, subtype, role)
            for subtype in range(len(SUBTYPE_NAMES))
            for role in range(len(ROLE_NAMES))
        ]
    raise ValueError(f"Unsupported non-fixed condition grid {mode!r}")


def build_condition_grid(
    bundle: LoadedMultiFieldCheckpoint,
    *,
    mode: str,
    samples_per_condition: int = 1,
    base_seed: int | None = None,
    limit: int | None = None,
) -> list[ConditionRecord]:
    if mode not in GRID_MODES:
        raise ValueError(f"grid mode must be one of {GRID_MODES}")
    if not 1 <= samples_per_condition <= 8:
        raise ValueError("samples_per_condition must be between 1 and 8")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")
    checkpoint_seed = int(bundle.payload["fixed_validation"]["generation_seed"])
    seed_root = checkpoint_seed if base_seed is None else int(base_seed)
    if not 0 <= seed_root <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("base_seed must be a nonnegative signed 64-bit integer")

    specifications: list[tuple[int, int, int, int, int]] = []
    if mode == "fixed":
        source_indices = list(
            map(int, bundle.payload["fixed_validation"]["generation_source_indices"])
        )
        if not source_indices:
            raise ValueError("Checkpoint has an empty fixed generation bank")
        for source_index in source_indices:
            if source_index not in set(map(int, bundle.validation_indices)):
                raise ValueError(
                    f"Fixed source index {source_index} is outside validation split"
                )
            for variation in range(samples_per_condition):
                specifications.append(
                    (
                        source_index,
                        int(bundle.corpus.morphologies[source_index]),
                        int(bundle.corpus.subtypes[source_index]),
                        int(bundle.corpus.roles[source_index]),
                        variation,
                    )
                )
    else:
        for morphology, subtype, role in _condition_axes(mode):
            for variation in range(samples_per_condition):
                source_index = _pick_source(
                    bundle,
                    morphology,
                    subtype,
                    role,
                    variation=variation,
                    base_seed=seed_root,
                )
                specifications.append(
                    (source_index, morphology, subtype, role, variation)
                )
    if limit is not None:
        specifications = specifications[:limit]

    records: list[ConditionRecord] = []
    for ordinal, (source_index, morphology, subtype, role, variation) in enumerate(
        specifications
    ):
        if mode == "fixed" and variation == 0 and samples_per_condition == 1:
            # Exactly preserve the trainer's scheduled generation-bank seed policy.
            sample_seed = (
                seed_root
                ^ int(bundle.corpus.seeds[source_index])
                ^ (ordinal * 0x9E3779B1)
            ) & 0x7FFFFFFFFFFFFFFF
        else:
            sample_seed = _derived_sample_seed(
                seed_root,
                grid_mode=mode,
                source_index=source_index,
                morphology=morphology,
                subtype=subtype,
                role=role,
                variation=variation,
            )
        records.append(
            ConditionRecord(
                ordinal=ordinal,
                grid_mode=mode,
                source_index=source_index,
                variation=variation,
                sample_seed=sample_seed,
                morphology=morphology,
                subtype=subtype,
                role=role,
            )
        )
    return records


def condition_batch(
    bundle: LoadedMultiFieldCheckpoint,
    records: Sequence[ConditionRecord],
) -> dict[str, Tensor]:
    if not records:
        raise ValueError("Cannot collate an empty condition batch")
    # Dataset forbids duplicate source indices, while repeated variations are a
    # valid evaluation use case.  Collate individual one-row views instead.
    samples = []
    for record in records:
        one = MorphologyCorpusDataset(
            bundle.corpus,
            [record.source_index],
            guide_policy=bundle.guide_policy,
        )
        sample = one[0]
        if (
            int(sample["morphology"]) != record.morphology
            or int(sample["subtype"]) != record.subtype
            or int(sample["role"]) != record.role
        ):
            raise ValueError("Condition record disagrees with its corpus source row")
        samples.append(sample)
    return default_collate(samples)


def validate_grid_coverage(
    records: Iterable[ConditionRecord], mode: str
) -> dict[str, Any]:
    records = list(records)
    families = sorted({item.morphology for item in records})
    subtypes = sorted({item.subtype for item in records})
    roles = sorted({item.role for item in records})
    family_role = sorted({(item.morphology, item.role) for item in records})
    report = {
        "format": CONDITION_GRID_FORMAT,
        "mode": mode,
        "samples": len(records),
        "family_ids": families,
        "subtype_ids": subtypes,
        "role_ids": roles,
        "family_role_pairs": len(family_role),
    }
    if mode in {"stratified", "exhaustive"}:
        report["covers_all_families"] = families == list(range(len(FAMILIES)))
        report["covers_all_subtypes"] = subtypes == list(range(len(SUBTYPE_NAMES)))
        report["covers_all_roles"] = roles == list(range(len(ROLE_NAMES)))
        report["covers_all_family_role_pairs"] = len(family_role) == len(FAMILIES) * len(
            ROLE_NAMES
        )
    return report
