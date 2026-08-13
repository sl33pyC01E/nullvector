from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..multifield_style.model import StyleCondition


@dataclass(frozen=True, slots=True)
class FusionGenome:
    specimen_id: str
    seed: int
    parent_a_ordinal: int
    parent_b_ordinal: int
    parent_a_sample_id: str
    parent_b_sample_id: str
    dominant_parent: str
    fusion_mode: str
    mutation_mode: str
    mutation_strength: int
    mirror_donor: bool
    condition: StyleCondition
    lineage_sha256: str


@dataclass(frozen=True, slots=True)
class FusionSpecimen:
    genome: FusionGenome
    part_owner: np.ndarray
    material: np.ndarray
    emission_level: np.ndarray
    provenance: np.ndarray
    guide: np.ndarray
    genes: np.ndarray
    legal_tuples: np.ndarray
    fields_sha256: str
    provenance_sha256: str
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "part_owner": (np.uint8, (48, 48)),
            "material": (np.uint8, (48, 48)),
            "emission_level": (np.uint8, (48, 48)),
            "provenance": (np.uint8, (48, 48)),
            "guide": (np.float32, (8, 48, 48)),
            "genes": (np.float32, (24,)),
        }
        for name, (dtype, shape) in expected.items():
            values = getattr(self, name)
            if values.dtype != dtype or values.shape != shape:
                raise ValueError(f"{name} must be {dtype} {shape}")
            values.setflags(write=False)
        if self.legal_tuples.dtype != np.uint8 or self.legal_tuples.ndim != 2 or self.legal_tuples.shape[1:] != (3,):
            raise ValueError("legal_tuples must be uint8 [N,3]")
        self.legal_tuples.setflags(write=False)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
