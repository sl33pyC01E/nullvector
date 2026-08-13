from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def readonly_array(values: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RepairSourceSample:
    sample_id: str
    ordinal: int
    family: str
    family_id: int
    subtype_id: int
    role_id: int
    corpus_seed: int
    sample_seed: int
    part_owner: np.ndarray
    material: np.ndarray
    emission_level: np.ndarray
    guide: np.ndarray
    genes: np.ndarray
    legal_tuples: np.ndarray
    raw_manifest_path: Path
    raw_manifest_bytes: int
    raw_manifest_sha256: str
    raw_archive_path: Path
    raw_archive_bytes: int
    raw_archive_sha256: str
    raw_fields_sha256: str
    compiled_fields_sha256: str
    static_palette_sha256: str

    def __post_init__(self) -> None:
        for name in ("part_owner", "material", "emission_level", "guide", "genes", "legal_tuples"):
            values = getattr(self, name)
            values.setflags(write=False)


@dataclass(frozen=True, slots=True)
class RepairSource:
    generation_manifest_path: Path
    generation_manifest_bytes: int
    generation_manifest_sha256: str
    style_manifest_path: Path
    style_manifest_bytes: int
    style_manifest_sha256: str
    legal_tuple_fingerprint: str
    samples: tuple[RepairSourceSample, ...]


@dataclass(frozen=True, slots=True)
class RepairAnchor:
    name: str
    kind: str
    driver: str
    source_point: tuple[int, int]
    point: tuple[int, int]
    support_point: tuple[int, int]
    displacement: float
    observed_owner: int
    policy: str

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "driver": self.driver,
            "source_point": list(self.source_point),
            "point": list(self.point),
            "support_point": list(self.support_point),
            "displacement": round(float(self.displacement), 9),
            "observed_owner": int(self.observed_owner),
            "policy": self.policy,
        }


@dataclass(frozen=True, slots=True)
class RepairedRigBinding:
    sample_id: str
    family: str
    family_id: int
    subtype_id: int
    role_id: int
    part_owner: np.ndarray
    material: np.ndarray
    emission_level: np.ndarray
    guide: np.ndarray
    genes: np.ndarray
    legal_tuples: np.ndarray
    driver_index: np.ndarray
    owner_masks: np.ndarray
    joints: Mapping[str, RepairAnchor]
    sockets: Mapping[str, RepairAnchor]
    plan: Mapping[str, Any]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "part_owner",
            "material",
            "emission_level",
            "guide",
            "genes",
            "legal_tuples",
            "driver_index",
            "owner_masks",
        ):
            getattr(self, name).setflags(write=False)
        object.__setattr__(self, "joints", MappingProxyType(dict(self.joints)))
        object.__setattr__(self, "sockets", MappingProxyType(dict(self.sockets)))
        object.__setattr__(self, "plan", MappingProxyType(dict(self.plan)))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def sha256(self) -> str:
        return str(self.manifest["binding_sha256"])

    @property
    def raw_fields_sha256(self) -> str:
        return str(self.manifest["raw_fields_sha256"])

    def reconstruct_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        part = np.zeros_like(self.part_owner)
        material = np.zeros_like(self.material)
        emission = np.zeros_like(self.emission_level)
        for owner_id in range(1, self.owner_masks.shape[0] + 1):
            mask = self.owner_masks[owner_id - 1].astype(bool)
            part[mask] = owner_id
            material[mask] = self.material[mask]
            emission[mask] = self.emission_level[mask]
        return part, material, emission

