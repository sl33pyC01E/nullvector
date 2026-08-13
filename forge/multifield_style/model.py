from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..morphology.constants import FAMILIES, ROLE_NAMES, SUBTYPE_NAMES


IMAGE_SIZE = 48
AURA_PART_ID = 16


@dataclass(frozen=True, slots=True)
class StyleCondition:
    sample_id: str
    ordinal: int
    sample_seed: int
    morphology_id: int
    morphology_name: str
    subtype_id: int
    subtype_name: str
    role_id: int
    role_name: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StyleCondition":
        try:
            condition = cls(
                sample_id=str(payload["sample_id"]),
                ordinal=int(payload["ordinal"]),
                sample_seed=int(payload["sample_seed"]),
                morphology_id=int(payload["morphology_id"]),
                morphology_name=str(payload["morphology_name"]),
                subtype_id=int(payload["subtype_id"]),
                subtype_name=str(payload["subtype_name"]),
                role_id=int(payload["role_id"]),
                role_name=str(payload["role_name"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed style condition: {error}") from error
        condition.validate()
        return condition

    def validate(self) -> None:
        if not 0 <= self.morphology_id < len(FAMILIES):
            raise ValueError("morphology_id is outside the canonical vocabulary")
        if self.morphology_name != FAMILIES[self.morphology_id]:
            raise ValueError("morphology name/id mismatch")
        if not 0 <= self.subtype_id < len(SUBTYPE_NAMES):
            raise ValueError("subtype_id is outside the canonical vocabulary")
        if self.subtype_name != SUBTYPE_NAMES[self.subtype_id]:
            raise ValueError("subtype name/id mismatch")
        if self.subtype_id // 4 != self.morphology_id:
            raise ValueError("subtype does not belong to the declared morphology")
        if not 0 <= self.role_id < len(ROLE_NAMES):
            raise ValueError("role_id is outside the canonical vocabulary")
        if self.role_name != ROLE_NAMES[self.role_id]:
            raise ValueError("role name/id mismatch")
        if self.ordinal < 0 or self.sample_seed < 0:
            raise ValueError("ordinal and sample_seed must be non-negative")
        if not self.sample_id or any(character in self.sample_id for character in "/\\"):
            raise ValueError("sample_id must be a safe filename component")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "ordinal": self.ordinal,
            "sample_seed": self.sample_seed,
            "morphology_id": self.morphology_id,
            "morphology_name": self.morphology_name,
            "subtype_id": self.subtype_id,
            "subtype_name": self.subtype_name,
            "role_id": self.role_id,
            "role_name": self.role_name,
        }


@dataclass(frozen=True, slots=True)
class CategoricalFields:
    part: np.ndarray
    material: np.ndarray
    emission: np.ndarray
    aligned_sha256: str

    def __post_init__(self) -> None:
        for name in ("part", "material", "emission"):
            values = getattr(self, name)
            if values.shape != (IMAGE_SIZE, IMAGE_SIZE) or values.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8 {(IMAGE_SIZE, IMAGE_SIZE)}")
            values.flags.writeable = False


@dataclass(frozen=True, slots=True)
class LoadedSourceSample:
    condition: StyleCondition
    fields: CategoricalFields
    raw_fields_sha256: str
    fields_artifact: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields_artifact", MappingProxyType(dict(self.fields_artifact)))


@dataclass(frozen=True, slots=True)
class LoadedGenerationBank:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_bytes: int
    manifest: Mapping[str, Any]
    samples: tuple[LoadedSourceSample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True, slots=True)
class BackgroundCrop:
    theme: str
    pack_manifest_path: Path
    pack_manifest_sha256: str
    image_path: Path
    image_sha256: str
    xy: tuple[int, int]
    rgb: np.ndarray

    def __post_init__(self) -> None:
        if self.rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or self.rgb.dtype != np.uint8:
            raise ValueError("Map crop must be RGB uint8 at native 48px")
        self.rgb.flags.writeable = False


@dataclass(frozen=True, slots=True)
class RenderedLayers:
    base: np.ndarray
    outline: np.ndarray
    emission_core: np.ndarray
    aura: np.ndarray
    bloom_r1: np.ndarray
    bloom_r2: np.ndarray
    composite: np.ndarray
    palette: Mapping[str, Any]
    masks: Mapping[str, np.ndarray]
    accent_pixels: int

