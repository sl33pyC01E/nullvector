from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


IMAGE_SIZE = 48
ATLAS_COLUMNS = 16
LAYER_NAMES = (
    "base",
    "outline",
    "emission_core",
    "aura",
    "bloom_r1",
    "bloom_r2",
    "composite",
)
JOINT_NAMES = (
    "root",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "appendage_base",
    "weapon_mount",
)
SOCKET_NAMES = (
    "focus",
    "muzzle",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
    "appendage_tip",
)


@dataclass(frozen=True, slots=True)
class MotionStyleCondition:
    sample_id: str
    ordinal: int
    sample_seed: int
    morphology_id: int
    morphology_name: str
    subtype_id: int
    subtype_name: str
    role_id: int
    role_name: str


@dataclass(frozen=True, slots=True)
class IdentityStyleFields:
    """Duck-typed public renderer input with a specimen-stable style key.

    ``aligned_sha256`` intentionally names the style identity, not the moving
    frame's categorical hash. The true frame hash is independently recorded
    and verified before and after rendering.
    """

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
class LoadedMotionBank:
    asset_root: Path
    asset_index_path: Path
    asset_index_sha256: str
    asset_index_bytes: int
    index: Mapping[str, Any]
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest_bytes: int
    source_manifest: Mapping[str, Any]
    source_archive_path: Path
    source_archive_sha256: str
    source_archive_bytes: int
    sources: Mapping[str, Mapping[str, Any]]
    atlases: Mapping[str, Mapping[str, Any]]
    clips_by_family: Mapping[str, tuple[Mapping[str, Any], ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", MappingProxyType(dict(self.index)))
        object.__setattr__(self, "source_manifest", MappingProxyType(dict(self.source_manifest)))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        object.__setattr__(self, "atlases", MappingProxyType(dict(self.atlases)))
        object.__setattr__(self, "clips_by_family", MappingProxyType(dict(self.clips_by_family)))


@dataclass(frozen=True, slots=True)
class FamilyPayload:
    family: str
    file_payloads: Mapping[str, bytes]
    family_manifest: Mapping[str, Any]
    frame_count: int
    clip_count: int


@dataclass(frozen=True, slots=True)
class FrameAudit:
    layers: Mapping[str, np.ndarray]
    palette: Mapping[str, Any]
    palette_sha256: str
    categorical_sha256: str
    joint_sha256: str
    socket_sha256: str
    authority_sha256: str
    presentation_sha256: tuple[str, ...]
    gates: Mapping[str, bool]
