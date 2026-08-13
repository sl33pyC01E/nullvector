from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


BINDING_FORMAT = "nullvector-neural-rig-binding-v1"
BINDER_VERSION = "neural-owner-rig-bridge-v1"
REPLAY_FORMAT = "nullvector-neural-rig-binding-replay-v1"
ADAPTER_FORMAT = "nullvector-neural-rig-motion-adapter-v1"
FRAME_FORMAT = "nullvector-bound-rig-frame-v1"

DRIVER_NAMES = (
    "body",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "appendage",
    "weapon",
)

DRIVER_INDEX = {name: index for index, name in enumerate(DRIVER_NAMES)}
BACKGROUND_DRIVER = np.uint8(255)
MIN_DRIVER_PIXELS = {
    "body": 8,
    "head": 3,
    "left_arm": 3,
    "right_arm": 3,
    "left_leg": 3,
    "right_leg": 3,
    "appendage": 3,
    "weapon": 3,
}

JOINT_DRIVER = {
    "root": "body",
    "head": "head",
    "left_shoulder": "left_arm",
    "right_shoulder": "right_arm",
    "left_hip": "left_leg",
    "right_hip": "right_leg",
    "appendage_base": "appendage",
    "weapon_mount": "weapon",
}

SOCKET_DRIVER = {
    "focus": "body",
    "muzzle": "weapon",
    "left_hand": "left_arm",
    "right_hand": "right_arm",
    "left_foot": "left_leg",
    "right_foot": "right_leg",
    "appendage_tip": "appendage",
}


class BindingRejected(ValueError):
    """Raised when neural fields cannot be bound without inventing anatomy."""

    def __init__(self, errors: list[str] | tuple[str, ...]):
        self.errors = tuple(str(error) for error in errors)
        super().__init__("Neural rig binding rejected: " + "; ".join(self.errors))


def readonly_array(values: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def readonly_points(
    values: Mapping[str, tuple[int, int] | list[int]],
) -> Mapping[str, tuple[int, int]]:
    if not isinstance(values, Mapping):
        raise ValueError("anatomy points must be a mapping")
    result: dict[str, tuple[int, int]] = {}
    for name, point in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("anatomy point names must be nonempty strings")
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError(f"anatomy point {name!r} must be an (x, y) pair")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in point
        ):
            raise ValueError(f"anatomy point {name!r} must contain integers")
        result[name] = (int(point[0]), int(point[1]))
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class DerivedAnatomy:
    """Named conditioning points only; it intentionally contains no pixels."""

    joints: Mapping[str, tuple[int, int]]
    sockets: Mapping[str, tuple[int, int]]
    source: str = "derived_anatomy"
    source_sha256: str | None = None

    @classmethod
    def from_mappings(
        cls,
        joints: Mapping[str, tuple[int, int] | list[int]],
        sockets: Mapping[str, tuple[int, int] | list[int]],
        *,
        source: str = "derived_anatomy",
        source_sha256: str | None = None,
    ) -> "DerivedAnatomy":
        return cls(
            joints=readonly_points(joints),
            sockets=readonly_points(sockets),
            source=source,
            source_sha256=source_sha256,
        )


@dataclass(frozen=True, slots=True)
class RigAnchor:
    name: str
    kind: str
    point: tuple[int, int]
    support_point: tuple[int, int]
    driver: str
    observed_owner: int
    source: str

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "point": [self.point[0], self.point[1]],
            "support_point": [self.support_point[0], self.support_point[1]],
            "driver": self.driver,
            "observed_owner": self.observed_owner,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class OwnerLayerBinding:
    owner_id: int
    owner_name: str
    pixel_count: int
    drivers: tuple[str, ...]
    tuple_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
            "pixel_count": self.pixel_count,
            "drivers": list(self.drivers),
            "tuple_sha256": self.tuple_sha256,
        }


@dataclass(frozen=True, slots=True)
class NeuralRigBinding:
    sample_id: str
    family: str
    family_id: int
    subtype_id: int
    role_id: int
    corpus_seed: int | None
    part_owner: np.ndarray
    material: np.ndarray
    emission_level: np.ndarray
    guide: np.ndarray
    genes: np.ndarray | None
    legal_tuples: np.ndarray
    owner_masks: np.ndarray
    driver_index: np.ndarray
    joints: Mapping[str, RigAnchor]
    sockets: Mapping[str, RigAnchor]
    owner_layers: tuple[OwnerLayerBinding, ...]
    anatomy: DerivedAnatomy
    upstream_hashes: Mapping[str, str]
    manifest: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.manifest["hashes"]["binding_sha256"])

    @property
    def raw_fields_sha256(self) -> str:
        return str(self.manifest["source"]["raw_fields_sha256"])

    def reconstruct_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recompose the rest fields from disjoint owner layers exactly."""
        part = np.zeros_like(self.part_owner)
        material = np.zeros_like(self.material)
        emission = np.zeros_like(self.emission_level)
        for owner_id in range(1, self.owner_masks.shape[0] + 1):
            mask = self.owner_masks[owner_id - 1].astype(bool)
            part[mask] = owner_id
            material[mask] = self.material[mask]
            emission[mask] = self.emission_level[mask]
        return part, material, emission


@dataclass(frozen=True, slots=True)
class BoundRigFrame:
    part_owner: np.ndarray
    material: np.ndarray
    emission_level: np.ndarray
    driver_index: np.ndarray
    manifest: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.manifest["hashes"]["frame_sha256"])
