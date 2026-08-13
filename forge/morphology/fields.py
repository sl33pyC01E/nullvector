from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .constants import (
    ARMOR,
    BODY,
    CANVAS_SIZE,
    CORE,
    DETAIL,
    EMISSION,
    EMISSION_LEVEL_NAMES,
    GUIDE_CHANNEL_NAMES,
    LAYER_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    STRUCTURAL_LAYERS,
    SUBTYPE_NAMES,
    WEAPON,
)
from .genome import MorphologyGenome


@dataclass(frozen=True, slots=True)
class MorphologyTrainingFields:
    guide: np.ndarray
    part_owner: np.ndarray
    material: np.ndarray
    emission_level: np.ndarray
    morphology_index: int
    subtype_id: int
    role_id: int
    genes: np.ndarray

    def arrays_hash(self) -> str:
        digest = hashlib.sha256()
        for values in (
            self.guide,
            self.part_owner,
            self.material,
            self.emission_level,
            self.genes,
        ):
            digest.update(values.dtype.str.encode("ascii"))
            digest.update(str(values.shape).encode("ascii"))
            digest.update(values.tobytes())
        digest.update(bytes((self.morphology_index, self.subtype_id, self.role_id)))
        return digest.hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "morphology_index": self.morphology_index,
            "subtype_id": self.subtype_id,
            "subtype_name": SUBTYPE_NAMES[self.subtype_id],
            "role_id": self.role_id,
            "role_name": ROLE_NAMES[self.role_id],
            "genes_shape": list(self.genes.shape),
            "guide": {
                "shape": list(self.guide.shape),
                "dtype": "float32",
                "channel_names": list(GUIDE_CHANNEL_NAMES),
            },
            "targets": {
                "part_owner": {
                    "shape": list(self.part_owner.shape),
                    "dtype": "uint8",
                    "vocabulary": list(PART_OWNER_NAMES),
                },
                "material": {
                    "shape": list(self.material.shape),
                    "dtype": "uint8",
                    "vocabulary": list(MATERIAL_NAMES),
                },
                "emission_level": {
                    "shape": list(self.emission_level.shape),
                    "dtype": "uint8",
                    "vocabulary": list(EMISSION_LEVEL_NAMES),
                },
            },
            "arrays_sha256": self.arrays_hash(),
        }


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    return np.logical_or.reduce(
        [
            padded[y : y + mask.shape[0], x : x + mask.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
    )


def _point_map(points: dict[str, list[int]], radius: int = 1) -> np.ndarray:
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(image)
    for x, y in points.values():
        draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=255)
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.float32)


def _skeleton_map(joints: dict[str, list[int]]) -> np.ndarray:
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(image)
    edges = (
        ("root", "head"),
        ("root", "left_shoulder"),
        ("root", "right_shoulder"),
        ("root", "left_hip"),
        ("root", "right_hip"),
        ("root", "appendage_base"),
        ("head", "weapon_mount"),
    )
    for first, second in edges:
        draw.line([tuple(joints[first]), tuple(joints[second])], fill=255, width=1)
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.float32)


def _guide(
    layers: np.ndarray,
    joints: dict[str, list[int]],
    sockets: dict[str, list[int]],
) -> np.ndarray:
    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    root_x, root_y = joints["root"]
    yy, xx = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
    horizontal = np.clip(
        0.5 + (xx.astype(np.float32) - float(root_x)) / CANVAS_SIZE,
        0.0,
        1.0,
    )
    maximum_distance = float(np.hypot(CANVAS_SIZE - 1, CANVAS_SIZE - 1))
    root_distance = 1.0 - np.hypot(xx - root_x, yy - root_y) / maximum_distance
    return np.stack(
        (
            structural.astype(np.float32),
            layers[BODY].astype(np.float32),
            _skeleton_map(joints),
            _point_map(joints),
            _point_map(sockets),
            layers[CORE].astype(np.float32),
            horizontal.astype(np.float32),
            root_distance.astype(np.float32),
        ),
        axis=0,
    ).astype(np.float32)


def _part_owner(
    layers: np.ndarray,
    genome: MorphologyGenome,
    joints: dict[str, list[int]],
    sockets: dict[str, list[int]],
) -> np.ndarray:
    owner = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    for index in range(len(LAYER_NAMES)):
        owner[layers[index] > 0] = index + 1
    # Core is a true owned anatomical region; emission is an aligned field, not
    # permission to erase the core category wherever the masks overlap.
    owner[layers[CORE] > 0] = CORE + 1
    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    ornament = (layers[DETAIL] > 0) & ~(layers[EMISSION] > 0)
    yy, xx = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
    ornament &= ((xx + yy + genome.seed) & 1) == 0
    owner[ornament] = 15
    aura = _dilate(layers[EMISSION] > 0) & ~structural
    owner[aura] = 16
    for x, y in joints.values():
        owner[y, x] = 13
    for x, y in sockets.values():
        owner[y, x] = 14
    return owner


def build_training_fields(
    layers: np.ndarray,
    genome: MorphologyGenome,
    joints: dict[str, list[int]],
    sockets: dict[str, list[int]],
) -> MorphologyTrainingFields:
    guide = _guide(layers, joints, sockets)
    part_owner = _part_owner(layers, genome, joints, sockets)
    structural = part_owner > 0
    base_material = (1, 2, 3, 7, 4)[genome.family]
    material = np.zeros_like(part_owner)
    material[structural] = base_material
    material[layers[ARMOR] > 0] = 5
    material[layers[WEAPON] > 0] = 6
    material[layers[CORE] > 0] = 7
    material[layers[DETAIL] > 0] = 8
    material[(layers[EMISSION] > 0) | (part_owner == 16)] = 9

    emission_level = np.zeros_like(part_owner)
    emission_level[layers[DETAIL] > 0] = 1
    emission_level[layers[EMISSION] > 0] = 2
    emission_level[(layers[CORE] > 0) | (part_owner == 16)] = 3
    result = MorphologyTrainingFields(
        guide=guide,
        part_owner=part_owner,
        material=material,
        emission_level=emission_level,
        morphology_index=genome.family,
        subtype_id=genome.subtype_id,
        role_id=genome.role_id,
        genes=genome.condition_vector(),
    )
    validate_training_fields(result)
    return result


def validate_training_fields(fields: MorphologyTrainingFields) -> None:
    expected_spatial = (CANVAS_SIZE, CANVAS_SIZE)
    if fields.guide.shape != (len(GUIDE_CHANNEL_NAMES), *expected_spatial):
        raise ValueError(f"guide has invalid shape {fields.guide.shape}")
    if fields.guide.dtype != np.float32 or not np.isfinite(fields.guide).all():
        raise ValueError("guide must be finite float32")
    if not ((fields.guide >= 0.0).all() and (fields.guide <= 1.0).all()):
        raise ValueError("guide values must stay in [0, 1]")
    arrays = (
        (fields.part_owner, len(PART_OWNER_NAMES), "part_owner"),
        (fields.material, len(MATERIAL_NAMES), "material"),
        (fields.emission_level, len(EMISSION_LEVEL_NAMES), "emission_level"),
    )
    for values, count, name in arrays:
        if values.shape != expected_spatial or values.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 {expected_spatial}")
        if int(values.max()) >= count:
            raise ValueError(f"{name} exceeds its categorical vocabulary")
    if fields.genes.shape != (24,) or fields.genes.dtype != np.float32:
        raise ValueError("genes must be a float32 vector of length 24")
    if not 0 <= fields.morphology_index < 5:
        raise ValueError("morphology_index must be in [0, 4]")
    if not 0 <= fields.subtype_id < len(SUBTYPE_NAMES):
        raise ValueError("subtype_id must be in [0, 19]")
    if not 0 <= fields.role_id < len(ROLE_NAMES):
        raise ValueError("role_id must be in [0, 7]")


def legal_field_tuples(
    specimens: list[MorphologyTrainingFields] | tuple[MorphologyTrainingFields, ...],
) -> np.ndarray:
    """Return the sorted categorical triples observed in a training split.

    Using only the train split avoids leaking validation examples while giving
    the sampler a hard compatibility table for part/material/emission tuples.
    """
    if not specimens:
        raise ValueError("At least one training specimen is required.")
    observed: set[tuple[int, int, int]] = set()
    for fields in specimens:
        validate_training_fields(fields)
        triples = np.stack(
            (fields.part_owner, fields.material, fields.emission_level), axis=-1
        ).reshape(-1, 3)
        observed.update(tuple(map(int, values)) for values in triples)
    return np.asarray(sorted(observed), dtype=np.uint8)
