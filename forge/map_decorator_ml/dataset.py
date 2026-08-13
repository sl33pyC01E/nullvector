from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..map_art.renderer import render_map_art
from ..map_decorator.catalog import (
    CATALOG_SHA256,
    LegalClassMasks,
    build_legal_class_masks,
    catalog_for,
    validate_decoration_fields,
)
from ..map_decorator.features import EncodedFeatures, encode_features
from ..map_decorator.hashing import array_sha256, json_sha256, named_arrays_sha256
from ..maps.io import MANIFEST_FILE, load_map_pack
from ..maps.model import MapData, Terrain, THEMES
from .contract import (
    FEATURE_CONTRACT_SHA256,
    HEAD_CLASS_COUNTS,
    HEAD_NAMES,
    SPLIT_POLICY_VERSION,
    TEACHER_PROJECTION_VERSION,
    global_condition_vector,
)


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class Crop:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if min(self.x, self.y) < 0:
            raise ValueError("Crop origin cannot be negative.")
        if not (32 <= self.width <= 256 and 32 <= self.height <= 256):
            raise ValueError("Crop dimensions must each be in [32, 256].")

    def slices(self, full_shape: tuple[int, int]) -> tuple[slice, slice]:
        full_height, full_width = full_shape
        if self.x + self.width > full_width or self.y + self.height > full_height:
            raise ValueError("Crop extends outside its complete source map.")
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class TeacherRecord:
    pack_path: Path
    public_seed: int
    crop: Crop | None = None


@dataclass(slots=True)
class TeacherSample:
    features: np.ndarray
    targets: dict[str, np.ndarray]
    legal_masks: dict[str, np.ndarray]
    hard_empty: np.ndarray
    global_conditions: np.ndarray
    theme_index: int
    split: str
    full_map_identity_sha256: str
    sample_identity_sha256: str
    source_semantic_sha256: str
    feature_tensor_sha256: str
    target_fields_sha256: str
    map_id: str
    crop: Crop | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.features.shape[-2], self.features.shape[-1]


def _manifest_for_pack(pack_path: Path) -> tuple[Path, dict[str, object]]:
    path = Path(pack_path)
    manifest_path = path if path.name == MANIFEST_FILE else path / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Strict map manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Map manifest root must be an object.")
    return manifest_path.parent, manifest


def canonical_full_map_identity(manifest: dict[str, object]) -> dict[str, object]:
    """Identity excludes crops and feature noise, pinning all views to one split."""
    generator = manifest.get("generator")
    topology = manifest.get("semantics")
    if not isinstance(generator, dict) or not isinstance(topology, dict):
        raise ValueError("Map manifest is missing generator or semantics identity.")
    topology_masks = topology.get("topology_masks")
    if not isinstance(topology_masks, dict):
        raise ValueError("Map manifest lacks authoritative persisted topology-mask identity.")
    required = (
        "schema_version",
        "map_id",
        "seed",
        "theme",
        "dimensions",
        "semantic_array_sha256",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"Map manifest identity is incomplete: {missing}")
    return {
        "schema_version": manifest["schema_version"],
        "map_id": manifest["map_id"],
        "seed": manifest["seed"],
        "theme": manifest["theme"],
        "dimensions": manifest["dimensions"],
        "generator_name": generator.get("name"),
        "generator_version": generator.get("version"),
        "generator_config": generator.get("config"),
        "semantic_array_sha256": manifest["semantic_array_sha256"],
        "topology_masks_sha256": topology_masks.get("combined_sha256"),
    }


def split_for_identity(full_map_identity_sha256: str) -> str:
    if not isinstance(full_map_identity_sha256, str) or len(full_map_identity_sha256) != 64:
        raise ValueError("full_map_identity_sha256 must be a hexadecimal SHA-256 string.")
    try:
        int(full_map_identity_sha256, 16)
        bucket = int(full_map_identity_sha256[:16], 16) % 100
    except ValueError as error:
        raise ValueError("full_map_identity_sha256 is not hexadecimal.") from error
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _teacher_targets(data: MapData) -> tuple[dict[str, np.ndarray], LegalClassMasks, dict[str, object]]:
    """Project deterministic map-art semantics into the safe theme-local catalog."""
    layers = render_map_art(data)
    variant = np.ascontiguousarray(layers.variant, dtype=np.uint8)
    decal = np.zeros(data.shape, dtype=np.uint8)
    prop = np.zeros(data.shape, dtype=np.uint8)
    occupied = np.zeros(data.shape, dtype=bool)
    catalog = catalog_for(data.theme)
    decal_by_source = {entry.catalog_index: entry for entry in catalog.decal_classes}
    prop_by_source = {entry.catalog_index: entry for entry in catalog.prop_classes}
    base_masks = build_legal_class_masks(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
    )
    # Stable source-catalog order gives a single winner if the legacy renderer placed
    # one decal and one prop on the same cell. Colliding props are absent by construction.
    for instance in sorted(layers.instances, key=lambda item: (item.catalog_index, item.instance_id)):
        x, y = instance.cell
        if occupied[y, x] or base_masks.hard_empty[y, x] or instance.collision:
            continue
        if instance.kind == "decal":
            entry = decal_by_source.get(instance.catalog_index)
            if entry is not None and base_masks.decal[entry.class_id, y, x]:
                decal[y, x] = entry.class_id
                occupied[y, x] = True
        elif instance.kind == "prop":
            entry = prop_by_source.get(instance.catalog_index)
            if entry is not None and base_masks.prop[entry.class_id, y, x]:
                prop[y, x] = entry.class_id
                occupied[y, x] = True

    conditional = build_legal_class_masks(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        selected_variant=variant,
        selected_decal=decal,
        selected_prop=prop,
    )
    emission = np.zeros(data.shape, dtype=np.uint8)
    capable = conditional.emission[1]
    emission[capable] = 1
    emission[(data.terrain == int(Terrain.CRYSTAL)) & capable] = 2
    for field, entries in ((decal, catalog.decal_classes), (prop, catalog.prop_classes)):
        for entry in entries:
            if entry.emission_capable:
                level = 3 if entry.color_role == "secondary" else 2
                emission[(field == entry.class_id) & capable] = level
    targets = {"variant": variant, "decal": decal, "prop": prop, "emission": emission}
    report = validate_decoration_fields(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        **targets,
    )
    if not report["passed"]:
        raise RuntimeError(f"Deterministic teacher projection violated legality: {report}")
    return targets, conditional, report


def load_teacher_sample(record: TeacherRecord) -> TeacherSample:
    pack_dir, manifest = _manifest_for_pack(record.pack_path)
    # The strict loader verifies schema, files, hashes, invariants, and deterministic replay.
    data = load_map_pack(pack_dir)
    if data.protected_backbone is None or data.required_clearance is None or data.decoration_forbidden is None:
        raise ValueError("Authoritative topology masks are required and may never be reconstructed.")
    full_identity = canonical_full_map_identity(manifest)
    full_hash = json_sha256(full_identity)
    split = split_for_identity(full_hash)
    encoded: EncodedFeatures = encode_features(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        public_seed=record.public_seed,
    )
    targets, legal, _ = _teacher_targets(data)

    crop_payload: dict[str, int] | None = None
    if record.crop is None:
        ys, xs = slice(None), slice(None)
    else:
        ys, xs = record.crop.slices(data.shape)
        crop_payload = record.crop.to_dict()
    features = np.ascontiguousarray(encoded.tensor[:, ys, xs], dtype=np.float32)
    cropped_targets = {
        name: np.ascontiguousarray(targets[name][ys, xs], dtype=np.uint8) for name in HEAD_NAMES
    }
    legal_masks = {
        name: np.ascontiguousarray(getattr(legal, name)[:, ys, xs], dtype=bool)
        for name in HEAD_NAMES
    }
    hard_empty = np.ascontiguousarray(legal.hard_empty[ys, xs], dtype=bool)
    source_hash = str(manifest["semantic_array_sha256"])
    sample_identity = {
        "full_map_identity_sha256": full_hash,
        "feature_public_seed": int(record.public_seed),
        "crop": crop_payload,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "teacher_projection_version": TEACHER_PROJECTION_VERSION,
        "feature_tensor_sha256": array_sha256(features),
        "target_fields_sha256": named_arrays_sha256(cropped_targets),
    }
    return TeacherSample(
        features=features,
        targets=cropped_targets,
        legal_masks=legal_masks,
        hard_empty=hard_empty,
        global_conditions=global_condition_vector(encoded),
        theme_index=THEMES.index(data.theme),
        split=split,
        full_map_identity_sha256=full_hash,
        sample_identity_sha256=json_sha256(sample_identity),
        source_semantic_sha256=source_hash,
        feature_tensor_sha256=array_sha256(features),
        target_fields_sha256=named_arrays_sha256(cropped_targets),
        map_id=data.map_id,
        crop=record.crop,
    )


class TeacherDataset(Dataset[TeacherSample]):
    def __init__(
        self,
        records: Sequence[TeacherRecord],
        *,
        expected_split: str | None = None,
    ) -> None:
        if not records:
            raise ValueError("TeacherDataset requires at least one record.")
        if expected_split is not None and expected_split not in SPLIT_NAMES:
            raise ValueError(f"Unknown expected split {expected_split!r}.")
        self.records = tuple(records)
        self.expected_split = expected_split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TeacherSample:
        sample = load_teacher_sample(self.records[index])
        if self.expected_split is not None and sample.split != self.expected_split:
            raise ValueError(
                f"Sample {sample.map_id} belongs to {sample.split}, not {self.expected_split}."
            )
        return sample


def corpus_identity(samples: Iterable[TeacherSample]) -> str:
    materialized = tuple(samples)
    assert_no_split_leakage(materialized)
    identities = sorted(sample.sample_identity_sha256 for sample in materialized)
    if not identities:
        raise ValueError("Cannot identify an empty teacher corpus.")
    if len(identities) != len(set(identities)):
        raise ValueError("Teacher corpus contains a duplicate sample identity.")
    return json_sha256(
        {
            "format": "nullvector-map-decorator-teacher-corpus-v1",
            "split_policy_version": SPLIT_POLICY_VERSION,
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "teacher_projection_version": TEACHER_PROJECTION_VERSION,
            "sample_identity_sha256": identities,
        }
    )


def assert_no_split_leakage(samples: Iterable[TeacherSample]) -> None:
    observed: dict[str, str] = {}
    for sample in samples:
        previous = observed.setdefault(sample.full_map_identity_sha256, sample.split)
        if previous != sample.split:
            raise ValueError(
                "Crops/augmentations from one complete map crossed split boundaries: "
                f"{sample.full_map_identity_sha256}."
            )
        expected = split_for_identity(sample.full_map_identity_sha256)
        if sample.split != expected:
            raise ValueError("A sample carries a split inconsistent with its full map identity.")


def collate_teacher_samples(samples: Sequence[TeacherSample]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    assert_no_split_leakage(samples)
    batch = len(samples)
    max_h = max(sample.shape[0] for sample in samples)
    max_w = max(sample.shape[1] for sample in samples)
    features = torch.zeros((batch, 53, max_h, max_w), dtype=torch.float32)
    targets = {
        name: torch.zeros((batch, max_h, max_w), dtype=torch.long) for name in HEAD_NAMES
    }
    legal = {
        name: torch.zeros((batch, classes, max_h, max_w), dtype=torch.bool)
        for name, classes in HEAD_CLASS_COUNTS.items()
    }
    valid = torch.zeros((batch, max_h, max_w), dtype=torch.bool)
    hard_empty = torch.ones((batch, max_h, max_w), dtype=torch.bool)
    global_conditions = torch.empty((batch, 8), dtype=torch.float32)
    themes = torch.empty((batch,), dtype=torch.long)
    for index, sample in enumerate(samples):
        height, width = sample.shape
        features[index, :, :height, :width] = torch.from_numpy(sample.features.copy())
        valid[index, :height, :width] = True
        hard_empty[index, :height, :width] = torch.from_numpy(sample.hard_empty.copy())
        for name in HEAD_NAMES:
            targets[name][index, :height, :width] = torch.from_numpy(
                sample.targets[name].copy().astype(np.int64)
            )
            legal[name][index, :, :height, :width] = torch.from_numpy(
                sample.legal_masks[name].copy()
            )
            legal[name][index, 0, height:, :] = True
            legal[name][index, 0, :, width:] = True
        global_conditions[index] = torch.from_numpy(sample.global_conditions.copy())
        themes[index] = sample.theme_index
    return {
        "features": features,
        "targets": targets,
        "legal_masks": legal,
        "valid_cells": valid,
        "hard_empty": hard_empty,
        "global_conditions": global_conditions,
        "theme_index": themes,
        "sample_identity_sha256": [sample.sample_identity_sha256 for sample in samples],
        "split": [sample.split for sample in samples],
    }
