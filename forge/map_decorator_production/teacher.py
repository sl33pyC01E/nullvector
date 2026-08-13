from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..map_art.hashing import bounded_hash
from ..map_art.objects import derive_instances
from ..map_art.styles import style_for
from ..map_decorator.catalog import (
    CATALOG_SHA256,
    LegalClassMasks,
    build_legal_class_masks,
    catalog_for,
    validate_decoration_fields,
)
from ..map_decorator.contract import FEATURE_CONTRACT_SHA256
from ..map_decorator.features import encode_features
from ..map_decorator.hashing import array_sha256, json_sha256, named_arrays_sha256
from ..map_decorator_ml.contract import (
    HEAD_NAMES,
    TEACHER_PROJECTION_VERSION,
    global_condition_vector,
)
from ..map_decorator_ml.dataset import split_for_identity
from ..maps.io import ARRAY_NAMES, array_digest
from ..maps.model import (
    GENERATOR_NAME,
    GENERATOR_VERSION,
    MAP_SCHEMA_VERSION,
    THEMES,
    TOPOLOGY_MASK_NAMES,
    MapData,
    Terrain,
)


SEMANTIC_TEACHER_VERSION = "map-art-semantic-only-legal-projection-v1"
_VARIANT_SALT = 0x56415249414E54


@dataclass(slots=True)
class ProductionSample:
    data: MapData
    feature_seed: int
    split: str
    full_map_identity_sha256: str
    sample_identity_sha256: str
    source_semantic_sha256: str
    topology_masks_sha256: str
    features: np.ndarray
    feature_tensor_sha256: str
    targets: dict[str, np.ndarray]
    target_fields_sha256: str
    legal_masks: dict[str, np.ndarray]
    hard_empty: np.ndarray
    legal_masks_sha256: str
    global_conditions: np.ndarray
    replay_sha256: str


def canonical_full_map_identity(data: MapData) -> dict[str, object]:
    arrays = data.arrays()
    topology = {name: arrays[name] for name in TOPOLOGY_MASK_NAMES}
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "map_id": data.map_id,
        "seed": int(data.seed),
        "theme": data.theme,
        "dimensions": {"width": data.config.width, "height": data.config.height},
        "generator_name": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "generator_config": data.config.to_dict(),
        "semantic_array_sha256": array_digest(arrays),
        "topology_masks_sha256": array_digest(topology),
    }


def full_map_identity_sha256(data: MapData) -> str:
    return json_sha256(canonical_full_map_identity(data))


def semantic_teacher_targets(
    data: MapData,
) -> tuple[dict[str, np.ndarray], LegalClassMasks, dict[str, object]]:
    """Extract the renderer's categorical semantics without allocating any RGB frames."""
    height, width = data.shape
    variant = np.empty(data.shape, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            variant[y, x] = bounded_hash(data.seed, x, y, _VARIANT_SALT, 8)

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
    instances = derive_instances(data, style_for(data.theme))
    for instance in sorted(instances, key=lambda item: (item.catalog_index, item.instance_id)):
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
    targets = {
        "variant": np.ascontiguousarray(variant),
        "decal": np.ascontiguousarray(decal),
        "prop": np.ascontiguousarray(prop),
        "emission": np.ascontiguousarray(emission),
    }
    report = validate_decoration_fields(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        **targets,
    )
    if not report["passed"]:
        raise RuntimeError(f"Semantic-only teacher violated legality: {report}")
    return targets, conditional, report


def _map_replay_identity(data: MapData) -> dict[str, object]:
    return {
        "semantic_arrays_sha256": array_digest(data.arrays()),
        "points": {
            "start": list(data.start),
            "exit": list(data.exit),
            "objectives": [list(point) for point in data.objectives],
            "spawns": [list(point) for point in data.spawns],
        },
        "repair_count": int(data.repair_count),
        "theme_parameters": data.metadata.get("theme_parameters", {}),
        "protected_backbone_segments": int(
            data.metadata.get("protected_backbone_segments", 1 + len(data.objectives))
        ),
    }


def build_production_sample(
    data: MapData,
    *,
    feature_seed: int,
    replay_data: MapData | None = None,
) -> ProductionSample:
    full_hash = full_map_identity_sha256(data)
    split = split_for_identity(full_hash)
    encoded = encode_features(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        public_seed=feature_seed,
    )
    targets, legal, _ = semantic_teacher_targets(data)
    legal_arrays = {
        name: np.ascontiguousarray(getattr(legal, name), dtype=bool) for name in HEAD_NAMES
    }
    hard_empty = np.ascontiguousarray(legal.hard_empty, dtype=bool)
    target_hash = named_arrays_sha256(targets)
    sample_identity = {
        "full_map_identity_sha256": full_hash,
        "feature_public_seed": int(feature_seed),
        "crop": None,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "teacher_projection_version": TEACHER_PROJECTION_VERSION,
        "semantic_teacher_version": SEMANTIC_TEACHER_VERSION,
        "feature_tensor_sha256": encoded.tensor_sha256,
        "target_fields_sha256": target_hash,
    }
    replay_hash = json_sha256(
        {
            "map": _map_replay_identity(data),
            "feature_tensor_sha256": encoded.tensor_sha256,
            "target_fields_sha256": target_hash,
            "legal_masks_sha256": legal.masks_sha256,
        }
    )
    if replay_data is not None:
        if _map_replay_identity(data) != _map_replay_identity(replay_data):
            raise RuntimeError("Generated map failed exact semantic/point replay.")
        replay_encoded = encode_features(
            replay_data,
            protected_backbone=replay_data.protected_backbone,
            required_clearance=replay_data.required_clearance,
            decoration_forbidden=replay_data.decoration_forbidden,
            public_seed=feature_seed,
        )
        replay_targets, replay_legal, _ = semantic_teacher_targets(replay_data)
        replay_identity = json_sha256(
            {
                "map": _map_replay_identity(replay_data),
                "feature_tensor_sha256": replay_encoded.tensor_sha256,
                "target_fields_sha256": named_arrays_sha256(replay_targets),
                "legal_masks_sha256": replay_legal.masks_sha256,
            }
        )
        if replay_identity != replay_hash:
            raise RuntimeError("Teacher sample failed exact feature/target/legality replay.")
    topology = {name: data.arrays()[name] for name in TOPOLOGY_MASK_NAMES}
    return ProductionSample(
        data=data,
        feature_seed=int(feature_seed),
        split=split,
        full_map_identity_sha256=full_hash,
        sample_identity_sha256=json_sha256(sample_identity),
        source_semantic_sha256=array_digest(
            {name: data.arrays()[name] for name in ARRAY_NAMES}
        ),
        topology_masks_sha256=array_digest(topology),
        features=np.ascontiguousarray(encoded.tensor, dtype=np.float32),
        feature_tensor_sha256=array_sha256(encoded.tensor),
        targets=targets,
        target_fields_sha256=target_hash,
        legal_masks=legal_arrays,
        hard_empty=hard_empty,
        legal_masks_sha256=legal.masks_sha256,
        global_conditions=np.ascontiguousarray(global_condition_vector(encoded), dtype=np.float32),
        replay_sha256=replay_hash,
    )


def sample_record(sample: ProductionSample) -> dict[str, object]:
    data = sample.data
    return {
        "map_id": data.map_id,
        "seed": int(data.seed),
        "feature_seed": sample.feature_seed,
        "theme": data.theme,
        "width": data.config.width,
        "height": data.config.height,
        "objective_count": len(data.objectives),
        "spawn_count": len(data.spawns),
        "generator_config": data.config.to_dict(),
        "split": sample.split,
        "full_map_identity_sha256": sample.full_map_identity_sha256,
        "sample_identity_sha256": sample.sample_identity_sha256,
        "source_semantic_sha256": sample.source_semantic_sha256,
        "topology_masks_sha256": sample.topology_masks_sha256,
        "feature_tensor_sha256": sample.feature_tensor_sha256,
        "target_fields_sha256": sample.target_fields_sha256,
        "legal_masks_sha256": sample.legal_masks_sha256,
        "replay_sha256": sample.replay_sha256,
        "points": {
            "start": list(data.start),
            "exit": list(data.exit),
            "objectives": [list(point) for point in data.objectives],
            "spawns": [list(point) for point in data.spawns],
        },
        "repair_count": int(data.repair_count),
        "theme_parameters": data.metadata.get("theme_parameters", {}),
        "protected_backbone_segments": int(
            data.metadata.get("protected_backbone_segments", 1 + len(data.objectives))
        ),
    }


def stacked_sample_arrays(samples: list[ProductionSample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("Cannot stack an empty production shard.")
    shape = samples[0].data.shape
    theme = samples[0].data.theme
    objective_count = len(samples[0].data.objectives)
    spawn_count = len(samples[0].data.spawns)
    if any(
        sample.data.shape != shape
        or sample.data.theme != theme
        or len(sample.data.objectives) != objective_count
        or len(sample.data.spawns) != spawn_count
        for sample in samples
    ):
        raise ValueError("Production shards must be homogeneous in shape/theme/point counts.")
    arrays: dict[str, np.ndarray] = {
        "features": np.ascontiguousarray(np.stack([sample.features for sample in samples])),
        "hard_empty": np.ascontiguousarray(np.stack([sample.hard_empty for sample in samples])),
        "global_conditions": np.ascontiguousarray(
            np.stack([sample.global_conditions for sample in samples]), dtype=np.float32
        ),
        "theme_index": np.full((len(samples),), THEMES.index(theme), dtype=np.int64),
        "seeds": np.asarray([sample.data.seed for sample in samples], dtype=np.uint64),
        "feature_seeds": np.asarray([sample.feature_seed for sample in samples], dtype=np.uint64),
        "start": np.asarray([sample.data.start for sample in samples], dtype=np.int16),
        "exit": np.asarray([sample.data.exit for sample in samples], dtype=np.int16),
        "objectives": np.asarray([sample.data.objectives for sample in samples], dtype=np.int16),
        "spawns": np.asarray([sample.data.spawns for sample in samples], dtype=np.int16),
        "repair_count": np.asarray([sample.data.repair_count for sample in samples], dtype=np.int16),
    }
    for name in HEAD_NAMES:
        arrays[f"target_{name}"] = np.ascontiguousarray(
            np.stack([sample.targets[name] for sample in samples]), dtype=np.uint8
        )
        arrays[f"legal_{name}"] = np.ascontiguousarray(
            np.stack([sample.legal_masks[name] for sample in samples]), dtype=bool
        )
    for name in ARRAY_NAMES:
        arrays[f"semantic_{name}"] = np.ascontiguousarray(
            np.stack([sample.data.arrays()[name] for sample in samples])
        )
    return arrays

