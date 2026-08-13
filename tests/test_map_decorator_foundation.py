from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from forge.map_art.autotile import EAST, NORTH, SOUTH, WEST, cardinal_match_mask, elevation_drop_mask
from forge.map_art.styles import style_for
from forge.map_decorator import (
    CATALOG_SHA256,
    CHANNEL_INDEX,
    EMISSION_CLASS_COUNT,
    FEATURE_CHANNELS,
    FEATURE_CONTRACT_SHA256,
    MAX_DECAL_CLASSES,
    MAX_PROP_CLASSES,
    VARIANT_CLASS_COUNT,
    FoundationCase,
    build_foundation,
    build_legal_class_masks,
    catalog_for,
    catalog_manifest,
    encode_features,
    feature_manifest,
    fuzz_foundation,
    validate_decoration_fields,
    validate_encoded_features,
    validate_feature_inputs,
)
from forge.map_decorator.hashing import json_sha256
from forge.maps import MapConfig, THEMES, generate_map, load_map_pack, write_map_pack


def _authoritative_masks(data: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact generator-captured masks; tests never reconstruct routes."""
    return (
        getattr(data, "protected_backbone"),
        getattr(data, "required_clearance"),
        getattr(data, "decoration_forbidden"),
    )


def _encode(data: object, seed: int = 0xDEC0A7E) -> object:
    protected, clearance, forbidden = _authoritative_masks(data)
    return encode_features(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        public_seed=seed,
    )


def test_feature_manifest_is_complete_versioned_and_self_hashing() -> None:
    manifest = feature_manifest()
    assert manifest["contract_version"] == "1.0.0"
    assert manifest["channel_count"] == 53 == len(FEATURE_CHANNELS)
    assert [channel.index for channel in FEATURE_CHANNELS] == list(range(53))
    assert len({channel.name for channel in FEATURE_CHANNELS}) == 53
    assert CHANNEL_INDEX == {channel.name: channel.index for channel in FEATURE_CHANNELS}
    assert FEATURE_CONTRACT_SHA256 == json_sha256(manifest)
    assert [channel.group for channel in FEATURE_CHANNELS].count("seeded_noise") == 8
    assert manifest["noise"]["scales"] == [1, 2, 4, 8]


@pytest.mark.parametrize("theme_index,theme", enumerate(THEMES))
def test_all_themes_consume_persisted_authoritative_masks(
    theme_index: int, theme: str, tmp_path: Path
) -> None:
    config = MapConfig(width=35 + theme_index, height=32 + (theme_index % 3), spawn_count=5)
    generated = generate_map(0xF0A7_0000 + theme_index, theme, config)
    pack = write_map_pack(generated, tmp_path, preview_scale=2)
    data = load_map_pack(pack)
    protected, clearance, forbidden = _authoritative_masks(data)
    assert protected is data.protected_backbone
    assert clearance is data.required_clearance
    assert forbidden is data.decoration_forbidden
    first = build_foundation(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        public_seed=0xA11CE + theme_index,
    )
    second = build_foundation(
        data,
        protected_backbone=protected.copy(),
        required_clearance=clearance.copy(),
        decoration_forbidden=forbidden.copy(),
        public_seed=0xA11CE + theme_index,
    )
    assert first.report["passed"], first.report
    assert second.report["passed"], second.report
    assert first.features.tensor.shape == (53, config.height, config.width)
    assert first.features.tensor.dtype == np.float32
    assert first.features.tensor.flags.c_contiguous
    assert not first.features.tensor.flags.writeable
    np.testing.assert_array_equal(first.features.tensor, second.features.tensor)
    assert first.features.tensor_sha256 == second.features.tensor_sha256
    assert first.legal_masks.masks_sha256 == second.legal_masks.masks_sha256
    assert np.isfinite(first.features.tensor).all()


def test_exact_adjacency_edges_zone_boundaries_points_and_distances() -> None:
    data = generate_map(0xED6E, "garden", MapConfig(width=39, height=34, spawn_count=6))
    encoded = _encode(data)
    terrain_bits = cardinal_match_mask(data.terrain)
    drop_bits = elevation_drop_mask(data.elevation, data.walkability)
    for direction, bit in zip(("north", "east", "south", "west"), (NORTH, EAST, SOUTH, WEST), strict=True):
        np.testing.assert_array_equal(
            encoded.channel(f"terrain_match.{direction}"), (terrain_bits & bit) != 0
        )
        np.testing.assert_array_equal(
            encoded.channel(f"elevation_drop.{direction}"), (drop_bits & bit) != 0
        )

    height, width = data.shape
    for direction, (dx, dy) in zip(
        ("north", "east", "south", "west"), ((0, -1), (1, 0), (0, 1), (-1, 0)), strict=True
    ):
        observed = encoded.channel(f"zone_boundary.{direction}")
        expected = np.ones(data.shape, dtype=np.float32)
        for y in range(height):
            for x in range(width):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    expected[y, x] = float(data.zone[y, x] != data.zone[ny, nx])
        np.testing.assert_array_equal(observed, expected)

    for name, points in {
        "start": (data.start,),
        "exit": (data.exit,),
        "objective": data.objectives,
        "spawn": data.spawns,
    }.items():
        point_mask = encoded.channel(f"required.{name}")
        assert int(point_mask.sum()) == len(points)
        assert all(point_mask[y, x] == 1.0 for x, y in points)
        distance = encoded.channel(f"distance.{name}")
        assert all(distance[y, x] == 0.0 for x, y in points)
        assert (distance >= 0.0).all() and (distance <= 1.0).all()


def test_public_seed_changes_only_the_eight_noise_channels() -> None:
    data = generate_map(991_177, "caves", MapConfig(width=37, height=33, spawn_count=4))
    first = _encode(data, 1)
    second = _encode(data, 2)
    first_noise = CHANNEL_INDEX["noise.scale_1.a"]
    np.testing.assert_array_equal(first.tensor[:first_noise], second.tensor[:first_noise])
    assert not np.array_equal(first.tensor[first_noise:], second.tensor[first_noise:])
    assert first.tensor_sha256 != second.tensor_sha256
    for channel in FEATURE_CHANNELS[first_noise:]:
        values = first.tensor[channel.index]
        assert (values >= -1.0).all() and (values <= 1.0).all()


def test_one_hot_groups_and_authoritative_masks_are_exact() -> None:
    data = generate_map(0x0A11_A5, "arena", MapConfig(width=38, height=33, spawn_count=5))
    protected, clearance, forbidden = _authoritative_masks(data)
    encoded = encode_features(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        public_seed=77,
    )
    terrain = encoded.tensor[CHANNEL_INDEX["terrain.0"] : CHANNEL_INDEX["terrain.8"] + 1]
    hazard = encoded.tensor[CHANNEL_INDEX["hazard.0"] : CHANNEL_INDEX["hazard.4"] + 1]
    np.testing.assert_array_equal(terrain.sum(axis=0), np.ones(data.shape, dtype=np.float32))
    np.testing.assert_array_equal(hazard.sum(axis=0), np.ones(data.shape, dtype=np.float32))
    np.testing.assert_array_equal(encoded.channel("protected_backbone"), protected)
    np.testing.assert_array_equal(encoded.channel("required_clearance"), clearance)
    np.testing.assert_array_equal(encoded.channel("decoration_forbidden"), forbidden)
    assert encoded.global_conditions["feature_contract_sha256"] == FEATURE_CONTRACT_SHA256
    assert encoded.global_conditions["catalog_sha256"] == CATALOG_SHA256


def test_feature_inputs_are_strict_and_cannot_hide_protected_or_required_cells() -> None:
    data = generate_map(5150, "rooms", MapConfig(width=34, height=34, spawn_count=4))
    protected, clearance, forbidden = _authoritative_masks(data)
    validate_feature_inputs(data, protected, clearance, forbidden)
    with pytest.raises(TypeError, match="uint8"):
        validate_feature_inputs(data, protected.astype(bool), clearance, forbidden)
    malformed = protected.copy()
    malformed[0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_feature_inputs(data, malformed, clearance, forbidden)
    with pytest.raises(ValueError, match="shape"):
        validate_feature_inputs(data, protected[:-1], clearance, forbidden)
    missing_protection = forbidden.copy()
    missing_protection[protected.astype(bool)] = 0
    with pytest.raises(ValueError, match="must include"):
        validate_feature_inputs(data, protected, clearance, missing_protection)
    missing_required = clearance.copy()
    missing_required[data.exit[1], data.exit[0]] = 0
    with pytest.raises(ValueError, match="every objective"):
        validate_feature_inputs(data, protected, missing_required, forbidden)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        encode_features(
            data,
            protected_backbone=protected,
            required_clearance=clearance,
            decoration_forbidden=forbidden,
            public_seed=-1,
        )


def test_encoded_feature_validator_detects_tensor_corruption() -> None:
    data = generate_map(8321, "anomaly", MapConfig(width=35, height=35, spawn_count=5))
    protected, clearance, forbidden = _authoritative_masks(data)
    encoded = _encode(data)
    clean = validate_encoded_features(
        data,
        encoded,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
    )
    assert clean["passed"], clean
    damaged_tensor = encoded.tensor.copy()
    damaged_tensor[0, 2, 2] = 0.25
    damaged = replace(encoded, tensor=damaged_tensor)
    report = validate_encoded_features(
        data,
        damaged,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
    )
    assert not report["passed"]
    assert "tensor_hash" in report["failures"]
    assert "tensor_exact" in report["failures"]


def test_catalog_manifest_tracks_source_and_excludes_every_colliding_prop() -> None:
    manifest = catalog_manifest()
    assert CATALOG_SHA256 == json_sha256(manifest)
    assert manifest["theme_order"] == list(THEMES)
    assert manifest["emission_capability_policy"]["pixels_are_never_inspected"] is True
    assert MAX_DECAL_CLASSES == 3
    assert MAX_PROP_CLASSES == 3
    for theme in THEMES:
        source = style_for(theme).props
        catalog = catalog_for(theme)
        assert catalog.theme == theme
        assert not any(entry.collision for entry in catalog.prop_classes)
        assert set(catalog.excluded_colliding_props) == {
            spec.key for spec in source if spec.kind == "prop" and spec.collision
        }
        assert {entry.key for entry in catalog.decal_classes} == {
            spec.key for spec in source if spec.kind == "decal"
        }
        assert {entry.key for entry in catalog.prop_classes} == {
            spec.key for spec in source if spec.kind == "prop" and not spec.collision
        }


@pytest.mark.parametrize("theme_index,theme", enumerate(THEMES))
def test_legal_masks_are_catalog_exact_and_force_empty_on_hard_cells(theme_index: int, theme: str) -> None:
    data = generate_map(0xCA7A_0000 + theme_index, theme, MapConfig(width=36, height=34, spawn_count=5))
    protected, clearance, forbidden = _authoritative_masks(data)
    masks = build_legal_class_masks(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
    )
    assert masks.variant.shape == (VARIANT_CLASS_COUNT, *data.shape)
    assert masks.decal.shape == (MAX_DECAL_CLASSES, *data.shape)
    assert masks.prop.shape == (MAX_PROP_CLASSES, *data.shape)
    assert masks.emission.shape == (EMISSION_CLASS_COUNT, *data.shape)
    assert masks.variant.all()  # Variant 0 is a real micro-variant, not an empty class.
    assert masks.decal[0].all() and masks.prop[0].all() and masks.emission[0].all()
    assert not masks.decal[1:, masks.hard_empty].any()
    assert not masks.prop[1:, masks.hard_empty].any()
    assert not masks.emission[1:, masks.hard_empty].any()
    catalog = catalog_for(theme)
    available = ~masks.hard_empty
    for entry in catalog.decal_classes:
        expected = available & np.isin(data.terrain, entry.allowed_terrain)
        np.testing.assert_array_equal(masks.decal[entry.class_id], expected)
    for entry in catalog.prop_classes:
        expected = available & (data.walkability != 0) & np.isin(data.terrain, entry.allowed_terrain)
        np.testing.assert_array_equal(masks.prop[entry.class_id], expected)
    for class_id in range(1 + len(catalog.decal_classes), MAX_DECAL_CLASSES):
        assert not masks.decal[class_id].any()
    for class_id in range(1 + len(catalog.prop_classes), MAX_PROP_CLASSES):
        assert not masks.prop[class_id].any()


def test_field_validation_rejects_illegal_hard_cell_missing_class_and_double_object() -> None:
    data = generate_map(44119, "arena", MapConfig(width=38, height=36, spawn_count=5))
    protected, clearance, forbidden = _authoritative_masks(data)
    masks = build_legal_class_masks(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
    )
    fields = {name: np.zeros(data.shape, dtype=np.uint8) for name in ("variant", "decal", "prop", "emission")}
    clean = validate_decoration_fields(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        **fields,
    )
    assert clean["passed"], clean

    hard_y, hard_x = np.argwhere(masks.hard_empty)[0]
    illegal_hard = {name: value.copy() for name, value in fields.items()}
    illegal_hard["prop"][hard_y, hard_x] = 1
    report = validate_decoration_fields(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        **illegal_hard,
    )
    assert not report["passed"] and "illegal.prop" in report["failures"]

    missing_class = {name: value.copy() for name, value in fields.items()}
    missing_class["prop"][0, 0] = 2  # Global slot exists, but arena exposes only prop class 1.
    report = validate_decoration_fields(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        **missing_class,
    )
    assert not report["passed"] and "illegal.prop" in report["failures"]

    shared = masks.decal[1] & masks.prop[1]
    y, x = np.argwhere(shared)[0]
    doubled = {name: value.copy() for name, value in fields.items()}
    doubled["decal"][y, x] = 1
    doubled["prop"][y, x] = 1
    report = validate_decoration_fields(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        **doubled,
    )
    assert not report["passed"]
    assert "objects.multiple_classes_per_cell" in report["failures"]


def test_emission_legality_is_refined_by_selected_catalog_class_without_rgb_inspection() -> None:
    data = generate_map(0xA2C4_1E, "archipelago", MapConfig(width=44, height=40, spawn_count=5))
    protected, clearance, forbidden = _authoritative_masks(data)
    potential = build_legal_class_masks(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
    )
    wave = next(entry for entry in catalog_for("archipelago").decal_classes if entry.key == "wave_glyph")
    water_candidates = potential.decal[wave.class_id] & (data.walkability == 0)
    y, x = np.argwhere(water_candidates)[0]
    fields = {name: np.zeros(data.shape, dtype=np.uint8) for name in ("variant", "decal", "prop", "emission")}
    fields["emission"][y, x] = 1
    without_object = validate_decoration_fields(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        **fields,
    )
    assert not without_object["passed"]
    assert "illegal.emission" in without_object["failures"]
    fields["decal"][y, x] = wave.class_id
    with_object = validate_decoration_fields(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        **fields,
    )
    assert with_object["passed"], with_object


def test_foundation_performs_no_disk_io(monkeypatch: pytest.MonkeyPatch) -> None:
    data = generate_map(0xD15C, "caves", MapConfig(width=32, height=32, spawn_count=4))
    protected, clearance, forbidden = _authoritative_masks(data)

    def reject_disk(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"foundation attempted disk I/O: {args!r} {kwargs!r}")

    monkeypatch.setattr("builtins.open", reject_disk)
    result = build_foundation(
        data,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        public_seed=1234,
    )
    assert result.report["passed"]


def test_small_all_theme_fuzz_has_unique_deterministic_contract_hashes() -> None:
    cases: list[FoundationCase] = []
    for index in range(18):
        theme = THEMES[index % len(THEMES)]
        data = generate_map(
            0xF022_0000 + index,
            theme,
            MapConfig(width=32 + index % 5, height=32 + index % 4, spawn_count=4 + index % 3),
        )
        protected, clearance, forbidden = _authoritative_masks(data)
        cases.append(
            FoundationCase(
                data=data,
                protected_backbone=protected,
                required_clearance=clearance,
                decoration_forbidden=forbidden,
                public_seed=0x5000 + index,
            )
        )
    report = fuzz_foundation(cases, require_all_themes=True)
    assert report["passed"], report
    assert report["case_count"] == 18
    assert report["unique_feature_tensors"] == 18
    assert report["disk_io"] is False
    assert all(report["per_theme"][theme] == 3 for theme in THEMES)
    signatures = {
        encode_features(
            case.data,
            protected_backbone=case.protected_backbone,
            required_clearance=case.required_clearance,
            decoration_forbidden=case.decoration_forbidden,
            public_seed=case.public_seed,
        ).tensor_sha256
        for case in cases
    }
    aggregate = hashlib.sha256("".join(sorted(signatures)).encode("ascii")).hexdigest()
    assert len(aggregate) == 64


def test_fuzz_reports_a_bad_case_without_aborting_following_cases() -> None:
    data = generate_map(0xBAD_C45E, "garden", MapConfig(width=32, height=32, spawn_count=4))
    protected, clearance, forbidden = _authoritative_masks(data)
    malformed = FoundationCase(
        data=data,
        protected_backbone=protected.astype(bool),
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        public_seed=1,
    )
    clean = FoundationCase(data, protected, clearance, forbidden, 2)
    report = fuzz_foundation((malformed, clean))
    assert not report["passed"]
    assert report["case_count"] == 2
    assert report["unique_feature_tensors"] == 1
    assert report["failures"][0]["failures"] == ["exception"]
    assert report["failures"][0]["error"].startswith("TypeError:")
