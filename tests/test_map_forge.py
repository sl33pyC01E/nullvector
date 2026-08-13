from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from forge.maps.cli import build_parser, fuzz_maps, fuzz_maps_isolated
from forge.maps.generator import generate_map
from forge.maps.io import ARRAY_NAMES, array_digest, file_sha256, load_map_pack, write_map_pack
from forge.maps.model import (
    GENERATOR_VERSION,
    MAP_SCHEMA_VERSION,
    TOPOLOGY_MASK_NAMES,
    Hazard,
    MapConfig,
    THEMES,
    Terrain,
)
from forge.maps.validate import validate_map, validate_pack


def _legacy_array_digest(arrays: dict[str, np.ndarray]) -> str:
    """Reference the published v1 digest byte contract for regression tests."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def test_array_digest_matches_published_legacy_byte_contract() -> None:
    arrays = {
        "bool_2d": np.asarray([[True, False], [False, True]], dtype=np.bool_),
        "float_3d": np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 7.0,
        "int_1d": np.asarray([-32768, -1, 0, 32767], dtype=np.int16),
        "scalar": np.asarray(0x12345678, dtype=np.uint32),
        "uint_2d": np.arange(15, dtype=np.uint8).reshape(3, 5),
    }
    assert array_digest(arrays) == _legacy_array_digest(arrays)


@pytest.mark.parametrize("theme_index,theme", enumerate(THEMES))
def test_all_themes_are_deterministic_and_valid(theme_index: int, theme: str) -> None:
    config = MapConfig(width=52, height=48, objective_count=3, spawn_count=10)
    seed = 90_000 + theme_index
    first = generate_map(seed, theme, config)
    second = generate_map(seed, theme, config)
    assert validate_map(first)["passed"]
    assert first.start == second.start
    assert first.exit == second.exit
    assert first.objectives == second.objectives
    assert first.spawns == second.spawns
    assert array_digest(first.arrays()) == array_digest(second.arrays())
    for name in first.arrays():
        np.testing.assert_array_equal(first.arrays()[name], second.arrays()[name])
    required = (first.start, first.exit, *first.objectives)
    all_points = (*required, *first.spawns)
    assert all(first.protected_backbone[y, x] == 1 for x, y in required)
    assert all(first.required_clearance[y, x] == 1 for x, y in all_points)
    expected_forbidden = (
        (first.protected_backbone != 0)
        | (first.required_clearance != 0)
        | (first.hazard != 0)
    ).astype(np.uint8)
    np.testing.assert_array_equal(first.decoration_forbidden, expected_forbidden)


def test_themes_have_distinct_semantic_signatures() -> None:
    config = MapConfig(width=56, height=56)
    maps = {theme: generate_map(0xBADC0DE, theme, config) for theme in THEMES}
    signatures = {theme: array_digest(data.arrays()) for theme, data in maps.items()}
    assert len(set(signatures.values())) == len(THEMES)
    assert np.any(maps["archipelago"].terrain == int(Terrain.WATER))
    assert np.any(maps["garden"].terrain == int(Terrain.GROWTH))
    assert np.any(maps["anomaly"].terrain == int(Terrain.CHASM))
    assert np.any(maps["anomaly"].terrain == int(Terrain.CRYSTAL))


def test_pack_round_trip_schema_hashes_and_preview(tmp_path: Path) -> None:
    data = generate_map(123_456_789, "archipelago", MapConfig(width=48, height=44))
    pack = write_map_pack(data, tmp_path, preview_scale=4)
    assert sorted(path.name for path in pack.iterdir()) == ["manifest.json", "preview.png", "semantics.npz"]
    report = validate_pack(pack)
    assert report["passed"], report
    assert report["replay_report"]["passed"]
    loaded = load_map_pack(pack)
    assert loaded.map_id == data.map_id
    assert array_digest(loaded.arrays()) == array_digest(data.arrays())
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MAP_SCHEMA_VERSION == "2.0.0"
    assert manifest["generator"]["version"] == GENERATOR_VERSION == "2.0.0"
    assert manifest["semantic_array_hash_algorithm"] == "sha256-canonical-named-arrays-v1"
    assert set(manifest["semantics"]["arrays"]) == set(ARRAY_NAMES)
    topology = manifest["semantics"]["topology_masks"]
    assert topology["capture_policy"].endswith("never reconstructed")
    assert topology["hash_algorithm"] == manifest["semantic_array_hash_algorithm"]
    assert set(topology["members"]) == set(TOPOLOGY_MASK_NAMES)
    for name in TOPOLOGY_MASK_NAMES:
        assert topology["members"][name]["cell_count"] == int((data.arrays()[name] != 0).sum())
        assert topology["members"][name]["hash_scope"] == "single_named_array"
        assert len(topology["members"][name]["sha256"]) == 64
    # Skip-existing is append-safe: it validates and reuses rather than overwriting.
    assert write_map_pack(data, tmp_path, preview_scale=4, skip_existing=True) == pack


def test_pack_writer_is_byte_deterministic_across_output_roots(tmp_path: Path) -> None:
    data = generate_map(0xB17E_5AFE, "anomaly", MapConfig(width=40, height=36, spawn_count=5))
    first = write_map_pack(data, tmp_path / "first", preview_scale=3)
    second = write_map_pack(data, tmp_path / "second", preview_scale=3)
    for name in ("manifest.json", "preview.png", "semantics.npz"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert validate_pack(first)["passed"]
    assert validate_pack(second)["passed"]


def test_validator_rejects_semantic_drift() -> None:
    original = generate_map(444, "rooms", MapConfig(width=48, height=48))
    broken_walkability = original.walkability.copy()
    broken_walkability[original.start[1], original.start[0]] = 0
    broken = replace(original, walkability=broken_walkability)
    report = validate_map(broken)
    assert not report["passed"]
    assert "semantic.walkability_matches_terrain" in report["failures"]
    assert "points.required_walkable" in report["failures"]


def test_validator_rejects_topology_mask_domain_subset_and_exact_union_drift() -> None:
    original = generate_map(0xA11CE, "garden", MapConfig(width=42, height=40, spawn_count=6))
    wrong_dtype = replace(
        original, protected_backbone=original.protected_backbone.astype(np.bool_)
    )
    report = validate_map(wrong_dtype)
    assert not report["passed"]
    assert "dtype.protected_backbone" in report["failures"]

    wrong_shape = replace(
        original, required_clearance=original.required_clearance[:-1, :]
    )
    report = validate_map(wrong_shape)
    assert not report["passed"]
    assert "shape.required_clearance" in report["failures"]

    malformed = original.protected_backbone.copy()
    malformed[original.start[1], original.start[0]] = 2
    report = validate_map(replace(original, protected_backbone=malformed))
    assert not report["passed"]
    assert "semantic.protected_backbone_binary" in report["failures"]

    missing_clearance = original.required_clearance.copy()
    missing_clearance[original.exit[1], original.exit[0]] = 0
    report = validate_map(replace(original, required_clearance=missing_clearance))
    assert not report["passed"]
    assert "safety.required_clearance_covers_points" in report["failures"]

    missing_forbidden = original.decoration_forbidden.copy()
    y, x = np.argwhere(original.protected_backbone != 0)[0]
    missing_forbidden[y, x] = 0
    report = validate_map(replace(original, decoration_forbidden=missing_forbidden))
    assert not report["passed"]
    assert "safety.decoration_forbidden_exact_union" in report["failures"]


def test_load_explicitly_rejects_legacy_schema_without_fabricating_masks(tmp_path: Path) -> None:
    data = generate_map(0x1E6AC7, "rooms", MapConfig(width=38, height=38, spawn_count=5))
    pack = write_map_pack(data, tmp_path, preview_scale=3)
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be fabricated"):
        load_map_pack(pack)
    report = validate_pack(pack)
    assert not report["passed"]
    assert report["schema_errors"]


def test_load_rejects_schema_valid_manifest_consistency_tamper(tmp_path: Path) -> None:
    data = generate_map(0x4A41_FE57, "arena", MapConfig(width=40, height=38, spawn_count=5))
    pack = write_map_pack(data, tmp_path, preview_scale=3)
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["statistics"]["protected_backbone_cells"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest disagrees"):
        load_map_pack(pack)
    report = validate_pack(pack)
    assert not report["passed"]
    assert "manifest statistics disagree with validated semantics" in report["artifact_errors"]

    manifest["statistics"]["protected_backbone_cells"] -= 1
    manifest["topology"]["protected_backbone_segments"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="generator metadata disagree"):
        load_map_pack(pack)
    report = validate_pack(pack)
    assert not report["passed"]
    assert "manifest protected backbone segment count is inconsistent" in report["artifact_errors"]


def test_fully_rehashed_mask_tamper_is_rejected_by_exact_replay(tmp_path: Path) -> None:
    data = generate_map(0x7A4E_E2, "caves", MapConfig(width=44, height=42, spawn_count=6))
    pack = write_map_pack(data, tmp_path, preview_scale=3)
    arrays = {name: value.copy() for name, value in data.arrays().items()}

    candidate: tuple[int, int] | None = None
    for y in range(1, data.config.height - 1):
        for x in range(1, data.config.width - 1):
            if (
                data.walkability[y, x]
                and data.hazard[y, x] == 0
                and data.decoration_forbidden[y, x] == 0
                and any(
                    data.protected_backbone[y + dy, x + dx]
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
            ):
                candidate = (x, y)
                break
        if candidate is not None:
            break
    assert candidate is not None
    x, y = candidate
    arrays["protected_backbone"][y, x] = 1
    arrays["decoration_forbidden"][y, x] = 1
    np.savez_compressed(pack / "semantics.npz", **{name: arrays[name] for name in ARRAY_NAMES})

    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["semantic_array_sha256"] = array_digest(arrays)
    manifest["artifacts"]["arrays"]["sha256"] = file_sha256(pack / "semantics.npz")
    topology_arrays = {name: arrays[name] for name in TOPOLOGY_MASK_NAMES}
    topology = manifest["semantics"]["topology_masks"]
    topology["combined_sha256"] = array_digest(topology_arrays)
    for name in TOPOLOGY_MASK_NAMES:
        topology["members"][name]["sha256"] = array_digest({name: arrays[name]})
        topology["members"][name]["cell_count"] = int((arrays[name] != 0).sum())
    manifest["statistics"]["protected_backbone_cells"] = int(
        (arrays["protected_backbone"] != 0).sum()
    )
    manifest["statistics"]["decoration_forbidden_cells"] = int(
        (arrays["decoration_forbidden"] != 0).sum()
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic replay"):
        load_map_pack(pack)
    report = validate_pack(pack)
    assert not report["passed"]
    assert report["map_report"]["passed"]
    assert not report["replay_report"]["passed"]
    assert "arrays.protected_backbone" in report["replay_report"]["failures"]
    assert "arrays.decoration_forbidden" in report["replay_report"]["failures"]


def test_validator_requires_a_hazard_free_mission_route() -> None:
    original = generate_map(7719, "rooms", MapConfig(width=48, height=48))
    hazards = np.where(
        original.walkability > 0, int(Hazard.LASER), int(Hazard.NONE)
    ).astype(np.uint8)
    for x, y in (original.start, original.exit, *original.objectives):
        hazards[y, x] = int(Hazard.NONE)
    report = validate_map(replace(original, hazard=hazards))
    assert not report["passed"]
    assert "topology.required_connected" not in report["failures"]
    assert "topology.required_hazard_free_connected" in report["failures"]


def test_unsigned_64_bit_seed_is_preserved() -> None:
    seed = (1 << 64) - 1
    data = generate_map(seed, "caves", MapConfig(width=48, height=48))
    assert data.seed == seed
    assert data.map_id.startswith("caves-ffffffffffffffff-")
    assert validate_map(data)["passed"]


def test_small_fuzz_matrix_has_full_validity_and_uniqueness() -> None:
    report = fuzz_maps(72, base_seed=0x5155414C495459, width=48, height=48)
    assert report["passed"], report["failures"]
    assert report["passed_count"] == 72
    assert report["unique_semantic_maps"] == 72
    assert all(count == 12 for count in report["per_theme"].values())


def test_isolated_fuzz_aggregation_exactly_matches_in_process_reference() -> None:
    kwargs = {
        "base_seed": 0x49534F4C41544544,
        "width": 40,
        "height": 44,
    }
    reference = fuzz_maps(24, include_hashes=True, **kwargs)
    isolated = fuzz_maps_isolated(
        24,
        chunk_size=7,
        worker_retries=0,
        worker_timeout_seconds=120,
        **kwargs,
    )
    assert isolated["passed"], isolated
    for key in (
        "passed_count",
        "failure_count",
        "per_theme",
        "unique_semantic_maps",
        "repair_total",
        "minimum_start_exit_path_length",
        "maximum_start_exit_path_length",
        "minimum_walkable_ratio",
        "maximum_walkable_ratio",
        "semantic_identity_sha256",
        "failures",
    ):
        assert isolated[key] == reference[key]
    assert isolated["isolation"]["processed_index_count"] == 24
    assert isolated["isolation"]["retry_count"] == 0


def test_isolated_fuzz_retries_an_abrupt_worker_exit_with_telemetry() -> None:
    report = fuzz_maps_isolated(
        12,
        base_seed=0x4352415348524554,
        width=40,
        height=40,
        chunk_size=6,
        worker_retries=1,
        worker_timeout_seconds=120,
        _inject_worker_exit_start_index=0,
    )
    assert report["passed"], report
    assert report["passed_count"] == 12
    assert report["unique_semantic_maps"] == 12
    isolation = report["isolation"]
    assert isolation["worker_count"] == 2
    assert isolation["attempt_count"] == 3
    assert isolation["retry_count"] == 1
    assert isolation["processed_index_count"] == 12
    first, retry = isolation["attempts"][:2]
    assert first["start_index"] == retry["start_index"] == 0
    assert first["attempt"] == 1 and first["exit_code"] == 86
    assert retry["attempt"] == 2 and retry["exit_code"] == 0


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        MapConfig(width=31)
    with pytest.raises(ValueError):
        MapConfig(objective_count=0)


def test_generate_default_output_root_is_schema_major_versioned() -> None:
    args = build_parser().parse_args(["generate"])
    assert Path(args.output).name == "maps_v2"
