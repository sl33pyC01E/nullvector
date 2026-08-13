from __future__ import annotations

import hashlib
import json
from pathlib import Path
from PIL import Image
import pytest

from forge.neural_workshop_sync import (
    DEFAULT_DESTINATION,
    EXPECTED_FAMILIES,
    EXPECTED_CENSUS_CATEGORIES,
    EXPECTED_CENSUS_FAMILY_COUNTS,
    EXPECTED_CENSUS_REJECTIONS,
    EXPECTED_NEURAL_MOTION_REPRESENTATIVES,
    INDEX_FORMAT,
    MAP_LAYERS,
    MAP_THEMES,
    MIN_FREE_BYTES,
    PLANNED_BYTES,
    PRESERVED_GAME_FILES,
    RUNTIME_EXTENSIONS,
    SourceContractError,
    STATIC_LAYERS,
    NEURAL_MOTION_REPLAY_SCHEMA,
    _validate_manifest_schema,
    _validate_neural_motion_census,
    asset_inventory,
    sync_neural_workshop_assets,
    validate_synced_assets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = DEFAULT_DESTINATION / "asset_index.json"
PROJECT_CONFIG = PROJECT_ROOT / "game" / "project.godot"
ARENA_SCENE = PROJECT_ROOT / "game" / "Arena.tscn"
WORKSHOP_SCENE = PROJECT_ROOT / "game" / "NeuralWorkshop.tscn"
WORKSHOP_SCRIPT = PROJECT_ROOT / "game" / "scripts" / "neural_workshop.gd"


def _load_index(path: Path = INDEX_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_in_workshop_bundle_is_complete_current_and_staged_safely() -> None:
    assert validate_synced_assets(INDEX_PATH) == []
    index = _load_index()
    assert index["format"] == INDEX_FORMAT
    assert index["status"] == "ready"
    assert index["pixel_filter"] == "nearest"
    assert index["native_scale_options"] == [1, 4]
    assert index["python_runtime_required"] is False
    assert index["runtime_asset_extensions"] == list(RUNTIME_EXTENSIONS)
    assert index["disk_budget"] == {
        "guard_passed": True,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "planned_bytes": PLANNED_BYTES,
    }
    motion = index["motion"]
    assert motion["status"] in {"staged", "rejected", "ready"}
    if motion["status"] != "ready":
        assert motion["available"] is False
        assert motion["neural_output"] is False
        assert motion["fail_closed"] is True
        assert motion["identities"] == []
        assert motion["clip_count"] == 0
        assert motion["frame_count"] == 0
        assert motion["expected"]["source_sample_count"] == 80
        assert motion["expected"]["bindable_count"] == 70
        assert motion["expected"]["rejected_count"] == 10
        assert motion["expected"]["representatives"] == [
            {"family": family, "sample_id": sample_id, "static_cell": cell}
            for family, sample_id, cell in EXPECTED_NEURAL_MOTION_REPRESENTATIVES
        ]


def test_static_neural_bank_covers_every_identity_filter_and_layer() -> None:
    static = _load_index()["static"]
    assert static["status"] == "ready"
    assert tuple(static["layers"]) == STATIC_LAYERS
    assert static["identity_count"] == 80
    identities = static["identities"]
    assert len(identities) == 80
    assert [identity["cell"] for identity in identities] == list(range(80))
    assert len({identity["sample_id"] for identity in identities}) == 80
    assert {identity["family"] for identity in identities} == set(EXPECTED_FAMILIES)
    assert {identity["subtype_id"] for identity in identities} == set(range(20))
    assert {identity["role_id"] for identity in identities} == set(range(8))
    assert {identity["variant"] for identity in identities} == {0, 1}
    assert {
        (identity["family"], identity["role"], identity["variant"])
        for identity in identities
    } == {
        (family, role, variant)
        for family in EXPECTED_FAMILIES
        for role in ("striker", "defender", "scout", "controller", "support", "artillery", "harvester", "disruptor")
        for variant in (0, 1)
    }
    for atlas in static["atlases"]:
        path = INDEX_PATH.parent / atlas["path"]
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            assert image.size == (768, 240)


def test_map_browser_bank_is_bound_to_exact_topology_v2_manifests() -> None:
    maps = _load_index()["maps"]
    assert maps["status"] == "ready"
    assert tuple(maps["themes"]) == MAP_THEMES
    assert tuple(maps["layers"]) == MAP_LAYERS
    assert maps["topology_schema_version"] == "2.0.0"
    for entry in maps["maps"]:
        topology = entry["topology_contract"]
        assert topology["schema_version"] == "2.0.0"
        assert topology["all_invariants_passed"] is True
        assert topology["invariant_count"] >= 50
        assert topology["semantic_array_sha256"] == entry["source_semantic_sha256"]
        topology_manifest = json.loads(
            (INDEX_PATH.parent / entry["topology_manifest"]["path"]).read_text(encoding="utf-8")
        )
        assert topology_manifest["schema_version"] == "2.0.0"
        assert topology_manifest["semantic_array_sha256"] == entry["source_semantic_sha256"]
        assert all(value["passed"] for value in topology_manifest["topology"]["invariants"])


def test_runtime_inventory_is_json_png_only_and_exactly_hashed() -> None:
    inventory = asset_inventory(INDEX_PATH)
    assert len(inventory) == _load_index()["asset_count"]
    assert len({record["path"] for record in inventory}) == len(inventory)
    assert {Path(record["path"]).suffix for record in inventory} <= set(RUNTIME_EXTENSIONS)
    assert sum(record["bytes"] for record in inventory) < 32 * 1024**2
    for record in inventory:
        path = INDEX_PATH.parent / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]


def test_sync_is_byte_reproducible_in_fresh_destinations(tmp_path: Path) -> None:
    a = sync_neural_workshop_assets(destination=tmp_path / "a")
    b = sync_neural_workshop_assets(destination=tmp_path / "b")
    assert a.index["bundle_id"] == b.index["bundle_id"]
    files_a = {
        path.relative_to(a.destination).as_posix(): path.read_bytes()
        for path in a.destination.rglob("*")
        if path.is_file()
    }
    files_b = {
        path.relative_to(b.destination).as_posix(): path.read_bytes()
        for path in b.destination.rglob("*")
        if path.is_file()
    }
    assert files_a == files_b


def test_motion_adapter_rejects_partial_or_untrusted_bank_fail_closed(tmp_path: Path) -> None:
    motion_root = tmp_path / "motion"
    motion_root.mkdir()
    (motion_root / "motion_style_neural_manifest.json").write_text(
        json.dumps(
            {
                "format": "nullvector-multifield-style-neural-motion-bank-v1",
                "status": "ready",
                "neural_output": True,
                "identity_count": 5,
                "clip_count": 520,
                "frame_count": 4720,
                "identities": [],
            }
        ),
        encoding="utf-8",
    )
    (motion_root / "verification_report.json").write_text(
        json.dumps(
            {
                "format": "nullvector-multifield-style-neural-motion-replay-v1",
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )
    result = sync_neural_workshop_assets(
        destination=tmp_path / "bundle",
        neural_motion_source=motion_root,
    )
    motion = result.index["motion"]
    assert motion["status"] == "rejected"
    assert motion["available"] is False
    assert motion["neural_output"] is False
    assert motion["identities"] == []
    assert motion["clip_count"] == 0
    assert motion["frame_count"] == 0
    assert "rejected fail-closed" in motion["reasons"][0]
    assert "multifield_style_neural_motion_bank.schema.json" in motion["reasons"][0]
    assert validate_synced_assets(result.index_path) == []


def test_require_neural_motion_ready_rejects_an_explicit_staged_source(
    tmp_path: Path,
) -> None:
    # Do not couple this rejection test to the mutable production output.  Once
    # that output is promoted to ready, the default source should (correctly)
    # pass the release gate.  An empty, explicit motion root is the canonical
    # staged fixture because both authoritative manifests are still absent.
    staged_motion = tmp_path / "staged-motion"
    staged_motion.mkdir()
    with pytest.raises(SourceContractError, match="Neural motion was required"):
        sync_neural_workshop_assets(
            destination=tmp_path / "bundle",
            neural_motion_source=staged_motion,
            require_neural_motion_ready=True,
        )


def _expected_census() -> dict:
    return {
        "format": "nullvector-neural-rig-binding-census-v1",
        "scope": "all-80-immutable-production-samples",
        "sample_count": 80,
        "bindable_count": 70,
        "rejected_count": 10,
        "family_counts": [dict(value) for value in EXPECTED_CENSUS_FAMILY_COUNTS],
        "rejection_categories": [dict(value) for value in EXPECTED_CENSUS_CATEGORIES],
        "rejections": [
            {
                "family": family,
                "candidate_ordinal_within_family": ordinal,
                "sample_id": sample_id,
                "category": category,
                "reason": f"exact {category} rejection",
            }
            for family, ordinal, sample_id, category in EXPECTED_CENSUS_REJECTIONS
        ],
        "animation_bank_scope": {
            "selected_identity_count": 5,
            "all_80_animated": False,
            "policy": "first-bank-ordered-full-matrix-valid-identity-per-family-v1",
            "binding_census_does_not_imply_animation": True,
        },
    }


def test_all_80_census_is_exact_and_bound_to_static_identity_cells() -> None:
    static_identities = _load_index()["static"]["identities"]
    census = _expected_census()
    _validate_neural_motion_census(census, static_identities)
    tampered = json.loads(json.dumps(census))
    tampered["family_counts"][2]["bindable_count"] = 14
    with pytest.raises(SourceContractError, match="family counts"):
        _validate_neural_motion_census(tampered, static_identities)
    tampered = json.loads(json.dumps(census))
    tampered["rejections"][0]["sample_id"] = "0000_f0_s00_r0_v00"
    with pytest.raises(SourceContractError, match="rejection registry"):
        _validate_neural_motion_census(tampered, static_identities)


def _strict_replay_payload() -> dict:
    return {
        "format": "nullvector-multifield-style-neural-motion-replay-v1",
        "status": "passed",
        "neural_output": True,
        "manifest": {
            "path": "outputs/multifield_style_neural_motion/motion_style_neural_manifest.json",
            "bytes": 123,
            "sha256": "1" * 64,
        },
        "compiler_source_sha256": "2" * 64,
        "identity_results": [
            {
                "family": family,
                "sample_id": sample_id,
                "exact": True,
                "artifact_count": 12,
                "bytes_compared": 100,
                "shard_count": 4,
            }
            for family, sample_id, _cell in EXPECTED_NEURAL_MOTION_REPRESENTATIVES
        ],
        "identity_count": 5,
        "clip_count": 520,
        "frame_count": 4720,
        "artifact_count_compared": 63,
        "bytes_compared": 1000,
        "exact_identity_replay": True,
        "exact_showcase_replay": True,
        "all_gates_passed": True,
    }


def test_replay_schema_requires_the_complete_ordered_63_artifact_proof() -> None:
    payload = _strict_replay_payload()
    _validate_manifest_schema(payload, NEURAL_MOTION_REPLAY_SCHEMA, label="test replay")
    tampered = json.loads(json.dumps(payload))
    tampered["identity_results"][0]["artifact_count"] = 11
    with pytest.raises(SourceContractError, match="artifact_count"):
        _validate_manifest_schema(tampered, NEURAL_MOTION_REPLAY_SCHEMA, label="test replay")
    tampered = json.loads(json.dumps(payload))
    tampered["exact_showcase_replay"] = False
    with pytest.raises(SourceContractError, match="exact_showcase_replay"):
        _validate_manifest_schema(tampered, NEURAL_MOTION_REPLAY_SCHEMA, label="test replay")


def test_validator_detects_runtime_byte_tampering(tmp_path: Path) -> None:
    result = sync_neural_workshop_assets(destination=tmp_path / "bundle")
    record = result.index["static"]["atlases"][0]
    path = result.destination / record["path"]
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    errors = validate_synced_assets(result.index_path)
    assert any("hash inventory" in error for error in errors)


def test_validator_requires_reference_inventory_closure(tmp_path: Path) -> None:
    result = sync_neural_workshop_assets(destination=tmp_path / "bundle")
    index = _load_index(result.index_path)
    index["static"]["atlases"][0]["sha256"] = "0" * 64
    result.index_path.write_text(
        json.dumps(index, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    errors = validate_synced_assets(result.index_path)
    assert any("referenced artifact record mismatch" in error for error in errors)


def test_workshop_is_additive_native_and_arena_main_is_byte_preserved() -> None:
    project = PROJECT_CONFIG.read_text(encoding="utf-8")
    scene = WORKSHOP_SCENE.read_text(encoding="utf-8")
    script = WORKSHOP_SCRIPT.read_text(encoding="utf-8")
    assert 'run/main_scene="res://Arena.tscn"' in project
    assert 'path="res://scripts/neural_workshop.gd"' in scene
    assert "res://generated/neural_workshop/v1/asset_index.json" in script
    assert "TEXTURE_FILTER_NEAREST" in script
    assert "NEURAL_WORKSHOP_SMOKE_OK" in script
    assert "REJECTED\\nFAIL-CLOSED" in script
    assert "_motion_playback_frame_count" in script
    assert "stored_count - 1" in script
    assert "motion_atlas_regions != 4720" in script
    assert "texture.get_width() != 768 or texture.get_height() != 2832" in script
    assert "_refresh_motion()" in script.split("func _toggle_scale()", 1)[1].split("func ", 1)[0]
    for family, sample_id, cell in EXPECTED_NEURAL_MOTION_REPRESENTATIVES:
        assert f'"{family}": ["{sample_id}", {cell}]' in script
    assert "OS.execute" not in script
    assert "python_runtime_required" in script
    game_root = PROJECT_ROOT / "game"
    for relative, expected_sha in PRESERVED_GAME_FILES.items():
        assert _sha256(game_root / relative) == expected_sha
    assert ARENA_SCENE.is_file()
