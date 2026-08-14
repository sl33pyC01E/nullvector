from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAME = ROOT / "game"
SCRIPT = GAME / "scripts" / "forge_hub.gd"
SCENE = GAME / "ForgeHub.tscn"


LABS = {
    "Arena.tscn": None,
    "ForgeLab.tscn": ("generated/v2/asset_index.json", "nullvector-forge-lab-assets-v1", "motion.clip_count", 520, False),
    "NeuralWorkshop.tscn": ("generated/neural_workshop/v1/asset_index.json", "nullvector-neural-workshop-assets-v1", "static.identity_count", 80, True),
    "NeuralGeneticsWorkshop.tscn": ("generated/neural_genetics/v3/asset_index.json", "nullvector-neural-genetics-workshop-assets-v3", "asset_count", 407, True),
    "RepairedMotionLab.tscn": ("generated/repaired_motion_lab/v1/catalog.json", "nullvector-repaired-motion-native-catalog-v1", "counts.frame_count", 75_520, True),
    "SubtypeMotionLab.tscn": ("generated/morphology_subtype_lab/v1/catalog.json", "nullvector-native-morphology-subtype-runtime-v1", "counts.identity_count", 20, True),
    "CellularMotionLab.tscn": ("generated/cellular_motion/v12/motion_catalog.json", "nullvector-cellular-neuromuscular-native-catalog-v7", "frame_count", 4_720, True),
    "CellularOrganismLab.tscn": ("generated/cellular_organism/v2/catalog.json", "nullvector-cellular-organism-native-catalog-v1", "totals.physical_cells", 34_178, True),
    "SymmetricOrganismLab.tscn": ("generated/cellular_symmetry/v1/catalog.json", "nullvector-cellular-organism-native-catalog-v1", "symmetry_summary.improved_samples", 45, True),
    "EvolvedOrganismLab.tscn": ("generated/evolved_cellular_organism/v1/catalog.json", "nullvector-cellular-organism-native-catalog-v1", "sample_count", 36, True),
    "CellularBreedingLab.tscn": ("generated/cellular_breeding/v1/catalog.json", "nullvector-cellular-organism-native-catalog-v1", "sample_count", 45, True),
    "CellularEcologyLab.tscn": ("generated/cellular_ecology/v6/ecology_catalog.json", "nullvector-cellular-ecology-native-catalog-v3", "resource_node_count", 120, True),
    "CellularOntogenyLab.tscn": ("generated/cellular_ontogeny/v6/ontogeny_catalog.json", "nullvector-cellular-ontogeny-native-catalog-v3", "program_count", 45, True),
    "NeuralDecoratedMapLab.tscn": ("generated/neural_decorated_maps/v1_1/catalog.json", "nullvector-neural-decorated-map-native-catalog/1.0.0", "atlas_frame_count", 90, True),
}


def _nested(value: dict, path: str):
    current = value
    for part in path.split("."):
        current = current[part]
    return current


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hub_inventory_covers_every_authoritative_native_lab() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    scene_text = SCENE.read_text(encoding="utf-8")
    assert 'res://scripts/forge_hub.gd' in scene_text
    assert source.count('"scene": "res://') == len(LABS)
    for scene_name, contract in LABS.items():
        assert (GAME / scene_name).is_file(), scene_name
        assert f'"scene": "res://{scene_name}"' in source
        if contract is not None:
            manifest, expected_format, _, _, _ = contract
            assert f'"manifest": "res://{manifest}"' in source
            assert expected_format in source


def test_every_hub_catalog_is_current_ready_and_python_free() -> None:
    for scene_name, contract in LABS.items():
        if contract is None:
            continue
        manifest_path, expected_format, metric_path, expected_metric, requires_ready = contract
        payload = json.loads((GAME / manifest_path).read_text(encoding="utf-8"))
        assert payload["format"] == expected_format, scene_name
        if requires_ready:
            assert payload["status"] == "ready", scene_name
        if "python_runtime_required" in payload:
            assert payload["python_runtime_required"] is False, scene_name
        assert not payload.get("errors", []), scene_name
        assert _nested(payload, metric_path) == expected_metric, scene_name
        bundle = payload.get("bundle_id")
        if bundle is not None:
            assert len(bundle) == 64 and set(bundle) <= set("0123456789abcdef"), scene_name


def test_hub_is_aesthetic_filterable_and_launches_in_place_or_detached() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for category in ("core", "neural", "motion", "organism", "ecology", "map"):
        assert f'"{category}"' in source
    for token in (
        "draw_circle",
        "draw_line",
        "ScrollContainer",
        "_apply_filter",
        "change_scene_to_file",
        "OS.create_instance",
        "OPEN",
        "DETACH",
    ):
        assert token in source


def test_hub_smoke_is_fail_closed_and_hashes_every_source() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "--forge-hub-smoke",
        "--forge-hub-report=",
        "FORGE_HUB_SMOKE_OK",
        "ResourceLoader.CACHE_MODE_IGNORE",
        "HashingContext.HASH_SHA256",
        'main_scene != "res://Arena.tscn"',
        '"python_runtime_required": false',
    ):
        assert token in source


def test_hub_is_additive_and_preserves_arena_authority() -> None:
    project = (GAME / "project.godot").read_text(encoding="utf-8")
    assert 'run/main_scene="res://Arena.tscn"' in project
    assert _sha(GAME / "project.godot") == "7397c7c032be468b94e072aa31ebaba42250342d5eae49fc9aa9972bffe245da"
    assert _sha(GAME / "Arena.tscn") == "c5f8b961297b43f683d40dff831cb576d89539c222f0a5a9abab3d29a1f67490"
    assert _sha(GAME / "scripts" / "arena_game.gd") == "0a5d2964cf9869bc292afada98ade98460cb25bcade1e50c4cd3a9766e419b22"
