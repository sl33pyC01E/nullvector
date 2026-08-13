from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from forge.forge_lab_sync import (
    ASSET_INDEX_FORMAT,
    EXPECTED_FAMILIES,
    MAP_LAYERS,
    MAP_THEMES,
    MIN_FREE_BYTES,
    asset_inventory,
    validate_synced_assets,
)
from forge.map_art.provenance import source_hash as map_art_source_hash
from forge.morphology import FACING_NAMES, MOTION_NAMES, MOTION_RENDERER_VERSION, RENDERER_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "game" / "generated" / "v2" / "asset_index.json"
PROJECT_CONFIG = PROJECT_ROOT / "game" / "project.godot"
SCENE_PATH = PROJECT_ROOT / "game" / "ForgeLab.tscn"
SCRIPT_PATH = PROJECT_ROOT / "game" / "scripts" / "forge_lab.gd"


def _index() -> dict:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def test_checked_in_forge_lab_bank_is_complete_and_current() -> None:
    assert validate_synced_assets(INDEX_PATH) == []
    index = _index()
    assert index["format"] == ASSET_INDEX_FORMAT
    assert index["pixel_filter"] == "nearest"
    assert index["python_runtime_required"] is False
    assert index["runtime_asset_extensions"] == [".json", ".png"]
    assert index["disk_budget"] == {
        "guard_passed": True,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "planned_bytes": 128 * 1024**2,
    }
    assert index["errors"] == []


def test_motion_bank_is_the_full_role_conditioned_direction_matrix() -> None:
    motion = _index()["motion"]
    assert tuple(motion["families"]) == EXPECTED_FAMILIES
    assert tuple(motion["motions"]) == MOTION_NAMES
    assert tuple(motion["facings"]) == FACING_NAMES
    assert motion["renderer"] == MOTION_RENDERER_VERSION
    assert motion["source_morphology_renderer"] == RENDERER_VERSION
    assert motion["clip_count"] == len(EXPECTED_FAMILIES) * len(MOTION_NAMES) * len(FACING_NAMES)
    assert len(motion["clips"]) == motion["clip_count"]
    assert {entry["source_role_name"] for entry in motion["atlases"]} == {
        "harvester",
        "support",
        "scout",
        "disruptor",
    }
    assert all(entry["source_renderer_version"] == RENDERER_VERSION for entry in motion["atlases"])
    keys = {(entry["family"], entry["motion"], entry["facing"]) for entry in motion["clips"]}
    expected = {
        (family, motion_name, facing)
        for family in EXPECTED_FAMILIES
        for motion_name in MOTION_NAMES
        for facing in FACING_NAMES
    }
    assert keys == expected


def test_map_bank_has_every_theme_layer_and_current_renderer() -> None:
    maps = _index()["maps"]
    assert tuple(maps["themes"]) == MAP_THEMES
    assert tuple(maps["layers"]) == MAP_LAYERS
    assert maps["renderer_source_sha256"] == map_art_source_hash()
    assert maps["map_count"] == len(MAP_THEMES)
    assert tuple(entry["theme"] for entry in maps["maps"]) == MAP_THEMES
    for entry in maps["maps"]:
        assert entry["renderer"]["source_sha256"] == map_art_source_hash()
        assert tuple(layer["name"] for layer in entry["layers"]) == MAP_LAYERS
        assert entry["layers"][-1]["frame_count"] == 8


def test_engine_bank_is_png_json_only_with_decodable_nearest_atlases() -> None:
    inventory = asset_inventory(INDEX_PATH)
    assert len(inventory) == 19
    assert {Path(record["path"]).suffix for record in inventory} == {".json", ".png"}
    assert sum(record["bytes"] for record in inventory) < 8 * 1024**2
    for record in inventory:
        if record["path"].endswith(".png"):
            with Image.open(INDEX_PATH.parent / record["path"]) as image:
                image.verify()


def test_forge_lab_is_additive_and_arena_remains_main() -> None:
    project = PROJECT_CONFIG.read_text(encoding="utf-8")
    scene = SCENE_PATH.read_text(encoding="utf-8")
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'run/main_scene="res://Arena.tscn"' in project
    assert "textures/canvas_textures/default_texture_filter=0" in project
    assert 'path="res://scripts/forge_lab.gd"' in scene
    assert "TEXTURE_FILTER_NEAREST" in script
    assert "FORGE_LAB_SMOKE_OK" in script
    assert "OS.execute" not in script
    assert "res://generated/v2/asset_index.json" in script
