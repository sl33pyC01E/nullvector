from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from forge.neural_decorated_map_sync import (
    DEFAULT_SOURCE,
    FORMAT,
    project_runtime,
    sync_runtime,
    validate_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = PROJECT_ROOT / "game" / "project.godot"
ARENA_SCENE = PROJECT_ROOT / "game" / "Arena.tscn"
LAB_SCENE = PROJECT_ROOT / "game" / "NeuralDecoratedMapLab.tscn"
LAB_SCRIPT = PROJECT_ROOT / "game" / "scripts" / "neural_decorated_map_lab.gd"
EXPECTED_AUTHORITY = {
    "variant": "deterministic_semantic_teacher",
    "decal": "accepted_neural_protected_selector",
    "prop": "accepted_neural_protected_selector",
    "emission": "conditional_semantic_projection",
}


def test_native_projection_is_exact_small_and_runtime_closed(tmp_path: Path) -> None:
    first = project_runtime(DEFAULT_SOURCE)
    second = project_runtime(DEFAULT_SOURCE)
    assert first == second
    assert set(first) == {"catalog.json", "neural_map_atlas.png"}
    assert sum(map(len, first.values())) < 2 * 1024**2

    destination = tmp_path / "native"
    report = sync_runtime(DEFAULT_SOURCE, destination)
    assert report["passed"] is True
    assert report["repeat_exact"] is True
    assert report["file_count"] == 2
    assert validate_runtime(destination)["passed"] is True


def test_native_catalog_carries_exact_authority_and_atlas_bounds() -> None:
    projected = project_runtime(DEFAULT_SOURCE)
    catalog = json.loads(projected["catalog.json"])
    assert catalog["format"] == FORMAT
    assert catalog["status"] == "ready"
    assert catalog["themes"] == ["arena", "rooms", "caves", "archipelago", "garden", "anomaly"]
    assert catalog["layers"] == ["composite", "base_color", "emissive", "objects", "variant", "emission_level", "topology", "hazard"]
    assert (catalog["theme_count"], catalog["layer_count"], catalog["atlas_frame_count"]) == (6, 8, 90)
    assert catalog["python_runtime_required"] is False
    assert catalog["checkpoint_shipped"] is False
    assert catalog["visual_inspection_passed"] is True
    assert all(entry["selection"]["field_authority"] == EXPECTED_AUTHORITY for entry in catalog["maps"])
    assert all(entry["selection"]["unsupported_neural_heads_cross_runtime_boundary"] is False for entry in catalog["maps"])

    atlas_path = DEFAULT_SOURCE / "neural_map_atlas.png"
    with Image.open(atlas_path) as image:
        assert image.size == tuple(catalog["atlas"]["size"])
        assert image.size == (1536, 8832)
        image.verify()
    capacity = catalog["atlas"]["columns"] * catalog["atlas"]["rows"]
    for entry in catalog["maps"]:
        for layer in entry["layers"]:
            assert 0 <= layer["start_cell"] < capacity
            assert layer["start_cell"] + layer["frame_count"] <= capacity


def test_godot_lab_is_additive_fail_closed_and_nearest_filtered() -> None:
    project = PROJECT_CONFIG.read_text(encoding="utf-8")
    scene = LAB_SCENE.read_text(encoding="utf-8")
    script = LAB_SCRIPT.read_text(encoding="utf-8")
    assert 'run/main_scene="res://Arena.tscn"' in project
    assert hashlib.sha256(ARENA_SCENE.read_bytes()).hexdigest() == "c5f8b961297b43f683d40dff831cb576d89539c222f0a5a9abab3d29a1f67490"
    assert 'path="res://scripts/neural_decorated_map_lab.gd"' in scene
    assert "TEXTURE_FILTER_NEAREST" in script
    assert "NEURAL_DECORATED_MAP_SMOKE_OK themes=6 layers=8 regions=48 frames=90" in script
    assert "unsupported_neural_heads_cross_runtime_boundary" in script
    assert "python_runtime_required" in script
    assert "OS.execute" not in script

