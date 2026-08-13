from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from forge.neural_genetics_workshop_sync import (
    DEFAULT_DESTINATION,
    EVOLUTION_COUNT,
    FORMAT,
    FUSION_COUNT,
    LATENT_COUNT,
    LAYERS,
    MIN_FREE_BYTES,
    sync_genetics_workshop,
    validate_genetics_workshop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = DEFAULT_DESTINATION / "asset_index.json"


def _load(path: Path = INDEX_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_checked_in_local_genetics_runtime_bank_is_current_and_truthful() -> None:
    assert validate_genetics_workshop(INDEX_PATH) == []
    index = _load()
    assert index["format"] == FORMAT
    assert index["status"] == "ready"
    assert index["pixel_filter"] == "nearest"
    assert index["python_runtime_required"] is False
    assert index["disk_budget"]["minimum_free_bytes"] == MIN_FREE_BYTES
    assert index["fusion"]["status"] == "ready"
    assert index["fusion"]["truth_label"] == "verified-categorical-fusion"
    assert index["fusion"]["specimen_count"] == FUSION_COUNT
    assert index["latent"]["status"] == "experimental"
    assert index["latent"]["truth_label"] == "learned-fsq-smoke-not-production"
    assert index["latent"]["specimen_count"] == LATENT_COUNT
    assert index["evolution"]["selected_count"] == EVOLUTION_COUNT


def test_all_native_genetics_atlases_and_regions_are_addressable() -> None:
    index = _load()
    assert index["fusion"]["clip_count"] == 70
    assert index["fusion"]["frame_count"] == 660
    assert index["latent"]["clip_count"] == 48
    assert index["latent"]["frame_count"] == 420
    for bank_name in ("fusion", "latent"):
        for specimen in index[bank_name]["specimens"]:
            layout = specimen["layout"]
            cursor = 0
            for clip in specimen["clips"]:
                assert clip["start_cell"] == cursor
                cursor += clip["frame_count"]
                assert cursor <= layout["columns"] * layout["rows"]
            assert cursor == layout["frame_count"]
            for layer in LAYERS:
                path = INDEX_PATH.parent / specimen["layers"][layer]["path"]
                with Image.open(path) as image:
                    assert image.mode == "RGBA"
                    assert image.size == (layout["columns"] * 48, layout["rows"] * 48)


def test_evolution_bank_keeps_twelve_survivors_and_all_families_per_generation() -> None:
    index = _load()
    for generation in (1, 2):
        specimens = [item for item in index["evolution"]["specimens"] if item["generation"] == generation]
        assert len(specimens) == 12
        assert len({item["family"] for item in specimens}) == 5
        assert [item["rank"] for item in specimens] == list(range(12))
        for specimen in specimens:
            assert len(specimen["parents"]) == 2
            assert specimen["score"]["motion_strength"] == 1.0
            with Image.open(INDEX_PATH.parent / specimen["image"]["path"]) as image:
                assert image.mode == "RGBA"
                assert image.size == (48, 48)


def test_runtime_sync_is_byte_exact_across_fresh_destinations(tmp_path: Path) -> None:
    first = sync_genetics_workshop(tmp_path / "a")
    second = sync_genetics_workshop(tmp_path / "b")
    assert first["bundle_id"] == second["bundle_id"]
    files_a = {path.relative_to(tmp_path / "a").as_posix(): path.read_bytes() for path in (tmp_path / "a").rglob("*") if path.is_file()}
    files_b = {path.relative_to(tmp_path / "b").as_posix(): path.read_bytes() for path in (tmp_path / "b").rglob("*") if path.is_file()}
    assert files_a == files_b


def test_runtime_inventory_is_png_json_only_exact_and_small() -> None:
    index = _load()
    assert index["asset_count"] == 179
    assert len(index["inventory"]) == 178
    assert {Path(item["path"]).suffix for item in index["inventory"]} == {".png"}
    assert sum(item["bytes"] for item in index["inventory"]) < 5 * 1024**2
    for record in index["inventory"]:
        path = INDEX_PATH.parent / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_native_scene_is_additive_and_smoke_is_exhaustive() -> None:
    project = (PROJECT_ROOT / "game" / "project.godot").read_text(encoding="utf-8")
    scene = (PROJECT_ROOT / "game" / "NeuralGeneticsWorkshop.tscn").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "game" / "scripts" / "neural_genetics_workshop.gd").read_text(encoding="utf-8")
    assert 'run/main_scene="res://Arena.tscn"' in project
    assert "neural_genetics_workshop.gd" in scene
    assert "NEURAL_GENETICS_SMOKE_OK" in script
    assert "atlas_count != 154" in script
    assert "clip_count != 118" in script
    assert "frame_count != 1080" in script
    assert "hash_count != 178" in script
