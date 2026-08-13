from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from forge.map_art.atlas import (
    build_hazard_atlases,
    build_object_atlases,
    build_terrain_atlases,
    frame_grid_metadata,
    pack_frame_grid,
)
from forge.map_art.autotile import EAST, NORTH, SOUTH, WEST, cardinal_match_mask, elevation_drop_mask
from forge.map_art.cli import build_showcase, fuzz_art
from forge.map_art.io import load_art_semantics, write_art_pack
from forge.map_art.model import HAZARD_FRAME_COUNT, TILE_SIZE
from forge.map_art.objects import render_object_sprite
from forge.map_art.provenance import source_hash
from forge.map_art.renderer import render_map_art
from forge.map_art.styles import STYLES, style_for
from forge.map_art.validate import validate_art_pack, validate_layers
from forge.maps.generator import generate_map
from forge.maps.io import array_digest, file_sha256, write_map_pack
from forge.maps.model import (
    MAP_SCHEMA_VERSION,
    TOPOLOGY_MASK_CONTRACT_VERSION,
    MapConfig,
    THEMES,
)


def _visual_digest(layers: object) -> str:
    digest = hashlib.sha256()
    for name in ("base_color", "emissive", "hazard_color_frames", "hazard_emissive_frames"):
        digest.update(getattr(layers, name).tobytes())
    digest.update(array_digest(getattr(layers, "semantic_arrays")()).encode("ascii"))
    digest.update(json.dumps([item.to_dict() for item in getattr(layers, "instances")], sort_keys=True).encode())
    return digest.hexdigest()


def test_cardinal_and_elevation_masks_are_exact() -> None:
    values = np.array([[1, 1, 2], [1, 1, 2], [3, 1, 2]], dtype=np.uint8)
    masks = cardinal_match_mask(values)
    assert masks[1, 1] == NORTH | SOUTH | WEST
    assert masks[0, 0] == EAST | SOUTH
    assert masks[2, 2] == NORTH
    elevation = np.array([[0, 0, 0], [0, 3, 1], [0, 2, 0]], dtype=np.int8)
    walkability = np.ones((3, 3), dtype=np.uint8)
    drops = elevation_drop_mask(elevation, walkability)
    assert drops[1, 1] == NORTH | EAST | SOUTH | WEST
    assert drops[1, 2] == NORTH | SOUTH


def test_all_theme_catalogs_and_atlases_are_complete() -> None:
    assert tuple(STYLES) == THEMES
    for theme in THEMES:
        style = style_for(theme)
        terrain, terrain_glow, terrain_entries = build_terrain_atlases(style)
        hazard, hazard_glow, hazard_entries = build_hazard_atlases(style)
        objects, object_glow, object_entries = build_object_atlases(style)
        assert terrain.shape == (9 * TILE_SIZE, 16 * TILE_SIZE, 3)
        assert terrain_glow.shape == terrain.shape
        assert len(terrain_entries) == 144
        assert hazard.shape == (4 * TILE_SIZE, HAZARD_FRAME_COUNT * TILE_SIZE, 4)
        assert hazard_glow.shape == (*hazard.shape[:2], 3)
        assert len(hazard_entries) == 32
        assert objects.shape == (4 * TILE_SIZE, len(style.props) * TILE_SIZE, 4)
        assert object_glow.shape == (*objects.shape[:2], 3)
        assert len(object_entries) == len(style.props) * 4
        assert object_glow.any(), theme
        for spec in style.props:
            sprite, glow = render_object_sprite(style, spec)
            assert sprite[..., 3].any(), (theme, spec.key)


@pytest.mark.parametrize("theme_index,theme", enumerate(THEMES))
def test_every_theme_renders_deterministically_and_validly(theme_index: int, theme: str) -> None:
    data = generate_map(0xA770000 + theme_index, theme, MapConfig(width=44, height=40, spawn_count=8))
    first = render_map_art(data)
    second = render_map_art(data)
    report = validate_layers(data, first)
    assert report["passed"], report
    assert _visual_digest(first) == _visual_digest(second)
    assert first.instances == second.instances
    assert np.array_equal(first.autotile_mask, cardinal_match_mask(data.terrain))
    assert np.array_equal(first.elevation_edge_mask, elevation_drop_mask(data.elevation, data.walkability))
    assert first.base_color.shape == (40 * TILE_SIZE, 44 * TILE_SIZE, 3)
    assert first.emissive.any()
    assert all(instance.cell not in {data.start, data.exit, *data.objectives} for instance in first.instances)


def test_theme_visual_signatures_are_distinct() -> None:
    signatures = set()
    for index, theme in enumerate(THEMES):
        data = generate_map(0x515151 + index, theme, MapConfig(width=40, height=40, spawn_count=8))
        signatures.add(_visual_digest(render_map_art(data)))
    assert len(signatures) == len(THEMES)


def test_hazard_animation_frame_grid_is_explicit_and_varied() -> None:
    style = style_for("anomaly")
    atlas, _, _ = build_hazard_atlases(style)
    for row in range(4):
        frames = [
            atlas[row * TILE_SIZE : (row + 1) * TILE_SIZE, frame * TILE_SIZE : (frame + 1) * TILE_SIZE]
            for frame in range(HAZARD_FRAME_COUNT)
        ]
        assert len({hashlib.sha256(frame.tobytes()).digest() for frame in frames}) >= 3
    synthetic = np.zeros((HAZARD_FRAME_COUNT, 7, 11, 4), dtype=np.uint8)
    for frame in range(HAZARD_FRAME_COUNT):
        synthetic[frame, :, :, 0] = frame
    sheet = pack_frame_grid(synthetic)
    assert sheet.shape == (14, 44, 4)
    meta = frame_grid_metadata(11, 7)
    assert meta["n_frames"] == HAZARD_FRAME_COUNT
    assert meta["grid"] == {"columns": 4, "rows": 2}
    assert meta["frames"][7]["x"] == 33
    assert meta["frames"][7]["y"] == 7


def test_art_pack_round_trip_schema_hashes_and_append_safety(tmp_path: Path) -> None:
    data = generate_map(0xDECAFBAD, "garden", MapConfig(width=40, height=36, spawn_count=8))
    pack = write_art_pack(data, tmp_path)
    report = validate_art_pack(pack, source_data=data)
    assert report["passed"], report
    expected_files = {
        "art_semantics.npz", "base_color.png", "emissive.png", "hazard_atlas.png",
        "hazard_emissive_atlas.png", "hazard_emissive_frames.png", "hazard_frames.meta.json",
        "hazard_frames.png", "instances.json", "manifest.json", "object_atlas.png",
        "object_emissive_atlas.png", "preview.png", "terrain_atlas.png", "terrain_emissive_atlas.png",
    }
    assert {path.name for path in pack.iterdir()} == expected_files
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["renderer"]["source_sha256"] == source_hash()
    assert manifest["source"]["semantic_array_sha256"] == array_digest(data.arrays())
    before = {path.name: file_sha256(path) for path in pack.iterdir()}
    assert write_art_pack(data, tmp_path, skip_existing=True) == pack
    after = {path.name: file_sha256(path) for path in pack.iterdir()}
    assert before == after
    arrays = load_art_semantics(pack)
    assert set(arrays) == {
        "autotile_mask", "elevation_edge_mask", "variant", "collision", "occlusion", "prop_id", "decal_id"
    }
    assert all(array.shape == data.shape for array in arrays.values())


def test_pack_rejects_a_different_source_map(tmp_path: Path) -> None:
    config = MapConfig(width=36, height=36, spawn_count=8)
    first = generate_map(101, "rooms", config)
    second = generate_map(102, "rooms", config)
    pack = write_art_pack(first, tmp_path)
    report = validate_art_pack(pack, source_data=second)
    assert not report["passed"]
    assert "source map semantic SHA-256 mismatch" in report["semantic_errors"]


def test_showcase_consumes_persisted_v2_maps_with_topology_provenance(tmp_path: Path) -> None:
    map_root = tmp_path / "maps_v2"
    sources = {}
    for index, theme in enumerate(THEMES):
        data = generate_map(0x5A0C0000 + index, theme, MapConfig(width=32, height=32, spawn_count=8))
        sources[theme] = data
        write_map_pack(data, map_root, preview_scale=2)

    report = build_showcase(tmp_path / "showcase", source_paths=[map_root])

    assert report["passed"], report
    assert report["source_mode"] == "persisted_map_packs"
    assert tuple(record["theme"] for record in report["source_maps"]) == THEMES
    packs = dict(zip(THEMES, (Path(path) for path in report["packs"]), strict=True))
    for record in report["source_maps"]:
        assert record["schema_version"] == MAP_SCHEMA_VERSION
        assert record["topology_masks"]["contract_version"] == TOPOLOGY_MASK_CONTRACT_VERSION
        assert set(record["topology_masks"]["members"]) == {
            "protected_backbone",
            "required_clearance",
            "decoration_forbidden",
        }
        manifest = Path(record["manifest"])
        assert file_sha256(manifest) == record["manifest_sha256"]
        assert validate_art_pack(
            packs[record["theme"]], source_data=sources[record["theme"]]
        )["passed"]


def test_small_visual_fuzz_matrix_is_valid_unique_and_balanced() -> None:
    report = fuzz_art(18, width=36, height=36)
    assert report["passed"], report["failures"]
    assert report["unique_visual_maps"] == 18
    assert all(count == 3 for count in report["per_theme"].values())
