from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from PIL import Image

from ..config import PROJECT_ROOT
from ..maps.io import array_digest, file_sha256
from ..maps.model import Hazard, MapData
from ..maps.validate import assert_valid
from .autotile import cardinal_match_mask, elevation_drop_mask
from .model import ArtLayers, HAZARD_FRAME_COUNT, RENDERER_NAME, TILE_SIZE
from .styles import style_for


SCHEMA_PATH = PROJECT_ROOT / "shared" / "schema" / "map_art_manifest.schema.json"
SEMANTIC_ARRAY_NAMES = (
    "autotile_mask",
    "elevation_edge_mask",
    "variant",
    "collision",
    "occlusion",
    "prop_id",
    "decal_id",
)


def validate_layers(data: MapData, layers: ArtLayers) -> dict[str, Any]:
    assert_valid(data)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    height, width = data.shape
    pixel_shape = (height * TILE_SIZE, width * TILE_SIZE)
    check("shape.base_color", layers.base_color.shape == (*pixel_shape, 3), f"expected {(*pixel_shape, 3)}")
    check("shape.emissive", layers.emissive.shape == (*pixel_shape, 3), f"expected {(*pixel_shape, 3)}")
    check(
        "shape.hazard_color_frames",
        layers.hazard_color_frames.shape == (HAZARD_FRAME_COUNT, *pixel_shape, 4),
        "hazard RGBA frames align with map pixels",
    )
    check(
        "shape.hazard_emissive_frames",
        layers.hazard_emissive_frames.shape == (HAZARD_FRAME_COUNT, *pixel_shape, 3),
        "hazard emission frames align with map pixels",
    )
    expected_dtypes = {
        "base_color": np.dtype(np.uint8),
        "emissive": np.dtype(np.uint8),
        "hazard_color_frames": np.dtype(np.uint8),
        "hazard_emissive_frames": np.dtype(np.uint8),
    }
    for name, dtype in expected_dtypes.items():
        observed = getattr(layers, name).dtype
        check(f"dtype.{name}", observed == dtype, f"expected {dtype}, observed {observed}")

    arrays = layers.semantic_arrays()
    for name, array in arrays.items():
        check(f"shape.{name}", array.shape == (height, width), "art semantics align with source cells")
    check(
        "autotile.exact",
        np.array_equal(layers.autotile_mask, cardinal_match_mask(data.terrain)),
        "four-neighbor masks are exactly derived from terrain",
    )
    check(
        "elevation_edges.exact",
        np.array_equal(layers.elevation_edge_mask, elevation_drop_mask(data.elevation, data.walkability)),
        "drop masks are exactly derived from elevation and walkability",
    )
    check("domain.autotile", bool((layers.autotile_mask <= 15).all()), "autotile masks fit four bits")
    check("domain.elevation_edges", bool((layers.elevation_edge_mask <= 15).all()), "edge masks fit four bits")
    check("domain.variant", bool((layers.variant <= 7).all()), "variant ids are in [0, 7]")
    check("domain.collision", bool(np.isin(layers.collision, (0, 1)).all()), "collision is binary")
    check("domain.occlusion", bool((layers.occlusion <= 3).all()), "occlusion is in [0, 3]")
    check(
        "collision.source_blockers",
        bool((layers.collision[data.walkability == 0] == 1).all()),
        "source blockers remain collidable",
    )
    required = (data.start, data.exit, *data.objectives)
    check(
        "collision.required_points_clear",
        all(layers.collision[y, x] == 0 for x, y in required),
        "start, exit, and objectives remain unobstructed",
    )

    instance_ids = [instance.instance_id for instance in layers.instances]
    instance_cells_by_kind: set[tuple[int, int, str]] = set()
    instances_valid = True
    catalog_valid = True
    style = style_for(data.theme)
    for instance in layers.instances:
        x, y = instance.cell
        if not (0 <= x < width and 0 <= y < height and int(data.hazard[y, x]) == int(Hazard.NONE)):
            instances_valid = False
        key = (x, y, instance.kind)
        if key in instance_cells_by_kind:
            instances_valid = False
        instance_cells_by_kind.add(key)
        if not 1 <= instance.catalog_index <= len(style.props):
            catalog_valid = False
        elif style.props[instance.catalog_index - 1].key != instance.key:
            catalog_valid = False
        field = layers.prop_id if instance.kind == "prop" else layers.decal_id
        if int(field[y, x]) != instance.catalog_index:
            instances_valid = False
    check("instances.unique_ids", len(instance_ids) == len(set(instance_ids)), "instance identifiers are unique")
    check("instances.cells_and_fields", instances_valid, "instances are in bounds, hazard-free, unique by kind, and indexed")
    check("instances.catalog", catalog_valid, "instance keys match the stable theme catalog")

    pixel_hazard = np.repeat(np.repeat(data.hazard != int(Hazard.NONE), TILE_SIZE, axis=0), TILE_SIZE, axis=1)
    outside_clear = bool((layers.hazard_color_frames[:, ~pixel_hazard, 3] == 0).all())
    outside_emission_clear = bool((layers.hazard_emissive_frames[:, ~pixel_hazard] == 0).all())
    check("hazards.color_locality", outside_clear, "animated color is transparent outside hazard cells")
    check("hazards.emission_locality", outside_emission_clear, "animated emission is black outside hazard cells")
    if bool(pixel_hazard.any()):
        frame_hashes = {
            hashlib.sha256(layers.hazard_color_frames[index].tobytes()).digest()
            for index in range(HAZARD_FRAME_COUNT)
        }
        check("hazards.temporal_variation", len(frame_hashes) >= 3, "hazard maps expose at least three distinct animation frames")
        check("hazards.visible", bool((layers.hazard_color_frames[..., 3] > 0).any()), "hazard color frames contain visible pixels")
    else:
        check("hazards.temporal_variation", True, "source map has no hazard cells")
        check("hazards.visible", True, "source map has no hazard cells")
    check("emissive.visible", bool((layers.emissive > 0).any()), "static map contains emissive pixels")

    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "passed": not failures,
        "map_id": data.map_id,
        "checks": checks,
        "failures": failures,
        "metrics": {
            "instance_count": len(layers.instances),
            "prop_count": sum(instance.kind == "prop" for instance in layers.instances),
            "decal_count": sum(instance.kind == "decal" for instance in layers.instances),
            "emissive_pixels": int((layers.emissive.max(axis=2) > 0).sum()),
            "animated_hazard_cells": int((data.hazard != int(Hazard.NONE)).sum()),
            "collision_cells": int(layers.collision.sum()),
            "occluding_cells": int((layers.occlusion > 0).sum()),
        },
    }


def assert_valid_layers(data: MapData, layers: ArtLayers) -> dict[str, Any]:
    report = validate_layers(data, layers)
    if not report["passed"]:
        raise ValueError(f"Map art failed invariants: {', '.join(report['failures'])}")
    return report


def validate_art_pack(path: Path, *, source_data: MapData | None = None) -> dict[str, Any]:
    path = Path(path)
    manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
    pack_dir = manifest_path.parent
    schema_errors: list[str] = []
    artifact_errors: list[str] = []
    semantic_errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(manifest),
            key=lambda item: list(item.path),
        )
        schema_errors = [error.message for error in errors]
    except Exception as error:
        return {
            "passed": False,
            "pack": str(pack_dir),
            "schema_errors": [f"manifest/schema could not be read: {error}"],
            "artifact_errors": [],
            "semantic_errors": [],
        }
    if schema_errors:
        return {
            "passed": False,
            "pack": str(pack_dir),
            "schema_errors": schema_errors,
            "artifact_errors": [],
            "semantic_errors": [],
        }

    cell_width = int(manifest["dimensions"]["cells"]["width"])
    cell_height = int(manifest["dimensions"]["cells"]["height"])
    pixel_width = cell_width * TILE_SIZE
    pixel_height = cell_height * TILE_SIZE
    catalog_count = int(manifest["statistics"]["catalog_count"])
    expected_image_sizes = {
        "base_color": [pixel_width, pixel_height],
        "emissive": [pixel_width, pixel_height],
        "preview": [pixel_width, pixel_height],
        "hazard_color_frames": [pixel_width * 4, pixel_height * 2],
        "hazard_emissive_frames": [pixel_width * 4, pixel_height * 2],
        "terrain_atlas": [16 * TILE_SIZE, 9 * TILE_SIZE],
        "terrain_emissive_atlas": [16 * TILE_SIZE, 9 * TILE_SIZE],
        "hazard_atlas": [HAZARD_FRAME_COUNT * TILE_SIZE, 4 * TILE_SIZE],
        "hazard_emissive_atlas": [HAZARD_FRAME_COUNT * TILE_SIZE, 4 * TILE_SIZE],
        "object_atlas": [catalog_count * TILE_SIZE, 4 * TILE_SIZE],
        "object_emissive_atlas": [catalog_count * TILE_SIZE, 4 * TILE_SIZE],
    }
    expected_atlas_entries = {
        "terrain": (16, 9, 144),
        "hazard": (HAZARD_FRAME_COUNT, 4, 32),
        "objects": (catalog_count, 4, catalog_count * 4),
    }
    for name, (columns, rows, entry_count) in expected_atlas_entries.items():
        atlas = manifest["atlases"][name]
        if (atlas["columns"], atlas["rows"], len(atlas["entries"])) != (columns, rows, entry_count):
            artifact_errors.append(f"{name}: atlas grid or entry count is inconsistent")
    for name, expected_size in expected_image_sizes.items():
        if manifest["artifacts"][name]["pixel_size"] != expected_size:
            artifact_errors.append(
                f"{name}: declared size {manifest['artifacts'][name]['pixel_size']} != semantic size {expected_size}"
            )

    for name, descriptor in manifest["artifacts"].items():
        artifact = pack_dir / descriptor["file"]
        if not artifact.is_file():
            artifact_errors.append(f"{name}: missing {descriptor['file']}")
            continue
        if file_sha256(artifact) != descriptor["sha256"]:
            artifact_errors.append(f"{name}: SHA-256 mismatch")
        if descriptor["format"].startswith("png"):
            try:
                with Image.open(artifact) as image:
                    image.verify()
                with Image.open(artifact) as image:
                    if image.mode != descriptor["mode"]:
                        artifact_errors.append(f"{name}: mode {image.mode} != {descriptor['mode']}")
                    if list(image.size) != descriptor["pixel_size"]:
                        artifact_errors.append(f"{name}: size {list(image.size)} != {descriptor['pixel_size']}")
            except Exception as error:
                artifact_errors.append(f"{name}: PNG decode failed: {error}")

    arrays: dict[str, np.ndarray] = {}
    semantics_path = pack_dir / manifest["artifacts"]["art_semantics"]["file"]
    if semantics_path.is_file():
        try:
            with np.load(semantics_path, allow_pickle=False) as archive:
                if set(archive.files) != set(SEMANTIC_ARRAY_NAMES):
                    semantic_errors.append(
                        f"art semantics arrays {sorted(archive.files)} != {sorted(SEMANTIC_ARRAY_NAMES)}"
                    )
                    arrays = {}
                else:
                    arrays = {name: np.ascontiguousarray(archive[name]) for name in SEMANTIC_ARRAY_NAMES}
            if arrays:
                shape = (cell_height, cell_width)
                for name, array in arrays.items():
                    if array.shape != shape:
                        semantic_errors.append(f"{name}: shape {array.shape} != {shape}")
                    descriptor = manifest["semantics"]["arrays"][name]
                    if descriptor["shape"] != list(array.shape) or descriptor["dtype"] != str(array.dtype):
                        semantic_errors.append(f"{name}: manifest array descriptor mismatch")
                if array_digest(arrays) != manifest["semantics"]["array_sha256"]:
                    semantic_errors.append("canonical art-semantic SHA-256 mismatch")
                if not bool((arrays["autotile_mask"] <= 15).all()):
                    semantic_errors.append("autotile mask exceeds four bits")
                if not bool((arrays["elevation_edge_mask"] <= 15).all()):
                    semantic_errors.append("elevation-edge mask exceeds four bits")
                if not bool(np.isin(arrays["collision"], (0, 1)).all()):
                    semantic_errors.append("collision mask is not binary")
        except Exception as error:
            semantic_errors.append(f"art semantics could not be loaded: {error}")

    frame_meta_path = pack_dir / manifest["artifacts"]["hazard_frame_metadata"]["file"]
    if frame_meta_path.is_file():
        try:
            frame_meta = json.loads(frame_meta_path.read_text(encoding="utf-8"))
            expected_meta = {
                "n_frames": HAZARD_FRAME_COUNT,
                "frame_w": pixel_width,
                "frame_h": pixel_height,
                "duration_ms": manifest["animation"]["duration_ms"],
                "loop": manifest["animation"]["loop"],
                "grid": manifest["animation"]["grid"],
            }
            for key, expected in expected_meta.items():
                if frame_meta.get(key) != expected:
                    semantic_errors.append(f"frame metadata {key!r} does not match manifest")
            if len(frame_meta.get("frames", [])) != HAZARD_FRAME_COUNT:
                semantic_errors.append("frame metadata does not enumerate all frames")
        except Exception as error:
            semantic_errors.append(f"hazard frame metadata could not be loaded: {error}")

    instances_path = pack_dir / manifest["artifacts"]["instances"]["file"]
    if instances_path.is_file():
        try:
            payload = json.loads(instances_path.read_text(encoding="utf-8"))
            if len(payload["instances"]) != manifest["statistics"]["instance_count"]:
                semantic_errors.append("instance count disagrees with manifest")
            identifiers = [item["instance_id"] for item in payload["instances"]]
            if len(identifiers) != len(set(identifiers)):
                semantic_errors.append("instance identifiers are not unique")
            if len(payload["catalog"]) != manifest["statistics"]["catalog_count"]:
                semantic_errors.append("catalog count disagrees with manifest")
            catalog_indices = {int(item["catalog_index"]) for item in payload["catalog"]}
            if catalog_indices != set(range(1, catalog_count + 1)):
                semantic_errors.append("catalog indices are not contiguous and one-based")
            seen_cells: set[tuple[int, int, str]] = set()
            for item in payload["instances"]:
                x, y = (int(value) for value in item["cell"])
                kind = str(item["kind"])
                if not (0 <= x < cell_width and 0 <= y < cell_height):
                    semantic_errors.append(f"instance {item['instance_id']} is outside map bounds")
                if (x, y, kind) in seen_cells:
                    semantic_errors.append(f"multiple {kind} instances occupy cell {(x, y)}")
                seen_cells.add((x, y, kind))
                catalog_index = int(item["catalog_index"])
                if catalog_index not in catalog_indices:
                    semantic_errors.append(f"instance {item['instance_id']} has unknown catalog index")
                if arrays and 0 <= x < cell_width and 0 <= y < cell_height:
                    field_name = "prop_id" if kind == "prop" else "decal_id"
                    if int(arrays[field_name][y, x]) != catalog_index:
                        semantic_errors.append(f"instance {item['instance_id']} disagrees with {field_name}")
        except Exception as error:
            semantic_errors.append(f"instances JSON could not be loaded: {error}")

    if source_data is not None:
        expected = array_digest(source_data.arrays())
        if manifest["source"]["semantic_array_sha256"] != expected:
            semantic_errors.append("source map semantic SHA-256 mismatch")
        if manifest["map_id"] != source_data.map_id:
            semantic_errors.append("source map id mismatch")
        if arrays:
            if not np.array_equal(arrays["autotile_mask"], cardinal_match_mask(source_data.terrain)):
                semantic_errors.append("art autotile masks disagree with source terrain")
            if not np.array_equal(
                arrays["elevation_edge_mask"],
                elevation_drop_mask(source_data.elevation, source_data.walkability),
            ):
                semantic_errors.append("art elevation edges disagree with source semantics")
            if not bool((arrays["collision"][source_data.walkability == 0] == 1).all()):
                semantic_errors.append("source blockers are absent from art collision")
            required = (source_data.start, source_data.exit, *source_data.objectives)
            if not all(arrays["collision"][y, x] == 0 for x, y in required):
                semantic_errors.append("required mission points are obstructed in art collision")
    passed = not schema_errors and not artifact_errors and not semantic_errors
    return {
        "passed": passed,
        "pack": str(pack_dir),
        "schema_errors": schema_errors,
        "artifact_errors": artifact_errors,
        "semantic_errors": semantic_errors,
    }
