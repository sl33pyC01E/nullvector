from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import uuid
from typing import Any

import numpy as np
from PIL import Image

from ..maps.io import array_digest
from ..maps.model import MapData
from ..safety import require_disk_floor, write_json_atomic
from .atlas import (
    FRAME_GRID_COLUMNS,
    FRAME_GRID_ROWS,
    build_hazard_atlases,
    build_object_atlases,
    build_terrain_atlases,
    compose_preview,
    frame_grid_metadata,
    pack_frame_grid,
)
from .model import HAZARD_FRAME_COUNT, RENDERER_NAME, RENDERER_VERSION, TILE_SIZE, ArtLayers
from .provenance import source_hash
from .renderer import render_map_art
from .styles import style_for
from .validate import SEMANTIC_ARRAY_NAMES, assert_valid_layers


BASE_COLOR_FILE = "base_color.png"
EMISSIVE_FILE = "emissive.png"
PREVIEW_FILE = "preview.png"
HAZARD_FRAMES_FILE = "hazard_frames.png"
HAZARD_EMISSIVE_FRAMES_FILE = "hazard_emissive_frames.png"
HAZARD_FRAME_META_FILE = "hazard_frames.meta.json"
TERRAIN_ATLAS_FILE = "terrain_atlas.png"
TERRAIN_EMISSIVE_ATLAS_FILE = "terrain_emissive_atlas.png"
HAZARD_ATLAS_FILE = "hazard_atlas.png"
HAZARD_EMISSIVE_ATLAS_FILE = "hazard_emissive_atlas.png"
OBJECT_ATLAS_FILE = "object_atlas.png"
OBJECT_EMISSIVE_ATLAS_FILE = "object_emissive_atlas.png"
SEMANTICS_FILE = "art_semantics.npz"
INSTANCES_FILE = "instances.json"
MANIFEST_FILE = "manifest.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **{name: arrays[name] for name in SEMANTIC_ARRAY_NAMES})
    return buffer.getvalue()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, planned_bytes=len(payload) + 1024 * 1024)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _image_descriptor(file: str, payload: bytes, array: np.ndarray) -> dict[str, Any]:
    channels = int(array.shape[2])
    mode = "RGBA" if channels == 4 else "RGB"
    return {
        "file": file,
        "format": f"png-{mode.lower()}-pixel-native",
        "mode": mode,
        "pixel_size": [int(array.shape[1]), int(array.shape[0])],
        "sha256": _sha256(payload),
    }


def _file_descriptor(file: str, format_name: str, payload: bytes) -> dict[str, str]:
    return {"file": file, "format": format_name, "sha256": _sha256(payload)}


def _instances_payload(data: MapData, layers: ArtLayers) -> dict[str, Any]:
    style = style_for(data.theme)
    catalog = []
    for catalog_index, spec in enumerate(style.props, start=1):
        catalog.append(
            {
                "catalog_index": catalog_index,
                "key": spec.key,
                "shape": spec.shape,
                "kind": spec.kind,
                "allowed_terrain": list(spec.allowed_terrain),
                "collision": {
                    "blocks_movement": spec.collision,
                    "cell_footprint": [[0, 0]] if spec.collision else [],
                },
                "occlusion": {
                    "class": spec.occlusion,
                    "height_pixels": spec.occlusion * TILE_SIZE,
                },
                "atlas": {
                    "file": OBJECT_ATLAS_FILE,
                    "column": catalog_index - 1,
                    "orientation_rows": 4,
                    "tile_size": TILE_SIZE,
                },
            }
        )
    instances = []
    for instance in layers.instances:
        entry = instance.to_dict()
        atlas_column, atlas_row = instance.atlas_cell
        entry["atlas_pixel_rect"] = [
            atlas_column * TILE_SIZE,
            atlas_row * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ]
        entry["world_pixel_anchor"] = [
            instance.cell[0] * TILE_SIZE,
            instance.cell[1] * TILE_SIZE,
        ]
        instances.append(entry)
    return {
        "schema_version": "1.0.0",
        "map_id": data.map_id,
        "coordinate_system": "cell [x,y], origin top-left; sprites anchor to tile top-left",
        "catalog": catalog,
        "instances": instances,
    }


def _build_payloads(data: MapData, layers: ArtLayers) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any]]:
    style = style_for(data.theme)
    terrain_atlas, terrain_emissive_atlas, terrain_entries = build_terrain_atlases(style)
    hazard_atlas, hazard_emissive_atlas, hazard_entries = build_hazard_atlases(style)
    object_atlas, object_emissive_atlas, object_entries = build_object_atlases(style)
    hazard_sheet = pack_frame_grid(layers.hazard_color_frames)
    hazard_emissive_sheet = pack_frame_grid(layers.hazard_emissive_frames)
    preview = compose_preview(layers, frame=0)
    frame_meta = frame_grid_metadata(layers.base_color.shape[1], layers.base_color.shape[0])
    instances = _instances_payload(data, layers)
    semantic_arrays = layers.semantic_arrays()

    image_arrays = {
        BASE_COLOR_FILE: layers.base_color,
        EMISSIVE_FILE: layers.emissive,
        PREVIEW_FILE: preview,
        HAZARD_FRAMES_FILE: hazard_sheet,
        HAZARD_EMISSIVE_FRAMES_FILE: hazard_emissive_sheet,
        TERRAIN_ATLAS_FILE: terrain_atlas,
        TERRAIN_EMISSIVE_ATLAS_FILE: terrain_emissive_atlas,
        HAZARD_ATLAS_FILE: hazard_atlas,
        HAZARD_EMISSIVE_ATLAS_FILE: hazard_emissive_atlas,
        OBJECT_ATLAS_FILE: object_atlas,
        OBJECT_EMISSIVE_ATLAS_FILE: object_emissive_atlas,
    }
    payloads = {name: _png_bytes(array) for name, array in image_arrays.items()}
    payloads[HAZARD_FRAME_META_FILE] = _json_bytes(frame_meta)
    payloads[INSTANCES_FILE] = _json_bytes(instances)
    payloads[SEMANTICS_FILE] = _npz_bytes(semantic_arrays)

    artifacts: dict[str, Any] = {}
    artifact_keys = {
        BASE_COLOR_FILE: "base_color",
        EMISSIVE_FILE: "emissive",
        PREVIEW_FILE: "preview",
        HAZARD_FRAMES_FILE: "hazard_color_frames",
        HAZARD_EMISSIVE_FRAMES_FILE: "hazard_emissive_frames",
        TERRAIN_ATLAS_FILE: "terrain_atlas",
        TERRAIN_EMISSIVE_ATLAS_FILE: "terrain_emissive_atlas",
        HAZARD_ATLAS_FILE: "hazard_atlas",
        HAZARD_EMISSIVE_ATLAS_FILE: "hazard_emissive_atlas",
        OBJECT_ATLAS_FILE: "object_atlas",
        OBJECT_EMISSIVE_ATLAS_FILE: "object_emissive_atlas",
    }
    for file, key in artifact_keys.items():
        artifacts[key] = _image_descriptor(file, payloads[file], image_arrays[file])
    artifacts["hazard_frame_metadata"] = _file_descriptor(
        HAZARD_FRAME_META_FILE, "json-frame-grid", payloads[HAZARD_FRAME_META_FILE]
    )
    artifacts["instances"] = _file_descriptor(INSTANCES_FILE, "json-object-instances", payloads[INSTANCES_FILE])
    artifacts["art_semantics"] = _file_descriptor(SEMANTICS_FILE, "npz-deflate", payloads[SEMANTICS_FILE])
    atlas_layout = {
        "terrain": {
            "tile_size": TILE_SIZE,
            "columns": 16,
            "rows": 9,
            "color_file": TERRAIN_ATLAS_FILE,
            "emissive_file": TERRAIN_EMISSIVE_ATLAS_FILE,
            "entries": terrain_entries,
        },
        "hazard": {
            "tile_size": TILE_SIZE,
            "columns": HAZARD_FRAME_COUNT,
            "rows": 4,
            "color_file": HAZARD_ATLAS_FILE,
            "emissive_file": HAZARD_EMISSIVE_ATLAS_FILE,
            "entries": hazard_entries,
        },
        "objects": {
            "tile_size": TILE_SIZE,
            "columns": len(style.props),
            "rows": 4,
            "color_file": OBJECT_ATLAS_FILE,
            "emissive_file": OBJECT_EMISSIVE_ATLAS_FILE,
            "entries": object_entries,
        },
    }
    return payloads, artifacts, atlas_layout


def build_manifest(
    data: MapData,
    layers: ArtLayers,
    validation: dict[str, Any],
    artifacts: dict[str, Any],
    atlas_layout: dict[str, Any],
) -> dict[str, Any]:
    height, width = data.shape
    arrays = layers.semantic_arrays()
    renderer_source_hash = source_hash()
    return {
        "schema_version": "1.0.0",
        "art_id": f"{data.map_id}-art-v1",
        "map_id": data.map_id,
        "seed": int(data.seed),
        "theme": data.theme,
        "source": {
            "semantic_array_sha256": array_digest(data.arrays()),
            "expected_array_file": "semantics.npz",
            "coordinate_system": "source arrays [y,x]; exported instance cells [x,y]; origin top-left",
        },
        "dimensions": {
            "cells": {"width": width, "height": height},
            "pixels": {"width": width * TILE_SIZE, "height": height * TILE_SIZE},
            "tile_size": TILE_SIZE,
        },
        "renderer": {
            "name": RENDERER_NAME,
            "version": RENDERER_VERSION,
            "deterministic": True,
            "random_access_hash": "splitmix64-coordinate-v1",
            "source_sha256": renderer_source_hash,
        },
        "animation": {
            "frame_count": HAZARD_FRAME_COUNT,
            "duration_ms": 110,
            "loop": "cyclic",
            "grid": {"columns": FRAME_GRID_COLUMNS, "rows": FRAME_GRID_ROWS},
        },
        "atlases": atlas_layout,
        "semantics": {
            "file": SEMANTICS_FILE,
            "format": "npz-deflate",
            "array_sha256": array_digest(arrays),
            "arrays": {
                name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                for name, array in arrays.items()
            },
        },
        "statistics": {
            **validation["metrics"],
            "catalog_count": len(style_for(data.theme).props),
            "terrain_atlas_entries": len(atlas_layout["terrain"]["entries"]),
            "hazard_atlas_entries": len(atlas_layout["hazard"]["entries"]),
            "object_atlas_entries": len(atlas_layout["objects"]["entries"]),
        },
        "validation": validation["checks"],
        "artifacts": artifacts,
    }


def write_art_pack(
    data: MapData,
    output_root: Path,
    *,
    skip_existing: bool = False,
) -> Path:
    """Render, stage, and atomically publish a semantic-bound art pack."""
    output_root = Path(output_root)
    art_id = f"{data.map_id}-art-v1"
    final = output_root / art_id
    if final.exists():
        if not skip_existing:
            raise FileExistsError(f"Map art pack already exists: {final}")
        from .validate import validate_art_pack

        report = validate_art_pack(final, source_data=data)
        if not report["passed"]:
            raise RuntimeError(f"Existing art pack is invalid and was left untouched: {report}")
        manifest = json.loads((final / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest["renderer"]["source_sha256"] != source_hash():
            raise RuntimeError("Existing art pack was made by different renderer source and was left untouched.")
        return final

    layers = render_map_art(data)
    validation = assert_valid_layers(data, layers)
    payloads, artifacts, atlas_layout = _build_payloads(data, layers)
    planned_bytes = sum(len(payload) for payload in payloads.values()) + 4 * 1024 * 1024
    require_disk_floor(output_root, planned_bytes=planned_bytes)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{art_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    for name in sorted(payloads):
        _atomic_bytes(staging / name, payloads[name])
    manifest = build_manifest(data, layers, validation, artifacts, atlas_layout)
    write_json_atomic(staging / MANIFEST_FILE, manifest)
    os.replace(staging, final)
    return final


def load_art_semantics(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    manifest_path = path if path.name == MANIFEST_FILE else path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    semantic_path = manifest_path.parent / manifest["artifacts"]["art_semantics"]["file"]
    with np.load(semantic_path, allow_pickle=False) as archive:
        if set(archive.files) != set(SEMANTIC_ARRAY_NAMES):
            raise ValueError(f"Unexpected art semantic arrays: {sorted(archive.files)}")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in SEMANTIC_ARRAY_NAMES}
    if array_digest(arrays) != manifest["semantics"]["array_sha256"]:
        raise ValueError("Canonical art-semantic SHA-256 mismatch.")
    return arrays

