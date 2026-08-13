from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from ..maps.model import THEMES
from ..map_art.provenance import source_hash as map_art_source_hash
from .hashing import sha256_file
from .model import BackgroundCrop, IMAGE_SIZE


CROPS_PER_THEME = 3


def _artifact_path(pack_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("Unsafe map-art artifact path")
    target = (pack_root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(pack_root.resolve())
    except ValueError as error:
        raise ValueError("Map-art artifact escapes its pack") from error
    if not target.is_file() or target.is_symlink():
        raise ValueError("Map-art artifact must be a regular non-symlink file")
    return target


def _crop_positions(image_sha256: str, width: int, height: int) -> list[tuple[int, int]]:
    cells_x, cells_y = width // IMAGE_SIZE, height // IMAGE_SIZE
    if cells_x <= 0 or cells_y <= 0:
        raise ValueError("Map-art base color is smaller than native sprite size")
    total = cells_x * cells_y
    if total < CROPS_PER_THEME:
        raise ValueError("Map-art base color has too few native crop cells")
    digest = hashlib.sha256(
        b"nullvector-style-map-crops-v1\0" + image_sha256.encode("ascii")
    ).digest()
    start = int.from_bytes(digest[:8], "little") % total
    stride = (int.from_bytes(digest[8:16], "little") % (total - 1)) + 1
    while total > 1 and _greatest_common_divisor(stride, total) != 1:
        stride = stride % total + 1
    indices: list[int] = []
    cursor = start
    while len(indices) < CROPS_PER_THEME:
        if cursor not in indices:
            indices.append(cursor)
        cursor = (cursor + stride) % total
    return [((index % cells_x) * IMAGE_SIZE, (index // cells_x) * IMAGE_SIZE) for index in indices]


def _greatest_common_divisor(first: int, second: int) -> int:
    while second:
        first, second = second, first % second
    return abs(first)


def load_background_crops(pack_root: Path) -> tuple[BackgroundCrop, ...]:
    pack_root = Path(pack_root).resolve()
    if not pack_root.is_dir():
        raise ValueError(f"Map-art pack root does not exist: {pack_root}")
    manifests = sorted(pack_root.glob("*/manifest.json"))
    by_theme: dict[str, tuple[Path, dict[str, Any]]] = {}
    current_renderer_source = map_art_source_hash()
    for manifest_path in manifests:
        if manifest_path.is_symlink():
            raise ValueError("Map-art manifests cannot be symlinks")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            json.dumps(payload, allow_nan=False)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"Malformed map-art manifest {manifest_path}: {error}") from error
        theme = payload.get("theme") if isinstance(payload, dict) else None
        if theme not in THEMES:
            continue
        renderer = payload.get("renderer", {}) if isinstance(payload, dict) else {}
        if (
            not isinstance(renderer, dict)
            or renderer.get("source_sha256") != current_renderer_source
        ):
            # Historical packs are intentionally retained for provenance, but
            # only the active renderer contract may supply canonical style
            # contrast crops. This also keeps additive map-art refreshes from
            # becoming falsely ambiguous.
            continue
        if theme in by_theme:
            raise ValueError(f"Ambiguous map-art packs for canonical theme {theme}")
        by_theme[str(theme)] = (manifest_path.resolve(), payload)
    missing = [theme for theme in THEMES if theme not in by_theme]
    if missing:
        raise ValueError(f"Missing representative map-art themes: {missing}")

    crops: list[BackgroundCrop] = []
    for theme in THEMES:
        manifest_path, manifest = by_theme[theme]
        artifacts = manifest.get("artifacts", {})
        base_record = artifacts.get("base_color", {}) if isinstance(artifacts, dict) else {}
        if not isinstance(base_record, dict):
            raise ValueError(f"Map-art base_color record missing for {theme}")
        image_path = _artifact_path(manifest_path.parent, str(base_record.get("file", "")))
        expected_sha = str(base_record.get("sha256", ""))
        actual_sha = sha256_file(image_path)
        if actual_sha != expected_sha:
            raise ValueError(f"Map-art base_color hash mismatch for {theme}")
        with Image.open(image_path) as opened:
            if opened.mode != "RGB":
                raise ValueError(f"Map-art base_color must be RGB for {theme}")
            rgb = np.asarray(opened, dtype=np.uint8).copy()
        height, width = rgb.shape[:2]
        declared_size = base_record.get("pixel_size")
        if declared_size != [width, height] or width % IMAGE_SIZE or height % IMAGE_SIZE:
            raise ValueError(f"Map-art base_color dimensions are not native-grid aligned for {theme}")
        for xy in _crop_positions(actual_sha, width, height):
            x, y = xy
            crop = rgb[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE].copy()
            crops.append(
                BackgroundCrop(
                    theme=theme,
                    pack_manifest_path=manifest_path,
                    pack_manifest_sha256=sha256_file(manifest_path),
                    image_path=image_path,
                    image_sha256=actual_sha,
                    xy=xy,
                    rgb=crop,
                )
            )
    return tuple(crops)


def background_catalog(crops: tuple[BackgroundCrop, ...], project_root: Path) -> list[dict[str, Any]]:
    project_root = Path(project_root).resolve()
    records: list[dict[str, Any]] = []
    for theme in THEMES:
        themed = [crop for crop in crops if crop.theme == theme]
        first = themed[0]
        records.append(
            {
                "theme": theme,
                "pack_manifest_path": first.pack_manifest_path.relative_to(project_root).as_posix(),
                "pack_manifest_sha256": first.pack_manifest_sha256,
                "base_color_path": first.image_path.relative_to(project_root).as_posix(),
                "base_color_sha256": first.image_sha256,
                "crop_xy": [list(crop.xy) for crop in themed],
            }
        )
    return records
