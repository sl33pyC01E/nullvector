from __future__ import annotations

import numpy as np

from ..maps.model import Hazard, Terrain
from .model import ArtLayers, HAZARD_FRAME_COUNT, TILE_SIZE, ThemeStyle
from .objects import render_object_sprite
from .renderer import render_hazard_tile, render_terrain_tile


FRAME_GRID_COLUMNS = 4
FRAME_GRID_ROWS = 2


def pack_frame_grid(frames: np.ndarray, *, columns: int = FRAME_GRID_COLUMNS) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError("Frames must have shape [frame, height, width, channels].")
    count, height, width, channels = frames.shape
    if columns < 1:
        raise ValueError("columns must be positive")
    rows = (count + columns - 1) // columns
    result = np.zeros((rows * height, columns * width, channels), dtype=frames.dtype)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        result[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return result


def frame_grid_metadata(pixel_width: int, pixel_height: int, *, duration_ms: int = 110) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "n_frames": HAZARD_FRAME_COUNT,
        "frame_w": int(pixel_width),
        "frame_h": int(pixel_height),
        "duration_ms": int(duration_ms),
        "loop": "cyclic",
        "grid": {"columns": FRAME_GRID_COLUMNS, "rows": FRAME_GRID_ROWS},
        "frames": [
            {
                "index": index,
                "x": (index % FRAME_GRID_COLUMNS) * pixel_width,
                "y": (index // FRAME_GRID_COLUMNS) * pixel_height,
                "width": pixel_width,
                "height": pixel_height,
            }
            for index in range(HAZARD_FRAME_COUNT)
        ],
    }


def build_terrain_atlases(style: ThemeStyle) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    rows = len(Terrain)
    columns = 16
    color = np.zeros((rows * TILE_SIZE, columns * TILE_SIZE, 3), dtype=np.uint8)
    emissive = np.zeros_like(color)
    entries: list[dict[str, int | str]] = []
    for terrain in Terrain:
        for mask in range(columns):
            tile, glow = render_terrain_tile(
                style,
                int(terrain),
                mask,
                0,
                (int(terrain) * 3 + mask) & 7,
                2 if int(terrain) != int(Terrain.VOID) else 0,
            )
            top, left = int(terrain) * TILE_SIZE, mask * TILE_SIZE
            color[top : top + TILE_SIZE, left : left + TILE_SIZE] = tile
            emissive[top : top + TILE_SIZE, left : left + TILE_SIZE] = glow
            entries.append(
                {
                    "terrain_id": int(terrain),
                    "terrain": terrain.name.lower(),
                    "cardinal_mask": mask,
                    "column": mask,
                    "row": int(terrain),
                }
            )
    return color, emissive, entries


def build_hazard_atlases(style: ThemeStyle) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    rows = len(Hazard) - 1
    columns = HAZARD_FRAME_COUNT
    color = np.zeros((rows * TILE_SIZE, columns * TILE_SIZE, 4), dtype=np.uint8)
    emissive = np.zeros((rows * TILE_SIZE, columns * TILE_SIZE, 3), dtype=np.uint8)
    entries: list[dict[str, int | str]] = []
    for hazard in tuple(Hazard)[1:]:
        row = int(hazard) - 1
        for frame in range(HAZARD_FRAME_COUNT):
            tile, glow = render_hazard_tile(style, int(hazard), frame, same_mask=15)
            top, left = row * TILE_SIZE, frame * TILE_SIZE
            color[top : top + TILE_SIZE, left : left + TILE_SIZE] = tile
            emissive[top : top + TILE_SIZE, left : left + TILE_SIZE] = glow
            entries.append(
                {
                    "hazard_id": int(hazard),
                    "hazard": hazard.name.lower(),
                    "frame": frame,
                    "column": frame,
                    "row": row,
                }
            )
    return color, emissive, entries


def build_object_atlases(style: ThemeStyle) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str | bool]]]:
    columns = len(style.props)
    rows = 4
    color = np.zeros((rows * TILE_SIZE, columns * TILE_SIZE, 4), dtype=np.uint8)
    emissive = np.zeros((rows * TILE_SIZE, columns * TILE_SIZE, 3), dtype=np.uint8)
    entries: list[dict[str, int | str | bool]] = []
    for column, spec in enumerate(style.props):
        for orientation in range(rows):
            tile, glow = render_object_sprite(style, spec, orientation=orientation)
            top, left = orientation * TILE_SIZE, column * TILE_SIZE
            color[top : top + TILE_SIZE, left : left + TILE_SIZE] = tile
            emissive[top : top + TILE_SIZE, left : left + TILE_SIZE] = glow
            entries.append(
                {
                    "catalog_index": column + 1,
                    "key": spec.key,
                    "kind": spec.kind,
                    "orientation_quarter_turns_ccw": orientation,
                    "column": column,
                    "row": orientation,
                    "collision": spec.collision,
                    "occlusion": spec.occlusion,
                }
            )
    return color, emissive, entries


def compose_preview(layers: ArtLayers, *, frame: int = 0) -> np.ndarray:
    if not 0 <= frame < HAZARD_FRAME_COUNT:
        raise ValueError(f"Frame must be in [0, {HAZARD_FRAME_COUNT - 1}].")
    base = layers.base_color.astype(np.uint16).copy()
    static_glow = layers.emissive.astype(np.uint16)
    base = np.minimum(base + static_glow // 3, 255)
    overlay = layers.hazard_color_frames[frame]
    alpha = overlay[..., 3:4].astype(np.uint16)
    base = (overlay[..., :3].astype(np.uint16) * alpha + base * (255 - alpha) + 127) // 255
    base = np.minimum(base + layers.hazard_emissive_frames[frame].astype(np.uint16) // 2, 255)
    return base.astype(np.uint8)

