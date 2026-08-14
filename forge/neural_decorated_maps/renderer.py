from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ..map_art.autotile import cardinal_match_mask, elevation_drop_mask
from ..map_art.hashing import bounded_hash
from ..map_art.model import ArtInstance, HAZARD_FRAME_COUNT, TILE_SIZE
from ..map_art.objects import render_object_sprite
from ..map_art.renderer import render_hazard_tile, render_terrain_tile
from ..map_art.styles import style_for
from ..map_decorator.catalog import catalog_for, validate_decoration_fields
from ..maps.model import Hazard, MapData, Terrain


_OBJECT_RENDER_SALT: Final[int] = 0x535052495445
_OBJECT_ORIENTATION_SALT: Final[int] = 0x4F424A454354
_HAZARD_SALT: Final[int] = 0x48415A415244


@dataclass(slots=True)
class SelectedMapLayers:
    base_color: np.ndarray
    emissive: np.ndarray
    hazard_color_frames: np.ndarray
    hazard_emissive_frames: np.ndarray
    collision: np.ndarray
    occlusion: np.ndarray
    prop_id: np.ndarray
    decal_id: np.ndarray
    instances: tuple[ArtInstance, ...]


def _alpha_composite(base: np.ndarray, overlay: np.ndarray) -> None:
    alpha = overlay[..., 3:4].astype(np.uint16)
    inverse = 255 - alpha
    base[:] = ((overlay[..., :3].astype(np.uint16) * alpha + base.astype(np.uint16) * inverse + 127) // 255).astype(np.uint8)


def _scaled_glow(glow: np.ndarray, level: int) -> np.ndarray:
    if level <= 0:
        return np.zeros_like(glow)
    return np.ascontiguousarray((glow.astype(np.uint16) * min(level, 3) // 3).astype(np.uint8))


def render_selected_map(data: MapData, fields: dict[str, np.ndarray]) -> SelectedMapLayers:
    report = validate_decoration_fields(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        **fields,
    )
    if not report["passed"]:
        raise ValueError(f"Selected map fields are illegal: {report['failures']}")
    style = style_for(data.theme)
    catalog = catalog_for(data.theme)
    height, width = data.shape
    pixel_height, pixel_width = height * TILE_SIZE, width * TILE_SIZE
    autotile = cardinal_match_mask(data.terrain)
    elevation_edges = elevation_drop_mask(data.elevation, data.walkability)
    hazard_autotile = cardinal_match_mask(data.hazard)
    base = np.zeros((pixel_height, pixel_width, 3), dtype=np.uint8)
    emissive = np.zeros_like(base)
    for y in range(height):
        for x in range(width):
            tile, glow = render_terrain_tile(
                style,
                int(data.terrain[y, x]),
                int(autotile[y, x]),
                int(elevation_edges[y, x]),
                int(fields["variant"][y, x]),
                int(data.elevation[y, x]),
            )
            top, left = y * TILE_SIZE, x * TILE_SIZE
            base[top : top + TILE_SIZE, left : left + TILE_SIZE] = tile
            emissive[top : top + TILE_SIZE, left : left + TILE_SIZE] = _scaled_glow(
                glow, int(fields["emission"][y, x])
            )

    prop_id = np.zeros((height, width), dtype=np.int16)
    decal_id = np.zeros((height, width), dtype=np.int16)
    collision = (data.walkability == 0).astype(np.uint8)
    occlusion = np.zeros((height, width), dtype=np.uint8)
    occlusion[data.walkability == 0] = 1
    occlusion[data.terrain == int(Terrain.WALL)] = 2
    instances: list[ArtInstance] = []
    for head, entries in (("decal", catalog.decal_classes), ("prop", catalog.prop_classes)):
        lookup = {entry.class_id: entry for entry in entries}
        for y, x in np.argwhere(fields[head] != 0):
            class_id = int(fields[head][y, x])
            entry = lookup[class_id]
            spec = style.props[entry.catalog_index - 1]
            orientation = bounded_hash(data.seed, int(x), int(y), _OBJECT_ORIENTATION_SALT + entry.catalog_index, 4)
            sprite, glow = render_object_sprite(
                style,
                spec,
                orientation=orientation,
                variant_seed=bounded_hash(data.seed, int(x), int(y), _OBJECT_RENDER_SALT, 1 << 16),
            )
            top, left = int(y) * TILE_SIZE, int(x) * TILE_SIZE
            _alpha_composite(base[top : top + TILE_SIZE, left : left + TILE_SIZE], sprite)
            region = emissive[top : top + TILE_SIZE, left : left + TILE_SIZE]
            np.maximum(region, _scaled_glow(glow, int(fields["emission"][y, x])), out=region)
            if head == "prop":
                prop_id[y, x] = entry.catalog_index
            else:
                decal_id[y, x] = entry.catalog_index
            occlusion[y, x] = max(int(occlusion[y, x]), entry.occlusion)
            instances.append(
                ArtInstance(
                    instance_id=f"{data.map_id}:neural:{head}:{class_id}:{int(x):03d}:{int(y):03d}",
                    catalog_index=entry.catalog_index,
                    key=entry.key,
                    kind=head,
                    cell=(int(x), int(y)),
                    atlas_cell=(entry.catalog_index - 1, orientation),
                    collision=False,
                    occlusion=entry.occlusion,
                    z_class="ground" if head == "decal" else ("high" if entry.occlusion >= 2 else "low"),
                )
            )

    hazard_color = np.zeros((HAZARD_FRAME_COUNT, pixel_height, pixel_width, 4), dtype=np.uint8)
    hazard_emissive = np.zeros((HAZARD_FRAME_COUNT, pixel_height, pixel_width, 3), dtype=np.uint8)
    for y, x in np.argwhere(data.hazard != int(Hazard.NONE)):
        hazard_id = int(data.hazard[y, x])
        phase = 0 if hazard_id == int(Hazard.LASER) else bounded_hash(
            data.seed, int(x), int(y), _HAZARD_SALT, HAZARD_FRAME_COUNT
        )
        top, left = int(y) * TILE_SIZE, int(x) * TILE_SIZE
        for frame in range(HAZARD_FRAME_COUNT):
            tile, glow = render_hazard_tile(style, hazard_id, frame, same_mask=int(hazard_autotile[y, x]), phase_offset=phase)
            hazard_color[frame, top : top + TILE_SIZE, left : left + TILE_SIZE] = tile
            hazard_emissive[frame, top : top + TILE_SIZE, left : left + TILE_SIZE] = glow
    return SelectedMapLayers(
        base_color=np.ascontiguousarray(base),
        emissive=np.ascontiguousarray(emissive),
        hazard_color_frames=hazard_color,
        hazard_emissive_frames=hazard_emissive,
        collision=collision,
        occlusion=occlusion,
        prop_id=prop_id,
        decal_id=decal_id,
        instances=tuple(instances),
    )


def composite_frame(layers: SelectedMapLayers, frame: int = 0) -> np.ndarray:
    result = layers.base_color.copy()
    _alpha_composite(result, layers.hazard_color_frames[frame % HAZARD_FRAME_COUNT])
    glow = np.maximum(layers.emissive, layers.hazard_emissive_frames[frame % HAZARD_FRAME_COUNT])
    result = np.clip(result.astype(np.uint16) + glow.astype(np.uint16) // 3, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(result)
