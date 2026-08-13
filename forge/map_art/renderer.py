from __future__ import annotations

from typing import Final

import numpy as np

from ..maps.model import Hazard, MapData, Terrain, WALKABLE_TERRAIN
from ..maps.validate import assert_valid
from .autotile import EAST, NORTH, SOUTH, WEST, cardinal_match_mask, elevation_drop_mask
from .hashing import bounded_hash
from .model import ArtLayers, HAZARD_FRAME_COUNT, TILE_SIZE, ThemeStyle
from .objects import derive_instances, render_object_sprite
from .styles import style_for


_VARIANT_SALT: Final[int] = 0x56415249414E54
_HAZARD_SALT: Final[int] = 0x48415A415244


def _adjust(color: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, channel + delta)) for channel in color)  # type: ignore[return-value]


def _emit_pixel(emissive: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < TILE_SIZE and 0 <= y < TILE_SIZE:
        emissive[y, x] = color


def render_terrain_tile(
    style: ThemeStyle,
    terrain_id: int,
    same_mask: int,
    elevation_edges: int,
    variant: int,
    elevation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one reusable four-neighbor terrain tile and its emission."""
    if not 0 <= terrain_id < len(style.terrain):
        raise ValueError(f"Unknown terrain id {terrain_id}")
    if not 0 <= same_mask <= 15 or not 0 <= elevation_edges <= 15:
        raise ValueError("Autotile masks must be four-bit values.")
    lift = min(max(int(elevation), 0), 5) * 3
    base = np.full((TILE_SIZE, TILE_SIZE, 3), _adjust(style.terrain[terrain_id], lift), dtype=np.uint8)
    emissive = np.zeros_like(base)
    detail = _adjust(style.terrain_detail[terrain_id], lift)
    shadow = style.terrain_shadow[terrain_id]
    variant = int(variant) & 7

    # Terrain-specific micro-patterns keep the tiles legible at native 8 px.
    if terrain_id == int(Terrain.FLOOR):
        if variant in (0, 3, 6):
            base[3, 1:7] = detail
            base[1:7, 3] = detail
            base[3, 3] = style.grid
    elif terrain_id == int(Terrain.WALL):
        base[2, :] = detail
        base[5, :] = shadow
        base[1:6, (variant % 3) + 2] = detail
    elif terrain_id == int(Terrain.WATER):
        row = 2 + (variant % 3)
        base[row, 1:4] = detail
        base[(row + 3) % 7, 4:7] = detail
    elif terrain_id == int(Terrain.BRIDGE):
        base[:, 1] = shadow
        base[:, 6] = shadow
        base[(variant % 3) + 2, 1:7] = detail
    elif terrain_id == int(Terrain.GROWTH):
        base[6, 1:7] = shadow
        base[5, 2] = detail
        base[4, 3] = detail
        base[3, 5] = detail
        if variant & 1:
            _emit_pixel(emissive, 5, 3, style.emission_primary)
    elif terrain_id == int(Terrain.CRYSTAL):
        base[6, 2:6] = shadow
        for x, y in ((2, 5), (3, 3), (4, 1), (5, 4)):
            base[y, x] = detail
            _emit_pixel(emissive, x, y, style.emission_secondary if (x + variant) & 1 else style.emission_primary)
    elif terrain_id == int(Terrain.CHASM):
        if variant in (1, 4, 7):
            base[variant % 6 + 1, (variant * 3) % 6 + 1] = detail
    elif terrain_id == int(Terrain.SAND):
        for index in range(3):
            x = (variant * 3 + index * 5) % 7
            y = (variant * 5 + index * 3) % 7
            base[y, x] = detail
    elif terrain_id == int(Terrain.VOID):
        if variant == 0:
            base[2, 5] = detail

    walkable = terrain_id in WALKABLE_TERRAIN
    border_light = style.edge_light if walkable else detail
    border_hot = style.edge_hot if terrain_id in (int(Terrain.CRYSTAL), int(Terrain.CHASM)) else border_light
    if not same_mask & NORTH:
        base[0, :] = border_light
        if walkable:
            emissive[0, variant % TILE_SIZE] = style.emission_primary
    if not same_mask & EAST:
        base[:, 7] = border_hot
        if walkable:
            emissive[(variant * 3) % TILE_SIZE, 7] = style.emission_secondary
    if not same_mask & SOUTH:
        base[7, :] = shadow
        if walkable:
            base[7, variant % TILE_SIZE] = border_hot
    if not same_mask & WEST:
        base[:, 0] = border_light
        if walkable:
            emissive[(variant * 5) % TILE_SIZE, 0] = style.emission_primary

    # Drop faces are deliberately two pixels thick so elevation reads top-down.
    if elevation_edges & NORTH:
        base[1, 1:7] = shadow
    if elevation_edges & EAST:
        base[1:7, 6] = shadow
    if elevation_edges & SOUTH:
        base[6:8, 1:7] = shadow
        base[6, 2 + variant % 4] = border_hot
    if elevation_edges & WEST:
        base[1:7, 1] = shadow
    return base, emissive


def render_hazard_tile(
    style: ThemeStyle,
    hazard_id: int,
    frame: int,
    *,
    same_mask: int = 0,
    phase_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a loopable transparent hazard tile plus additive emission."""
    if not 1 <= hazard_id <= 4:
        raise ValueError("Hazard id must be in [1, 4].")
    if not 0 <= frame < HAZARD_FRAME_COUNT:
        raise ValueError(f"Frame must be in [0, {HAZARD_FRAME_COUNT - 1}].")
    color = style.hazard[hazard_id]
    hot = (255, 250, 255)
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    emissive = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    phase = (int(frame) + int(phase_offset)) % HAZARD_FRAME_COUNT

    def put(x: int, y: int, rgb: tuple[int, int, int], alpha: int = 255, glow: bool = True) -> None:
        if 0 <= x < TILE_SIZE and 0 <= y < TILE_SIZE:
            rgba[y, x, :3] = rgb
            rgba[y, x, 3] = alpha
            if glow:
                emissive[y, x] = color

    if hazard_id == int(Hazard.LASER):
        width = 2 if phase in (2, 3, 4) else 1
        connected = same_mask & 15
        horizontal = bool(connected & (EAST | WEST)) or not bool(connected & (NORTH | SOUTH))
        vertical = bool(connected & (NORTH | SOUTH)) or not bool(connected & (EAST | WEST))
        if horizontal:
            for offset in range(width):
                y = 3 + offset
                for x in range(TILE_SIZE):
                    put(x, y, hot if offset == 0 and phase in (3, 4) else color)
        if vertical:
            for offset in range(width):
                x = 3 + offset
                for y in range(TILE_SIZE):
                    put(x, y, hot if offset == 0 and phase in (3, 4) else color)
        put(3, 3, hot)
    elif hazard_id == int(Hazard.LAVA):
        pool = ((1, 3), (2, 2), (3, 2), (4, 2), (5, 3), (6, 3), (1, 4), (2, 5), (3, 5), (4, 5), (5, 5), (6, 4))
        for x, y in pool:
            put(x, y, color, 180)
        for index in range(3):
            x = 2 + ((phase + index * 2) % 4)
            y = 2 + ((phase * 3 + index) % 4)
            put(x, y, hot if index == 0 else style.edge_hot)
    elif hazard_id == int(Hazard.SPORES):
        centers = ((1, 2), (3, 1), (5, 3), (2, 5), (6, 6), (4, 5))
        for index, (x, y) in enumerate(centers):
            drift = (phase + index) % 3 - 1
            put(x, max(0, min(7, y + drift)), hot if (phase + index) % 4 == 0 else color, 210)
    else:  # ARC
        direction = -1 if phase & 1 else 1
        points = []
        for y in range(TILE_SIZE):
            x = 3 + direction * (((y + phase) % 3) - 1)
            points.append((x, y))
        for index, (x, y) in enumerate(points):
            put(x, y, hot if index in (2, 5) else color)
        put(1 + phase % 2, 3, color, 190)
        put(6 - phase % 2, 5, color, 190)
    return rgba, emissive


def _alpha_composite_rgb(base: np.ndarray, overlay: np.ndarray) -> None:
    alpha = overlay[..., 3:4].astype(np.uint16)
    inverse = 255 - alpha
    blended = (
        overlay[..., :3].astype(np.uint16) * alpha
        + base.astype(np.uint16) * inverse
        + 127
    ) // 255
    base[:] = blended.astype(np.uint8)


def render_map_art(data: MapData) -> ArtLayers:
    assert_valid(data)
    style = style_for(data.theme)
    height, width = data.shape
    pixel_height, pixel_width = height * TILE_SIZE, width * TILE_SIZE
    autotile = cardinal_match_mask(data.terrain)
    elevation_edges = elevation_drop_mask(data.elevation, data.walkability)
    hazard_autotile = cardinal_match_mask(data.hazard)
    variant = np.empty((height, width), dtype=np.uint8)
    base = np.zeros((pixel_height, pixel_width, 3), dtype=np.uint8)
    emissive = np.zeros_like(base)

    for y in range(height):
        for x in range(width):
            tile_variant = bounded_hash(data.seed, x, y, _VARIANT_SALT, 8)
            variant[y, x] = tile_variant
            tile, glow = render_terrain_tile(
                style,
                int(data.terrain[y, x]),
                int(autotile[y, x]),
                int(elevation_edges[y, x]),
                tile_variant,
                int(data.elevation[y, x]),
            )
            top, left = y * TILE_SIZE, x * TILE_SIZE
            base[top : top + TILE_SIZE, left : left + TILE_SIZE] = tile
            emissive[top : top + TILE_SIZE, left : left + TILE_SIZE] = glow

    instances = derive_instances(data, style)
    prop_id = np.zeros((height, width), dtype=np.int16)
    decal_id = np.zeros((height, width), dtype=np.int16)
    collision = (data.walkability == 0).astype(np.uint8)
    occlusion = np.zeros((height, width), dtype=np.uint8)
    occlusion[data.walkability == 0] = 1
    occlusion[data.terrain == int(Terrain.WALL)] = 2
    spec_by_index = {index: spec for index, spec in enumerate(style.props, start=1)}
    for instance in instances:
        x, y = instance.cell
        spec = spec_by_index[instance.catalog_index]
        sprite, glow = render_object_sprite(
            style,
            spec,
            orientation=instance.atlas_cell[1],
            variant_seed=bounded_hash(data.seed, x, y, _OBJECT_RENDER_SALT, 1 << 16),
        )
        top, left = y * TILE_SIZE, x * TILE_SIZE
        target = base[top : top + TILE_SIZE, left : left + TILE_SIZE]
        _alpha_composite_rgb(target, sprite)
        emissive_region = emissive[top : top + TILE_SIZE, left : left + TILE_SIZE]
        np.maximum(emissive_region, glow, out=emissive_region)
        if instance.kind == "prop":
            prop_id[y, x] = instance.catalog_index
        else:
            decal_id[y, x] = instance.catalog_index
        if instance.collision:
            collision[y, x] = 1
        occlusion[y, x] = max(int(occlusion[y, x]), instance.occlusion)

    hazard_color = np.zeros(
        (HAZARD_FRAME_COUNT, pixel_height, pixel_width, 4), dtype=np.uint8
    )
    hazard_emissive = np.zeros(
        (HAZARD_FRAME_COUNT, pixel_height, pixel_width, 3), dtype=np.uint8
    )
    for y, x in np.argwhere(data.hazard != int(Hazard.NONE)):
        hazard_id = int(data.hazard[y, x])
        phase_offset = 0 if hazard_id == int(Hazard.LASER) else bounded_hash(
            data.seed, int(x), int(y), _HAZARD_SALT, HAZARD_FRAME_COUNT
        )
        top, left = int(y) * TILE_SIZE, int(x) * TILE_SIZE
        for frame in range(HAZARD_FRAME_COUNT):
            tile, glow = render_hazard_tile(
                style,
                hazard_id,
                frame,
                same_mask=int(hazard_autotile[y, x]),
                phase_offset=phase_offset,
            )
            hazard_color[frame, top : top + TILE_SIZE, left : left + TILE_SIZE] = tile
            hazard_emissive[frame, top : top + TILE_SIZE, left : left + TILE_SIZE] = glow

    return ArtLayers(
        base_color=base,
        emissive=emissive,
        hazard_color_frames=hazard_color,
        hazard_emissive_frames=hazard_emissive,
        autotile_mask=autotile,
        elevation_edge_mask=elevation_edges,
        variant=variant,
        collision=collision,
        occlusion=occlusion,
        prop_id=prop_id,
        decal_id=decal_id,
        instances=instances,
    )


_OBJECT_RENDER_SALT: Final[int] = 0x535052495445

