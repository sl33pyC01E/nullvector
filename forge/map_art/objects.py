from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..maps.model import MapData
from .hashing import bounded_hash, coordinate_hash
from .model import ArtInstance, PropSpec, ThemeStyle, TILE_SIZE


_OBJECT_SALT = 0x4F424A454354


def _adjacent_walkable(data: MapData, x: int, y: int) -> bool:
    height, width = data.shape
    return any(
        0 <= nx < width and 0 <= ny < height and bool(data.walkability[ny, nx])
        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
    )


def _reserved_cells(data: MapData) -> set[tuple[int, int]]:
    anchors = (data.start, data.exit, *data.objectives, *data.spawns)
    reserved: set[tuple[int, int]] = set()
    for x, y in anchors:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                reserved.add((x + dx, y + dy))
    return reserved


def derive_instances(data: MapData, style: ThemeStyle) -> tuple[ArtInstance, ...]:
    """Place sparse objects without consuming a mutable RNG stream."""
    height, width = data.shape
    reserved = _reserved_cells(data)
    instances: list[ArtInstance] = []
    occupied_kinds: set[tuple[int, int, str]] = set()
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            terrain = int(data.terrain[y, x])
            if int(data.hazard[y, x]) != 0 or (x, y) in reserved:
                continue
            for catalog_index, spec in enumerate(style.props, start=1):
                if terrain not in spec.allowed_terrain or (x, y, spec.kind) in occupied_kinds:
                    continue
                slot = bounded_hash(
                    data.seed,
                    x,
                    y,
                    _OBJECT_SALT + catalog_index * 17,
                    spec.placement_modulus,
                )
                if slot not in spec.placement_slots:
                    continue
                walkable = bool(data.walkability[y, x])
                if spec.collision and (walkable or not _adjacent_walkable(data, x, y)):
                    continue
                if spec.kind == "prop" and not spec.collision and not walkable:
                    continue
                orientation = bounded_hash(data.seed, x, y, _OBJECT_SALT + catalog_index, 4)
                instances.append(
                    ArtInstance(
                        instance_id=(
                            f"{data.map_id}:object:{catalog_index:02d}:{x:03d}:{y:03d}"
                        ),
                        catalog_index=catalog_index,
                        key=spec.key,
                        kind=spec.kind,
                        cell=(x, y),
                        atlas_cell=(catalog_index - 1, orientation),
                        collision=spec.collision,
                        occlusion=spec.occlusion,
                        z_class="ground" if spec.kind == "decal" else ("high" if spec.occlusion >= 2 else "low"),
                    )
                )
                occupied_kinds.add((x, y, spec.kind))
    return tuple(instances)


def _role_colors(style: ThemeStyle, spec: PropSpec) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if spec.color_role == "secondary":
        return style.edge_hot, style.emission_secondary
    return style.edge_light, style.emission_primary


def _pixel(
    rgba: np.ndarray,
    emissive: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
    glow: tuple[int, int, int] | None = None,
    *,
    alpha: int = 255,
) -> None:
    if 0 <= x < TILE_SIZE and 0 <= y < TILE_SIZE:
        rgba[y, x, :3] = color
        rgba[y, x, 3] = alpha
        if glow is not None:
            emissive[y, x] = glow


def _line(
    rgba: np.ndarray,
    emissive: np.ndarray,
    points: Iterable[tuple[int, int]],
    color: tuple[int, int, int],
    glow: tuple[int, int, int] | None = None,
    *,
    alpha: int = 255,
) -> None:
    for x, y in points:
        _pixel(rgba, emissive, x, y, color, glow, alpha=alpha)


def render_object_sprite(
    style: ThemeStyle,
    spec: PropSpec,
    *,
    orientation: int = 0,
    variant_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return crisp RGBA color and RGB additive-emission object tiles."""
    rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    emissive = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    color, glow = _role_colors(style, spec)
    shadow = tuple(max(0, channel // 3) for channel in color)
    hot = (245, 248, 255)
    shape = spec.shape

    if shape in {"pylon", "obelisk", "beacon"}:
        _line(rgba, emissive, ((3, y) for y in range(1, 7)), shadow)
        _line(rgba, emissive, ((4, y) for y in range(1, 7)), color)
        _line(rgba, emissive, ((2, 6), (3, 6), (4, 6), (5, 6)), shadow)
        _line(rgba, emissive, ((3, 1), (4, 1), (3, 2), (4, 2)), hot, glow)
        if shape == "obelisk":
            _pixel(rgba, emissive, 2, 3, color, glow)
            _pixel(rgba, emissive, 5, 4, color, glow)
        elif shape == "beacon":
            _line(rgba, emissive, ((2, 2), (5, 2), (2, 3), (5, 3)), color, glow)
    elif shape in {"console", "crate", "hatch"}:
        for y in range(2, 7):
            for x in range(1, 7):
                edge = x in (1, 6) or y in (2, 6)
                _pixel(rgba, emissive, x, y, color if edge else shadow)
        if shape == "console":
            _line(rgba, emissive, ((2, 3), (3, 3), (4, 3), (5, 3)), hot, glow)
            _pixel(rgba, emissive, 5, 5, color, glow)
        elif shape == "crate":
            _line(rgba, emissive, ((2, 3), (3, 4), (4, 4), (5, 5)), color)
            _line(rgba, emissive, ((5, 3), (4, 4), (3, 4), (2, 5)), color)
        else:
            _line(rgba, emissive, ((2, 4), (3, 3), (4, 3), (5, 4), (4, 5), (3, 5)), color, glow)
    elif shape in {"stalagmite", "prism"}:
        rows = ((3, 1), (2, 2), (4, 2), (2, 3), (3, 3), (4, 3), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4))
        _line(rgba, emissive, rows, color)
        _line(rgba, emissive, ((2, 5), (3, 5), (4, 5), (5, 5), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)), shadow)
        if shape == "prism":
            _line(rgba, emissive, ((3, 1), (2, 3), (3, 3), (4, 3)), hot, glow)
    elif shape in {"fungus", "flower", "coral"}:
        _line(rgba, emissive, ((3, 4), (3, 5), (3, 6)), shadow)
        if shape == "fungus":
            _line(rgba, emissive, ((1, 3), (2, 2), (3, 2), (4, 2), (5, 3), (2, 3), (3, 3), (4, 3)), color, glow)
        elif shape == "flower":
            _line(rgba, emissive, ((3, 1), (1, 3), (5, 3), (3, 5), (2, 2), (4, 2), (2, 4), (4, 4)), color, glow)
            _pixel(rgba, emissive, 3, 3, hot, glow)
        else:
            _line(rgba, emissive, ((3, 5), (2, 4), (1, 3), (4, 4), (5, 3), (2, 2), (5, 1)), color, glow)
    elif shape == "ring":
        _line(rgba, emissive, ((2, 2), (3, 1), (4, 1), (5, 2), (6, 3), (6, 4), (5, 5), (4, 6), (3, 6), (2, 5), (1, 4), (1, 3)), color, glow, alpha=180)
    elif shape in {"circuit", "crack", "vine", "wave"}:
        if shape == "wave":
            points = ((0, 3), (1, 2), (2, 2), (3, 3), (4, 4), (5, 4), (6, 3), (7, 3))
        elif shape == "vine":
            points = ((1, 6), (2, 5), (2, 4), (3, 3), (4, 3), (5, 2), (6, 1))
        elif shape == "crack":
            points = ((1, 1), (2, 2), (3, 2), (3, 3), (4, 4), (4, 6), (6, 7))
        else:
            points = ((0, 4), (2, 4), (2, 2), (4, 2), (4, 5), (6, 5), (6, 7))
        _line(rgba, emissive, points, color, glow, alpha=180)
    elif shape in {"rune", "eye"}:
        if shape == "eye":
            _line(rgba, emissive, ((1, 3), (2, 2), (3, 2), (4, 2), (5, 3), (4, 4), (3, 4), (2, 4)), color, glow, alpha=190)
            _pixel(rgba, emissive, 3, 3, hot, glow, alpha=220)
        else:
            _line(rgba, emissive, ((1, 1), (6, 1), (4, 3), (6, 6), (1, 6), (3, 3), (1, 1)), color, glow, alpha=180)
    elif shape == "dots":
        offsets = ((1, 2), (4, 1), (6, 3), (2, 5), (5, 6))
        _line(rgba, emissive, offsets, color, glow, alpha=180)
    else:
        raise ValueError(f"Unknown object sprite shape {shape!r}")

    # Four exact rotations share one atlas row and retain nearest-neighbor pixels.
    turns = int(orientation) % 4
    if turns:
        rgba = np.ascontiguousarray(np.rot90(rgba, turns))
        emissive = np.ascontiguousarray(np.rot90(emissive, turns))
    return rgba, emissive

