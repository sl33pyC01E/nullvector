from __future__ import annotations

from ..maps.model import Terrain, THEMES
from .model import PropSpec, RGB, ThemeStyle


def _shade(color: RGB, delta: int) -> RGB:
    return tuple(max(0, min(255, channel + delta)) for channel in color)  # type: ignore[return-value]


def _style(
    name: str,
    terrain: tuple[RGB, ...],
    *,
    edge_light: RGB,
    edge_hot: RGB,
    grid: RGB,
    emission_primary: RGB,
    emission_secondary: RGB,
    hazard: tuple[RGB, ...],
    props: tuple[PropSpec, ...],
) -> ThemeStyle:
    return ThemeStyle(
        name=name,
        terrain=terrain,
        terrain_detail=tuple(_shade(color, 13) for color in terrain),
        terrain_shadow=tuple(_shade(color, -12) for color in terrain),
        edge_light=edge_light,
        edge_hot=edge_hot,
        grid=grid,
        emission_primary=emission_primary,
        emission_secondary=emission_secondary,
        hazard=hazard,
        props=props,
    )


FLOORISH = (
    int(Terrain.FLOOR),
    int(Terrain.BRIDGE),
    int(Terrain.GROWTH),
    int(Terrain.CRYSTAL),
    int(Terrain.SAND),
)
BLOCKED = (int(Terrain.WALL), int(Terrain.VOID), int(Terrain.CHASM), int(Terrain.WATER))


_COMMON_HAZARDS: tuple[RGB, ...] = (
    (0, 0, 0),
    (255, 42, 177),
    (255, 85, 20),
    (111, 255, 101),
    (48, 222, 255),
)


STYLES: dict[str, ThemeStyle] = {
    "arena": _style(
        "arena",
        (
            (2, 2, 10), (8, 15, 34), (28, 12, 48), (4, 20, 48), (67, 40, 35),
            (10, 40, 35), (48, 20, 67), (4, 1, 14), (55, 47, 36),
        ),
        edge_light=(38, 225, 255), edge_hot=(245, 42, 230), grid=(22, 71, 96),
        emission_primary=(35, 226, 255), emission_secondary=(248, 38, 223),
        hazard=_COMMON_HAZARDS,
        props=(
            PropSpec("holo_pylon", "pylon", "prop", BLOCKED, 83, (0,), True, 3, "secondary"),
            PropSpec("relay_console", "console", "prop", FLOORISH, 137, (0,), False, 2, "primary"),
            PropSpec("circuit_sigil", "circuit", "decal", FLOORISH, 41, (0,), False, 0, "secondary"),
            PropSpec("target_ring", "ring", "decal", FLOORISH, 67, (0,), False, 0, "primary"),
        ),
    ),
    "rooms": _style(
        "rooms",
        (
            (3, 5, 11), (18, 25, 39), (42, 34, 58), (8, 25, 50), (83, 60, 43),
            (22, 49, 40), (57, 30, 68), (4, 3, 12), (69, 58, 43),
        ),
        edge_light=(77, 198, 255), edge_hot=(255, 174, 54), grid=(42, 73, 92),
        emission_primary=(55, 189, 255), emission_secondary=(255, 160, 42),
        hazard=_COMMON_HAZARDS,
        props=(
            PropSpec("bulkhead_terminal", "console", "prop", FLOORISH, 113, (0,), False, 2, "primary"),
            PropSpec("cargo_stack", "crate", "prop", BLOCKED, 73, (0,), True, 2, "secondary"),
            PropSpec("service_hatch", "hatch", "decal", FLOORISH, 53, (0,), False, 0, "secondary"),
            PropSpec("cable_run", "circuit", "decal", FLOORISH, 47, (0,), False, 0, "primary"),
        ),
    ),
    "caves": _style(
        "caves",
        (
            (2, 3, 8), (17, 20, 29), (34, 25, 42), (5, 20, 44), (74, 48, 30),
            (17, 48, 35), (52, 27, 70), (3, 1, 10), (63, 49, 34),
        ),
        edge_light=(83, 172, 214), edge_hot=(222, 69, 255), grid=(42, 53, 69),
        emission_primary=(83, 209, 255), emission_secondary=(214, 70, 255),
        hazard=_COMMON_HAZARDS,
        props=(
            PropSpec("stalagmite", "stalagmite", "prop", BLOCKED, 61, (0,), True, 3, "primary"),
            PropSpec("glow_fungus", "fungus", "prop", FLOORISH, 97, (0,), False, 1, "secondary"),
            PropSpec("mineral_seam", "crack", "decal", FLOORISH, 37, (0,), False, 0, "primary"),
            PropSpec("fossil_mark", "rune", "decal", FLOORISH, 89, (0,), False, 0, "secondary"),
        ),
    ),
    "archipelago": _style(
        "archipelago",
        (
            (2, 8, 18), (24, 42, 45), (45, 38, 50), (4, 38, 67), (104, 67, 38),
            (21, 63, 43), (55, 31, 72), (3, 3, 13), (102, 82, 49),
        ),
        edge_light=(47, 230, 220), edge_hot=(255, 92, 196), grid=(25, 96, 104),
        emission_primary=(37, 231, 220), emission_secondary=(255, 83, 186),
        hazard=_COMMON_HAZARDS,
        props=(
            PropSpec("coral_fan", "coral", "prop", BLOCKED, 67, (0,), True, 2, "secondary"),
            PropSpec("tide_beacon", "beacon", "prop", FLOORISH, 149, (0,), False, 2, "primary"),
            PropSpec("shell_scatter", "dots", "decal", FLOORISH, 43, (0,), False, 0, "secondary"),
            PropSpec("wave_glyph", "wave", "decal", (int(Terrain.WATER), int(Terrain.SAND)), 31, (0,), False, 0, "primary"),
        ),
    ),
    "garden": _style(
        "garden",
        (
            (2, 7, 8), (11, 33, 28), (29, 25, 44), (5, 27, 45), (69, 53, 34),
            (18, 69, 42), (50, 29, 68), (3, 2, 11), (62, 57, 38),
        ),
        edge_light=(69, 255, 155), edge_hot=(255, 77, 202), grid=(31, 94, 59),
        emission_primary=(65, 255, 148), emission_secondary=(255, 74, 194),
        hazard=_COMMON_HAZARDS,
        props=(
            PropSpec("neon_blossom", "flower", "prop", FLOORISH, 79, (0,), False, 1, "secondary"),
            PropSpec("root_lantern", "beacon", "prop", FLOORISH, 157, (0,), False, 2, "primary"),
            PropSpec("vine_knot", "vine", "decal", FLOORISH, 29, (0,), False, 0, "primary"),
            PropSpec("pollen_ring", "dots", "decal", FLOORISH, 47, (0,), False, 0, "secondary"),
        ),
    ),
    "anomaly": _style(
        "anomaly",
        (
            (1, 1, 8), (11, 9, 31), (33, 9, 56), (3, 17, 44), (63, 37, 41),
            (14, 40, 38), (61, 17, 83), (3, 0, 12), (54, 43, 42),
        ),
        edge_light=(43, 242, 255), edge_hot=(255, 35, 226), grid=(43, 35, 93),
        emission_primary=(38, 238, 255), emission_secondary=(255, 34, 219),
        hazard=_COMMON_HAZARDS,
        props=(
            PropSpec("rift_obelisk", "obelisk", "prop", BLOCKED, 53, (0,), True, 3, "secondary"),
            PropSpec("floating_prism", "prism", "prop", FLOORISH, 109, (0,), False, 2, "primary"),
            PropSpec("impossible_eye", "eye", "decal", FLOORISH, 73, (0,), False, 0, "secondary"),
            PropSpec("fracture_rune", "rune", "decal", FLOORISH, 31, (0,), False, 0, "primary"),
        ),
    ),
}


if tuple(STYLES) != THEMES:
    raise RuntimeError("Map art styles must preserve the canonical theme order.")


def style_for(theme: str) -> ThemeStyle:
    try:
        return STYLES[theme]
    except KeyError as error:
        raise ValueError(f"Unknown map-art theme {theme!r}; expected one of {tuple(STYLES)}") from error

