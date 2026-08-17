from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..nature_world_scale_v1.atlas import BIOMES
from .contract import CLASSES, GRID_SIZE, PURPOSES, CityCondition


CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}


@dataclass(frozen=True, slots=True)
class CityExample:
    identity: str
    seed: int
    condition: CityCondition
    target: np.ndarray


def _condition(seed: int) -> CityCondition:
    rng = np.random.default_rng(seed ^ 0x434F4E444954494F)
    family = int(rng.integers(5))
    family_bias = np.asarray((
        (.68, .63, .28, .58, .28, .62, .48, .12),
        (.56, .42, .34, .55, .24, .36, .72, .10),
        (.78, .38, .12, .65, .58, .31, .93, .08),
        (.34, .73, .31, .32, .76, .28, .19, .94),
        (.66, .59, .48, .24, .15, .91, .12, .21),
    ), np.float32)[family]
    culture = tuple(np.clip(family_bias + rng.normal(0, .09, 8), 0, 1).astype(float))
    population = int(rng.integers(3, 33))
    target = int(np.clip(round(2 + population / 4 + rng.normal(0, 1.2)), 2, 12))
    return CityCondition(
        family=family,
        culture=culture,
        project=PURPOSES[int(rng.integers(len(PURPOSES)))],
        biome=BIOMES[int(rng.integers(len(BIOMES)))],
        style=tuple(rng.uniform(0, 1, 16).astype(float)),
        population=population,
        wealth=float(rng.uniform(.1, 5)),
        technology=float(rng.uniform(0, 1)),
        building_target=target,
    )


def render_teacher_city(seed: int, condition: CityCondition | None = None) -> np.ndarray:
    condition = condition or _condition(seed)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)
    buildings, roads = _expand_teacher_settlement(condition=condition)
    for x, y in roads:
        grid[y % GRID_SIZE, x % GRID_SIZE] = CLASS_INDEX["road"]
    for building in buildings:
        for x, y, material in building:
            grid[y % GRID_SIZE, x % GRID_SIZE] = CLASS_INDEX[material]
    return grid


def _teacher_building(condition: CityCondition, index: int, origin: tuple[int, int], purpose: str) -> tuple[tuple[int, int, str], ...]:
    """Learnable local projection of the safe Qud-style building grammar.

    Keeping this tiny projection local avoids importing the gameplay society
    package (which imports NatureWorld back through session saves) into an
    offline training process. Geometry is driven directly by the recorded
    style latent instead of an opaque PRNG sequence.
    """
    style = condition.style
    width = 8 + int(round(style[(index * 2 + 2) % 16] * 10)); height = 8 + int(round(style[(index * 2 + 3) % 16] * 9)); width += width % 2; height += height % 2
    ox, oy = origin; door_x = width // 2; cells = []
    for y in range(height):
        for x in range(width):
            material = "wall" if x < 2 or y < 2 or x >= width - 2 or y >= height - 2 else "floor"
            if y >= height - 2 and door_x - 1 <= x <= door_x: material = "door"
            cells.append((ox + x, oy + y, material))
    feature = {"clinic": "utility", "granary": "storage", "observatory": "utility", "graft_house": "utility", "battery_hall": "utility", "market": "storage"}.get(purpose, "garden" if purpose == "habitat" else "utility")
    stride = 3 + int(style[(index * 3 + 7) % 16] * 3); phase = int(style[(index * 5 + 11) % 16] * stride)
    for cell_index, (x, y, material) in enumerate(cells):
        if material == "floor" and (x + y + phase) % stride == 0 and (cell_index + index) % 2 == 0:
            cells[cell_index] = (x, y, feature)
    return tuple(cells)


def _expand_teacher_settlement(*, condition: CityCondition) -> tuple[list[tuple[tuple[int, int, str], ...]], set[tuple[int, int]]]:
    buildings = []; roads = set(); center = GRID_SIZE // 2; golden = math.pi * (3 - math.sqrt(5)); style = condition.style
    radius_gain = 12 + style[1] * 7
    for index in range(condition.building_target):
        radius = 7 + math.sqrt(index) * radius_gain; angle = index * golden + style[0] * math.tau + condition.family * .09
        origin = (int(round(center + math.cos(angle) * radius)), int(round(center + math.sin(angle) * radius)))
        purpose = condition.project if index == 0 else PURPOSES[(index + condition.family) % len(PURPOSES)]; building = _teacher_building(condition, index, origin, purpose); buildings.append(building)
        doors = [(x, y) for x, y, material in building if material == "door"]
        door = doors[0]; x = center; y = center
        while x != door[0]: roads.add((x, y)); x += 1 if door[0] > x else -1
        while y != door[1]: roads.add((x, y)); y += 1 if door[1] > y else -1
        roads.add(door)
    return buildings, roads


def build_corpus(count: int, *, seed: int) -> tuple[CityExample, ...]:
    if not 64 <= count <= 1_000_000:
        raise ValueError("City corpus size is outside the bounded contract.")
    examples = []
    for index in range(count):
        example_seed = int((seed + index * 0x9E3779B97F4A7C15) & ((1 << 63) - 1))
        condition = _condition(example_seed)
        target = render_teacher_city(example_seed, condition)
        identity = hashlib.sha256(condition.vector().tobytes() + target.tobytes() + example_seed.to_bytes(8, "little")).hexdigest()
        examples.append(CityExample(identity, example_seed, condition, target))
    if len({item.identity for item in examples}) != count:
        raise ValueError("City corpus contains duplicate identities.")
    return tuple(examples)


def _neighbors(y: int, x: int):
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
            yield ny, nx


def compile_city_layout(raw: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if raw.shape != (GRID_SIZE, GRID_SIZE) or raw.dtype != np.uint8 or int(raw.max()) >= len(CLASSES):
        raise ValueError("Raw city layout shape/dtype/vocabulary drifted.")
    result = raw.copy()
    edits = 0
    # Unsupported isolated wall/door pixels are construction hazards. Convert
    # them to floor while retaining the neural proposal as a separate artifact.
    structural = np.isin(result, (CLASS_INDEX["wall"], CLASS_INDEX["door"]))
    for y, x in np.argwhere(structural):
        support = sum(bool(structural[ny, nx]) for ny, nx in _neighbors(int(y), int(x)))
        if support == 0:
            result[y, x] = CLASS_INDEX["floor"]
            edits += 1
    # Every occupied component needs a road or door interface. Add the minimum
    # one-cell road bridge to the center cross when a component has none.
    occupied = result != 0
    seen = np.zeros_like(occupied)
    components = []
    for start_y, start_x in np.argwhere(occupied):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        component = []
        while stack:
            y, x = stack.pop(); component.append((y, x))
            for ny, nx in _neighbors(y, x):
                if occupied[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; stack.append((ny, nx))
        components.append(component)
    center = GRID_SIZE // 2
    for component in components:
        if any(result[y, x] in (CLASS_INDEX["road"], CLASS_INDEX["door"]) for y, x in component):
            continue
        y, x = min(component, key=lambda point: abs(point[0] - center) + abs(point[1] - center))
        while x != center:
            x += 1 if center > x else -1
            if result[y, x] == 0: result[y, x] = CLASS_INDEX["road"]; edits += 1
        while y != center:
            y += 1 if center > y else -1
            if result[y, x] == 0: result[y, x] = CLASS_INDEX["road"]; edits += 1
    return result, {"edited_cells": edits, "occupied_cells": int(occupied.sum())}


def validate_compiled_city(layout: np.ndarray) -> dict[str, object]:
    if layout.shape != (GRID_SIZE, GRID_SIZE) or layout.dtype != np.uint8:
        raise ValueError("Compiled city layout shape/dtype drifted.")
    occupied = layout != 0
    if not bool(occupied.any()) or int(layout.max()) >= len(CLASSES):
        raise ValueError("Compiled city layout is empty or outside vocabulary.")
    isolated_walls = 0
    for y, x in np.argwhere(np.isin(layout, (CLASS_INDEX["wall"], CLASS_INDEX["door"]))):
        if not any(layout[ny, nx] in (CLASS_INDEX["wall"], CLASS_INDEX["door"]) for ny, nx in _neighbors(int(y), int(x))):
            isolated_walls += 1
    return {
        "passed": isolated_walls == 0,
        "isolated_structural_cells": isolated_walls,
        "occupied_fraction": float(occupied.mean()),
        "road_fraction": float((layout == CLASS_INDEX["road"]).mean()),
    }
