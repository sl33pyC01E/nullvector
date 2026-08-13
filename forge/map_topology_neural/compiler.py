from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Any, Final, Iterable

import numpy as np

from ..maps.model import (
    THEMES,
    WALKABLE_TERRAIN,
    Hazard,
    MapConfig,
    MapData,
    Point,
    Terrain,
)
from ..maps.validate import assert_valid
from .contract import CONTRACT_SHA256, FIELD_CLASS_COUNTS
from .hashing import json_sha256, named_arrays_sha256
from .provenance import compiler_source_sha256


COMPILER_NAME: Final[str] = "nullvector-neural-topology-deterministic-compiler"
COMPILER_VERSION: Final[str] = "1.0.0"
LEDGER_FORMAT: Final[str] = "nullvector-neural-topology-edit-ledger-v1"
MAX_LEDGER_ENTRIES: Final[int] = 2_000_000
THEME_HAZARDS: Final[dict[str, tuple[int, ...]]] = {
    "arena": (int(Hazard.LASER),),
    "rooms": (int(Hazard.ARC),),
    "caves": (int(Hazard.LAVA),),
    "archipelago": (int(Hazard.ARC),),
    "garden": (int(Hazard.SPORES),),
    "anomaly": (int(Hazard.LASER), int(Hazard.ARC)),
}
THEME_HAZARD_FRACTION: Final[dict[str, float]] = {
    "arena": 0.08,
    "rooms": 0.08,
    "caves": 0.12,
    "archipelago": 0.10,
    "garden": 0.12,
    "anomaly": 0.14,
}
NAV_BASE: Final[dict[int, float]] = {
    int(Terrain.FLOOR): 1.0,
    int(Terrain.BRIDGE): 1.15,
    int(Terrain.GROWTH): 1.55,
    int(Terrain.CRYSTAL): 1.7,
    int(Terrain.SAND): 1.35,
}


@dataclass(frozen=True, slots=True)
class RawTopology:
    terrain: np.ndarray
    hazard: np.ndarray
    elevation: np.ndarray
    raw_sha256: str

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "terrain": self.terrain,
            "hazard": self.hazard,
            "elevation": self.elevation,
        }


@dataclass(frozen=True, slots=True)
class CompileResult:
    data: MapData
    ledger: tuple[dict[str, object], ...]
    ledger_sha256: str
    raw_sha256: str
    compiler_source_sha256: str
    compiled_arrays_sha256: str
    report: dict[str, object]


def make_raw_topology(
    terrain: np.ndarray,
    hazard: np.ndarray,
    elevation: np.ndarray,
    *,
    shape: tuple[int, int] | None = None,
) -> RawTopology:
    expected = shape if shape is not None else getattr(terrain, "shape", None)
    fields = {
        "terrain": (terrain, np.dtype(np.uint8)),
        "hazard": (hazard, np.dtype(np.uint8)),
        "elevation": (elevation, np.dtype(np.int8)),
    }
    copies: dict[str, np.ndarray] = {}
    for name, (array, dtype) in fields.items():
        if not isinstance(array, np.ndarray) or array.ndim != 2 or array.shape != expected:
            raise TypeError(f"{name} must be a two-dimensional ndarray with shape {expected}.")
        if array.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}.")
        if not bool(((array >= 0) & (array < FIELD_CLASS_COUNTS[name])).all()):
            raise ValueError(f"{name} contains an illegal categorical ID.")
        copied = np.ascontiguousarray(array.copy())
        copied.setflags(write=False)
        copies[name] = copied
    return RawTopology(
        terrain=copies["terrain"],
        hazard=copies["hazard"],
        elevation=copies["elevation"],
        raw_sha256=named_arrays_sha256(copies),
    )


def _neighbors4(point: Point, width: int, height: int) -> tuple[Point, ...]:
    x, y = point
    candidates = ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1))
    return tuple(
        (nx, ny)
        for nx, ny in candidates
        if 1 <= nx < width - 1 and 1 <= ny < height - 1
    )


def _distance_map(walkable: np.ndarray, sources: Iterable[Point]) -> np.ndarray:
    height, width = walkable.shape
    distances = np.full((height, width), -1, dtype=np.int32)
    queue: deque[Point] = deque()
    for x, y in sources:
        if 0 <= x < width and 0 <= y < height and walkable[y, x] and distances[y, x] < 0:
            distances[y, x] = 0
            queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        next_distance = int(distances[y, x]) + 1
        for nx, ny in _neighbors4((x, y), width, height):
            if walkable[ny, nx] and distances[ny, nx] < 0:
                distances[ny, nx] = next_distance
                queue.append((nx, ny))
    return distances


def _components(walkable: np.ndarray) -> list[list[Point]]:
    height, width = walkable.shape
    unseen = walkable.copy()
    result: list[list[Point]] = []
    for y in range(height):
        for x in range(width):
            if not unseen[y, x]:
                continue
            unseen[y, x] = False
            queue: deque[Point] = deque([(x, y)])
            cells: list[Point] = []
            while queue:
                point = queue.popleft()
                cells.append(point)
                for nx, ny in _neighbors4(point, width, height):
                    if unseen[ny, nx]:
                        unseen[ny, nx] = False
                        queue.append((nx, ny))
            result.append(cells)
    result.sort(key=lambda cells: (-len(cells), min((y, x) for x, y in cells)))
    return result


def _full_neighbors4(index: int, width: int, size: int) -> tuple[int, ...]:
    neighbors: list[int] = []
    if index >= width:
        neighbors.append(index - width)
    if index + width < size:
        neighbors.append(index + width)
    if index % width:
        neighbors.append(index - 1)
    if index % width != width - 1:
        neighbors.append(index + 1)
    return tuple(neighbors)


def _full_distance_map(mask: np.ndarray, sources: Iterable[Point]) -> np.ndarray:
    allowed_array = np.ascontiguousarray(mask, dtype=np.uint8)
    height, width = allowed_array.shape
    size = height * width
    allowed = memoryview(allowed_array).cast("B")
    distances = [-1] * size
    queue: deque[int] = deque()
    for x, y in sources:
        if not (0 <= x < width and 0 <= y < height):
            continue
        index = y * width + x
        if allowed[index] and distances[index] < 0:
            distances[index] = 0
            queue.append(index)
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in _full_neighbors4(current, width, size):
            if allowed[neighbor] and distances[neighbor] < 0:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return np.asarray(distances, dtype=np.int32).reshape(height, width)


def _full_component_count(mask: np.ndarray) -> int:
    active = np.ascontiguousarray(mask, dtype=np.uint8)
    height, width = active.shape
    size = height * width
    allowed = memoryview(active).cast("B")
    seen = bytearray(size)
    count = 0
    for start in range(size):
        if not allowed[start] or seen[start]:
            continue
        count += 1
        queue = [start]
        seen[start] = 1
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for neighbor in _full_neighbors4(current, width, size):
                if allowed[neighbor] and not seen[neighbor]:
                    seen[neighbor] = 1
                    queue.append(neighbor)
    return count


def _square_erode(mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    height, width = active.shape
    padded = np.pad(active, 1, mode="constant", constant_values=False)
    return np.logical_and.reduce(
        [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ]
    )


def _articulation_points(
    mask: np.ndarray,
    required_component: np.ndarray,
    start: Point,
    required_targets: tuple[Point, ...],
) -> tuple[set[int], set[int]]:
    """Iterative Tarjan articulation census with mission-critical filtering."""
    active = np.ascontiguousarray(mask & required_component, dtype=np.uint8)
    height, width = active.shape
    size = height * width
    allowed = memoryview(active).cast("B")
    discovery = [-1] * size
    low = [0] * size
    parent = [-1] * size
    child_count = [0] * size
    target_count = [0] * size
    articulation: set[int] = set()
    mission_critical: set[int] = set()
    timestamp = 0
    root_index = start[1] * width + start[0]
    target_indices = {
        y * width + x
        for x, y in required_targets
        if 0 <= x < width and 0 <= y < height
    }
    if not allowed[root_index]:
        target_indices.clear()
    roots = (root_index, *range(size))
    for root in roots:
        if not allowed[root] or discovery[root] >= 0:
            continue
        discovery[root] = low[root] = timestamp
        target_count[root] = int(root in target_indices)
        timestamp += 1
        stack: list[tuple[int, tuple[int, ...], int]] = [
            (root, _full_neighbors4(root, width, size), 0)
        ]
        while stack:
            node, neighbors, cursor = stack[-1]
            if cursor < len(neighbors):
                neighbor = neighbors[cursor]
                stack[-1] = (node, neighbors, cursor + 1)
                if not allowed[neighbor]:
                    continue
                if discovery[neighbor] < 0:
                    parent[neighbor] = node
                    child_count[node] += 1
                    discovery[neighbor] = low[neighbor] = timestamp
                    target_count[neighbor] = int(neighbor in target_indices)
                    timestamp += 1
                    stack.append((neighbor, _full_neighbors4(neighbor, width, size), 0))
                elif neighbor != parent[node]:
                    low[node] = min(low[node], discovery[neighbor])
                continue
            stack.pop()
            ancestor = parent[node]
            if ancestor < 0:
                if child_count[node] > 1:
                    articulation.add(node)
            else:
                low[ancestor] = min(low[ancestor], low[node])
                target_count[ancestor] += target_count[node]
                if parent[ancestor] >= 0 and low[node] >= discovery[ancestor]:
                    articulation.add(ancestor)
                    if target_count[node] > 0:
                        mission_critical.add(ancestor)
    mission_critical.discard(root_index)
    mission_critical.difference_update(target_indices)
    return articulation, mission_critical


def _quality_metrics(
    terrain: np.ndarray,
    hazard: np.ndarray,
    *,
    start: Point,
    exit_point: Point,
    objectives: tuple[Point, ...],
) -> dict[str, object]:
    walkable = np.isin(terrain, tuple(WALKABLE_TERRAIN))
    required = (start, exit_point, *objectives)
    distances = _full_distance_map(walkable, (start,))
    required_connected = all(
        0 <= x < walkable.shape[1]
        and 0 <= y < walkable.shape[0]
        and distances[y, x] >= 0
        for x, y in required
    )
    safe = walkable & (hazard == int(Hazard.NONE))
    safe_distances = _full_distance_map(safe, (start,))
    safe_connected = all(
        0 <= x < walkable.shape[1]
        and 0 <= y < walkable.shape[0]
        and safe_distances[y, x] >= 0
        for x, y in required
    )
    radius_one = _square_erode(safe)
    radius_distances = _full_distance_map(radius_one, (start,))
    agent_connected = all(
        0 <= x < walkable.shape[1]
        and 0 <= y < walkable.shape[0]
        and radius_distances[y, x] >= 0
        for x, y in required
    )
    component = radius_distances >= 0
    if agent_connected:
        articulations, mission_articulations = _articulation_points(
            radius_one,
            component,
            start,
            (exit_point, *objectives),
        )
    else:
        articulations, mission_articulations = set(), set()
    return {
        "walkable_cells": int(walkable.sum()),
        "walkable_fraction": round(float(walkable.mean()), 9),
        "hazard_cells": int((hazard != int(Hazard.NONE)).sum()),
        "hazard_fraction": round(float((hazard != int(Hazard.NONE)).mean()), 9),
        "component_count": _full_component_count(walkable),
        "mission_connected": required_connected,
        "hazard_free_mission_connected": safe_connected,
        "agent_scale_mission_connected": agent_connected,
        "agent_scale_articulation_cell_count": len(articulations),
        "agent_scale_mission_articulation_count": len(mission_articulations),
        "start_exit_path_length": int(distances[exit_point[1], exit_point[0]]),
    }


def _minimum_edit_path(
    walkable: np.ndarray,
    start: Point,
    goal: Point,
    *,
    boundary_margin: int,
) -> tuple[Point, ...]:
    """Lexicographically stable path minimizing blocked cells, then length."""
    height, width = walkable.shape
    if boundary_margin not in (1, 2):
        raise ValueError("Route boundary margin must be one or two cells.")
    if not all(
        boundary_margin <= x < width - boundary_margin
        and boundary_margin <= y < height - boundary_margin
        for x, y in (start, goal)
    ):
        raise ValueError("Route endpoint violates its required boundary margin.")
    size = height * width
    infinity = (size + 1, size * 4 + 1)
    best = [infinity] * size
    predecessor = [-1] * size
    start_i = start[1] * width + start[0]
    goal_i = goal[1] * width + goal[0]
    best[start_i] = (0, 0)
    frontier: list[tuple[int, int, int, int]] = [(0, 0, start[1], start[0])]
    while frontier:
        edits, steps, y, x = heapq.heappop(frontier)
        index = y * width + x
        if (edits, steps) != best[index]:
            continue
        if index == goal_i:
            path: list[Point] = []
            current = index
            while True:
                path.append((current % width, current // width))
                if current == start_i:
                    break
                current = predecessor[current]
                if current < 0:
                    raise RuntimeError("Minimum-edit predecessor chain is incomplete.")
            return tuple(reversed(path))
        candidates = ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1))
        for nx, ny in candidates:
            if not (
                boundary_margin <= nx < width - boundary_margin
                and boundary_margin <= ny < height - boundary_margin
            ):
                continue
            neighbor = ny * width + nx
            candidate = (edits + int(not walkable[ny, nx]), steps + 1)
            if candidate < best[neighbor]:
                best[neighbor] = candidate
                predecessor[neighbor] = index
                heapq.heappush(frontier, (candidate[0], candidate[1], ny, nx))
    raise RuntimeError(f"No interior route exists from {start} to {goal}.")


def _assign_zones(walkable: np.ndarray, seeds: tuple[Point, ...]) -> np.ndarray:
    height, width = walkable.shape
    zone = np.full((height, width), -1, dtype=np.int16)
    distance = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)
    queue: deque[Point] = deque()
    for zone_id, (x, y) in enumerate(seeds):
        if walkable[y, x] and distance[y, x] > 0:
            zone[y, x] = zone_id
            distance[y, x] = 0
            queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        candidate_distance = int(distance[y, x]) + 1
        candidate_zone = int(zone[y, x])
        for nx, ny in _neighbors4((x, y), width, height):
            if not walkable[ny, nx]:
                continue
            if candidate_distance < distance[ny, nx] or (
                candidate_distance == distance[ny, nx]
                and candidate_zone < zone[ny, nx]
            ):
                distance[ny, nx] = candidate_distance
                zone[ny, nx] = candidate_zone
                queue.append((nx, ny))
    next_zone = len(seeds)
    for y in range(height):
        for x in range(width):
            if not walkable[y, x] or zone[y, x] >= 0:
                continue
            zone[y, x] = next_zone
            queue = deque([(x, y)])
            while queue:
                point = queue.popleft()
                for nx, ny in _neighbors4(point, width, height):
                    if walkable[ny, nx] and zone[ny, nx] < 0:
                        zone[ny, nx] = next_zone
                        queue.append((nx, ny))
            next_zone += 1
    return np.ascontiguousarray(zone)


def _nav_cost(terrain: np.ndarray, hazard: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    nav = np.zeros(terrain.shape, dtype=np.float32)
    for terrain_id, cost in NAV_BASE.items():
        nav[terrain == terrain_id] = cost
    nav += np.where(hazard == int(Hazard.NONE), 0.0, 4.0).astype(np.float32)
    nav += np.where(nav > 0.0, elevation.astype(np.float32) * 0.08, 0.0)
    return np.ascontiguousarray(nav, dtype=np.float32)


def _mix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _nearest_untouched(mask: np.ndarray, point: Point) -> int:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return -1
    return int(np.min(np.abs(cells[:, 1] - point[0]) + np.abs(cells[:, 0] - point[1])))


class _Editor:
    def __init__(self, terrain: np.ndarray, hazard: np.ndarray, elevation: np.ndarray) -> None:
        self.terrain = terrain
        self.hazard = hazard
        self.elevation = elevation
        self.protected = np.zeros(terrain.shape, dtype=np.uint8)
        self.clearance = np.zeros(terrain.shape, dtype=np.uint8)
        self.ledger: list[dict[str, object]] = []

    def set(self, field: str, x: int, y: int, value: int, phase: str, reason: str) -> bool:
        attribute = {
            "protected_backbone": "protected",
            "required_clearance": "clearance",
        }.get(field, field)
        array = getattr(self, attribute)
        before = int(array[y, x])
        after = int(value)
        if before == after:
            return False
        if len(self.ledger) >= MAX_LEDGER_ENTRIES:
            raise RuntimeError("Compiler edit ledger exceeded its strict entry bound.")
        array[y, x] = after
        self.ledger.append(
            {
                "sequence": len(self.ledger),
                "phase": phase,
                "field": field,
                "x": x,
                "y": y,
                "before": before,
                "after": after,
                "reason": reason,
            }
        )
        return True

    def walkable(self) -> np.ndarray:
        return np.isin(self.terrain, tuple(WALKABLE_TERRAIN))

    def carve_value(self, x: int, y: int) -> int:
        current = int(self.terrain[y, x])
        if current in WALKABLE_TERRAIN:
            return current
        if current in (int(Terrain.WATER), int(Terrain.CHASM)):
            return int(Terrain.BRIDGE)
        return int(Terrain.FLOOR)


def _validate_immutable_points(
    config: MapConfig,
    start: Point,
    exit_point: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
) -> None:
    if len(objectives) != config.objective_count or len(spawns) != config.spawn_count:
        raise ValueError("Immutable point counts disagree with MapConfig.")
    required = (start, exit_point, *objectives)
    if len(set(required)) != len(required) or len(set(spawns)) != len(spawns):
        raise ValueError("Immutable required/spawn point groups contain duplicates.")
    if set(required) & set(spawns):
        raise ValueError("Spawn points may not overlap required mission points.")
    for point in (*required, *spawns):
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in point)
        ):
            raise TypeError("Immutable points must be exact integer tuples.")
        x, y = point
        margin = 2 if point in required else 1
        if not (
            margin <= x < config.width - margin
            and margin <= y < config.height - margin
        ):
            label = "two-cell mission" if margin == 2 else "one-cell spawn"
            raise ValueError(f"Immutable point {point!r} lacks the required {label} boundary margin.")
    if abs(start[0] - exit_point[0]) + abs(start[1] - exit_point[1]) < config.effective_min_separation:
        raise ValueError("Immutable start/exit points cannot prove the configured minimum separation.")
    for spawn in spawns:
        if abs(spawn[0] - start[0]) + abs(spawn[1] - start[1]) < config.spawn_clearance_start:
            raise ValueError("Immutable spawn cannot satisfy start clearance.")
        if min(
            abs(spawn[0] - point[0]) + abs(spawn[1] - point[1])
            for point in (exit_point, *objectives)
        ) < config.spawn_clearance_objective:
            raise ValueError("Immutable spawn cannot satisfy objective/exit clearance.")


def _record_capture(editor: _Editor, field: str, cells: Iterable[Point], phase: str, reason: str) -> None:
    for x, y in sorted(set(cells), key=lambda point: (point[1], point[0])):
        editor.set(field, x, y, 1, phase, reason)


def compile_topology(
    raw: RawTopology,
    *,
    seed: int,
    theme: str,
    config: MapConfig,
    start: Point,
    exit: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
) -> CompileResult:
    if theme not in THEMES:
        raise ValueError(f"Unknown theme {theme!r}.")
    if raw.terrain.shape != (config.height, config.width):
        raise ValueError("Raw topology shape disagrees with MapConfig.")
    _validate_immutable_points(config, start, exit, objectives, spawns)
    seed = int(seed)
    if seed < 0 or seed >= 1 << 64:
        raise ValueError("Compiler seed must be an unsigned 64-bit integer.")
    pre_quality = _quality_metrics(
        raw.terrain,
        raw.hazard,
        start=start,
        exit_point=exit,
        objectives=objectives,
    )
    editor = _Editor(raw.terrain.copy(), raw.hazard.copy(), raw.elevation.copy())
    raw_walkable = np.isin(raw.terrain, tuple(WALKABLE_TERRAIN))
    raw_components = _components(raw_walkable)
    mission_set = {start, exit, *objectives}
    primary_component = max(
        raw_components,
        key=lambda cells: (
            sum(point in mission_set for point in cells),
            len(cells),
            tuple(-value for value in min((y, x) for x, y in cells)),
        ),
        default=[],
    )

    # Phase 1: seal the boundary. All changed categorical cells are ledgered.
    boundary = sorted(
        {
            *((x, 0) for x in range(config.width)),
            *((x, config.height - 1) for x in range(config.width)),
            *((0, y) for y in range(config.height)),
            *((config.width - 1, y) for y in range(config.height)),
        },
        key=lambda point: (point[1], point[0]),
    )
    for x, y in boundary:
        editor.set("terrain", x, y, int(Terrain.WALL), "boundary", "seal_outer_boundary")
        editor.set("hazard", x, y, int(Hazard.NONE), "boundary", "boundary_hazard_forbidden")
        editor.set("elevation", x, y, 0, "boundary", "blocked_elevation_zero")

    # Phase 2: normalize legal IDs into legal tuples without inventing derived fields.
    allowed_hazards = set(THEME_HAZARDS[theme])
    default_hazard = min(allowed_hazards)
    for y in range(config.height):
        for x in range(config.width):
            walkable = int(editor.terrain[y, x]) in WALKABLE_TERRAIN
            if not walkable:
                editor.set("hazard", x, y, int(Hazard.NONE), "normalize", "hazard_requires_walkable")
                editor.set("elevation", x, y, 0, "normalize", "blocked_elevation_zero")
            elif int(editor.hazard[y, x]) != int(Hazard.NONE) and int(editor.hazard[y, x]) not in allowed_hazards:
                editor.set("hazard", x, y, default_hazard, "normalize", "theme_local_hazard_retype")

    # Phase 3: connect mission and spawn sockets. Every actual carve write captures
    # the same cell in protected_backbone, including idempotent terrain writes.
    route_centerline: set[Point] = set()
    route_carved: set[Point] = set()
    for target_index, target in enumerate((exit, *objectives, *spawns)):
        mission_target = target_index < 1 + len(objectives)
        path = _minimum_edit_path(
            editor.walkable(),
            start,
            target,
            boundary_margin=2 if mission_target else 1,
        )
        route_centerline.update(path)
        phase = "route" if mission_target else "spawn_route"
        for x, y in path:
            if editor.set("terrain", x, y, editor.carve_value(x, y), phase, "minimum_edit_route"):
                route_carved.add((x, y))
            editor.set("hazard", x, y, int(Hazard.NONE), phase, "route_hazard_clear")
            editor.set("protected_backbone", x, y, 1, phase, "capture_actual_route_write")

    # Phase 4: square radius-one widening. The exact write footprint is captured.
    widened: set[Point] = set()
    widening_cells: set[Point] = set()
    widening_carved: set[Point] = set()
    for x, y in route_centerline:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if not (1 <= nx < config.width - 1 and 1 <= ny < config.height - 1):
                    continue
                widened.add((nx, ny))
                if (nx, ny) not in route_centerline:
                    widening_cells.add((nx, ny))
    for x, y in sorted(widened, key=lambda point: (point[1], point[0])):
        if editor.set("terrain", x, y, editor.carve_value(x, y), "widen", "square_radius_one"):
            widening_carved.add((x, y))
        editor.set("hazard", x, y, int(Hazard.NONE), "widen", "agent_corridor_hazard_clear")
        editor.set("protected_backbone", x, y, 1, "widen", "capture_actual_widen_write")

    # Phase 5: safe square disks around every immutable socket.
    safe_cells: set[Point] = set()
    for x, y in (start, exit, *objectives, *spawns):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 1 <= nx < config.width - 1 and 1 <= ny < config.height - 1:
                    safe_cells.add((nx, ny))
    for x, y in sorted(safe_cells, key=lambda point: (point[1], point[0])):
        editor.set("terrain", x, y, editor.carve_value(x, y), "safe_disk", "immutable_point_safe_disk")
        editor.set("hazard", x, y, int(Hazard.NONE), "safe_disk", "point_hazard_clear")
        editor.set("required_clearance", x, y, 1, "safe_disk", "capture_safe_disk_write")
    spawn_hazard_cells: set[Point] = set()
    radius = max(0, config.spawn_clearance_hazard - 1)
    for x, y in spawns:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < config.width and 0 <= ny < config.height:
                    spawn_hazard_cells.add((nx, ny))
    for x, y in sorted(spawn_hazard_cells, key=lambda point: (point[1], point[0])):
        editor.set("hazard", x, y, int(Hazard.NONE), "safe_disk", "spawn_hazard_clearance")
        editor.set("required_clearance", x, y, 1, "safe_disk", "capture_spawn_hazard_disk")

    # Phase 6: exact final tuple normalization and bounded theme-local hazard density.
    walkable = editor.walkable()
    for y in range(config.height):
        for x in range(config.width):
            if not walkable[y, x]:
                editor.set("hazard", x, y, int(Hazard.NONE), "finalize", "hazard_requires_walkable")
                editor.set("elevation", x, y, 0, "finalize", "blocked_elevation_zero")
            if editor.protected[y, x] or editor.clearance[y, x]:
                editor.set("hazard", x, y, int(Hazard.NONE), "finalize", "protected_or_clearance_hazard_free")
    candidates = [
        (x, y)
        for y in range(config.height)
        for x in range(config.width)
        if int(editor.hazard[y, x]) != int(Hazard.NONE)
    ]
    cap = int(np.floor(int(walkable.sum()) * THEME_HAZARD_FRACTION[theme]))
    if len(candidates) > cap:
        ranked = sorted(
            candidates,
            key=lambda point: (
                _mix64(seed ^ (point[0] * 0xD6E8FEB86659FD93) ^ (point[1] * 0xA5A3564E27F8862D)),
                point[1],
                point[0],
            ),
        )
        remove = set(ranked[cap:])
        for x, y in sorted(remove, key=lambda point: (point[1], point[0])):
            editor.set("hazard", x, y, int(Hazard.NONE), "hazard_budget", "theme_density_cap")

    terrain = np.ascontiguousarray(editor.terrain, dtype=np.uint8)
    hazard = np.ascontiguousarray(editor.hazard, dtype=np.uint8)
    elevation = np.ascontiguousarray(editor.elevation, dtype=np.int8)
    walkability = np.ascontiguousarray(
        np.isin(terrain, tuple(WALKABLE_TERRAIN)).astype(np.uint8)
    )
    protected = np.ascontiguousarray(editor.protected, dtype=np.uint8)
    clearance = np.ascontiguousarray(editor.clearance, dtype=np.uint8)
    forbidden = np.ascontiguousarray(
        ((protected != 0) | (clearance != 0) | (hazard != 0)).astype(np.uint8)
    )
    zone = _assign_zones(walkability.astype(bool), (start, exit, *objectives))
    nav = _nav_cost(terrain, hazard, elevation)
    data = MapData(
        seed=seed,
        theme=theme,
        config=config,
        terrain=terrain,
        walkability=walkability,
        hazard=hazard,
        elevation=elevation,
        zone=zone,
        nav_cost=nav,
        protected_backbone=protected,
        required_clearance=clearance,
        decoration_forbidden=forbidden,
        start=start,
        exit=exit,
        objectives=objectives,
        spawns=spawns,
        repair_count=sum(
            entry["field"] in {"terrain", "hazard", "elevation"}
            for entry in editor.ledger
        ),
        metadata={
            "compiler": COMPILER_NAME,
            "compiler_version": COMPILER_VERSION,
            "compiler_seed": seed,
            "raw_topology_sha256": raw.raw_sha256,
            "protected_backbone_segments": 1 + len(objectives) + len(spawns),
            "topology_mask_capture": "captured at compiler mutation sites; never reconstructed",
        },
    )
    validation = assert_valid(data)
    changed_fields = {
        "terrain": raw.terrain != terrain,
        "hazard": raw.hazard != hazard,
        "elevation": raw.elevation != elevation,
    }
    changed_cell = np.logical_or.reduce(tuple(changed_fields.values()))
    untouched = ~changed_cell
    field_hamming = {name: int(mask.sum()) for name, mask in changed_fields.items()}
    hazard_cleared = sum(
        entry["field"] == "hazard" and entry["after"] == 0 and entry["before"] != 0
        for entry in editor.ledger
    )
    hazard_retyped = sum(
        entry["field"] == "hazard"
        and entry["after"] != 0
        and entry["before"] != entry["after"]
        for entry in editor.ledger
    )
    ledger_payload = {
        "format": LEDGER_FORMAT,
        "entries": editor.ledger,
    }
    ledger_hash = json_sha256(ledger_payload)
    compiled_hash = named_arrays_sha256(data.arrays())
    post_quality = _quality_metrics(
        terrain,
        hazard,
        start=start,
        exit_point=exit,
        objectives=objectives,
    )
    points_report = {
        label: {
            "point": list(point),
            "nearest_untouched_neural_cell_manhattan": _nearest_untouched(untouched, point),
        }
        for label, point in (
            ("start", start),
            ("exit", exit),
            *((f"objective_{index:02d}", point) for index, point in enumerate(objectives)),
        )
    }
    cell_count = terrain.size
    field_slots = cell_count * len(changed_fields)
    report: dict[str, object] = {
        "format": "nullvector-neural-topology-compile-report-v1",
        "passed": True,
        "compiler": {"name": COMPILER_NAME, "version": COMPILER_VERSION},
        "compiler_source_sha256": compiler_source_sha256(),
        "tensor_contract_sha256": CONTRACT_SHA256,
        "seed": seed,
        "theme": theme,
        "raw_sha256": raw.raw_sha256,
        "compiled_arrays_sha256": compiled_hash,
        "ledger_sha256": ledger_hash,
        "ledger_entry_count": len(editor.ledger),
        "raw_primary_component": {
            "component_count": len(raw_components),
            "selected_cells": len(primary_component),
            "selected_mission_support": sum(point in mission_set for point in primary_component),
        },
        "quality": {
            "pre": pre_quality,
            "post": post_quality,
            "agent_scale_mission_articulation_delta": int(
                post_quality["agent_scale_mission_articulation_count"]
            )
            - int(pre_quality["agent_scale_mission_articulation_count"]),
        },
        "costs": {
            "raw_to_compiled_hamming": field_hamming,
            "raw_to_compiled_hamming_total": sum(field_hamming.values()),
            "terrain_cells_changed": field_hamming["terrain"],
            "added_walkable_cells": int((~raw_walkable & (walkability != 0)).sum()),
            "removed_walkable_cells": int((raw_walkable & (walkability == 0)).sum()),
            "route_cells_carved": len(route_carved),
            "route_centerline_cells": len(route_centerline),
            "radius_one_widening_footprint_cells": len(widening_cells),
            "radius_one_widening_cells": len(widening_carved),
            "hazard_cells_cleared": hazard_cleared,
            "hazard_cells_retyped": hazard_retyped,
            "neural_preservation_fraction": round(1.0 - sum(field_hamming.values()) / max(field_slots, 1), 9),
            "cell_preservation_fraction": round(1.0 - int(changed_cell.sum()) / max(cell_count, 1), 9),
            "repair_fraction": round(int(changed_cell.sum()) / max(cell_count, 1), 9),
        },
        "mission_point_untouched_distance": points_report,
        "validation": validation,
    }
    return CompileResult(
        data=data,
        ledger=tuple(editor.ledger),
        ledger_sha256=ledger_hash,
        raw_sha256=raw.raw_sha256,
        compiler_source_sha256=str(report["compiler_source_sha256"]),
        compiled_arrays_sha256=compiled_hash,
        report=report,
    )


def replay_identity(result: CompileResult) -> dict[str, object]:
    return {
        "compiler": {"name": COMPILER_NAME, "version": COMPILER_VERSION},
        "compiler_source_sha256": result.compiler_source_sha256,
        "raw_sha256": result.raw_sha256,
        "compiled_arrays_sha256": result.compiled_arrays_sha256,
        "ledger_sha256": result.ledger_sha256,
        "report_sha256": json_sha256(result.report),
        "points": {
            "start": list(result.data.start),
            "exit": list(result.data.exit),
            "objectives": [list(point) for point in result.data.objectives],
            "spawns": [list(point) for point in result.data.spawns],
        },
        "config": result.data.config.to_dict(),
        "seed": int(result.data.seed),
        "theme": result.data.theme,
    }


def assert_exact_compiler_replay(
    expected: CompileResult,
    raw: RawTopology,
) -> CompileResult:
    replay = compile_topology(
        raw,
        seed=expected.data.seed,
        theme=expected.data.theme,
        config=expected.data.config,
        start=expected.data.start,
        exit=expected.data.exit,
        objectives=expected.data.objectives,
        spawns=expected.data.spawns,
    )
    if replay_identity(replay) != replay_identity(expected):
        raise ValueError("Neural topology compiler replay identity drifted.")
    if replay.ledger != expected.ledger or replay.report != expected.report:
        raise ValueError("Neural topology compiler replay ledger/report drifted.")
    for name, array in expected.data.arrays().items():
        if not np.array_equal(array, replay.data.arrays()[name]):
            raise ValueError(f"Neural topology compiler replay array drifted: {name}.")
    return replay
