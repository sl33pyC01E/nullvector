from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Iterable

import numpy as np

from .model import (
    Hazard,
    MapConfig,
    MapData,
    Point,
    Terrain,
    THEMES,
    TOPOLOGY_MASK_CAPTURE_POLICY,
    WALKABLE_TERRAIN,
)


_UINT64_MASK = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """Stable unsigned seed mixing; unlike hash(), this never varies by process."""
    value = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _walkability(terrain: np.ndarray) -> np.ndarray:
    return np.isin(terrain, tuple(WALKABLE_TERRAIN))


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], passes: int = 3) -> np.ndarray:
    field = rng.random(shape, dtype=np.float32)
    for _ in range(passes):
        padded = np.pad(field, 1, mode="edge")
        field = sum(
            padded[dy : dy + shape[0], dx : dx + shape[1]]
            for dy in range(3)
            for dx in range(3)
        ) / 9.0
    lo = float(field.min())
    span = max(float(field.max()) - lo, 1e-6)
    return ((field - lo) / span).astype(np.float32)


def _disk(mask: np.ndarray, x: int, y: int, radius: int, value: bool = True) -> None:
    height, width = mask.shape
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    local = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    mask[y0:y1, x0:x1][local] = value


def _theme_arena(rng: np.random.Generator, cfg: MapConfig) -> tuple[np.ndarray, dict[str, object]]:
    height, width = cfg.height, cfg.width
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    nx = (xx - cx) / max(width * 0.47, 1.0)
    ny = (yy - cy) / max(height * 0.47, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    terrain = np.full((height, width), int(Terrain.WALL), dtype=np.uint8)
    terrain[radius < 1.0] = int(Terrain.FLOOR)

    pillar_count = int(rng.integers(5, 9))
    pillar_radius = max(2, min(width, height) // 18)
    for index in range(pillar_count):
        angle = (index / pillar_count) * math.tau + float(rng.uniform(-0.13, 0.13))
        distance = float(rng.uniform(0.34, 0.63))
        px = int(round(cx + math.cos(angle) * distance * width * 0.42))
        py = int(round(cy + math.sin(angle) * distance * height * 0.42))
        pillar = np.zeros_like(terrain, dtype=bool)
        _disk(pillar, px, py, pillar_radius)
        terrain[pillar] = int(Terrain.WALL)

    # Four cardinal gates make the silhouette and navigation legible.
    gate_half = max(2, min(width, height) // 20)
    terrain[max(1, int(cy) - gate_half) : min(height - 1, int(cy) + gate_half + 1), 1 : int(cx)] = int(Terrain.FLOOR)
    terrain[max(1, int(cy) - gate_half) : min(height - 1, int(cy) + gate_half + 1), int(cx) : width - 1] = int(Terrain.FLOOR)
    terrain[1 : int(cy), max(1, int(cx) - gate_half) : min(width - 1, int(cx) + gate_half + 1)] = int(Terrain.FLOOR)
    terrain[int(cy) : height - 1, max(1, int(cx) - gate_half) : min(width - 1, int(cx) + gate_half + 1)] = int(Terrain.FLOOR)
    return terrain, {"pillar_count": pillar_count, "layout": "elliptic_cardinal"}


def _carve_rect(terrain: np.ndarray, rect: tuple[int, int, int, int], value: Terrain = Terrain.FLOOR) -> None:
    x, y, width, height = rect
    terrain[y : y + height, x : x + width] = int(value)


def _carve_corridor(terrain: np.ndarray, start: Point, end: Point, horizontal_first: bool) -> None:
    x1, y1 = start
    x2, y2 = end

    def horizontal(y: int, xa: int, xb: int) -> None:
        lo, hi = sorted((xa, xb))
        terrain[max(1, y - 1) : min(terrain.shape[0] - 1, y + 2), max(1, lo) : min(terrain.shape[1] - 1, hi + 1)] = int(Terrain.FLOOR)

    def vertical(x: int, ya: int, yb: int) -> None:
        lo, hi = sorted((ya, yb))
        terrain[max(1, lo) : min(terrain.shape[0] - 1, hi + 1), max(1, x - 1) : min(terrain.shape[1] - 1, x + 2)] = int(Terrain.FLOOR)

    if horizontal_first:
        horizontal(y1, x1, x2)
        vertical(x2, y1, y2)
    else:
        vertical(x1, y1, y2)
        horizontal(y2, x1, x2)


def _theme_rooms(rng: np.random.Generator, cfg: MapConfig) -> tuple[np.ndarray, dict[str, object]]:
    height, width = cfg.height, cfg.width
    terrain = np.full((height, width), int(Terrain.WALL), dtype=np.uint8)
    rooms: list[tuple[int, int, int, int]] = []
    desired = max(8, (width * height) // 390)
    for _ in range(desired * 12):
        room_w = int(rng.integers(6, max(7, min(15, width // 3)) + 1))
        room_h = int(rng.integers(6, max(7, min(13, height // 3)) + 1))
        x = int(rng.integers(2, max(3, width - room_w - 1)))
        y = int(rng.integers(2, max(3, height - room_h - 1)))
        candidate = (x, y, room_w, room_h)
        overlaps = any(
            x < ox + ow + 2 and x + room_w + 2 > ox and y < oy + oh + 2 and y + room_h + 2 > oy
            for ox, oy, ow, oh in rooms
        )
        if not overlaps:
            rooms.append(candidate)
            _carve_rect(terrain, candidate)
            if len(rooms) >= desired:
                break

    if len(rooms) < 4:
        rooms = []
        for gx, gy in ((1, 1), (2, 1), (1, 2), (2, 2)):
            room_w, room_h = max(6, width // 5), max(6, height // 5)
            x = int(gx * width / 3 - room_w / 2)
            y = int(gy * height / 3 - room_h / 2)
            rect = (max(2, x), max(2, y), room_w, room_h)
            rooms.append(rect)
            _carve_rect(terrain, rect)

    centers = [(x + room_w // 2, y + room_h // 2) for x, y, room_w, room_h in rooms]
    connected = {0}
    while len(connected) < len(centers):
        best: tuple[int, int, int] | None = None
        for source in sorted(connected):
            for target in range(len(centers)):
                if target in connected:
                    continue
                distance = abs(centers[source][0] - centers[target][0]) + abs(centers[source][1] - centers[target][1])
                candidate = (distance, source, target)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, source, target = best
        _carve_corridor(terrain, centers[source], centers[target], bool(rng.integers(0, 2)))
        connected.add(target)

    # A small number of optional wall loops keep rooms from becoming one tree.
    extra_edges = min(3, max(0, len(centers) // 4))
    for _ in range(extra_edges):
        source = int(rng.integers(0, len(centers)))
        target = int(rng.integers(0, len(centers) - 1))
        if target >= source:
            target += 1
        _carve_corridor(terrain, centers[source], centers[target], bool(rng.integers(0, 2)))
    return terrain, {"room_count": len(rooms), "extra_corridors": extra_edges, "layout": "room_mst"}


def _cellular_step(open_mask: np.ndarray) -> np.ndarray:
    padded = np.pad(open_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    neighbors = sum(
        padded[dy : dy + open_mask.shape[0], dx : dx + open_mask.shape[1]]
        for dy in range(3)
        for dx in range(3)
        if not (dx == 1 and dy == 1)
    )
    return neighbors >= 4


def _theme_caves(rng: np.random.Generator, cfg: MapConfig) -> tuple[np.ndarray, dict[str, object]]:
    height, width = cfg.height, cfg.width
    open_mask = rng.random((height, width)) > float(rng.uniform(0.43, 0.48))
    open_mask[[0, -1], :] = False
    open_mask[:, [0, -1]] = False
    steps = int(rng.integers(4, 7))
    for _ in range(steps):
        open_mask = _cellular_step(open_mask)
        open_mask[[0, -1], :] = False
        open_mask[:, [0, -1]] = False
    _disk(open_mask, width // 2, height // 2, max(4, min(width, height) // 10))
    terrain = np.full((height, width), int(Terrain.WALL), dtype=np.uint8)
    terrain[open_mask] = int(Terrain.FLOOR)
    return terrain, {"cellular_steps": steps, "layout": "cellular_cavern"}


def _theme_archipelago(rng: np.random.Generator, cfg: MapConfig) -> tuple[np.ndarray, dict[str, object]]:
    height, width = cfg.height, cfg.width
    noise = _smooth_noise(rng, (height, width), passes=5)
    land = noise > float(rng.uniform(0.49, 0.56))
    island_count = int(rng.integers(5, 9))
    for _ in range(island_count):
        x = int(rng.integers(4, width - 4))
        y = int(rng.integers(4, height - 4))
        _disk(land, x, y, int(rng.integers(3, max(4, min(width, height) // 10))))
    land[[0, -1], :] = False
    land[:, [0, -1]] = False

    terrain = np.full((height, width), int(Terrain.WATER), dtype=np.uint8)
    terrain[land] = int(Terrain.FLOOR)
    padded = np.pad(land, 1, mode="constant", constant_values=False)
    cardinal = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    terrain[land & ~cardinal] = int(Terrain.SAND)
    return terrain, {"island_stamps": island_count, "layout": "noise_archipelago"}


def _theme_garden(rng: np.random.Generator, cfg: MapConfig) -> tuple[np.ndarray, dict[str, object]]:
    height, width = cfg.height, cfg.width
    terrain = np.full((height, width), int(Terrain.FLOOR), dtype=np.uint8)
    terrain[[0, -1], :] = int(Terrain.WALL)
    terrain[:, [0, -1]] = int(Terrain.WALL)
    spacing = int(rng.integers(7, 11))

    for x in range(spacing, width - 1, spacing):
        terrain[2 : height - 2, x] = int(Terrain.WALL)
        gap_count = max(2, height // 22)
        for gap in rng.choice(np.arange(3, height - 3), size=gap_count, replace=False):
            terrain[max(1, int(gap) - 1) : min(height - 1, int(gap) + 2), x] = int(Terrain.FLOOR)
    for y in range(spacing, height - 1, spacing):
        terrain[y, 2 : width - 2] = int(Terrain.WALL)
        gap_count = max(2, width // 22)
        for gap in rng.choice(np.arange(3, width - 3), size=gap_count, replace=False):
            terrain[y, max(1, int(gap) - 1) : min(width - 1, int(gap) + 2)] = int(Terrain.FLOOR)

    texture = _smooth_noise(rng, (height, width), passes=2)
    terrain[(texture > 0.76) & (terrain == int(Terrain.FLOOR))] = int(Terrain.GROWTH)
    pool_count = int(rng.integers(2, 5))
    for _ in range(pool_count):
        pool = np.zeros_like(terrain, dtype=bool)
        _disk(
            pool,
            int(rng.integers(5, width - 5)),
            int(rng.integers(5, height - 5)),
            int(rng.integers(2, 4)),
        )
        terrain[pool] = int(Terrain.WATER)
    terrain[[0, -1], :] = int(Terrain.WALL)
    terrain[:, [0, -1]] = int(Terrain.WALL)
    return terrain, {"hedge_spacing": spacing, "pool_count": pool_count, "layout": "orthogonal_garden"}


def _theme_anomaly(rng: np.random.Generator, cfg: MapConfig) -> tuple[np.ndarray, dict[str, object]]:
    height, width = cfg.height, cfg.width
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) * float(rng.uniform(0.42, 0.58))
    cy = (height - 1) * float(rng.uniform(0.42, 0.58))
    radial = np.sqrt(((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2)
    angle = np.arctan2(yy - cy, xx - cx)
    noise = _smooth_noise(rng, (height, width), passes=3)
    wave = np.sin(radial * float(rng.uniform(45.0, 62.0)) + angle * int(rng.integers(2, 6)))
    open_mask = (wave + (noise - 0.5) * 2.1) > -0.28
    open_mask &= radial < 0.68
    open_mask[[0, -1], :] = False
    open_mask[:, [0, -1]] = False
    terrain = np.full((height, width), int(Terrain.CHASM), dtype=np.uint8)
    terrain[open_mask] = int(Terrain.FLOOR)
    terrain[open_mask & (wave > 0.72)] = int(Terrain.CRYSTAL)
    return terrain, {"wave_frequency": "seeded", "layout": "polar_fracture"}


_THEME_BUILDERS = {
    "arena": _theme_arena,
    "rooms": _theme_rooms,
    "caves": _theme_caves,
    "archipelago": _theme_archipelago,
    "garden": _theme_garden,
    "anomaly": _theme_anomaly,
}


def _neighbors4(x: int, y: int, width: int, height: int) -> tuple[Point, ...]:
    """Return only in-bounds cardinal neighbors, never the source cell.

    This function intentionally materializes its tiny result without a yield or
    comprehension frame.  The hot graph traversals below use flat Python state,
    but keeping this coordinate helper equally strict prevents boundary
    self-neighbors from becoming an accidental termination dependency.
    """
    result: list[Point] = []
    if x > 0:
        result.append((x - 1, y))
    if x + 1 < width:
        result.append((x + 1, y))
    if y > 0:
        result.append((x, y - 1))
    if y + 1 < height:
        result.append((x, y + 1))
    return tuple(result)


def _flat_neighbors4(index: int, width: int, size: int) -> tuple[int, ...]:
    """Return cardinal neighbor indices using Python integers only."""
    x = index % width
    result: list[int] = []
    if x > 0:
        result.append(index - 1)
    if x + 1 < width:
        result.append(index + 1)
    if index >= width:
        result.append(index - width)
    if index + width < size:
        result.append(index + width)
    return tuple(result)


def _components(walkable: np.ndarray) -> list[list[Point]]:
    height, width = walkable.shape
    size = height * width
    # The long Windows fuzz gate previously reached a fatal access violation in
    # repeated NumPy scalar indexing.  Convert graph state to owned Python
    # values once, traverse with bytearray/list storage, and create NumPy output
    # only at API boundaries.
    walkable_cells = np.asarray(walkable, dtype=np.bool_).reshape(-1).tolist()
    seen = bytearray(size)
    result: list[list[Point]] = []
    for index, is_walkable in enumerate(walkable_cells):
        if not is_walkable or seen[index]:
            continue
        queue: deque[int] = deque([index])
        seen[index] = 1
        cells: list[Point] = []
        while queue:
            current = queue.popleft()
            px, py = current % width, current // width
            cells.append((px, py))
            for neighbor in _flat_neighbors4(current, width, size):
                if walkable_cells[neighbor] and not seen[neighbor]:
                    seen[neighbor] = 1
                    queue.append(neighbor)
        result.append(cells)
    result.sort(key=lambda cells: (-len(cells), cells[0][1], cells[0][0]))
    return result


def _distance_map(walkable: np.ndarray, sources: Iterable[Point]) -> np.ndarray:
    height, width = walkable.shape
    size = height * width
    walkable_cells = np.asarray(walkable, dtype=np.bool_).reshape(-1).tolist()
    distances = [-1] * size
    queue: deque[int] = deque()
    for x, y in sources:
        if not (0 <= x < width and 0 <= y < height):
            continue
        index = y * width + x
        if walkable_cells[index] and distances[index] < 0:
            distances[index] = 0
            queue.append(index)
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in _flat_neighbors4(current, width, size):
            if walkable_cells[neighbor] and distances[neighbor] < 0:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return np.asarray(distances, dtype=np.int32).reshape(height, width)


def _grid_distance(shape: tuple[int, int], sources: Iterable[Point]) -> np.ndarray:
    height, width = shape
    size = height * width
    distances = [1_000_000] * size
    queue: deque[int] = deque()
    for x, y in sources:
        if not (0 <= x < width and 0 <= y < height):
            continue
        index = y * width + x
        if distances[index] != 0:
            distances[index] = 0
            queue.append(index)
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in _flat_neighbors4(current, width, size):
            if distances[neighbor] > next_distance:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return np.asarray(distances, dtype=np.int32).reshape(height, width)


def _astar(terrain: np.ndarray, start: Point, goal: Point) -> list[Point]:
    height, width = terrain.shape
    size = height * width
    # Fixed-point costs avoid a rare CPython 3.12/Windows access violation seen
    # in long fuzz runs while repeatedly mutating float-heavy dict/heap state.
    # Units are tenths of one floor step and preserve the previous ordering.
    costs = (90, 10, 55, 35, 10, 14, 18, 75, 13)
    terrain_cells = np.asarray(terrain, dtype=np.uint8).reshape(-1).tolist()
    frontier: list[tuple[int, int, int, int]] = [(0, 0, start[1], start[0])]
    came_from = [-1] * size
    best = [2**31 - 1] * size
    start_index = start[1] * width + start[0]
    goal_index = goal[1] * width + goal[0]
    best[start_index] = 0
    while frontier:
        _, distance, y, x = heapq.heappop(frontier)
        current = y * width + x
        if current == goal_index:
            path: list[Point] = []
            while True:
                path.append((current % width, current // width))
                if current == start_index:
                    break
                current = came_from[current]
                if current < 0:
                    raise RuntimeError("A* predecessor chain is incomplete.")
            path.reverse()
            return path
        if distance > best[current]:
            continue
        for neighbor in _flat_neighbors4(current, width, size):
            nx, ny = neighbor % width, neighbor // width
            if nx == 0 or ny == 0 or nx == width - 1 or ny == height - 1:
                continue
            terrain_id = int(terrain_cells[neighbor])
            step = costs[terrain_id] if 0 <= terrain_id < len(costs) else 90
            next_distance = distance + step
            if next_distance >= best[neighbor]:
                continue
            best[neighbor] = next_distance
            came_from[neighbor] = current
            heuristic = (abs(goal[0] - nx) + abs(goal[1] - ny)) * 10
            heapq.heappush(frontier, (next_distance + heuristic, next_distance, ny, nx))
    raise RuntimeError(f"A* could not repair route from {start} to {goal}.")


def _carve_path(
    terrain: np.ndarray,
    hazard: np.ndarray,
    path: Iterable[Point],
    width: int = 1,
    *,
    capture_mask: np.ndarray | None = None,
) -> None:
    height, map_width = terrain.shape
    cells: set[Point] = set()
    for x, y in path:
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                if abs(dx) + abs(dy) > width:
                    continue
                nx, ny = x + dx, y + dy
                if not (1 <= nx < map_width - 1 and 1 <= ny < height - 1):
                    continue
                cells.add((nx, ny))
    if not cells:
        return
    ordered = sorted(cells, key=lambda point: (point[1], point[0]))
    xs = np.fromiter((point[0] for point in ordered), dtype=np.intp, count=len(ordered))
    ys = np.fromiter((point[1] for point in ordered), dtype=np.intp, count=len(ordered))
    original = terrain[ys, xs]
    crosses_void = np.isin(original, (int(Terrain.WATER), int(Terrain.CHASM)))
    terrain[ys, xs] = np.where(crosses_void, int(Terrain.BRIDGE), int(Terrain.FLOOR)).astype(np.uint8)
    hazard[ys, xs] = int(Hazard.NONE)
    if capture_mask is not None:
        if capture_mask.shape != terrain.shape or capture_mask.dtype != np.dtype(np.uint8):
            raise TypeError("capture_mask must be a uint8 array matching the terrain shape.")
        capture_mask[ys, xs] = np.uint8(1)


def _component_representative(cells: list[Point]) -> Point:
    center_x = sum(point[0] for point in cells) / len(cells)
    center_y = sum(point[1] for point in cells) / len(cells)
    return min(cells, key=lambda point: ((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2, point[1], point[0]))


def _place_hazards(
    rng: np.random.Generator,
    theme: str,
    terrain: np.ndarray,
) -> np.ndarray:
    height, width = terrain.shape
    hazard = np.zeros((height, width), dtype=np.uint8)
    walkable = _walkability(terrain)
    noise = _smooth_noise(rng, (height, width), passes=2)
    yy, xx = np.mgrid[0:height, 0:width]
    if theme == "arena":
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        radius = np.sqrt(((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2)
        mask = (np.abs(radius - 0.29) < 0.012) & (((xx + yy) % 3) != 0)
        hazard[mask & walkable] = int(Hazard.LASER)
    elif theme == "rooms":
        wall_neighbors = sum(
            np.roll(terrain == int(Terrain.WALL), shift, axis=axis)
            for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1))
        )
        mask = (noise > 0.89) & (wall_neighbors >= 1)
        hazard[mask & walkable] = int(Hazard.ARC)
    elif theme == "caves":
        mask = (noise > 0.78) & walkable
        hazard[mask] = int(Hazard.LAVA)
    elif theme == "archipelago":
        mask = (noise > 0.84) & walkable
        hazard[mask] = int(Hazard.ARC)
    elif theme == "garden":
        growth = terrain == int(Terrain.GROWTH)
        hazard[growth & (noise > 0.61)] = int(Hazard.SPORES)
    elif theme == "anomaly":
        mask = (noise > 0.72) & walkable
        hazard[mask] = np.where(((xx + yy) % 2)[mask], int(Hazard.ARC), int(Hazard.LASER))
    hazard[~walkable] = int(Hazard.NONE)
    return hazard


def _clear_safe_disk(
    terrain: np.ndarray,
    hazard: np.ndarray,
    point: Point,
    radius: int,
    *,
    capture_mask: np.ndarray | None = None,
) -> None:
    x, y = point
    height, width = terrain.shape
    if capture_mask is not None and (
        capture_mask.shape != terrain.shape
        or capture_mask.dtype != np.dtype(np.uint8)
    ):
        raise TypeError("capture_mask must be a uint8 array matching the terrain shape.")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                if int(terrain[ny, nx]) not in WALKABLE_TERRAIN:
                    terrain[ny, nx] = int(Terrain.FLOOR)
                hazard[ny, nx] = int(Hazard.NONE)
                if capture_mask is not None:
                    capture_mask[ny, nx] = np.uint8(1)


def _farthest_point(walkable: np.ndarray, source: Point) -> tuple[Point, int]:
    distances = _distance_map(walkable, [source])
    maximum = int(distances.max())
    cells = np.argwhere(distances == maximum)
    y, x = cells[0]
    return (int(x), int(y)), maximum


def _select_objectives(
    rng: np.random.Generator,
    walkable: np.ndarray,
    start: Point,
    exit_point: Point,
    count: int,
) -> tuple[Point, ...]:
    distance_fields = [_distance_map(walkable, [start]), _distance_map(walkable, [exit_point])]
    selected: list[Point] = []
    valid = walkable.copy()
    valid[:2, :] = False
    valid[-2:, :] = False
    valid[:, :2] = False
    valid[:, -2:] = False
    for point in (start, exit_point):
        valid[point[1], point[0]] = False
    for _ in range(count):
        score = np.minimum.reduce(distance_fields)
        score[~valid] = -1
        maximum = int(score.max())
        if maximum < 1:
            raise RuntimeError("Map does not have enough distinct reachable objective cells.")
        candidates = np.argwhere(score == maximum)
        y, x = candidates[int(rng.integers(0, len(candidates)))]
        point = (int(x), int(y))
        selected.append(point)
        valid[y, x] = False
        distance_fields.append(_distance_map(walkable, [point]))
    return tuple(selected)


def _select_spawns(
    rng: np.random.Generator,
    terrain: np.ndarray,
    hazard: np.ndarray,
    cfg: MapConfig,
    start: Point,
    exit_point: Point,
    objectives: tuple[Point, ...],
    required_clearance: np.ndarray,
) -> tuple[Point, ...]:
    if (
        required_clearance.shape != terrain.shape
        or required_clearance.dtype != np.dtype(np.uint8)
    ):
        raise TypeError(
            "required_clearance must be a uint8 array matching the terrain shape."
        )
    if cfg.spawn_count == 0:
        return ()
    walkable = _walkability(terrain)
    start_distance = _distance_map(walkable, [start])
    objective_distance = _distance_map(walkable, [exit_point, *objectives])
    eligible = (
        walkable
        & (start_distance >= cfg.spawn_clearance_start)
        & (objective_distance >= cfg.spawn_clearance_objective)
    )
    candidates = np.argwhere(eligible)
    if len(candidates) < cfg.spawn_count:
        raise RuntimeError(
            f"Only {len(candidates)} safe spawn cells exist; {cfg.spawn_count} requested."
        )

    # Greedy farthest sampling using vectorized distance updates. Besides being
    # much faster than nested Python generators, this avoids a CPython 3.12
    # access violation observed during long Windows fuzz runs at this site.
    order = rng.permutation(len(candidates))
    pool_y = candidates[order, 0].astype(np.int32, copy=True)
    pool_x = candidates[order, 1].astype(np.int32, copy=True)
    active = np.ones(len(candidates), dtype=bool)
    nearest_selected = np.full(len(candidates), np.iinfo(np.int32).max, dtype=np.int32)
    selected: list[Point] = []
    for _ in range(cfg.spawn_count):
        if not selected:
            scores = np.where(active, start_distance[pool_y, pool_x], -1)
            best_distance = int(scores.max())
            tied = np.flatnonzero(active & (scores == best_distance))
        else:
            best_spacing = int(nearest_selected[active].max())
            tied = np.flatnonzero(active & (nearest_selected == best_spacing))
            if tied.size > 1:
                distances = start_distance[pool_y[tied], pool_x[tied]]
                tied = tied[distances == distances.max()]
        # Stable y/x tie-breaking preserves the previous deterministic contract.
        index = int(tied[np.lexsort((pool_x[tied], pool_y[tied]))[0]])
        point = (int(pool_x[index]), int(pool_y[index]))
        selected.append(point)
        active[index] = False
        distance = np.abs(pool_x - point[0]) + np.abs(pool_y - point[1])
        nearest_selected = np.minimum(nearest_selected, distance)

    # Hazards are repaired around the final spawn sockets, so the stated clearance is exact.
    for point in selected:
        x, y = point
        radius = max(0, cfg.spawn_clearance_hazard - 1)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < cfg.width and 0 <= ny < cfg.height:
                        hazard[ny, nx] = int(Hazard.NONE)
                        required_clearance[ny, nx] = np.uint8(1)
    return tuple(selected)


def _assign_zones(walkable: np.ndarray, seeds: tuple[Point, ...]) -> np.ndarray:
    height, width = walkable.shape
    size = height * width
    walkable_cells = np.asarray(walkable, dtype=np.bool_).reshape(-1).tolist()
    zone = [-1] * size
    distance = [1_000_000] * size
    queue: deque[int] = deque()
    for zone_id, (x, y) in enumerate(seeds):
        index = y * width + x
        if walkable_cells[index] and distance[index] > 0:
            zone[index] = zone_id
            distance[index] = 0
            queue.append(index)
    while queue:
        current = queue.popleft()
        next_distance = distance[current] + 1
        next_zone = zone[current]
        for neighbor in _flat_neighbors4(current, width, size):
            if walkable_cells[neighbor] and (
                next_distance < distance[neighbor]
                or (next_distance == distance[neighbor] and next_zone < zone[neighbor])
            ):
                distance[neighbor] = next_distance
                zone[neighbor] = next_zone
                queue.append(neighbor)

    # Optional disconnected scenery receives deterministic component zones too.
    next_zone = len(seeds)
    for index, is_walkable in enumerate(walkable_cells):
        if not is_walkable or zone[index] >= 0:
            continue
        queue = deque([index])
        zone[index] = next_zone
        while queue:
            current = queue.popleft()
            for neighbor in _flat_neighbors4(current, width, size):
                if walkable_cells[neighbor] and zone[neighbor] < 0:
                    zone[neighbor] = next_zone
                    queue.append(neighbor)
        next_zone += 1
    return np.asarray(zone, dtype=np.int16).reshape(height, width)


def _make_elevation(rng: np.random.Generator, terrain: np.ndarray) -> np.ndarray:
    noise = _smooth_noise(rng, terrain.shape, passes=4)
    elevation = np.clip(np.floor(noise * 6.0), 0, 5).astype(np.int8)
    elevation[~_walkability(terrain)] = 0
    return elevation


def _make_nav_cost(terrain: np.ndarray, hazard: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    base = np.zeros(terrain.shape, dtype=np.float32)
    for terrain_id, cost in {
        Terrain.FLOOR: 1.0,
        Terrain.BRIDGE: 1.15,
        Terrain.GROWTH: 1.55,
        Terrain.CRYSTAL: 1.7,
        Terrain.SAND: 1.35,
    }.items():
        base[terrain == int(terrain_id)] = cost
    base += np.where(hazard == int(Hazard.NONE), 0.0, 4.0).astype(np.float32)
    base += np.where(base > 0.0, elevation.astype(np.float32) * 0.08, 0.0)
    return np.ascontiguousarray(base, dtype=np.float32)


def generate_map(seed: int, theme: str, config: MapConfig | None = None) -> MapData:
    """Generate one deterministic semantic map and repair all required routes."""
    if theme not in THEMES:
        raise ValueError(f"Unknown theme {theme!r}; expected one of {THEMES}.")
    cfg = config or MapConfig()
    seed = int(seed) & _UINT64_MASK
    # Independent streams keep topology stable if decoration or spawn logic evolves.
    layout_rng = np.random.Generator(np.random.PCG64(splitmix64(seed ^ 0x4C41594F5554)))
    hazard_rng = np.random.Generator(np.random.PCG64(splitmix64(seed ^ 0x48415A415244)))
    objective_rng = np.random.Generator(np.random.PCG64(splitmix64(seed ^ 0x4F424A454354)))
    spawn_rng = np.random.Generator(np.random.PCG64(splitmix64(seed ^ 0x535041574E)))
    elevation_rng = np.random.Generator(np.random.PCG64(splitmix64(seed ^ 0x454C45564154)))
    terrain, theme_metadata = _THEME_BUILDERS[theme](layout_rng, cfg)
    hazard = _place_hazards(hazard_rng, theme, terrain)
    protected_backbone = np.zeros(terrain.shape, dtype=np.uint8)
    required_clearance = np.zeros(terrain.shape, dtype=np.uint8)

    walkable = _walkability(terrain)
    components = _components(walkable)
    if not components:
        raise RuntimeError(f"Theme {theme} produced no walkable cells for seed {seed}.")
    desired_anchors = min(len(components), cfg.objective_count + 2)
    anchors = [_component_representative(component) for component in components[:desired_anchors]]
    root = anchors[0]
    repairs = 0
    for anchor in anchors[1:]:
        if _distance_map(_walkability(terrain), [root])[anchor[1], anchor[0]] < 0:
            _carve_path(terrain, hazard, _astar(terrain, root, anchor), width=1)
            repairs += 1

    walkable = _walkability(terrain)
    first, _ = _farthest_point(walkable, root)
    start, _ = _farthest_point(walkable, first)
    exit_point, separation = _farthest_point(walkable, start)

    geographic_separation = abs(start[0] - exit_point[0]) + abs(start[1] - exit_point[1])
    endpoint_clear = all(
        2 <= x < cfg.width - 2 and 2 <= y < cfg.height - 2
        for x, y in (start, exit_point)
    )
    if (
        separation < cfg.effective_min_separation
        or geographic_separation < cfg.effective_min_separation
        or not endpoint_clear
    ):
        start = (2, cfg.height // 2)
        exit_point = (cfg.width - 3, cfg.height // 2)
        _carve_path(terrain, hazard, _astar(terrain, start, exit_point), width=1)
        _carve_path(terrain, hazard, _astar(terrain, root, start), width=1)
        repairs += 2
        walkable = _walkability(terrain)
        separation = int(_distance_map(walkable, [start])[exit_point[1], exit_point[0]])
    if separation < cfg.effective_min_separation:
        raise RuntimeError(
            f"Could not achieve start/exit separation {cfg.effective_min_separation}; got {separation}."
        )

    objectives = _select_objectives(objective_rng, walkable, start, exit_point, cfg.objective_count)

    # This is the authoritative mission backbone. A two-cell carve radius leaves a
    # connected centerline even after one-cell agent-radius erosion. Decoration was
    # already placed and can only be cleared here, never overwrite these routes.
    for target in (exit_point, *objectives):
        _carve_path(
            terrain,
            hazard,
            _astar(terrain, start, target),
            width=2,
            capture_mask=protected_backbone,
        )
    _clear_safe_disk(
        terrain,
        hazard,
        start,
        radius=3,
        capture_mask=required_clearance,
    )
    _clear_safe_disk(
        terrain,
        hazard,
        exit_point,
        radius=2,
        capture_mask=required_clearance,
    )
    for objective in objectives:
        _clear_safe_disk(
            terrain,
            hazard,
            objective,
            radius=2,
            capture_mask=required_clearance,
        )
    terrain[[0, -1], :] = int(Terrain.WALL)
    terrain[:, [0, -1]] = int(Terrain.WALL)
    hazard[[0, -1], :] = int(Hazard.NONE)
    hazard[:, [0, -1]] = int(Hazard.NONE)
    walkable = _walkability(terrain)

    spawns = _select_spawns(
        spawn_rng,
        terrain,
        hazard,
        cfg,
        start,
        exit_point,
        objectives,
        required_clearance,
    )
    decoration_forbidden = np.ascontiguousarray(
        (
            (protected_backbone != 0)
            | (required_clearance != 0)
            | (hazard != int(Hazard.NONE))
        ).astype(np.uint8)
    )
    elevation = _make_elevation(elevation_rng, terrain)
    zone = _assign_zones(walkable, (start, exit_point, *objectives))
    nav_cost = _make_nav_cost(terrain, hazard, elevation)
    final_separation = int(_distance_map(walkable, [start])[exit_point[1], exit_point[0]])
    metadata = {
        "theme_parameters": theme_metadata,
        "required_route_repairs": repairs,
        "protected_backbone_segments": 1 + len(objectives),
        "topology_mask_capture": TOPOLOGY_MASK_CAPTURE_POLICY,
        "start_exit_path_length": final_separation,
        "rng": "numpy.random.PCG64",
    }
    return MapData(
        seed=seed,
        theme=theme,
        config=cfg,
        terrain=np.ascontiguousarray(terrain, dtype=np.uint8),
        walkability=np.ascontiguousarray(walkable, dtype=np.uint8),
        hazard=np.ascontiguousarray(hazard, dtype=np.uint8),
        elevation=np.ascontiguousarray(elevation, dtype=np.int8),
        zone=np.ascontiguousarray(zone, dtype=np.int16),
        nav_cost=nav_cost,
        protected_backbone=np.ascontiguousarray(protected_backbone, dtype=np.uint8),
        required_clearance=np.ascontiguousarray(required_clearance, dtype=np.uint8),
        decoration_forbidden=decoration_forbidden,
        start=start,
        exit=exit_point,
        objectives=objectives,
        spawns=spawns,
        repair_count=repairs,
        metadata=metadata,
    )
