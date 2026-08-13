from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import mean
from typing import Any, Iterable, Sequence

import jsonschema
import numpy as np

from ..maps.io import array_digest, file_sha256, load_map_pack
from ..maps.model import (
    GENERATOR_VERSION,
    MAP_SCHEMA_VERSION,
    MapData,
    Point,
    THEMES,
)
from ..maps.validate import assert_valid, validate_pack
from ..safety import write_json_atomic


QUALITY_FORMAT = "nullvector-map-quality-audit-v1"
QUALITY_BANK_FORMAT = "nullvector-map-quality-bank-v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "schema"
    / "map_quality_report.schema.json"
)


_AUDIT_DEPENDENCIES = (
    "forge/map_quality/audit.py",
    "forge/maps/generator.py",
    "forge/maps/io.py",
    "forge/maps/model.py",
    "forge/maps/render.py",
    "forge/maps/validate.py",
    "shared/schema/map_manifest.schema.json",
    "shared/schema/map_quality_report.schema.json",
)


def audit_source_hash() -> str:
    project = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    digest.update(b"nullvector-map-quality-source-v1\0")
    for relative in _AUDIT_DEPENDENCIES:
        path = project / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _contract_payload() -> dict[str, Any]:
    return {
        "audit_source_sha256": audit_source_hash(),
        "map_generator_version": GENERATOR_VERSION,
        "map_schema_version": MAP_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _neighbors4(index: int, width: int, size: int) -> tuple[int, ...]:
    result: list[int] = []
    if index >= width:
        result.append(index - width)
    if index + width < size:
        result.append(index + width)
    if index % width:
        result.append(index - 1)
    if index % width != width - 1:
        result.append(index + 1)
    return tuple(result)


def _neighbors8(index: int, width: int, height: int) -> tuple[int, ...]:
    x = index % width
    y = index // width
    result: list[int] = []
    for dy in (-1, 0, 1):
        ny = y + dy
        if not 0 <= ny < height:
            continue
        for dx in (-1, 0, 1):
            nx = x + dx
            if (dx or dy) and 0 <= nx < width:
                result.append(ny * width + nx)
    return tuple(result)


def _distances(mask: np.ndarray, sources: Iterable[Point]) -> np.ndarray:
    walkable = np.asarray(mask, dtype=bool)
    height, width = walkable.shape
    size = height * width
    allowed = memoryview(np.ascontiguousarray(walkable, dtype=np.uint8)).cast("B")
    distance = [-1] * size
    queue: deque[int] = deque()
    for x, y in sources:
        if not (0 <= x < width and 0 <= y < height):
            continue
        index = y * width + x
        if allowed[index] and distance[index] < 0:
            distance[index] = 0
            queue.append(index)
    while queue:
        current = queue.popleft()
        next_distance = distance[current] + 1
        for neighbor in _neighbors4(current, width, size):
            if allowed[neighbor] and distance[neighbor] < 0:
                distance[neighbor] = next_distance
                queue.append(neighbor)
    return np.asarray(distance, dtype=np.int32).reshape(height, width)


def _shortest_path(
    mask: np.ndarray, source: Point, target: Point
) -> tuple[Point, ...]:
    walkable = np.asarray(mask, dtype=bool)
    height, width = walkable.shape
    size = height * width
    if not (
        0 <= source[0] < width
        and 0 <= source[1] < height
        and 0 <= target[0] < width
        and 0 <= target[1] < height
    ):
        return ()
    start = source[1] * width + source[0]
    goal = target[1] * width + target[0]
    allowed = memoryview(np.ascontiguousarray(walkable, dtype=np.uint8)).cast("B")
    if not allowed[start] or not allowed[goal]:
        return ()
    parent = [-1] * size
    queue: deque[int] = deque([start])
    parent[start] = start
    while queue and parent[goal] < 0:
        current = queue.popleft()
        for neighbor in _neighbors4(current, width, size):
            if allowed[neighbor] and parent[neighbor] < 0:
                parent[neighbor] = current
                queue.append(neighbor)
    if parent[goal] < 0:
        return ()
    indices = [goal]
    while indices[-1] != start:
        indices.append(parent[indices[-1]])
    indices.reverse()
    return tuple((index % width, index // width) for index in indices)


def _shortest_path_optima(
    mask: np.ndarray,
    source: Point,
    target: Point,
    clearance: np.ndarray,
    cell_cost: np.ndarray,
) -> dict[str, int]:
    """Intrinsic optima over every shortest path, independent of BFS ties."""
    allowed = np.asarray(mask, dtype=bool)
    height, width = allowed.shape
    start_distance = _distances(allowed, (source,))
    target_distance = _distances(allowed, (target,))
    length = int(start_distance[target[1], target[0]])
    if length < 0:
        return {"length": -1, "widest_clearance_radius": -1, "minimum_cell_cost": -1}
    best_clearance = np.full((height, width), -1, dtype=np.int32)
    minimum_cost = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)
    best_clearance[source[1], source[0]] = int(clearance[source[1], source[0]])
    minimum_cost[source[1], source[0]] = int(cell_cost[source[1], source[0]])
    for depth in range(1, length + 1):
        on_level = np.argwhere(
            (start_distance == depth)
            & (target_distance == length - depth)
            & allowed
        )
        for y_value, x_value in on_level:
            y = int(y_value)
            x = int(x_value)
            index = y * width + x
            for predecessor in _neighbors4(index, width, height * width):
                py, px = divmod(predecessor, width)
                if start_distance[py, px] != depth - 1:
                    continue
                if target_distance[py, px] != length - depth + 1:
                    continue
                if best_clearance[py, px] >= 0:
                    candidate = min(
                        int(best_clearance[py, px]), int(clearance[y, x])
                    )
                    best_clearance[y, x] = max(best_clearance[y, x], candidate)
                if minimum_cost[py, px] != np.iinfo(np.int32).max:
                    minimum_cost[y, x] = min(
                        minimum_cost[y, x],
                        int(minimum_cost[py, px]) + int(cell_cost[y, x]),
                    )
    return {
        "length": length,
        "widest_clearance_radius": int(best_clearance[target[1], target[0]]),
        "minimum_cell_cost": int(minimum_cost[target[1], target[0]]),
    }


def _component_count(mask: np.ndarray) -> int:
    active = np.asarray(mask, dtype=bool)
    height, width = active.shape
    size = height * width
    allowed = memoryview(np.ascontiguousarray(active, dtype=np.uint8)).cast("B")
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
            for neighbor in _neighbors4(current, width, size):
                if allowed[neighbor] and not seen[neighbor]:
                    seen[neighbor] = 1
                    queue.append(neighbor)
    return count


def _articulation_points(
    mask: np.ndarray,
    required_component: np.ndarray,
    start: Point,
    required_targets: Sequence[Point],
) -> tuple[set[int], set[int]]:
    """Return all and mission-critical articulation cells.

    The second set contains only vertices whose removal separates the DFS root
    (the mission start) from at least one exit/objective. Dead-end decorative
    branches therefore do not masquerade as mission chokepoints.
    """
    active = np.asarray(mask & required_component, dtype=bool)
    height, width = active.shape
    size = height * width
    allowed = memoryview(np.ascontiguousarray(active, dtype=np.uint8)).cast("B")
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
        point[1] * width + point[0]
        for point in required_targets
        if 0 <= point[0] < width and 0 <= point[1] < height
    }
    if not allowed[root_index]:
        # There is no meaningful start-to-target articulation question when
        # the start itself is absent from this derived navigation graph.
        target_indices.clear()
    roots = (root_index, *range(size))
    for root in roots:
        if not allowed[root] or discovery[root] >= 0:
            continue
        discovery[root] = low[root] = timestamp
        target_count[root] = int(root in target_indices)
        timestamp += 1
        stack: list[tuple[int, tuple[int, ...], int]] = [
            (root, _neighbors4(root, width, size), 0)
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
                    stack.append((neighbor, _neighbors4(neighbor, width, size), 0))
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


def _clearance_radius(walkable: np.ndarray) -> np.ndarray:
    """Chebyshev distance to the closest blocked tile.

    This exactly matches the square erosion used for the radius-one agent
    contract: a walkable cell survives that erosion iff this radius is >= 2.
    """
    walkable = np.asarray(walkable, dtype=bool)
    height, width = walkable.shape
    size = height * width
    distance = [-1] * size
    queue: deque[int] = deque()
    for y, x in np.argwhere(~walkable):
        index = int(y) * width + int(x)
        distance[index] = 0
        queue.append(index)
    if not queue:
        # The map validator seals the boundary, but keep this helper total.
        return np.full((height, width), min(height, width), dtype=np.int32)
    while queue:
        current = queue.popleft()
        next_distance = distance[current] + 1
        for neighbor in _neighbors8(current, width, height):
            if distance[neighbor] < 0:
                distance[neighbor] = next_distance
                queue.append(neighbor)
    return np.asarray(distance, dtype=np.int32).reshape(height, width)


def _square_erode(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return np.logical_and.reduce(
        [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ]
    )


def _path_metrics(
    data: MapData,
    mask: np.ndarray,
    safe_mask: np.ndarray,
    radius_one_safe_mask: np.ndarray,
    clearance: np.ndarray,
    safe_clearance: np.ndarray,
    hazard_distance: np.ndarray,
    target: Point,
) -> dict[str, Any]:
    path = _shortest_path(mask, data.start, target)
    if not path:
        raise ValueError(f"validated required target {target} has no audit path")
    safe_path = _shortest_path(safe_mask, data.start, target)
    radius_one_safe_path = _shortest_path(radius_one_safe_mask, data.start, target)
    points = np.asarray([(y, x) for x, y in path], dtype=np.int64)
    path_clearance = clearance[points[:, 0], points[:, 1]]
    path_hazard = hazard_distance[points[:, 0], points[:, 1]]
    path_elevation = data.elevation[points[:, 0], points[:, 1]].astype(np.int64)
    elevation_changes = int(np.abs(np.diff(path_elevation)).sum()) if len(path) > 1 else 0
    direct = abs(target[0] - data.start[0]) + abs(target[1] - data.start[1])
    hazard_cost = np.asarray(data.hazard != 0, dtype=np.int32)
    zero_cost = np.zeros_like(hazard_cost)
    geometric_optima = _shortest_path_optima(
        mask, data.start, target, clearance, hazard_cost
    )
    safe_optima = _shortest_path_optima(
        safe_mask, data.start, target, safe_clearance, zero_cost
    )
    radius_one_optima = _shortest_path_optima(
        radius_one_safe_mask, data.start, target, safe_clearance, zero_cost
    )
    return {
        "target": list(target),
        "geometric_length": len(path) - 1,
        "manhattan_direct": int(direct),
        "geometric_detour_ratio": round((len(path) - 1) / max(1, direct), 6),
        "safe_length": len(safe_path) - 1 if safe_path else -1,
        "safe_detour_ratio_vs_geometric": (
            round((len(safe_path) - 1) / max(1, len(path) - 1), 6)
            if safe_path
            else -1.0
        ),
        "radius_one_safe_length": (
            len(radius_one_safe_path) - 1 if radius_one_safe_path else -1
        ),
        "radius_one_safe_detour_ratio_vs_geometric": (
            round((len(radius_one_safe_path) - 1) / max(1, len(path) - 1), 6)
            if radius_one_safe_path
            else -1.0
        ),
        "canonical_geometric_minimum_clearance_radius": int(path_clearance.min()),
        "canonical_geometric_median_clearance_radius": round(
            float(np.median(path_clearance)), 6
        ),
        "canonical_geometric_narrow_cell_fraction": round(
            float((path_clearance <= 2).mean()), 6
        ),
        "canonical_geometric_minimum_hazard_distance": (
            int(path_hazard.min()) if (data.hazard != 0).any() else -1
        ),
        "canonical_geometric_hazard_nearby_fraction": (
            round(float((path_hazard <= 2).mean()), 6)
            if (data.hazard != 0).any()
            else 0.0
        ),
        "canonical_geometric_hazard_fraction": round(
            float(
                np.count_nonzero(data.hazard[points[:, 0], points[:, 1]])
            )
            / len(path),
            6,
        ),
        "canonical_geometric_elevation_change_sum": elevation_changes,
        "canonical_geometric_elevation_levels_used": int(
            len(np.unique(path_elevation))
        ),
        "geometric_widest_shortest_clearance_radius": geometric_optima[
            "widest_clearance_radius"
        ],
        "geometric_minimum_hazard_cells_on_shortest_path": geometric_optima[
            "minimum_cell_cost"
        ],
        "geometric_minimum_hazard_fraction_on_shortest_path": round(
            geometric_optima["minimum_cell_cost"] / max(1, len(path)), 6
        ),
        "safe_widest_shortest_clearance_radius": safe_optima[
            "widest_clearance_radius"
        ],
        "radius_one_safe_widest_shortest_clearance_radius": radius_one_optima[
            "widest_clearance_radius"
        ],
    }


def audit_map(
    data: MapData,
    *,
    source_manifest_sha256: str | None = None,
    source_semantic_sha256: str | None = None,
) -> dict[str, Any]:
    assert_valid(data)
    actual_semantic_sha256 = array_digest(data.arrays())
    claimed_semantic_sha256 = data.metadata.get("semantic_array_sha256")
    if (
        claimed_semantic_sha256 is not None
        and claimed_semantic_sha256 != actual_semantic_sha256
    ):
        raise ValueError("map metadata semantic-array hash disagrees with actual arrays")
    if (
        source_semantic_sha256 is not None
        and source_semantic_sha256 != actual_semantic_sha256
    ):
        raise ValueError("source semantic-array hash disagrees with actual arrays")
    walkable = data.walkability.astype(bool)
    safe = walkable & (data.hazard == 0)
    radius_one_walkable = _square_erode(walkable)
    radius_one_safe = _square_erode(safe)
    start_distance = _distances(walkable, [data.start])
    required_component = start_distance >= 0
    required = (data.exit, *data.objectives)
    articulations, mission_articulations = _articulation_points(
        walkable, required_component, data.start, required
    )
    required_indices = {
        point[1] * data.config.width + point[0] for point in (data.start, *required)
    }
    relevant_articulations = mission_articulations - required_indices
    radius_one_start_distance = _distances(radius_one_safe, [data.start])
    radius_one_component = radius_one_start_distance >= 0
    (
        radius_one_articulations,
        radius_one_mission_articulations,
    ) = _articulation_points(
        radius_one_safe, radius_one_component, data.start, required
    )
    radius_one_relevant = radius_one_mission_articulations - required_indices
    protected = data.protected_backbone.astype(bool)
    clearance = _clearance_radius(walkable)
    safe_clearance = _clearance_radius(safe)
    hazard_distance = _distances(
        np.ones_like(walkable, dtype=bool),
        [(int(x), int(y)) for y, x in np.argwhere(data.hazard != 0)],
    )
    path_metrics = [
        _path_metrics(
            data,
            walkable,
            safe,
            radius_one_safe,
            clearance,
            safe_clearance,
            hazard_distance,
            target,
        )
        for target in required
    ]
    protected_clearance = clearance[protected]
    spawn_distances = []
    for index, first in enumerate(data.spawns):
        if index + 1 < len(data.spawns):
            spawn_distances.extend(
                abs(first[0] - other[0]) + abs(first[1] - other[1])
                for other in data.spawns[index + 1 :]
            )
    zones = np.unique(data.zone[walkable])
    elevations = np.unique(data.elevation[walkable])
    metrics: dict[str, Any] = {
        "walkable_component_count": _component_count(walkable),
        "required_component_fraction": round(
            float(required_component.sum()) / max(1, int(walkable.sum())), 6
        ),
        "articulation_cell_count": len(articulations),
        "mission_relevant_articulation_count": len(relevant_articulations),
        "mission_relevant_articulation_fraction": round(
            len(relevant_articulations) / max(1, int(required_component.sum())), 6
        ),
        "agent_scale_articulation_cell_count": len(radius_one_articulations),
        "agent_scale_mission_articulation_count": len(radius_one_relevant),
        "agent_scale_mission_articulation_fraction": round(
            len(radius_one_relevant) / max(1, int(radius_one_component.sum())), 6
        ),
        "radius_one_walkable_fraction": round(
            float(radius_one_walkable.sum()) / max(1, int(walkable.sum())), 6
        ),
        "radius_one_safe_walkable_fraction": round(
            float(radius_one_safe.sum()) / max(1, int(walkable.sum())), 6
        ),
        "protected_minimum_clearance_radius": int(protected_clearance.min()),
        "protected_median_clearance_radius": round(
            float(np.median(protected_clearance)), 6
        ),
        "protected_narrow_cell_fraction": round(
            float((protected_clearance <= 2).mean()), 6
        ),
        "safe_walkable_fraction": round(
            float(safe.sum()) / max(1, int(walkable.sum())), 6
        ),
        "hazard_density_on_walkable": round(
            float(np.count_nonzero((data.hazard != 0) & walkable))
            / max(1, int(walkable.sum())),
            6,
        ),
        "zone_count": int(len(zones)),
        "elevation_levels_used": int(len(elevations)),
        "elevation_entropy_bits": round(
            float(
                -sum(
                    probability * math.log2(probability)
                    for count in np.unique(data.elevation[walkable], return_counts=True)[1]
                    for probability in [float(count) / max(1, int(walkable.sum()))]
                    if probability > 0
                )
            ),
            6,
        ),
        "spawn_pair_minimum_manhattan": min(spawn_distances, default=-1),
        "spawn_pair_median_manhattan": (
            round(float(np.median(spawn_distances)), 6) if spawn_distances else -1.0
        ),
        "required_paths": path_metrics,
        "maximum_required_detour_ratio": max(
            record["geometric_detour_ratio"] for record in path_metrics
        ),
        "minimum_canonical_geometric_path_clearance": min(
            record["canonical_geometric_minimum_clearance_radius"]
            for record in path_metrics
        ),
        "maximum_canonical_geometric_path_narrow_fraction": max(
            record["canonical_geometric_narrow_cell_fraction"]
            for record in path_metrics
        ),
        "maximum_safe_detour_ratio_vs_geometric": max(
            record["safe_detour_ratio_vs_geometric"] for record in path_metrics
        ),
        "maximum_radius_one_safe_detour_ratio_vs_geometric": max(
            record["radius_one_safe_detour_ratio_vs_geometric"]
            for record in path_metrics
        ),
        "maximum_canonical_geometric_path_hazard_fraction": max(
            record["canonical_geometric_hazard_fraction"]
            for record in path_metrics
        ),
        "minimum_canonical_geometric_path_hazard_distance": min(
            record["canonical_geometric_minimum_hazard_distance"]
            for record in path_metrics
            if record["canonical_geometric_minimum_hazard_distance"] >= 0
        ) if (data.hazard != 0).any() else -1,
        "minimum_widest_shortest_clearance_radius": min(
            record["geometric_widest_shortest_clearance_radius"]
            for record in path_metrics
        ),
        "maximum_minimum_hazard_fraction_on_shortest_path": max(
            record["geometric_minimum_hazard_fraction_on_shortest_path"]
            for record in path_metrics
        ),
    }
    radius_one_safe_route_exists = all(
        record["radius_one_safe_length"] >= 0 for record in path_metrics
    )
    diagnostics = {
        "mission_has_alternate_agent_scale_routes": (
            radius_one_safe_route_exists and len(radius_one_relevant) == 0
        ),
        "radius_one_safe_route_exists": radius_one_safe_route_exists,
        "required_paths_not_excessively_detoured": metrics[
            "maximum_required_detour_ratio"
        ] <= 4.0,
        "spawn_distribution_noncollapsed": (
            metrics["spawn_pair_minimum_manhattan"] >= 2
            if len(data.spawns) > 1
            else True
        ),
        "elevation_is_nontrivial": metrics["elevation_levels_used"] >= 2,
    }
    report: dict[str, Any] = {
        "format": QUALITY_FORMAT,
        "audit_source_sha256": audit_source_hash(),
        "contract": _contract_payload(),
        "map_id": data.map_id,
        "seed": int(data.seed),
        "theme": data.theme,
        "dimensions": [data.config.width, data.config.height],
        "source_semantic_sha256": actual_semantic_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "metrics": metrics,
        "diagnostics": diagnostics,
        # Quality diagnostics are intentionally descriptive. Core map validity
        # is the hard gate; a chokepoint can be a legitimate design choice.
        "hard_validity_preserved": True,
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def audit_pack(path: Path) -> dict[str, Any]:
    path = Path(path)
    manifest = path if path.name == "manifest.json" else path / "manifest.json"
    manifest_sha256_before = file_sha256(manifest)
    validation = validate_pack(manifest.parent)
    if not validation["passed"]:
        failures = [
            *validation.get("schema_errors", ()),
            *validation.get("artifact_errors", ()),
            *(validation.get("replay_report") or {}).get("failures", ()),
        ]
        raise ValueError(
            f"map pack failed authoritative validation: {manifest.parent}: "
            + "; ".join(str(item) for item in failures[:8])
        )
    data = load_map_pack(manifest.parent, verify_hashes=True)
    manifest_bytes = manifest.read_bytes()
    manifest_sha256_after = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256_before != manifest_sha256_after:
        raise ValueError("map manifest changed during quality audit")
    payload = json.loads(manifest_bytes)
    return audit_map(
        data,
        source_manifest_sha256=manifest_sha256_after,
        source_semantic_sha256=payload["semantic_array_sha256"],
    )


def _aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    numeric_names = (
        "mission_relevant_articulation_fraction",
        "agent_scale_mission_articulation_fraction",
        "radius_one_walkable_fraction",
        "radius_one_safe_walkable_fraction",
        "protected_minimum_clearance_radius",
        "protected_narrow_cell_fraction",
        "safe_walkable_fraction",
        "hazard_density_on_walkable",
        "elevation_levels_used",
        "elevation_entropy_bits",
        "maximum_required_detour_ratio",
        "minimum_canonical_geometric_path_clearance",
        "maximum_canonical_geometric_path_narrow_fraction",
        "maximum_safe_detour_ratio_vs_geometric",
        "maximum_radius_one_safe_detour_ratio_vs_geometric",
        "maximum_canonical_geometric_path_hazard_fraction",
        "minimum_widest_shortest_clearance_radius",
        "maximum_minimum_hazard_fraction_on_shortest_path",
    )
    result: dict[str, Any] = {}
    for name in numeric_names:
        values = [float(record["metrics"][name]) for record in records]
        result[name] = {
            "minimum": round(min(values), 6),
            "mean": round(mean(values), 6),
            "median": round(float(np.median(values)), 6),
            "maximum": round(max(values), 6),
        }
    return result


def audit_packs(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one map pack is required")
    records = sorted(
        (audit_pack(path) for path in paths),
        key=lambda record: (
            record["theme"],
            record["map_id"],
            record["source_semantic_sha256"] or "",
            record["source_manifest_sha256"] or "",
            record["report_sha256"],
        ),
    )
    identities = [record["source_semantic_sha256"] for record in records]
    by_theme = {
        theme: _aggregate([record for record in records if record["theme"] == theme])
        for theme in THEMES
        if any(record["theme"] == theme for record in records)
    }
    report: dict[str, Any] = {
        "format": QUALITY_BANK_FORMAT,
        "audit_source_sha256": audit_source_hash(),
        "contract": _contract_payload(),
        "status": "ready",
        "map_count": len(records),
        "unique_semantic_count": len(set(identities)),
        "themes": sorted({record["theme"] for record in records}),
        "all_hard_valid": all(record["hard_validity_preserved"] for record in records),
        "aggregate": _aggregate(records),
        "by_theme": by_theme,
        "maps": records,
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def _assert_record_hash(record: dict[str, Any]) -> None:
    expected = record.get("report_sha256")
    unhashed = {key: value for key, value in record.items() if key != "report_sha256"}
    if not isinstance(expected, str) or expected != _canonical_hash(unhashed):
        raise ValueError(f"map quality record hash mismatch: {record.get('map_id')}")


def _assert_record_relationships(record: dict[str, Any]) -> None:
    if record.get("contract") != _contract_payload():
        raise ValueError(f"map quality contract mismatch: {record.get('map_id')}")
    expected_id = (
        f"{record['theme']}-{record['seed']:016x}-"
        f"{record['dimensions'][0]}x{record['dimensions'][1]}"
    )
    if record["map_id"] != expected_id:
        raise ValueError("map quality map identity is not canonical")
    metrics = record["metrics"]
    paths = metrics["required_paths"]
    derived = {
        "maximum_required_detour_ratio": max(
            path["geometric_detour_ratio"] for path in paths
        ),
        "minimum_canonical_geometric_path_clearance": min(
            path["canonical_geometric_minimum_clearance_radius"] for path in paths
        ),
        "maximum_canonical_geometric_path_narrow_fraction": max(
            path["canonical_geometric_narrow_cell_fraction"] for path in paths
        ),
        "maximum_safe_detour_ratio_vs_geometric": max(
            path["safe_detour_ratio_vs_geometric"] for path in paths
        ),
        "maximum_radius_one_safe_detour_ratio_vs_geometric": max(
            path["radius_one_safe_detour_ratio_vs_geometric"] for path in paths
        ),
        "maximum_canonical_geometric_path_hazard_fraction": max(
            path["canonical_geometric_hazard_fraction"] for path in paths
        ),
        "minimum_widest_shortest_clearance_radius": min(
            path["geometric_widest_shortest_clearance_radius"] for path in paths
        ),
        "maximum_minimum_hazard_fraction_on_shortest_path": max(
            path["geometric_minimum_hazard_fraction_on_shortest_path"]
            for path in paths
        ),
    }
    for name, expected in derived.items():
        if metrics.get(name) != expected:
            raise ValueError(f"map quality record derived field mismatch: {name}")
    diagnostics = record["diagnostics"]
    expected_diagnostics = {
        "mission_has_alternate_agent_scale_routes": (
            all(path["radius_one_safe_length"] >= 0 for path in paths)
            and metrics["agent_scale_mission_articulation_count"] == 0
        ),
        "radius_one_safe_route_exists": all(
            path["radius_one_safe_length"] >= 0 for path in paths
        ),
        "required_paths_not_excessively_detoured": (
            metrics["maximum_required_detour_ratio"] <= 4.0
        ),
        "spawn_distribution_noncollapsed": (
            metrics["spawn_pair_minimum_manhattan"] == -1
            or metrics["spawn_pair_minimum_manhattan"] >= 2
        ),
        "elevation_is_nontrivial": metrics["elevation_levels_used"] >= 2,
    }
    if diagnostics != expected_diagnostics:
        raise ValueError("map quality diagnostics disagree with metrics")


def assert_valid_audit_report(report: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(report),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        raise ValueError(
            "map quality report schema failed: "
            + "; ".join(error.message for error in errors[:8])
        )
    current_source = audit_source_hash()
    if report.get("audit_source_sha256") != current_source:
        raise ValueError("map quality report audit-source hash is stale")
    if report["format"] == QUALITY_FORMAT:
        _assert_record_hash(report)
        _assert_record_relationships(report)
        return
    if report.get("contract") != _contract_payload():
        raise ValueError("map quality bank contract mismatch")
    records = report["maps"]
    for record in records:
        _assert_record_hash(record)
        _assert_record_relationships(record)
        if record["audit_source_sha256"] != current_source:
            raise ValueError(f"stale nested audit source: {record['map_id']}")
        if not record["source_manifest_sha256"]:
            raise ValueError(f"bank record lacks manifest provenance: {record['map_id']}")
    semantic_identities = [record["source_semantic_sha256"] for record in records]
    themes = sorted({record["theme"] for record in records})
    expected_by_theme = {
        theme: _aggregate([record for record in records if record["theme"] == theme])
        for theme in THEMES
        if theme in themes
    }
    exact_fields = {
        "map_count": len(records),
        "unique_semantic_count": len(set(semantic_identities)),
        "themes": themes,
        "all_hard_valid": all(record["hard_validity_preserved"] for record in records),
        "aggregate": _aggregate(records),
        "by_theme": expected_by_theme,
    }
    for name, expected in exact_fields.items():
        if report.get(name) != expected:
            raise ValueError(f"map quality report derived field mismatch: {name}")
    expected_order = sorted(
        records,
        key=lambda record: (
            record["theme"],
            record["map_id"],
            record["source_semantic_sha256"] or "",
            record["source_manifest_sha256"] or "",
            record["report_sha256"],
        ),
    )
    if records != expected_order:
        raise ValueError("map quality records are not in canonical order")
    expected_hash = report.get("report_sha256")
    unhashed = {key: value for key, value in report.items() if key != "report_sha256"}
    if expected_hash != _canonical_hash(unhashed):
        raise ValueError("map quality bank hash mismatch")


def assert_exact_audit_replay(
    report: dict[str, Any], source_packs: Sequence[Path]
) -> None:
    """Re-audit authoritative packs and require exact report equality."""
    assert_valid_audit_report(report)
    if report["format"] == QUALITY_FORMAT:
        if len(source_packs) != 1:
            raise ValueError("single-map audit replay requires exactly one source pack")
        expected = audit_pack(Path(source_packs[0]))
    else:
        expected = audit_packs(tuple(Path(path) for path in source_packs))
    if report != expected:
        raise ValueError("map quality report is not an exact source-pack replay")


def write_audit_report(report: dict[str, Any], path: Path) -> Path:
    assert_valid_audit_report(report)
    write_json_atomic(Path(path), report)
    return Path(path)
