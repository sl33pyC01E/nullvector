from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import jsonschema
import numpy as np
from PIL import Image

from ..config import PROJECT_ROOT
from .model import (
    HAZARD_NAMES,
    TERRAIN_NAMES,
    WALKABLE_TERRAIN,
    MapData,
    MapInvariantError,
    Point,
)


SCHEMA_PATH = PROJECT_ROOT / "shared" / "schema" / "map_manifest.schema.json"


def _neighbors4(x: int, y: int, width: int, height: int) -> tuple[Point, ...]:
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


def _distance_map(walkable: np.ndarray, sources: Iterable[Point], *, ignore_walls: bool = False) -> np.ndarray:
    height, width = walkable.shape
    size = height * width
    walkable_cells = np.asarray(walkable, dtype=np.bool_).reshape(-1).tolist()
    distances = [-1] * size
    queue: deque[int] = deque()
    for x, y in sources:
        if not (0 <= x < width and 0 <= y < height):
            continue
        index = y * width + x
        if (ignore_walls or walkable_cells[index]) and distances[index] < 0:
            distances[index] = 0
            queue.append(index)
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in _flat_neighbors4(current, width, size):
            if distances[neighbor] < 0 and (
                ignore_walls or walkable_cells[neighbor]
            ):
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return np.asarray(distances, dtype=np.int32).reshape(height, width)


def validate_map(data: MapData) -> dict[str, Any]:
    """Return a complete invariant report; no check is silently downgraded."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_shape = (data.config.height, data.config.width)
    arrays = {
        "terrain": data.terrain,
        "walkability": data.walkability,
        "hazard": data.hazard,
        "elevation": data.elevation,
        "zone": data.zone,
        "nav_cost": data.nav_cost,
        "protected_backbone": data.protected_backbone,
        "required_clearance": data.required_clearance,
        "decoration_forbidden": data.decoration_forbidden,
    }
    expected_dtypes = {
        "terrain": np.dtype(np.uint8),
        "walkability": np.dtype(np.uint8),
        "hazard": np.dtype(np.uint8),
        "elevation": np.dtype(np.int8),
        "zone": np.dtype(np.int16),
        "nav_cost": np.dtype(np.float32),
        "protected_backbone": np.dtype(np.uint8),
        "required_clearance": np.dtype(np.uint8),
        "decoration_forbidden": np.dtype(np.uint8),
    }
    for name, array in arrays.items():
        check(
            f"shape.{name}",
            array.shape == expected_shape,
            f"expected {expected_shape}, observed {array.shape}",
        )
        check(
            f"dtype.{name}",
            array.dtype == expected_dtypes[name],
            f"expected {expected_dtypes[name]}, observed {array.dtype}",
        )

    if any(array.shape != expected_shape for array in arrays.values()):
        return {
            "passed": False,
            "map_id": data.map_id,
            "checks": checks,
            "failures": [item["name"] for item in checks if not item["passed"]],
            "metrics": {},
        }

    terrain_legal = np.isin(data.terrain, tuple(TERRAIN_NAMES))
    hazard_legal = np.isin(data.hazard, tuple(HAZARD_NAMES))
    check("semantic.terrain_ids", bool(terrain_legal.all()), "all terrain IDs are declared")
    check("semantic.hazard_ids", bool(hazard_legal.all()), "all hazard IDs are declared")

    expected_walkability = np.isin(data.terrain, tuple(WALKABLE_TERRAIN)).astype(np.uint8)
    check(
        "semantic.walkability_matches_terrain",
        bool(np.array_equal(data.walkability, expected_walkability)),
        "walkability must be exactly derivable from terrain semantics",
    )
    walkable = data.walkability.astype(bool)
    protected = data.protected_backbone != 0
    clearance = data.required_clearance != 0
    forbidden = data.decoration_forbidden != 0
    for name, mask in (
        ("protected_backbone", data.protected_backbone),
        ("required_clearance", data.required_clearance),
        ("decoration_forbidden", data.decoration_forbidden),
    ):
        check(
            f"semantic.{name}_binary",
            bool(np.isin(mask, (0, 1)).all()),
            f"{name} contains only uint8 domain values 0 and 1",
        )
    check(
        "topology.protected_backbone_walkable",
        bool(walkable[protected].all()),
        "every protected backbone cell is navigable",
    )
    check(
        "topology.protected_backbone_hazard_free",
        bool((data.hazard[protected] == 0).all()),
        "every protected backbone cell is hazard-free",
    )
    check(
        "safety.required_clearance_hazard_free",
        bool((data.hazard[clearance] == 0).all()),
        "every required-clearance cell is hazard-free",
    )
    expected_forbidden = protected | clearance | (data.hazard != 0)
    check(
        "safety.decoration_forbidden_exact_union",
        bool(np.array_equal(forbidden, expected_forbidden)),
        "decoration_forbidden exactly equals backbone, clearance, and final hazards",
    )
    boundary_clear = not (
        walkable[0, :].any()
        or walkable[-1, :].any()
        or walkable[:, 0].any()
        or walkable[:, -1].any()
    )
    check("topology.sealed_boundary", boundary_clear, "outermost cells must be non-walkable")

    def in_bounds(point: Point) -> bool:
        return 0 <= point[0] < data.config.width and 0 <= point[1] < data.config.height

    required = (data.start, data.exit, *data.objectives)
    check(
        "points.required_count",
        len(data.objectives) == data.config.objective_count,
        f"expected {data.config.objective_count} objectives, observed {len(data.objectives)}",
    )
    check(
        "points.spawn_count",
        len(data.spawns) == data.config.spawn_count,
        f"expected {data.config.spawn_count} spawns, observed {len(data.spawns)}",
    )
    check("points.required_unique", len(set(required)) == len(required), "start, exit, and objectives must be unique")
    check("points.spawns_unique", len(set(data.spawns)) == len(data.spawns), "spawn coordinates must be unique")
    check("points.required_in_bounds", all(in_bounds(point) for point in required), "all required points are in bounds")
    check("points.spawns_in_bounds", all(in_bounds(point) for point in data.spawns), "all spawns are in bounds")
    required_walkable = all(in_bounds(point) and walkable[point[1], point[0]] for point in required)
    spawn_walkable = all(in_bounds(point) and walkable[point[1], point[0]] for point in data.spawns)
    check("points.required_walkable", required_walkable, "all required points must be walkable")
    check("points.spawns_walkable", spawn_walkable, "all spawn points must be walkable")
    clearance_points = (*required, *data.spawns)
    clearance_covers_points = all(
        in_bounds(point) and clearance[point[1], point[0]] for point in clearance_points
    )
    check(
        "safety.required_clearance_covers_points",
        clearance_covers_points,
        "required_clearance contains start, exit, every objective, and every spawn",
    )
    backbone_covers_required = all(
        in_bounds(point) and protected[point[1], point[0]] for point in required
    )
    check(
        "topology.protected_backbone_covers_required",
        backbone_covers_required,
        "protected_backbone contains start, exit, and every objective",
    )
    protected_distances = (
        _distance_map(protected, [data.start])
        if in_bounds(data.start) and protected[data.start[1], data.start[0]]
        else np.full(expected_shape, -1, dtype=np.int32)
    )
    backbone_connected = backbone_covers_required and all(
        protected_distances[y, x] >= 0 for x, y in required
    )
    check(
        "topology.protected_backbone_connected",
        backbone_connected,
        "the captured backbone itself connects start to exit and every objective",
    )

    start_distances = _distance_map(walkable, [data.start]) if in_bounds(data.start) else np.full(expected_shape, -1)
    all_required_connected = required_walkable and all(start_distances[y, x] >= 0 for x, y in required)
    check("topology.required_connected", all_required_connected, "start can flood-fill to exit and every objective")

    safe_walkable = walkable & (data.hazard == 0)
    safe_distances = _distance_map(safe_walkable, [data.start]) if in_bounds(data.start) else np.full(expected_shape, -1)
    all_required_safe = required_walkable and all(
        safe_walkable[y, x] and safe_distances[y, x] >= 0 for x, y in required
    )
    check(
        "topology.required_hazard_free_connected",
        all_required_safe,
        "start can reach exit and every objective without crossing a hazard",
    )

    padded = np.pad(walkable, 1, mode="constant", constant_values=False)
    eroded = np.logical_and.reduce(
        [
            padded[dy : dy + expected_shape[0], dx : dx + expected_shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
    )
    eroded_distances = _distance_map(eroded, [data.start])
    agent_radius_connected = all(
        in_bounds(point) and eroded[point[1], point[0]] and eroded_distances[point[1], point[0]] >= 0
        for point in required
    )
    check(
        "topology.agent_radius_one_connected",
        agent_radius_connected,
        "required mission backbone survives a one-cell square erosion",
    )
    separation = int(start_distances[data.exit[1], data.exit[0]]) if in_bounds(data.exit) else -1
    check(
        "topology.start_exit_separation",
        separation >= data.config.effective_min_separation,
        f"path length {separation}, minimum {data.config.effective_min_separation}",
    )

    objective_distances = _distance_map(walkable, [data.exit, *data.objectives])
    spawn_start_clear = all(start_distances[y, x] >= data.config.spawn_clearance_start for x, y in data.spawns)
    spawn_objective_clear = all(
        objective_distances[y, x] >= data.config.spawn_clearance_objective for x, y in data.spawns
    )
    hazard_points = [(int(x), int(y)) for y, x in np.argwhere(data.hazard != 0)]
    if hazard_points:
        hazard_distances = _distance_map(np.ones(expected_shape, dtype=bool), hazard_points, ignore_walls=True)
        spawn_hazard_clear = all(
            hazard_distances[y, x] >= data.config.spawn_clearance_hazard for x, y in data.spawns
        )
        min_spawn_hazard = min((int(hazard_distances[y, x]) for x, y in data.spawns), default=-1)
    else:
        spawn_hazard_clear = True
        min_spawn_hazard = -1
    check(
        "safety.spawn_start_clearance",
        spawn_start_clear,
        f"minimum required graph distance {data.config.spawn_clearance_start}",
    )
    check(
        "safety.spawn_objective_clearance",
        spawn_objective_clear,
        f"minimum required graph distance {data.config.spawn_clearance_objective}",
    )
    check(
        "safety.spawn_hazard_clearance",
        spawn_hazard_clear,
        f"minimum required Manhattan distance {data.config.spawn_clearance_hazard}",
    )

    hazard_walkable = bool((data.hazard[~walkable] == 0).all())
    check("semantic.hazards_walkable", hazard_walkable, "hazards may only occupy navigable cells")
    elevation_valid = bool(((data.elevation >= 0) & (data.elevation <= 5)).all())
    check("semantic.elevation_range", elevation_valid, "elevation is quantized to [0, 5]")
    check(
        "semantic.elevation_nonwalkable_zero",
        bool((data.elevation[~walkable] == 0).all()),
        "non-walkable elevation must be zero",
    )
    zone_valid = bool((data.zone[walkable] >= 0).all() and (data.zone[~walkable] == -1).all())
    check("semantic.zone_coverage", zone_valid, "walkable cells have zones; blocked cells use -1")
    nav_valid = bool(
        np.isfinite(data.nav_cost).all()
        and (data.nav_cost[walkable] > 0.0).all()
        and (data.nav_cost[~walkable] == 0.0).all()
    )
    check("semantic.nav_cost_domain", nav_valid, "nav cost is finite/positive on walkable cells and zero elsewhere")

    walkable_count = int(walkable.sum())
    reachable_count = int((start_distances >= 0).sum())
    metrics = {
        "walkable_cells": walkable_count,
        "walkable_ratio": round(walkable_count / walkable.size, 6),
        "required_component_cells": reachable_count,
        "required_component_ratio": round(reachable_count / max(walkable_count, 1), 6),
        "hazard_free_required_component_cells": int((safe_distances >= 0).sum()),
        "hazard_cells": int((data.hazard != 0).sum()),
        "protected_backbone_cells": int(protected.sum()),
        "required_clearance_cells": int(clearance.sum()),
        "decoration_forbidden_cells": int(forbidden.sum()),
        "zone_count": int(len(np.unique(data.zone[data.zone >= 0]))),
        "start_exit_path_length": separation,
        "minimum_spawn_hazard_distance": min_spawn_hazard,
        "repair_count": int(data.repair_count),
    }
    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "passed": not failures,
        "map_id": data.map_id,
        "checks": checks,
        "failures": failures,
        "metrics": metrics,
    }


def assert_valid(data: MapData) -> dict[str, Any]:
    report = validate_map(data)
    if not report["passed"]:
        joined = ", ".join(report["failures"])
        raise MapInvariantError(f"Map {data.map_id} failed invariants: {joined}")
    return report


def validate_pack(path: Path) -> dict[str, Any]:
    """Validate schema, artifacts, topology masks, and deterministic replay."""
    from .generator import generate_map
    from .io import ARRAY_FILE, PREVIEW_FILE, array_digest, file_sha256, load_map_pack
    from .model import TOPOLOGY_MASK_MEANINGS, TOPOLOGY_MASK_NAMES
    from .render import preview_png_bytes

    path = Path(path)
    manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
    pack_dir = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "passed": False,
            "pack": str(pack_dir),
            "schema_errors": [f"manifest could not be loaded: {error}"],
            "artifact_errors": [],
            "map_report": None,
            "replay_report": None,
        }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(manifest), key=lambda item: list(item.path))
    schema_messages = [error.message for error in schema_errors]
    if schema_errors:
        return {
            "passed": False,
            "pack": str(pack_dir),
            "schema_errors": schema_messages,
            "artifact_errors": [],
            "map_report": None,
            "replay_report": None,
        }

    artifact_errors: list[str] = []
    try:
        data = load_map_pack(pack_dir, verify_hashes=False)
    except Exception as error:
        return {
            "passed": False,
            "pack": str(pack_dir),
            "schema_errors": schema_messages,
            "artifact_errors": [f"semantic pack could not be loaded: {error}"],
            "map_report": None,
            "replay_report": None,
        }
    arrays_path = pack_dir / ARRAY_FILE
    preview_path = pack_dir / PREVIEW_FILE
    arrays = data.arrays()
    try:
        if file_sha256(arrays_path) != manifest["artifacts"]["arrays"]["sha256"]:
            artifact_errors.append("arrays file SHA-256 mismatch")
        if file_sha256(preview_path) != manifest["artifacts"]["preview"]["sha256"]:
            artifact_errors.append("preview file SHA-256 mismatch")
    except Exception as error:
        artifact_errors.append(f"artifact hashing failed: {error}")
    if array_digest(arrays) != manifest["semantic_array_sha256"]:
        artifact_errors.append("semantic array canonical SHA-256 mismatch")

    topology_contract = manifest["semantics"]["topology_masks"]
    topology_arrays = {name: arrays[name] for name in TOPOLOGY_MASK_NAMES}
    if array_digest(topology_arrays) != topology_contract["combined_sha256"]:
        artifact_errors.append("combined topology-mask SHA-256 mismatch")
    for name in TOPOLOGY_MASK_NAMES:
        member = topology_contract["members"][name]
        if member["meaning"] != TOPOLOGY_MASK_MEANINGS[name]:
            artifact_errors.append(f"topology-mask meaning mismatch for {name}")
        if array_digest({name: arrays[name]}) != member["sha256"]:
            artifact_errors.append(f"topology-mask SHA-256 mismatch for {name}")
        if int((arrays[name] != 0).sum()) != member["cell_count"]:
            artifact_errors.append(f"topology-mask cell count mismatch for {name}")

    try:
        with Image.open(preview_path) as preview:
            preview.verify()
        with Image.open(preview_path) as preview:
            expected = (data.config.width * manifest["artifacts"]["preview"]["scale"], data.config.height * manifest["artifacts"]["preview"]["scale"])
            if preview.size != expected:
                artifact_errors.append(f"preview dimensions {preview.size} do not match {expected}")
            if preview.mode != "RGB":
                artifact_errors.append(f"preview mode {preview.mode} is not RGB")
        replay_preview_hash = hashlib.sha256(
            preview_png_bytes(data, scale=manifest["artifacts"]["preview"]["scale"])
        ).hexdigest()
        if replay_preview_hash != manifest["artifacts"]["preview"]["sha256"]:
            artifact_errors.append("preview bytes disagree with deterministic rendering replay")
    except Exception as error:  # Pillow supplies useful corruption details.
        artifact_errors.append(f"preview could not be decoded: {error}")

    map_report = validate_map(data)
    if manifest["statistics"] != map_report["metrics"]:
        artifact_errors.append("manifest statistics disagree with validated semantics")
    topology_manifest = manifest["topology"]
    if topology_manifest["invariants"] != map_report["checks"]:
        artifact_errors.append("manifest invariant records disagree with validated semantics")
    if topology_manifest["start_exit_path_length"] != map_report["metrics"].get(
        "start_exit_path_length"
    ):
        artifact_errors.append("manifest start/exit path length disagrees with validated semantics")
    if topology_manifest["required_route_repairs"] != data.repair_count:
        artifact_errors.append("manifest repair count disagrees with loaded semantics")
    if topology_manifest["protected_backbone_segments"] != 1 + len(data.objectives):
        artifact_errors.append("manifest protected backbone segment count is inconsistent")
    replay_failures: list[str] = []
    try:
        replay = generate_map(data.seed, data.theme, data.config)
        if replay.map_id != data.map_id:
            replay_failures.append("map_id")
        if replay.start != data.start:
            replay_failures.append("points.start")
        if replay.exit != data.exit:
            replay_failures.append("points.exit")
        if replay.objectives != data.objectives:
            replay_failures.append("points.objectives")
        if replay.spawns != data.spawns:
            replay_failures.append("points.spawns")
        if replay.repair_count != data.repair_count:
            replay_failures.append("topology.required_route_repairs")
        if replay.metadata.get("theme_parameters") != data.metadata.get("theme_parameters"):
            replay_failures.append("generator.theme_parameters")
        replay_arrays = replay.arrays()
        for name in arrays:
            if not np.array_equal(replay_arrays[name], arrays[name]):
                replay_failures.append(f"arrays.{name}")
    except Exception as error:
        replay_failures.append(f"exception:{type(error).__name__}:{error}")
    replay_report = {
        "passed": not replay_failures,
        "failures": replay_failures,
        "policy": "exact deterministic regeneration from manifest seed, theme, and config",
    }
    return {
        "passed": not schema_errors and not artifact_errors and map_report["passed"] and replay_report["passed"],
        "pack": str(pack_dir),
        "schema_errors": schema_messages,
        "artifact_errors": artifact_errors,
        "map_report": map_report,
        "replay_report": replay_report,
    }
