from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from ..map_decorator.hashing import json_sha256
from ..morphology.constants import MATERIAL_NAMES, PART_OWNER_NAMES
from ..multifield_style.compiler import load_style_manifest
from ..multifield_style.hashing import sha256_file
from ..multifield_style.source import load_generation_bank
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    png_bytes,
    sha256_bytes,
)
from ..safety import require_disk_floor
from .contract import (
    ARRAY_FORMAT,
    CANVAS_SIZE,
    CELLULAR_CONTRACT_SHA256,
    DISK_FLOOR_GIB,
    FLUID_BY_FAMILY,
    FORMAT,
    SPECIES_FORMAT,
    TISSUE_NAMES,
    CellFlag,
    SimulationDefaults,
    TissueType,
    contract_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ARRAY_KEYS = frozenset(
    {
        "format",
        "position_xy",
        "part_owner",
        "material",
        "emission",
        "tissue",
        "organ_id",
        "cell_flags",
        "max_health",
        "fluid_capacity",
        "fluid_initial",
        "nutrient_initial",
        "energy_initial",
        "mass",
        "stiffness",
        "bond_ab",
        "bond_kind",
        "bond_rest",
        "bond_strength",
        "bond_conductance",
    }
)


def _stable_float(seed: int, label: str, low: float, high: float) -> float:
    digest = hashlib.sha256(f"{seed}:{label}".encode("ascii")).digest()
    unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    return round(low + (high - low) * unit, 7)


def _atomic_publish(root: Path, files: Mapping[str, bytes]) -> None:
    if root.exists():
        raise FileExistsError(root)
    require_disk_floor(root.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=sum(map(len, files.values())) + 512 * 1024**2)
    staging = root.parent / f".{root.name}.tmp-{os.getpid()}-{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for relative, payload in sorted(files.items()):
            pure = PurePosixPath(relative)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise ValueError(f"Unsafe output path: {relative!r}")
            target = staging.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        os.replace(staging, root)
    except BaseException:
        raise


def _neighbors(x: int, y: int, mask: np.ndarray, *, diagonal: bool = True):
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        offsets += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for dx, dy in offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < CANVAS_SIZE and 0 <= ny < CANVAS_SIZE and bool(mask[ny, nx]):
            yield nx, ny


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    remaining = {(int(x), int(y)) for y, x in np.argwhere(mask)}
    result: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining, key=lambda point: (point[1], point[0]))
        remaining.remove(start)
        queue = deque([start])
        component = [start]
        while queue:
            x, y = queue.popleft()
            for point in _neighbors(x, y, mask):
                if point in remaining:
                    remaining.remove(point)
                    component.append(point)
                    queue.append(point)
        result.append(sorted(component, key=lambda point: (point[1], point[0])))
    return sorted(result, key=lambda group: (-len(group), group[0][1], group[0][0]))


def _closest_cell(points: np.ndarray, target: tuple[float, float]) -> int:
    dx = points[:, 0].astype(np.float64) - target[0]
    dy = points[:, 1].astype(np.float64) - target[1]
    distance = dx * dx + dy * dy
    order = np.lexsort((points[:, 0], points[:, 1], distance))
    return int(order[0])


def _select_spaced(points: np.ndarray, count: int, *, y_bias: float = 0.0) -> list[int]:
    if not len(points):
        return []
    center_x = float(points[:, 0].mean())
    span = max(1.0, float(points[:, 0].max() - points[:, 0].min()))
    targets = np.linspace(center_x - span * 0.3, center_x + span * 0.3, count)
    selected: list[int] = []
    for target_x in targets:
        available = [index for index in range(len(points)) if index not in selected]
        if not available:
            break
        index = min(
            available,
            key=lambda item: (
                abs(float(points[item, 0]) - target_x) + y_bias * float(points[item, 1]),
                int(points[item, 1]),
                int(points[item, 0]),
            ),
        )
        selected.append(index)
    return selected


def _shortest_path(mask: np.ndarray, start: tuple[int, int], goals: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if start in goals:
        return [start]
    queue = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found: tuple[int, int] | None = None
    while queue and found is None:
        current = queue.popleft()
        for neighbor in _neighbors(*current, mask):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            if neighbor in goals:
                found = neighbor
                break
            queue.append(neighbor)
    if found is None:
        return []
    path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = found
    while node is not None:
        path.append(node)
        node = previous[node]
    return list(reversed(path))


def _genome(condition: Any) -> dict[str, object]:
    seed = int(condition.sample_seed)
    family = int(condition.morphology_id)
    base_metabolism = (1.0, 0.92, 0.62, 1.18, 0.78)[family]
    reproduction = (0.72, 0.67, 0.58, 0.76, 0.83)[family]
    return {
        "genome_seed": seed,
        "generation": 0,
        "metabolic_rate": round(base_metabolism * _stable_float(seed, "metabolism", 0.86, 1.14), 7),
        "digestion_efficiency": _stable_float(seed, "digestion", 0.58, 0.91),
        "fluid_regeneration_rate": _stable_float(seed, "fluid_regeneration", 0.012, 0.042),
        "tissue_regeneration_rate": _stable_float(seed, "tissue_regeneration", 0.004, 0.018),
        "basal_energy_capacity": _stable_float(seed, "energy_capacity", 80.0, 145.0),
        "reproduction_energy_threshold": round(reproduction * _stable_float(seed, "reproduction", 105.0, 150.0), 7),
        "gestation_seconds": _stable_float(seed, "gestation", 7.0, 19.0),
        "offspring_energy_fraction": _stable_float(seed, "offspring_fraction", 0.28, 0.46),
        "mutation_rate": _stable_float(seed, "mutation", 0.012, 0.085),
        "mutation_scale": _stable_float(seed, "mutation_scale", 0.025, 0.16),
        "litter_size": 1 + int(hashlib.sha256(f"{seed}:litter".encode()).digest()[0] % (2 if family in (1, 2, 3) else 1)),
        "diet": ("omnivore", "predator", "phototroph", "energivore", "mineralivore")[family],
        "reproduction_mode": ("live_division", "egg_bud", "spore_bloom", "phase_fission", "fabrication")[family],
    }


def _compile_arrays(sample: Any) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, object]]:
    part = sample.fields.part
    material_field = sample.fields.material
    emission_field = sample.fields.emission
    physical = (part > 0) & (part != 16)
    coordinates_yx = np.argwhere(physical)
    if len(coordinates_yx) < 24:
        raise ValueError(f"Species {sample.condition.sample_id} has too few physical cells.")
    position_xy = np.ascontiguousarray(coordinates_yx[:, ::-1], dtype=np.int16)
    index_grid = np.full((CANVAS_SIZE, CANVAS_SIZE), -1, dtype=np.int32)
    index_grid[coordinates_yx[:, 0], coordinates_yx[:, 1]] = np.arange(len(position_xy), dtype=np.int32)
    owners = np.ascontiguousarray(part[physical], dtype=np.uint8)
    materials = np.ascontiguousarray(material_field[physical], dtype=np.uint8)
    emissions = np.ascontiguousarray(emission_field[physical], dtype=np.uint8)
    tissue = np.full(len(position_xy), int(TissueType.EPIDERMIS), dtype=np.uint8)
    tissue[np.isin(owners, (4, 5, 6, 7, 8))] = int(TissueType.CONTRACTILE)
    tissue[np.isin(owners, (2, 13))] = int(TissueType.STRUCTURAL)
    tissue[owners == 3] = int(TissueType.NEURAL)
    tissue[owners == 10] = int(TissueType.VASCULAR)
    tissue[np.isin(owners, (11, 14, 15))] = int(TissueType.SENSORY)
    tissue[owners == 9] = int(TissueType.WEAPON)
    tissue[owners == 12] = int(TissueType.EMITTER)
    tissue[materials == 5] = int(TissueType.ARMOR)

    organ_names = ["integument", "musculature", "circulatory_core", "neural_core", "digestive_vacuole", "reproductive_nexus"]
    organ_kinds = ["integument", "contractile", "circulatory", "neural", "digestive", "reproductive"]
    essential = [True, False, True, True, True, True]
    organ_id = np.ones(len(position_xy), dtype=np.uint16)
    organ_id[np.isin(owners, (4, 5, 6, 7, 8))] = 2

    center = (float(position_xy[:, 0].mean()), float(position_xy[:, 1].mean()))
    core_candidates = np.where(owners == 10)[0]
    heart_index = int(core_candidates[0]) if len(core_candidates) else _closest_cell(position_xy, center)
    heart_xy = tuple(map(int, position_xy[heart_index]))
    heart_region = [index for index, xy in enumerate(position_xy) if (int(xy[0]) - heart_xy[0]) ** 2 + (int(xy[1]) - heart_xy[1]) ** 2 <= 2]
    organ_id[heart_region] = 3
    tissue[heart_region] = int(TissueType.VASCULAR)

    head_candidates = np.where(owners == 3)[0]
    if not len(head_candidates):
        threshold = float(np.quantile(position_xy[:, 1], 0.28))
        head_candidates = np.where(position_xy[:, 1] <= threshold)[0]
    brain_center = int(head_candidates[np.argmin(np.abs(position_xy[head_candidates, 0] - center[0]))])
    brain_xy = tuple(map(int, position_xy[brain_center]))
    brain_region = [index for index in head_candidates if (int(position_xy[index, 0]) - brain_xy[0]) ** 2 + (int(position_xy[index, 1]) - brain_xy[1]) ** 2 <= 3]
    if not brain_region:
        brain_region = [brain_center]
    organ_id[brain_region] = 4
    tissue[brain_region] = int(TissueType.NEURAL)

    body_candidates = np.where(np.isin(owners, (1, 10, 11, 12, 13)))[0]
    if len(body_candidates) < 2:
        body_candidates = np.arange(len(position_xy))
    gut_target = (center[0], min(float(position_xy[:, 1].max()), center[1] + 3.0))
    gut_index = int(body_candidates[_closest_cell(position_xy[body_candidates], gut_target)])
    repro_target = (center[0] + (1.0 if sample.condition.ordinal & 1 else -1.0), min(float(position_xy[:, 1].max()), center[1] + 6.0))
    repro_index = int(body_candidates[_closest_cell(position_xy[body_candidates], repro_target)])
    gut_region = [index for index in body_candidates if np.abs(position_xy[index] - position_xy[gut_index]).sum() <= 1][:5]
    repro_region = [index for index in body_candidates if np.abs(position_xy[index] - position_xy[repro_index]).sum() <= 1][:4]
    organ_id[gut_region] = 5
    tissue[gut_region] = int(TissueType.DIGESTIVE)
    organ_id[repro_region] = 6
    tissue[repro_region] = int(TissueType.REPRODUCTIVE)

    flags = np.zeros(len(position_xy), dtype=np.uint8)
    flags[heart_region] |= int(CellFlag.CIRCULATORY_CORE)
    flags[repro_region] |= int(CellFlag.REPRODUCTIVE)
    if sample.condition.morphology_id == 2:
        flags |= np.where(np.isin(tissue, (TissueType.EPIDERMIS, TissueType.EMITTER)), int(CellFlag.PHOTOSYNTHETIC), 0).astype(np.uint8)
    flags[np.isin(tissue, (TissueType.STEM,))] |= int(CellFlag.STEM)
    flags[owners == 9] |= int(CellFlag.WEAPON)
    flags[(owners == 12) | (emissions > 0)] |= int(CellFlag.EMITTER)

    eye_counts = (2, 2, 3, 5, 2)
    preferred_eye_candidates = head_candidates[organ_id[head_candidates] <= 2]
    # A very compact neural owner region can be entirely claimed by the brain.
    # Falling back to that same region would turn every neural-core cell into
    # an eye and silently delete an essential organ.  Use unclaimed upper
    # integument instead; the phenotype remains immutable and the derived
    # anatomy keeps sensory and neural systems physically distinct.
    eye_candidates = preferred_eye_candidates
    if not len(eye_candidates):
        unclaimed = np.where(organ_id <= 2)[0]
        if len(unclaimed):
            upper_limit = float(np.quantile(position_xy[unclaimed, 1], 0.35))
            upper = unclaimed[position_xy[unclaimed, 1] <= upper_limit]
            eye_candidates = upper if len(upper) else unclaimed
    local_eye_indices = _select_spaced(position_xy[eye_candidates], eye_counts[sample.condition.morphology_id], y_bias=0.06)
    eye_indices: list[int] = []
    for eye_number, local in enumerate(local_eye_indices, start=1):
        index = int(eye_candidates[local])
        if index in eye_indices:
            continue
        eye_indices.append(index)
        organ_names.append(f"eye_{eye_number}")
        organ_kinds.append("sensory")
        essential.append(eye_number == 1)
        organ_id[index] = len(organ_names)
        tissue[index] = int(TissueType.SENSORY)
        flags[index] |= int(CellFlag.EYE)

    mouth_candidates = eye_candidates[organ_id[eye_candidates] <= 2]
    if not len(mouth_candidates):
        mouth_candidates = np.where(organ_id <= 2)[0]
    eye_mean_y = float(position_xy[eye_indices, 1].mean()) if eye_indices else float(position_xy[brain_center, 1])
    mouth_target = (center[0], eye_mean_y + 2.0)
    mouth_index = int(mouth_candidates[_closest_cell(position_xy[mouth_candidates], mouth_target)])
    organ_names.append("mouth")
    organ_kinds.append("digestive")
    essential.append(True)
    organ_id[mouth_index] = len(organ_names)
    tissue[mouth_index] = int(TissueType.DIGESTIVE)
    flags[mouth_index] |= int(CellFlag.MOUTH)

    owner_organs = {
        2: ("armor_shell", "armor", TissueType.ARMOR),
        4: ("left_appendage", "appendage", TissueType.CONTRACTILE),
        5: ("right_appendage", "appendage", TissueType.CONTRACTILE),
        6: ("left_locomotor", "locomotor", TissueType.CONTRACTILE),
        7: ("right_locomotor", "locomotor", TissueType.CONTRACTILE),
        8: ("auxiliary_appendage", "appendage", TissueType.CONTRACTILE),
        9: ("weapon_organ", "weapon", TissueType.WEAPON),
        12: ("emitter_organ", "emitter", TissueType.EMITTER),
    }
    for owner, (name, kind, tissue_type) in owner_organs.items():
        members = np.where((owners == owner) & (organ_id <= 2))[0]
        if not len(members):
            continue
        organ_names.append(name)
        organ_kinds.append(kind)
        essential.append(False)
        organ_id[members] = len(organ_names)
        tissue[members] = int(tissue_type)

    organ_records: list[dict[str, object]] = []
    for organ_index, (name, kind, is_essential) in enumerate(zip(organ_names, organ_kinds, essential, strict=True), start=1):
        members = np.where(organ_id == organ_index)[0]
        if not len(members):
            continue
        organ_records.append(
            {
                "id": organ_index,
                "name": name,
                "kind": kind,
                "cell_count": int(len(members)),
                "center_xy": [round(float(position_xy[members, 0].mean()), 4), round(float(position_xy[members, 1].mean()), 4)],
                "essential": bool(is_essential),
            }
        )

    organ_groups = {record["id"]: set(map(tuple, position_xy[organ_id == record["id"]].tolist())) for record in organ_records}
    for record in organ_records:
        if record["kind"] in {"circulatory", "integument"}:
            continue
        path = _shortest_path(physical, heart_xy, organ_groups[int(record["id"])])
        for x, y in path[1:-1]:
            index = int(index_grid[y, x])
            if index >= 0 and tissue[index] in (TissueType.EPIDERMIS, TissueType.CONTRACTILE, TissueType.STRUCTURAL):
                tissue[index] = int(TissueType.VASCULAR)

    max_health = np.full(len(position_xy), 1.0, dtype=np.float32)
    max_health += np.where(tissue == TissueType.ARMOR, 1.2, 0.0).astype(np.float32)
    max_health += np.where(tissue == TissueType.STRUCTURAL, 0.55, 0.0).astype(np.float32)
    max_health -= np.where(np.isin(tissue, (TissueType.SENSORY, TissueType.VASCULAR)), 0.18, 0.0).astype(np.float32)
    mass = np.full(len(position_xy), 1.0, dtype=np.float32)
    mass += np.where(np.isin(materials, (4, 5, 6)), 0.55, 0.0).astype(np.float32)
    mass -= np.where(materials == 9, 0.3, 0.0).astype(np.float32)
    stiffness = np.full(len(position_xy), 0.62, dtype=np.float32)
    stiffness += np.where(np.isin(tissue, (TissueType.ARMOR, TissueType.STRUCTURAL)), 0.28, 0.0).astype(np.float32)
    stiffness -= np.where(np.isin(tissue, (TissueType.VASCULAR, TissueType.DIGESTIVE, TissueType.REPRODUCTIVE)), 0.18, 0.0).astype(np.float32)
    stiffness = np.clip(stiffness, 0.25, 1.2).astype(np.float32)
    fluid_capacity = np.full(len(position_xy), 0.32, dtype=np.float32)
    fluid_capacity += np.where(tissue == TissueType.VASCULAR, 0.75, 0.0).astype(np.float32)
    fluid_capacity += np.where(np.isin(tissue, (TissueType.DIGESTIVE, TissueType.REPRODUCTIVE, TissueType.STORAGE)), 0.42, 0.0).astype(np.float32)
    fluid_capacity -= np.where(tissue == TissueType.ARMOR, 0.2, 0.0).astype(np.float32)
    fluid_capacity = np.clip(fluid_capacity, 0.08, 1.25).astype(np.float32)
    fluid_initial = (fluid_capacity * 0.88).astype(np.float32)
    nutrient_initial = np.full(len(position_xy), 0.05, dtype=np.float32)
    nutrient_initial[tissue == TissueType.DIGESTIVE] = 0.95
    nutrient_initial[tissue == TissueType.STORAGE] = 0.7
    energy_initial = np.full(len(position_xy), 0.32, dtype=np.float32)
    energy_initial[tissue == TissueType.VASCULAR] = 0.7
    energy_initial[tissue == TissueType.REPRODUCTIVE] = 0.85
    energy_initial[(flags & int(CellFlag.PHOTOSYNTHETIC)) != 0] = 0.75

    bond_pairs: list[tuple[int, int]] = []
    bond_kind: list[int] = []
    for index, (x_value, y_value) in enumerate(position_xy):
        x, y = int(x_value), int(y_value)
        # Both diagonal directions are explicit. Sparse checkerboard bracing can
        # leave a diagonally touching one-pixel appendage unbound, which is not
        # acceptable when every visible source pixel is a physical cell.
        candidates = [(x + 1, y), (x, y + 1), (x + 1, y + 1), (x - 1, y + 1)]
        for nx, ny in candidates:
            if 0 <= nx < CANVAS_SIZE and 0 <= ny < CANVAS_SIZE:
                neighbor = int(index_grid[ny, nx])
                if neighbor >= 0:
                    bond_pairs.append((index, neighbor))
                    bond_kind.append(0)
    components = _components(physical)
    main_points = components[0]
    for component in components[1:]:
        left, right = min(
            ((a, b) for a in main_points for b in component),
            key=lambda pair: ((pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2, pair),
        )
        bond_pairs.append((int(index_grid[left[1], left[0]]), int(index_grid[right[1], right[0]])))
        bond_kind.append(1)
        main_points = main_points + component
    bond_ab = np.asarray(bond_pairs, dtype=np.uint16).reshape(-1, 2)
    bond_kind_array = np.asarray(bond_kind, dtype=np.uint8)
    delta = position_xy[bond_ab[:, 0]].astype(np.float32) - position_xy[bond_ab[:, 1]].astype(np.float32)
    bond_rest = np.linalg.norm(delta, axis=1).astype(np.float32)
    endpoint_stiffness = np.minimum(stiffness[bond_ab[:, 0]], stiffness[bond_ab[:, 1]])
    cross_organ = organ_id[bond_ab[:, 0]] != organ_id[bond_ab[:, 1]]
    bond_strength = (0.85 + 1.5 * endpoint_stiffness).astype(np.float32)
    bond_strength[cross_organ] *= 0.78
    bond_strength[bond_kind_array == 1] = 0.55
    bond_conductance = (0.12 + 0.62 * np.minimum(fluid_capacity[bond_ab[:, 0]], fluid_capacity[bond_ab[:, 1]])).astype(np.float32)
    bond_conductance[bond_kind_array == 1] *= 0.08

    arrays = {
        "format": np.asarray([ARRAY_FORMAT]),
        "position_xy": position_xy,
        "part_owner": owners,
        "material": materials,
        "emission": emissions,
        "tissue": tissue,
        "organ_id": organ_id,
        "cell_flags": flags,
        "max_health": max_health,
        "fluid_capacity": fluid_capacity,
        "fluid_initial": fluid_initial,
        "nutrient_initial": nutrient_initial,
        "energy_initial": energy_initial,
        "mass": mass,
        "stiffness": stiffness,
        "bond_ab": bond_ab,
        "bond_kind": bond_kind_array,
        "bond_rest": bond_rest,
        "bond_strength": bond_strength,
        "bond_conductance": bond_conductance,
    }
    summary = {
        "physical_cell_count": len(position_xy),
        "source_effect_pixel_count": int(np.count_nonzero(part == 16)),
        "bond_count": len(bond_ab),
        "structural_bond_count": int(np.count_nonzero(bond_kind_array == 0)),
        "phase_tether_count": int(np.count_nonzero(bond_kind_array == 1)),
        "organ_count": len(organ_records),
        "eye_count": int(np.count_nonzero((flags & int(CellFlag.EYE)) != 0)),
        "appendage_organ_count": sum(record["kind"] in {"appendage", "locomotor", "weapon"} for record in organ_records),
        "initial_fluid": round(float(fluid_initial.sum()), 6),
        "fluid_capacity": round(float(fluid_capacity.sum()), 6),
        "initial_nutrients": round(float(nutrient_initial.sum()), 6),
        "initial_cell_energy": round(float(energy_initial.sum()), 6),
        "graph_component_count_after_tethers": 1,
    }
    return arrays, organ_records, summary


def _palette(style_root: Path, style_record: Mapping[str, Any]) -> dict[str, object]:
    artifact = style_record["presentation"]["artifacts"]["palette"]
    path = (style_root / artifact["path"]).resolve()
    if sha256_file(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
        raise ValueError("Style palette artifact hash/size mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    material_colors = [[0, 0, 0]] * len(MATERIAL_NAMES)
    for name, record in payload["materials"].items():
        material_colors[int(record["id"])] = list(map(int, record["mid"]))
    return {
        "material_mid_rgb": material_colors,
        "fluid_rgb": list(map(int, payload["effects"]["role_hot"])),
        "nutrient_rgb": list(map(int, payload["effects"]["role_accent"])),
        "outline_rgb": list(map(int, payload["effects"]["outline_shadow"])),
        "emission_rgb": list(map(int, payload["effects"]["emission_levels"][-1])),
    }


def _render_panels(arrays: Mapping[str, np.ndarray], palette: Mapping[str, object]) -> np.ndarray:
    positions = arrays["position_xy"]
    scale = 2
    panels = np.zeros((CANVAS_SIZE * scale, CANVAS_SIZE * scale * 3, 3), dtype=np.uint8)
    material_colors = np.asarray(palette["material_mid_rgb"], dtype=np.uint8)
    base = material_colors[arrays["material"]]
    organ = np.empty_like(base)
    for index, organ_id in enumerate(arrays["organ_id"]):
        digest = hashlib.sha256(f"organ:{int(organ_id)}".encode()).digest()
        organ[index] = np.asarray((48 + digest[0] % 190, 48 + digest[1] % 190, 48 + digest[2] % 190), dtype=np.uint8)
    fluid_rgb = np.asarray(palette["fluid_rgb"], dtype=np.float32)
    ratio = arrays["fluid_initial"] / np.maximum(arrays["fluid_capacity"], 1e-6)
    fluid = np.clip(12.0 + ratio[:, None] * fluid_rgb[None], 0, 255).astype(np.uint8)
    for panel_index, colors in enumerate((base, organ, fluid)):
        for (x, y), color in zip(positions, colors, strict=True):
            x0 = panel_index * CANVAS_SIZE * scale + int(x) * scale
            y0 = int(y) * scale
            panels[y0 : y0 + scale, x0 : x0 + scale] = color
    return panels


def _contact_sheet(previews: list[tuple[str, str, np.ndarray]]) -> bytes:
    cell_w, cell_h = CANVAS_SIZE * 2 * 3, CANVAS_SIZE * 2
    columns, rows = 4, math.ceil(len(previews) / 4)
    label_h = 18
    canvas = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), (3, 9, 14))
    draw = ImageDraw.Draw(canvas)
    for index, (sample_id, family, pixels) in enumerate(previews):
        x = (index % columns) * cell_w
        y = (index // columns) * (cell_h + label_h)
        canvas.paste(Image.fromarray(pixels), (x, y))
        draw.text((x + 3, y + cell_h + 2), f"{family.upper()} // {sample_id}", fill=(120, 240, 230))
    output = np.asarray(canvas, dtype=np.uint8)
    return png_bytes(output)


def _compile_all(generation_manifest: Path, style_manifest: Path) -> tuple[list[dict[str, object]], dict[str, bytes], bytes]:
    bank = load_generation_bank(generation_manifest)
    style = load_style_manifest(style_manifest)
    if style["parent"]["manifest_sha256"] != bank.manifest_sha256:
        raise ValueError("Style bank and categorical generation bank are not aligned.")
    style_samples = {record["condition"]["sample_id"]: record for record in style["samples"]}
    if set(style_samples) != {sample.condition.sample_id for sample in bank.samples}:
        raise ValueError("Style and generation sample identities differ.")
    files: dict[str, bytes] = {}
    records: list[dict[str, object]] = []
    previews: list[tuple[str, str, np.ndarray]] = []
    for sample in bank.samples:
        sample_id = sample.condition.sample_id
        arrays, organs, summary = _compile_arrays(sample)
        validate_species_arrays(arrays, organs, summary)
        arrays_bytes = deterministic_npz_bytes(arrays)
        relative = f"species/{sample_id}/cellular_anatomy.npz"
        files[relative] = arrays_bytes
        palette = _palette(Path(style_manifest).resolve().parent, style_samples[sample_id])
        genome = _genome(sample.condition)
        anatomy_sha = json_sha256(
            {
                "sample_id": sample_id,
                "source_fields_sha256": sample.fields.aligned_sha256,
                "arrays_sha256": sha256_bytes(arrays_bytes),
                "organs": organs,
                "genome": genome,
            }
        )
        record = {
            "format": SPECIES_FORMAT,
            "sample_id": sample_id,
            "ordinal": sample.condition.ordinal,
            "family": sample.condition.morphology_name,
            "family_id": sample.condition.morphology_id,
            "subtype": sample.condition.subtype_name,
            "subtype_id": sample.condition.subtype_id,
            "role": sample.condition.role_name,
            "role_id": sample.condition.role_id,
            "source_fields_sha256": sample.fields.aligned_sha256,
            "anatomy_sha256": anatomy_sha,
            "arrays": artifact_record_from_bytes(relative, arrays_bytes),
            "fluid": {
                "name": FLUID_BY_FAMILY[sample.condition.morphology_id],
                "closed_loop_initially": True,
                "spills_when_cells_or_bonds_fail": True,
                "pressure_drives_diffusion": True,
            },
            "genome": genome,
            "organs": organs,
            "palette": palette,
            "summary": summary,
            "capabilities": {
                "damage": True,
                "bond_fracture": True,
                "cell_ablation": True,
                "fluid_leakage": True,
                "feeding": True,
                "metabolism": True,
                "healing": True,
                "reproduction": True,
                "heritable_mutation": True,
            },
        }
        records.append(record)
        previews.append((sample_id, sample.condition.morphology_name, _render_panels(arrays, palette)))
    return records, files, _contact_sheet(previews)


def build_bank(generation_manifest: Path, style_manifest: Path, destination: Path) -> dict[str, object]:
    records, files, contact = _compile_all(generation_manifest, style_manifest)
    contact_path = "cellular_organism_contact_sheet.png"
    files[contact_path] = contact
    family_counts = {family: sum(record["family"] == family for record in records) for family in ("humanoid", "animalian", "plantlike", "anomaly", "machine")}
    manifest: dict[str, object] = {
        "format": FORMAT,
        "status": "ready",
        "contract": contract_manifest(),
        "contract_sha256": CELLULAR_CONTRACT_SHA256,
        "source": {
            "generation_manifest": Path(generation_manifest).resolve().relative_to(PROJECT_ROOT).as_posix(),
            "generation_manifest_sha256": sha256_file(Path(generation_manifest)),
            "style_manifest": Path(style_manifest).resolve().relative_to(PROJECT_ROOT).as_posix(),
            "style_manifest_sha256": sha256_file(Path(style_manifest)),
        },
        "sample_count": len(records),
        "family_counts": family_counts,
        "totals": {
            "physical_cells": sum(int(record["summary"]["physical_cell_count"]) for record in records),
            "organs": sum(int(record["summary"]["organ_count"]) for record in records),
            "eyes": sum(int(record["summary"]["eye_count"]) for record in records),
            "bonds": sum(int(record["summary"]["bond_count"]) for record in records),
            "phase_tethers": sum(int(record["summary"]["phase_tether_count"]) for record in records),
        },
        "simulation": SimulationDefaults().to_dict(),
        "contact_sheet": artifact_record_from_bytes(contact_path, contact),
        "species": records,
    }
    manifest["semantic_sha256"] = json_sha256(manifest)
    files["cellular_organism_manifest.json"] = canonical_json_bytes(manifest)
    _atomic_publish(Path(destination).resolve(), files)
    validate_bank(Path(destination) / "cellular_organism_manifest.json")
    return manifest


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Cellular species NPZ exceeds its size bound.")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARRAY_KEYS:
            raise ValueError("Cellular species NPZ keys drifted.")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def validate_species_arrays(arrays: Mapping[str, np.ndarray], organs: list[dict[str, object]], summary: Mapping[str, object]) -> None:
    if set(arrays) != EXPECTED_ARRAY_KEYS:
        raise ValueError("Cellular species arrays are incomplete or unexpected.")
    if arrays["format"].shape != (1,) or str(arrays["format"][0]) != ARRAY_FORMAT:
        raise ValueError("Cellular array format marker is invalid.")
    n = len(arrays["position_xy"])
    m = len(arrays["bond_ab"])
    if arrays["position_xy"].shape != (n, 2) or arrays["position_xy"].dtype != np.int16 or n < 24:
        raise ValueError("Cell position array violates its contract.")
    if len(set(map(tuple, arrays["position_xy"].tolist()))) != n or np.any(arrays["position_xy"] < 0) or np.any(arrays["position_xy"] >= CANVAS_SIZE):
        raise ValueError("Cell positions are duplicated or outside the canvas.")
    dtypes = {
        "part_owner": np.uint8,
        "material": np.uint8,
        "emission": np.uint8,
        "tissue": np.uint8,
        "organ_id": np.uint16,
        "cell_flags": np.uint8,
    }
    for name, dtype in dtypes.items():
        if arrays[name].shape != (n,) or arrays[name].dtype != dtype:
            raise ValueError(f"Cell array {name} violates shape/dtype.")
    if np.any(arrays["part_owner"] == 0) or np.any(arrays["part_owner"] == 16):
        raise ValueError("Physical cells cannot be background or aura effects.")
    if np.any(arrays["tissue"] < 1) or np.any(arrays["tissue"] >= len(TISSUE_NAMES)) or np.any(arrays["organ_id"] < 1):
        raise ValueError("Every physical cell must have a tissue and organ.")
    float_cells = ("max_health", "fluid_capacity", "fluid_initial", "nutrient_initial", "energy_initial", "mass", "stiffness")
    for name in float_cells:
        if arrays[name].shape != (n,) or arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all() or np.any(arrays[name] < 0):
            raise ValueError(f"Cell scalar {name} is invalid.")
    if np.any(arrays["fluid_initial"] > arrays["fluid_capacity"] + 1e-6):
        raise ValueError("Initial fluid exceeds cell capacity.")
    if arrays["bond_ab"].shape != (m, 2) or arrays["bond_ab"].dtype != np.uint16 or m < n - 1:
        raise ValueError("Bond endpoint array violates its contract.")
    pairs = [tuple(map(int, pair)) for pair in arrays["bond_ab"]]
    normalized = [tuple(sorted(pair)) for pair in pairs]
    if any(a == b or a >= n or b >= n for a, b in pairs) or len(normalized) != len(set(normalized)):
        raise ValueError("Bond endpoints are invalid or duplicated.")
    for name, dtype in {"bond_kind": np.uint8, "bond_rest": np.float32, "bond_strength": np.float32, "bond_conductance": np.float32}.items():
        if arrays[name].shape != (m,) or arrays[name].dtype != dtype:
            raise ValueError(f"Bond array {name} violates shape/dtype.")
    if not all(np.isfinite(arrays[name]).all() and np.all(arrays[name] > 0) for name in ("bond_rest", "bond_strength", "bond_conductance")):
        raise ValueError("Bond physical scalars must be finite and positive.")
    adjacency = [[] for _ in range(n)]
    for a, b in pairs:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = {0}
    queue = deque([0])
    while queue:
        for neighbor in adjacency[queue.popleft()]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(seen) != n:
        raise ValueError("Bond graph is not connected after explicit phase tethers.")
    kinds = {str(record["kind"]) for record in organs}
    required = {"circulatory", "neural", "digestive", "reproductive", "sensory"}
    if not required <= kinds or int(summary["eye_count"]) < 1:
        raise ValueError("Species is missing an essential organ system or eye.")
    if sum(int(record["cell_count"]) for record in organs) != n:
        raise ValueError("Organ membership counts do not exhaust physical cells.")


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("format") != FORMAT or payload.get("status") != "ready" or payload.get("contract_sha256") != CELLULAR_CONTRACT_SHA256:
        raise ValueError("Cellular organism bank header is invalid.")
    semantic = payload.pop("semantic_sha256")
    if semantic != json_sha256(payload):
        raise ValueError("Cellular organism manifest semantic hash mismatch.")
    payload["semantic_sha256"] = semantic
    root = manifest_path.parent
    if payload["sample_count"] != 80 or len(payload["species"]) != 80:
        raise ValueError("Cellular organism bank must cover all 80 neural identities.")
    ids: set[str] = set()
    totals = {"physical_cells": 0, "organs": 0, "eyes": 0, "bonds": 0, "phase_tethers": 0}
    for record in payload["species"]:
        sample_id = record["sample_id"]
        if sample_id in ids:
            raise ValueError("Cellular organism sample IDs are duplicated.")
        ids.add(sample_id)
        artifact = record["arrays"]
        path = root.joinpath(*PurePosixPath(artifact["path"]).parts).resolve()
        path.relative_to(root)
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("Cellular organism array artifact hash/size mismatch.")
        arrays = _load_arrays(path)
        validate_species_arrays(arrays, record["organs"], record["summary"])
        if sha256_bytes(deterministic_npz_bytes(arrays)) != artifact["sha256"]:
            raise ValueError("Cellular anatomy NPZ is not the canonical deterministic encoding.")
        totals["physical_cells"] += int(record["summary"]["physical_cell_count"])
        totals["organs"] += int(record["summary"]["organ_count"])
        totals["eyes"] += int(record["summary"]["eye_count"])
        totals["bonds"] += int(record["summary"]["bond_count"])
        totals["phase_tethers"] += int(record["summary"]["phase_tether_count"])
    if totals != payload["totals"]:
        raise ValueError("Cellular organism aggregate totals drifted.")
    contact = payload["contact_sheet"]
    contact_path = root / contact["path"]
    if contact_path.stat().st_size != contact["bytes"] or sha256_file(contact_path) != contact["sha256"]:
        raise ValueError("Cellular organism contact sheet hash/size mismatch.")
    return {
        "passed": True,
        "manifest_sha256": sha256_bytes(raw),
        "semantic_sha256": semantic,
        "sample_count": len(ids),
        "totals": totals,
    }


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation = PROJECT_ROOT / existing["source"]["generation_manifest"]
    style = PROJECT_ROOT / existing["source"]["style_manifest"]
    records, files, contact = _compile_all(generation, style)
    if records != existing["species"]:
        raise ValueError("Cellular organism semantic replay differs from the manifest.")
    expected_files = {record["arrays"]["path"]: record["arrays"]["sha256"] for record in records}
    for relative, expected_sha in expected_files.items():
        if sha256_bytes(files[relative]) != expected_sha or files[relative] != (manifest_path.parent / relative).read_bytes():
            raise ValueError(f"Cellular organism artifact replay differs: {relative}")
    if contact != (manifest_path.parent / existing["contact_sheet"]["path"]).read_bytes():
        raise ValueError("Cellular organism contact sheet replay differs.")
    return {
        "passed": True,
        "sample_count": len(records),
        "artifact_count": len(expected_files) + 1,
        "exact_artifact_replay": True,
        "manifest_sha256": sha256_file(manifest_path),
    }
