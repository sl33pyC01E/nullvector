from __future__ import annotations

from collections import deque
import math
from typing import Iterable

import numpy as np

from ..morphology.constants import FAMILIES, ROLE_NAMES, SUBTYPE_NAMES
from ..multifield_style.model import StyleCondition
from ..neural_rig_bridge.hashing import aligned_fields_hash
from ..neural_rig_repair.model import RepairSourceSample
from .hashing import array_sha256, canonical_json_bytes, sha256_bytes
from .model import FusionGenome, FusionSpecimen


FUSION_MODES = ("chimera", "mosaic", "symbiote", "bipolar", "invasive")
MUTATION_MODES = ("none", "armorize", "phase_bloom", "mycelial_growth", "scar", "bilateral_break")
PROVENANCE_NAMES = ("background", "parent_a", "parent_b", "mutation", "connective_repair")
SAFE_MARGIN = 4


def _readonly(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _shift(values: np.ndarray, dx: int, dy: int, fill: int = 0) -> np.ndarray:
    result = np.full_like(values, fill)
    y0 = max(0, dy)
    y1 = min(values.shape[-2], values.shape[-2] + dy)
    x0 = max(0, dx)
    x1 = min(values.shape[-1], values.shape[-1] + dx)
    sy0, sy1 = y0 - dy, y1 - dy
    sx0, sx1 = x0 - dx, x1 - dx
    if values.ndim == 2:
        result[y0:y1, x0:x1] = values[sy0:sy1, sx0:sx1]
    else:
        result[..., y0:y1, x0:x1] = values[..., sy0:sy1, sx0:sx1]
    return result


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    points = np.argwhere(mask)
    if not len(points):
        return 23.5, 23.5
    return float(points[:, 1].mean()), float(points[:, 0].mean())


def _mirror_fields(part: np.ndarray, material: np.ndarray, emission: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mirrored_part = np.fliplr(part).copy()
    for left, right in ((4, 5), (6, 7)):
        left_mask = mirrored_part == left
        right_mask = mirrored_part == right
        mirrored_part[left_mask] = right
        mirrored_part[right_mask] = left
    return mirrored_part, np.fliplr(material).copy(), np.fliplr(emission).copy()


def _noise(seed: int) -> np.ndarray:
    y, x = np.mgrid[0:48, 0:48]
    value = np.sin((x + (seed & 31)) * 0.347) + np.cos((y + ((seed >> 5) & 31)) * 0.283)
    value += 0.7 * np.sin((x + y + ((seed >> 10) & 63)) * 0.173)
    value += 0.45 * np.cos((2 * x - y + ((seed >> 16) & 63)) * 0.119)
    return value.astype(np.float32)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(active)
    result: list[np.ndarray] = []
    for sy, sx in np.argwhere(active):
        if seen[sy, sx]:
            continue
        component = np.zeros_like(active)
        queue = deque([(int(sy), int(sx))])
        seen[sy, sx] = True
        while queue:
            y, x = queue.popleft()
            component[y, x] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 48 and 0 <= nx < 48 and active[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        result.append(component)
    result.sort(key=lambda component: -int(component.sum()))
    return result


def _line(a: tuple[int, int], b: tuple[int, int]) -> Iterable[tuple[int, int]]:
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _nearest_legal(
    legal: np.ndarray,
    owner: int,
    material: int,
    emission: int,
) -> tuple[int, int, int]:
    rows = legal[legal[:, 0] == owner]
    if not len(rows):
        rows = legal[legal[:, 0] == 1]
    score = 3 * np.abs(rows[:, 1].astype(int) - material) + 5 * np.abs(rows[:, 2].astype(int) - emission)
    row = rows[int(np.argmin(score))]
    return tuple(map(int, row))


def _connect(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    provenance: np.ndarray,
    legal: np.ndarray,
) -> int:
    repairs = 0
    while True:
        components = _components(part != 0)
        if len(components) <= 1:
            return repairs
        dominant = np.argwhere(components[0])
        detached = np.argwhere(components[1])
        delta = detached[:, None, :] - dominant[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        di, bi = np.unravel_index(int(np.argmin(squared)), squared.shape)
        dy, dx = map(int, detached[di])
        by, bx = map(int, dominant[bi])
        owner, mat, emit = _nearest_legal(legal, 1, int(material[by, bx]), 0)
        for x, y in _line((dx, dy), (bx, by)):
            if part[y, x] == 0:
                part[y, x] = owner
                material[y, x] = mat
                emission[y, x] = emit
                provenance[y, x] = 4
                repairs += 1


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    return np.logical_or.reduce(
        [padded[1 + dy : 49 + dy, 1 + dx : 49 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    )


def _mutate(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    provenance: np.ndarray,
    legal: np.ndarray,
    mode: str,
    strength: int,
    seed: int,
) -> int:
    if mode == "none" or strength == 0:
        return 0
    noise = _noise(seed ^ 0xA55A5AA5)
    changed = 0
    if mode == "armorize":
        eligible = np.isin(part, (1, 3, 4, 5, 6, 7, 8)) & (noise > 0.35 - strength * 0.18)
        for y, x in np.argwhere(eligible):
            owner, mat, emit = _nearest_legal(legal, int(part[y, x]), 5, int(emission[y, x]))
            if (mat, emit) != (int(material[y, x]), int(emission[y, x])):
                part[y, x], material[y, x], emission[y, x] = owner, mat, emit
                provenance[y, x] = 3
                changed += 1
    elif mode == "phase_bloom":
        source = (emission > 0) | (part == 10)
        ring = _dilate(source) & ~source & (part == 0)
        ring[:SAFE_MARGIN] = ring[-SAFE_MARGIN:] = False
        ring[:, :SAFE_MARGIN] = ring[:, -SAFE_MARGIN:] = False
        candidates = np.argwhere(ring & (noise > 0.1 - strength * 0.25))
        for y, x in candidates:
            part[y, x], material[y, x], emission[y, x] = (16, 9, 3)
            provenance[y, x] = 3
            changed += 1
    elif mode == "mycelial_growth":
        boundary = _dilate(part != 0) & (part == 0)
        boundary[:SAFE_MARGIN] = boundary[-SAFE_MARGIN:] = False
        boundary[:, :SAFE_MARGIN] = boundary[:, -SAFE_MARGIN:] = False
        for y, x in np.argwhere(boundary & (noise > 0.45 - strength * 0.3)):
            neighbors = [(ny, nx) for ny in range(max(0, y - 1), min(48, y + 2)) for nx in range(max(0, x - 1), min(48, x + 2)) if part[ny, nx] != 0]
            if not neighbors:
                continue
            ny, nx = min(neighbors, key=lambda point: (abs(point[0] - y) + abs(point[1] - x), point))
            part[y, x], material[y, x], emission[y, x] = part[ny, nx], material[ny, nx], emission[ny, nx]
            provenance[y, x] = 3
            changed += 1
    elif mode == "scar":
        visible = part != 0
        diagonal = np.abs((np.arange(48)[:, None] - np.arange(48)[None, :]) - ((seed % 13) - 6)) <= max(0, strength - 1)
        for y, x in np.argwhere(visible & diagonal):
            part[y, x], material[y, x], emission[y, x] = (11, 8, 1)
            provenance[y, x] = 3
            changed += 1
    elif mode == "bilateral_break":
        side = np.indices((48, 48))[1] > 23
        eligible = side & np.isin(part, (4, 5, 6, 7, 8, 11, 15)) & (noise > 0.15 - strength * 0.2)
        for y, x in np.argwhere(eligible):
            mirror_x = 47 - x
            if part[y, mirror_x] != 0:
                part[y, x], material[y, x], emission[y, x] = part[y, mirror_x], material[y, mirror_x], emission[y, mirror_x]
                provenance[y, x] = 3
                changed += 1
    else:
        raise ValueError(f"unsupported mutation mode {mode!r}")
    return changed


def _guide(part: np.ndarray, parent_a: RepairSourceSample, parent_b: RepairSourceSample, donor_shift: tuple[int, int]) -> np.ndarray:
    result = (0.55 * parent_a.guide + 0.45 * _shift(parent_b.guide, donor_shift[0], donor_shift[1])).astype(np.float32)
    visible = part != 0
    result[0] = visible.astype(np.float32)
    result[1] = np.isin(part, (1, 2, 3, 10)).astype(np.float32)
    result[5] = (part == 10).astype(np.float32)
    x = np.linspace(0.0, 1.0, 48, dtype=np.float32)
    result[6] = np.broadcast_to(x[None, :], (48, 48))
    points = np.argwhere(visible)
    if len(points):
        cy, cx = points.mean(axis=0)
        yy, xx = np.mgrid[0:48, 0:48]
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        result[7] = np.clip(1.0 - distance / 34.0, 0.0, 1.0).astype(np.float32)
    return result


def fuse_specimen(
    parent_a: RepairSourceSample,
    parent_b: RepairSourceSample,
    *,
    seed: int,
    fusion_mode: str,
    mutation_mode: str,
    mutation_strength: int = 2,
    dominant_parent: str = "a",
) -> FusionSpecimen:
    if parent_a.sample_id == parent_b.sample_id:
        raise ValueError("fusion parents must be distinct")
    if fusion_mode not in FUSION_MODES or mutation_mode not in MUTATION_MODES:
        raise ValueError("fusion or mutation mode is not canonical")
    if dominant_parent not in {"a", "b"} or not 0 <= mutation_strength <= 3:
        raise ValueError("fusion dominance/strength contract failed")
    base, donor = (parent_a, parent_b) if dominant_parent == "a" else (parent_b, parent_a)
    mirror = bool((seed >> 7) & 1)
    dp, dm, de = donor.part_owner.copy(), donor.material.copy(), donor.emission_level.copy()
    if mirror:
        dp, dm, de = _mirror_fields(dp, dm, de)
    base_center = _centroid(np.isin(base.part_owner, (1, 10)))
    donor_center = _centroid(np.isin(dp, (1, 10)))
    dx = int(np.clip(round(base_center[0] - donor_center[0]), -5, 5))
    dy = int(np.clip(round(base_center[1] - donor_center[1]), -5, 5))
    dp, dm, de = _shift(dp, dx, dy), _shift(dm, dx, dy), _shift(de, dx, dy)

    part = base.part_owner.copy()
    material = base.material.copy()
    emission = base.emission_level.copy()
    provenance = np.where(part != 0, 1 if dominant_parent == "a" else 2, 0).astype(np.uint8)
    donor_provenance = 2 if dominant_parent == "a" else 1
    y, x = np.mgrid[0:48, 0:48]
    noise = _noise(seed)
    donor_visible = dp != 0
    protected = np.isin(part, (1, 3, 10)) & (noise < 1.1)
    if fusion_mode == "chimera":
        selector = donor_visible & (np.isin(dp, (8, 9, 11, 12, 15, 16)) | ((x > 23) & np.isin(dp, (4, 5, 6, 7))))
    elif fusion_mode == "mosaic":
        selector = donor_visible & (noise > -0.05) & ~protected
    elif fusion_mode == "symbiote":
        selector = donor_visible & (np.isin(dp, (8, 11, 12, 15, 16)) | ((noise > 0.65) & ~protected))
    elif fusion_mode == "bipolar":
        axis = (x - 23.5) * math.cos((seed % 360) * math.pi / 180.0) + (y - 23.5) * math.sin((seed % 360) * math.pi / 180.0)
        selector = donor_visible & (axis + 2.0 * noise > 0.0) & ~protected
    else:
        selector = donor_visible & ((noise > -0.45) | np.isin(dp, (8, 9, 12, 16))) & ~protected
    selector[:SAFE_MARGIN] = selector[-SAFE_MARGIN:] = False
    selector[:, :SAFE_MARGIN] = selector[:, -SAFE_MARGIN:] = False
    part[selector], material[selector], emission[selector] = dp[selector], dm[selector], de[selector]
    provenance[selector] = donor_provenance

    visible = part != 0
    visible[:SAFE_MARGIN] = visible[-SAFE_MARGIN:] = False
    visible[:, :SAFE_MARGIN] = visible[:, -SAFE_MARGIN:] = False
    part[~visible] = material[~visible] = emission[~visible] = provenance[~visible] = 0
    legal = np.ascontiguousarray(parent_a.legal_tuples, dtype=np.uint8)
    changed = _mutate(part, material, emission, provenance, legal, mutation_mode, mutation_strength, seed)
    repairs = _connect(part, material, emission, provenance, legal)

    tuples = np.stack((part, material, emission), axis=-1).reshape(-1, 3)
    legal_set = {tuple(map(int, row)) for row in legal}
    if any(tuple(map(int, row)) not in legal_set for row in tuples):
        raise ValueError("fusion produced a tuple outside the authoritative neural vocabulary")
    components = _components(part != 0)
    occupancy = float((part != 0).mean())
    contribution_a = int((provenance == 1).sum())
    contribution_b = int((provenance == 2).sum())
    if len(components) != 1 or not 0.02 <= occupancy <= 0.60 or min(contribution_a, contribution_b) < 20:
        raise ValueError("fusion failed connectivity, occupancy, or ancestry contribution gates")
    child_seed = int(seed & 0x7FFFFFFF)
    dominant = base
    condition = StyleCondition(
        sample_id="fusion_pending",
        ordinal=0,
        sample_seed=child_seed,
        morphology_id=dominant.family_id,
        morphology_name=FAMILIES[dominant.family_id],
        subtype_id=dominant.subtype_id,
        subtype_name=SUBTYPE_NAMES[dominant.subtype_id],
        role_id=(parent_a.role_id if ((seed >> 3) & 1) == 0 else parent_b.role_id),
        role_name=ROLE_NAMES[parent_a.role_id if ((seed >> 3) & 1) == 0 else parent_b.role_id],
    )
    lineage = {
        "parent_a": parent_a.sample_id,
        "parent_b": parent_b.sample_id,
        "dominant_parent": dominant_parent,
        "fusion_mode": fusion_mode,
        "mutation_mode": mutation_mode,
        "mutation_strength": mutation_strength,
        "mirror_donor": mirror,
        "seed": seed,
        "fields_sha256": aligned_fields_hash(part, material, emission),
    }
    lineage_sha = sha256_bytes(canonical_json_bytes(lineage))
    specimen_id = f"fx_{parent_a.ordinal:02d}_{parent_b.ordinal:02d}_{lineage_sha[:12]}"
    condition = StyleCondition(
        sample_id=specimen_id,
        ordinal=0,
        sample_seed=child_seed,
        morphology_id=condition.morphology_id,
        morphology_name=condition.morphology_name,
        subtype_id=condition.subtype_id,
        subtype_name=condition.subtype_name,
        role_id=condition.role_id,
        role_name=condition.role_name,
    )
    genes = np.clip(0.5 * parent_a.genes + 0.5 * parent_b.genes + ((_noise(seed).reshape(-1)[:24] * 0.015)), 0.0, 1.0).astype(np.float32)
    guide = _guide(part, parent_a, parent_b, (dx, dy))
    fields_hash = aligned_fields_hash(part, material, emission)
    metrics = {
        "visible_pixels": int((part != 0).sum()),
        "occupancy": round(occupancy, 9),
        "component_count": len(components),
        "parent_a_pixels": contribution_a,
        "parent_b_pixels": contribution_b,
        "mutation_pixels": int((provenance == 3).sum()),
        "connective_repair_pixels": int((provenance == 4).sum()),
        "mutation_operations": changed,
        "connective_repair_operations": repairs,
        "unique_tuples": int(len(np.unique(tuples, axis=0))),
        "margin": SAFE_MARGIN,
        "provenance_names": list(PROVENANCE_NAMES),
    }
    return FusionSpecimen(
        genome=FusionGenome(
            specimen_id=specimen_id,
            seed=seed,
            parent_a_ordinal=parent_a.ordinal,
            parent_b_ordinal=parent_b.ordinal,
            parent_a_sample_id=parent_a.sample_id,
            parent_b_sample_id=parent_b.sample_id,
            dominant_parent=dominant_parent,
            fusion_mode=fusion_mode,
            mutation_mode=mutation_mode,
            mutation_strength=mutation_strength,
            mirror_donor=mirror,
            condition=condition,
            lineage_sha256=lineage_sha,
        ),
        part_owner=_readonly(part, np.uint8),
        material=_readonly(material, np.uint8),
        emission_level=_readonly(emission, np.uint8),
        provenance=_readonly(provenance, np.uint8),
        guide=_readonly(guide, np.float32),
        genes=_readonly(genes, np.float32),
        legal_tuples=_readonly(legal, np.uint8),
        fields_sha256=fields_hash,
        provenance_sha256=array_sha256("fusion_provenance", provenance),
        metrics=metrics,
    )
