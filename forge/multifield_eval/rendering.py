from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..morphology.constants import CANVAS_SIZE, MATERIAL_NAMES, SAFETY_MARGIN
from ..morphology.genome import genome_from_seed
from ..morphology.render import palette_for_genome
from ..safety import require_disk_floor
from .conditions import ConditionRecord


RENDER_FORMAT = "nullvector-multifield-raster-v1"
POSTPROCESS_FORMAT = "nullvector-bounded-field-postprocess-v1"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_fields_hash(
    part: np.ndarray, material: np.ndarray, emission: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-aligned-fields-v1\0")
    for name, values in (
        ("part", part),
        ("material", material),
        ("emission", emission),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def palette_for_condition(
    corpus_seed: int, record: ConditionRecord
) -> dict[str, tuple[int, int, int, int]]:
    genome = genome_from_seed(int(corpus_seed), record.morphology)
    genome = replace(
        genome,
        silhouette_variant=record.subtype % 4,
        subtype_id=record.subtype,
        role_id=record.role,
    )
    return palette_for_genome(genome)


def _mix(
    first: np.ndarray, second: np.ndarray, amount: np.ndarray | float
) -> np.ndarray:
    return first * (1.0 - amount) + second * amount


def fields_to_rgba(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    palette: Mapping[str, tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Render aligned categorical fields without changing their silhouette."""

    expected = (CANVAS_SIZE, CANVAS_SIZE)
    for values, name in (
        (part, "part"),
        (material, "material"),
        (emission, "emission"),
    ):
        if values.shape != expected or values.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 {expected}")

    base = {
        0: (0, 0, 0, 0),
        1: palette["organic"],       # tissue
        2: palette["primary"],       # chitin
        3: palette["organic"],       # bark
        4: palette["secondary"],     # alloy
        5: palette["primary"],       # armor
        6: palette["secondary"],     # weapon
        7: palette["hot"],           # organ
        8: palette["secondary"],     # marking
        9: palette["hot"],           # energy
    }
    if len(base) != len(MATERIAL_NAMES):
        raise RuntimeError("Material render table disagrees with vocabulary")
    rgb = np.zeros((*expected, 3), dtype=np.float32)
    for material_id, color in base.items():
        rgb[material == material_id] = np.asarray(color[:3], dtype=np.float32)

    # Ownership remains visually legible even when materials match.  The small
    # fixed modulation is categorical, deterministic, and never changes alpha.
    modulation = 0.82 + (part.astype(np.float32) % 5.0) * 0.045
    rgb *= modulation[..., None]
    hot = np.asarray(palette["hot"][:3], dtype=np.float32)
    emission_amount = emission.astype(np.float32) / 3.0
    rgb = _mix(rgb, hot, (emission_amount * 0.82)[..., None])
    visible = part != 0
    rgba = np.zeros((*expected, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(visible, 255, 0).astype(np.uint8)

    glow = np.zeros_like(rgba)
    glow[..., :3] = np.asarray(palette["hot"][:3], dtype=np.uint8)
    glow[..., 3] = np.rint(emission_amount * 255.0).astype(np.uint8)
    glow[~visible] = 0
    return rgba, glow


def _components(mask: np.ndarray) -> list[np.ndarray]:
    active = mask.astype(bool)
    visited = np.zeros_like(active)
    groups: list[np.ndarray] = []
    height, width = active.shape
    for y in range(height):
        for x in range(width):
            if not active[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            points: list[tuple[int, int]] = []
            while stack:
                py, px = stack.pop()
                points.append((py, px))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = py + dy, px + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and active[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            groups.append(np.asarray(points, dtype=np.int16))
    return groups


def bounded_postprocess_fields(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    *,
    max_delta_fraction: float = 0.03,
    maximum_speckle_pixels: int = 3,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Remove only tiny detached/unsafe speckles under an explicit pixel budget.

    No pixel is synthesized and no surviving tuple is rewritten.  Raw fields
    are always the authoritative neural output; this optional cleanup is kept
    separate and its exact delta is recorded.
    """

    if not 0.0 <= max_delta_fraction <= 0.10:
        raise ValueError("max_delta_fraction must be between 0 and 0.10")
    if maximum_speckle_pixels < 0:
        raise ValueError("maximum_speckle_pixels cannot be negative")
    result = [np.array(values, dtype=np.uint8, copy=True) for values in (part, material, emission)]
    visible = result[0] != 0
    budget = int(np.floor(visible.size * max_delta_fraction))
    candidates: list[tuple[int, int, int]] = []
    groups = _components(visible)
    largest_index = int(np.argmax([len(group) for group in groups])) if groups else -1
    for component_index, group in enumerate(groups):
        if component_index == largest_index or len(group) > maximum_speckle_pixels:
            continue
        for y, x in group:
            unsafe = (
                int(y) < SAFETY_MARGIN
                or int(y) >= CANVAS_SIZE - SAFETY_MARGIN
                or int(x) < SAFETY_MARGIN
                or int(x) >= CANVAS_SIZE - SAFETY_MARGIN
            )
            candidates.append((0 if unsafe else 1, int(y), int(x)))
    candidates.sort()
    selected = candidates[:budget]
    for _, y, x in selected:
        result[0][y, x] = 0
        result[1][y, x] = 0
        result[2][y, x] = 0
    changed = (
        (result[0] != part) | (result[1] != material) | (result[2] != emission)
    )
    report = {
        "format": POSTPROCESS_FORMAT,
        "policy": "remove-detached-speckles-only",
        "max_delta_fraction": float(max_delta_fraction),
        "maximum_speckle_pixels": int(maximum_speckle_pixels),
        "pixel_budget": budget,
        "candidate_pixels": len(candidates),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "raw_fields_sha256": aligned_fields_hash(part, material, emission),
        "processed_fields_sha256": aligned_fields_hash(*result),
    }
    if report["changed_fraction"] > max_delta_fraction + 1.0e-12:
        raise RuntimeError("Postprocess exceeded its explicit delta budget")
    return (result[0], result[1], result[2]), report


def save_png_atomic(path: Path, pixels: np.ndarray) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, planned_bytes=max(int(pixels.nbytes * 2), 256 * 1024))
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        Image.fromarray(pixels).save(temporary, format="PNG", optimize=False)
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Temporary PNG is empty")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    planned = sum(int(np.asarray(values).nbytes) for values in arrays.values()) * 2
    require_disk_floor(path.parent, planned_bytes=max(planned, 1024 * 1024))
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Temporary NPZ is empty")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_contact_sheet(
    images: Sequence[np.ndarray],
    records: Sequence[ConditionRecord],
    *,
    validation: Sequence[Mapping[str, Any]] | None = None,
    columns: int = 8,
    scale: int = 3,
) -> Image.Image:
    if len(images) != len(records) or not images:
        raise ValueError("Contact sheet needs one non-empty image per record")
    if validation is not None and len(validation) != len(images):
        raise ValueError("Validation count must match images")
    if columns <= 0 or scale <= 0:
        raise ValueError("columns and scale must be positive")
    cell_width = CANVAS_SIZE * scale
    label_height = 28
    cell_height = cell_width + label_height
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGBA", (columns * cell_width, rows * cell_height), (2, 4, 10, 255)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (pixels, record) in enumerate(zip(images, records)):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sprite = Image.fromarray(pixels).resize(
            (cell_width, cell_width), Image.Resampling.NEAREST
        )
        sheet.alpha_composite(sprite, (x, y))
        accepted = validation[index].get("accepted", False) if validation else None
        marker = "OK" if accepted else ("NO" if accepted is not None else "--")
        line1 = f"{record.sample_id} {marker}"
        line2 = f"{record.morphology}:{record.subtype % 4}  role {record.role}"
        draw.rectangle(
            (x, y + cell_width, x + cell_width - 1, y + cell_height - 1),
            fill=(2, 4, 10, 255),
        )
        draw.text((x + 3, y + cell_width + 2), line1, fill=(225, 244, 255, 255))
        draw.text((x + 3, y + cell_width + 14), line2, fill=(116, 222, 255, 255))
    return sheet

