from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
from PIL import Image

from ..morphology.constants import MATERIAL_NAMES
from .model import AURA_PART_ID, CategoricalFields, IMAGE_SIZE, RenderedLayers, StyleCondition
from .palette import palette_for_condition


LAYER_FORMAT = "nullvector-multifield-style-layers-v1"


def dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    if active.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Chebyshev dilation mask must be native 48px")
    if not 0 <= radius <= 8:
        raise ValueError("Chebyshev dilation radius must be between zero and eight")
    if radius == 0:
        return active.copy()
    padded = np.pad(active, radius, mode="constant", constant_values=False)
    output = np.zeros_like(active)
    for offset_y in range(2 * radius + 1):
        for offset_x in range(2 * radius + 1):
            output |= padded[
                offset_y : offset_y + IMAGE_SIZE,
                offset_x : offset_x + IMAGE_SIZE,
            ]
    return output


def chebyshev_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        raise ValueError("Ring radius must be positive")
    outer = dilate_chebyshev(mask, radius)
    inner = dilate_chebyshev(mask, radius - 1)
    return outer & ~inner


def _rgba_layer(mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> np.ndarray:
    output = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
    output[mask, :3] = np.asarray(color, dtype=np.uint8)
    output[mask, 3] = np.uint8(alpha)
    return output


def _composite(layers: list[np.ndarray]) -> np.ndarray:
    image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
    for values in layers:
        image = Image.alpha_composite(image, Image.fromarray(values))
    return np.asarray(image, dtype=np.uint8).copy()


def _style_seed(condition: StyleCondition, fields_sha256: str) -> int:
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-render-seed-v1\0")
    digest.update(str(condition.sample_seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(fields_sha256.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def _shade_indices(
    part: np.ndarray,
    body: np.ndarray,
    condition: StyleCondition,
    seed: int,
) -> np.ndarray:
    yy, xx = np.indices(part.shape, dtype=np.int32)
    seed_term = int(seed & 0x7FFFFFFF)
    pattern = (xx * 5 + yy * 3 + part.astype(np.int32) * 7 + seed_term) % 13
    shades = np.ones_like(part, dtype=np.uint8)
    shades[pattern <= 2] = 0
    shades[pattern >= 10] = 2

    family = condition.morphology_id
    if family == 0:  # humanoid: axial panels catch the key light
        active_x = xx[body]
        center = int(math.floor(float(np.median(active_x)))) if active_x.size else IMAGE_SIZE // 2
        shades[body & (np.abs(xx - center) <= 1) & ((yy + seed_term) % 4 != 0)] = 2
    elif family == 1:  # animalian: broken diagonal dorsal bands
        shades[body & ((xx + yy + seed_term) % 7 == 0)] = 2
        shades[body & ((xx + yy + seed_term) % 7 == 4)] = 0
    elif family == 2:  # plantlike: uneven growth rings
        radial = (xx - IMAGE_SIZE // 2) ** 2 + (yy - IMAGE_SIZE // 2) ** 2
        shades[body & ((radial + seed_term) % 11 <= 1)] = 2
    elif family == 3:  # anomaly: phase checker
        phase = ((xx ^ yy ^ seed_term) & 3)
        shades[body & (phase == 0)] = 2
        shades[body & (phase == 3)] = 0
    else:  # machine: panel grid
        shades[body & (((xx + seed_term) % 5 == 0) | ((yy + seed_term // 7) % 6 == 0))] = 0
        shades[body & ((xx + 2 * yy + seed_term) % 11 == 0)] = 2
    return shades


def _accent_mask(body: np.ndarray, condition: StyleCondition, seed: int) -> np.ndarray:
    result = np.zeros_like(body, dtype=bool)
    points = np.argwhere(body)
    if len(points) == 0:
        return result
    center_x = float(points[:, 1].mean())
    choose_right = bool(seed & 1)
    ranked: list[tuple[int, int, int]] = []
    for y_value, x_value in points:
        y, x = int(y_value), int(x_value)
        on_selected_side = x > center_x if choose_right else x < center_x
        if not on_selected_side:
            continue
        rank = (
            x * 73856093
            ^ y * 19349663
            ^ condition.role_id * 83492791
            ^ int(seed & 0xFFFFFFFF)
        ) & 0xFFFFFFFF
        ranked.append((rank, y, x))
    if not ranked:
        ranked = [(0, int(points[0, 0]), int(points[0, 1]))]
    ranked.sort()
    limit = max(1, min(len(ranked), int(math.ceil(int(body.sum()) * 0.025))))
    for _, y, x in ranked[:limit]:
        result[y, x] = True
    return result


def render_layers(
    fields: CategoricalFields,
    condition: StyleCondition,
) -> RenderedLayers:
    """Compile categorical fields into presentation-only native pixel layers.

    The input arrays are read-only and are never copied back or rewritten.
    Part-owner id 16 is explicitly treated as categorical aura/effect support,
    never as opaque body or collision presentation.
    """

    part, material, emission = fields.part, fields.material, fields.emission
    body = (part != 0) & (part != AURA_PART_ID)
    categorical_aura = part == AURA_PART_ID
    if not body.any():
        raise ValueError("Style compilation requires at least one non-aura body pixel")
    if np.any(body & (material == 0)):
        raise ValueError("Opaque body pixels cannot use the void material")

    palette = palette_for_condition(condition, fields.aligned_sha256)
    seed = _style_seed(condition, fields.aligned_sha256)
    shade_indices = _shade_indices(part, body, condition, seed)
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
    for material_id, material_name in enumerate(MATERIAL_NAMES):
        if material_id == 0:
            continue
        active = body & (material == material_id)
        material_palette = palette["materials"][material_name]
        for shade_id, shade_name in enumerate(("shadow", "mid", "highlight")):
            shade_mask = active & (shade_indices == shade_id)
            base[shade_mask, :3] = np.asarray(material_palette[shade_name], dtype=np.uint8)
    base[body, 3] = 255

    accents = _accent_mask(body, condition, seed)
    base[accents, :3] = np.asarray(palette["effects"]["role_accent"], dtype=np.uint8)
    base[accents, 3] = 255

    outline_mask = chebyshev_ring(body, 1)
    outline = _rgba_layer(
        outline_mask,
        tuple(palette["effects"]["outline_shadow"]),
        255,
    )
    yy, xx = np.indices(body.shape, dtype=np.int32)
    chromatic = outline_mask & ((xx * 3 + yy * 5 + int(seed & 0xFFFF)) % 9 <= 1)
    outline[chromatic, :3] = np.asarray(
        palette["effects"]["outline_chromatic"], dtype=np.uint8
    )

    core_support = body & (emission > 0)
    emission_core = np.zeros_like(base)
    for level in (1, 2, 3):
        active = core_support & (emission == level)
        emission_core[active, :3] = np.asarray(
            palette["effects"]["emission_levels"][level - 1], dtype=np.uint8
        )
        emission_core[active, 3] = 255

    emission_effect_source = (part != 0) & (emission > 0)
    bloom_r1_mask = chebyshev_ring(emission_effect_source, 1) if emission_effect_source.any() else np.zeros_like(body)
    bloom_r2_mask = chebyshev_ring(emission_effect_source, 2) if emission_effect_source.any() else np.zeros_like(body)
    maximum_level = int(emission[emission_effect_source].max()) if emission_effect_source.any() else 0
    bloom_r1 = _rgba_layer(
        bloom_r1_mask,
        tuple(palette["effects"]["bloom"]),
        44 + maximum_level * 20,
    )
    bloom_r2 = _rgba_layer(
        bloom_r2_mask,
        tuple(palette["effects"]["bloom"]),
        18 + maximum_level * 9,
    )

    body_ring_1 = chebyshev_ring(body, 1)
    body_ring_2 = chebyshev_ring(body, 2)
    generated_aura = (body_ring_1 | body_ring_2) if emission_effect_source.any() else np.zeros_like(body)
    aura_mask = (categorical_aura | generated_aura) & ~body
    aura = _rgba_layer(aura_mask, tuple(palette["effects"]["aura"]), 52)
    aura[body_ring_1 & aura_mask, 3] = 88
    aura[categorical_aura, 3] = np.where(
        emission[categorical_aura] > 0,
        112,
        72,
    ).astype(np.uint8)

    composite = _composite(
        [bloom_r2, bloom_r1, aura, outline, base, emission_core]
    )
    masks: Mapping[str, np.ndarray] = {
        "body": body,
        "categorical_aura": categorical_aura,
        "outline": outline_mask,
        "emission_core": core_support,
        "emission_effect_source": emission_effect_source,
        "bloom_r1": bloom_r1_mask,
        "bloom_r2": bloom_r2_mask,
        "aura": aura_mask,
        "aura_allowed": categorical_aura | body_ring_1 | body_ring_2,
    }
    return RenderedLayers(
        base=base,
        outline=outline,
        emission_core=emission_core,
        aura=aura,
        bloom_r1=bloom_r1,
        bloom_r2=bloom_r2,
        composite=composite,
        palette=palette,
        masks=masks,
        accent_pixels=int(accents.sum()),
    )
