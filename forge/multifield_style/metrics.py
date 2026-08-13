from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .backgrounds import CROPS_PER_THEME
from .color import rgb_array_to_oklab
from .model import BackgroundCrop, CategoricalFields, RenderedLayers, StyleCondition
from .rendering import chebyshev_ring


METRICS_FORMAT = "nullvector-multifield-style-metrics-v1"
MAX_BASE_PALETTE_COLORS = 31
MAX_COMPOSITE_PALETTE_COLORS = 40
MAX_CLIPPED_WHITE_FRACTION = 0.015
MIN_MATERIAL_DELTA_E = 0.045
MIN_EMISSION_LIGHTNESS_STEP = 0.055
MIN_THEME_MEDIAN_CONTRAST_DELTA_E = 0.105
MIN_GLOBAL_P10_CONTRAST_DELTA_E = 0.050


def _support(layer: np.ndarray) -> np.ndarray:
    return np.asarray(layer[..., 3] > 0, dtype=bool)


def _component_sizes(mask: np.ndarray) -> list[int]:
    active = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(active)
    sizes: list[int] = []
    height, width = active.shape
    for y in range(height):
        for x in range(width):
            if not active[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            size = 0
            while stack:
                current_y, current_x = stack.pop()
                size += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        next_y, next_x = current_y + dy, current_x + dx
                        if (
                            0 <= next_y < height
                            and 0 <= next_x < width
                            and active[next_y, next_x]
                            and not visited[next_y, next_x]
                        ):
                            visited[next_y, next_x] = True
                            stack.append((next_y, next_x))
            sizes.append(size)
    return sizes


def _contrast_metrics(
    rendered: RenderedLayers,
    crops: Iterable[BackgroundCrop],
) -> dict[str, Any]:
    body = rendered.masks["body"]
    foreground_lab = rgb_array_to_oklab(rendered.composite[..., :3][body])
    by_theme: dict[str, list[np.ndarray]] = defaultdict(list)
    crop_counts: dict[str, int] = defaultdict(int)
    for crop in crops:
        background_lab = rgb_array_to_oklab(crop.rgb[body])
        distances = np.linalg.norm(foreground_lab - background_lab, axis=-1)
        by_theme[crop.theme].append(distances)
        crop_counts[crop.theme] += 1
    if not by_theme or any(count != CROPS_PER_THEME for count in crop_counts.values()):
        raise ValueError("Contrast metrics require the complete representative crop catalog")
    theme_metrics: dict[str, Any] = {}
    global_values: list[np.ndarray] = []
    for theme in sorted(by_theme):
        values = np.concatenate(by_theme[theme])
        global_values.append(values)
        theme_metrics[theme] = {
            "samples": int(values.size),
            "p10_delta_e_oklab": round(float(np.quantile(values, 0.10)), 9),
            "median_delta_e_oklab": round(float(np.median(values)), 9),
            "mean_delta_e_oklab": round(float(values.mean()), 9),
        }
    combined = np.concatenate(global_values)
    return {
        "themes": theme_metrics,
        "minimum_theme_median_delta_e_oklab": round(
            min(record["median_delta_e_oklab"] for record in theme_metrics.values()), 9
        ),
        "global_p10_delta_e_oklab": round(float(np.quantile(combined, 0.10)), 9),
        "global_median_delta_e_oklab": round(float(np.median(combined)), 9),
    }


def evaluate_style(
    fields: CategoricalFields,
    condition: StyleCondition,
    rendered: RenderedLayers,
    backgrounds: tuple[BackgroundCrop, ...],
    *,
    fields_hash_after_render: str,
) -> dict[str, Any]:
    masks = rendered.masks
    body = masks["body"]
    base_support = _support(rendered.base)
    outline_support = _support(rendered.outline)
    core_support = _support(rendered.emission_core)
    bloom_r1_support = _support(rendered.bloom_r1)
    bloom_r2_support = _support(rendered.bloom_r2)
    aura_support = _support(rendered.aura)
    expected_outline = chebyshev_ring(body, 1)
    emission_source = masks["emission_effect_source"]
    expected_bloom_r1 = chebyshev_ring(emission_source, 1) if emission_source.any() else np.zeros_like(body)
    expected_bloom_r2 = chebyshev_ring(emission_source, 2) if emission_source.any() else np.zeros_like(body)

    base_colors = np.unique(rendered.base[..., :3][body], axis=0)
    composite_visible = rendered.composite[..., 3] > 0
    composite_colors = np.unique(
        rendered.composite[..., :3][composite_visible], axis=0
    )
    clipped_white = np.all(rendered.composite[..., :3][body] >= 250, axis=1)
    clipped_white_fraction = float(clipped_white.mean()) if clipped_white.size else 0.0
    body_component_sizes = sorted(_component_sizes(body), reverse=True)
    body_components = len(body_component_sizes)
    detached_body_pixels = sum(body_component_sizes[1:])
    body_connected_through_categorical_aura = (
        len(_component_sizes(body | masks["categorical_aura"])) == 1
    )
    material_separation = float(
        rendered.palette["diagnostics"]["minimum_material_mid_delta_e_oklab"]
    )
    emission_lightness = [
        float(value) for value in rendered.palette["diagnostics"]["emission_oklab_lightness"]
    ]
    emission_steps = [
        emission_lightness[index + 1] - emission_lightness[index]
        for index in range(len(emission_lightness) - 1)
    ]
    contrast = _contrast_metrics(rendered, backgrounds)
    aura_alphas = rendered.aura[..., 3][aura_support]
    bloom_alphas = np.concatenate(
        (rendered.bloom_r1[..., 3][bloom_r1_support], rendered.bloom_r2[..., 3][bloom_r2_support])
    )
    source_points = np.argwhere(emission_source)
    if len(source_points):
        minimum_effect_margin = int(
            min(
                source_points[:, 0].min(),
                source_points[:, 1].min(),
                47 - source_points[:, 0].max(),
                47 - source_points[:, 1].max(),
            )
        )
    else:
        minimum_effect_margin = 48

    measurements = {
        "body_pixels": int(body.sum()),
        "categorical_aura_pixels": int(masks["categorical_aura"].sum()),
        "body_component_count_8_connected": body_components,
        "detached_body_pixels_8_connected": detached_body_pixels,
        "body_connected_through_categorical_aura": body_connected_through_categorical_aura,
        "outline_pixels": int(outline_support.sum()),
        "emission_core_pixels": int(core_support.sum()),
        "aura_pixels": int(aura_support.sum()),
        "bloom_radius_1_pixels": int(bloom_r1_support.sum()),
        "bloom_radius_2_pixels": int(bloom_r2_support.sum()),
        "accent_pixels": rendered.accent_pixels,
        "accent_fraction_of_body": round(rendered.accent_pixels / max(int(body.sum()), 1), 9),
        "base_palette_color_count": int(len(base_colors)),
        "composite_palette_color_count": int(len(composite_colors)),
        "clipped_white_fraction": round(clipped_white_fraction, 9),
        "minimum_material_mid_delta_e_oklab": round(material_separation, 9),
        "emission_oklab_lightness": [round(value, 9) for value in emission_lightness],
        "minimum_emission_lightness_step": round(min(emission_steps), 9),
        "maximum_aura_alpha": int(aura_alphas.max(initial=0)),
        "maximum_bloom_alpha": int(bloom_alphas.max(initial=0)),
        "minimum_effect_source_canvas_margin": minimum_effect_margin,
        "map_background_contrast": contrast,
    }

    dimensions_native = all(
        layer.shape == (48, 48, 4) and layer.dtype == np.uint8
        for layer in (
            rendered.base,
            rendered.outline,
            rendered.emission_core,
            rendered.aura,
            rendered.bloom_r1,
            rendered.bloom_r2,
            rendered.composite,
        )
    )
    gates = {
        "categorical_fields_hash_unchanged": fields_hash_after_render == fields.aligned_sha256,
        "native_48px_rgba": dimensions_native,
        "base_support_exact_non_aura_body": bool(np.array_equal(base_support, body)),
        "outline_exact_chebyshev_radius_1": bool(np.array_equal(outline_support, expected_outline)),
        "emission_core_support_exact": bool(np.array_equal(core_support, masks["emission_core"])),
        "bloom_radius_1_support_exact": bool(np.array_equal(bloom_r1_support, expected_bloom_r1)),
        "bloom_radius_2_support_exact": bool(np.array_equal(bloom_r2_support, expected_bloom_r2)),
        "effect_rings_unclipped": minimum_effect_margin >= 2,
        "aura_support_bounded": bool(np.all(~aura_support | masks["aura_allowed"])),
        "aura_does_not_become_body": bool(not np.any(aura_support & body)),
        "aura_partial_alpha_only": bool(aura_alphas.size == 0 or (aura_alphas.min() > 0 and aura_alphas.max() < 255)),
        "bloom_partial_alpha_only": bool(bloom_alphas.size == 0 or (bloom_alphas.min() > 0 and bloom_alphas.max() < 255)),
        # Categorical validity is authoritative for topology.  Aura is an
        # effect owner and is intentionally removed from the solid body; that
        # can expose a single terminal/ornament pixel which remains connected
        # in the accepted categorical visible union.  Permit at most one such
        # presentation-only singleton while still rejecting substantive body
        # islands or any detached multi-pixel component.  Require the removed
        # categorical aura to be the actual bridge, so an unrelated stray
        # singleton cannot use this exception.
        "no_body_islands": detached_body_pixels <= 1
        and all(size == 1 for size in body_component_sizes[1:])
        and body_connected_through_categorical_aura,
        "bounded_rgb_only_accents": bool(
            rendered.accent_pixels >= 1
            and rendered.accent_pixels <= math_ceil_2_5_percent(int(body.sum()))
            and np.array_equal(rendered.base[..., 3] > 0, body)
        ),
        "base_palette_size": 2 <= len(base_colors) <= MAX_BASE_PALETTE_COLORS,
        "composite_palette_size": 2 <= len(composite_colors) <= MAX_COMPOSITE_PALETTE_COLORS,
        "clipped_white_fraction": clipped_white_fraction <= MAX_CLIPPED_WHITE_FRACTION,
        "ten_distinct_material_ramps": bool(
            len(rendered.palette["materials"]) == 10
            and len({tuple(record["mid"]) for record in rendered.palette["materials"].values()}) == 10
            and material_separation >= MIN_MATERIAL_DELTA_E
        ),
        "emission_monotonicity": min(emission_steps) >= MIN_EMISSION_LIGHTNESS_STEP,
        "map_background_contrast": bool(
            contrast["minimum_theme_median_delta_e_oklab"]
            >= MIN_THEME_MEDIAN_CONTRAST_DELTA_E
            and contrast["global_p10_delta_e_oklab"]
            >= MIN_GLOBAL_P10_CONTRAST_DELTA_E
        ),
    }
    return {
        "format": METRICS_FORMAT,
        "sample_id": condition.sample_id,
        "thresholds": {
            "maximum_base_palette_colors": MAX_BASE_PALETTE_COLORS,
            "maximum_composite_palette_colors": MAX_COMPOSITE_PALETTE_COLORS,
            "maximum_clipped_white_fraction": MAX_CLIPPED_WHITE_FRACTION,
            "minimum_material_delta_e_oklab": MIN_MATERIAL_DELTA_E,
            "minimum_emission_lightness_step": MIN_EMISSION_LIGHTNESS_STEP,
            "minimum_theme_median_contrast_delta_e_oklab": MIN_THEME_MEDIAN_CONTRAST_DELTA_E,
            "minimum_global_p10_contrast_delta_e_oklab": MIN_GLOBAL_P10_CONTRAST_DELTA_E,
        },
        "measurements": measurements,
        "gates": gates,
        "passed": all(gates.values()),
    }


def math_ceil_2_5_percent(value: int) -> int:
    return max(1, (value * 25 + 999) // 1000)
