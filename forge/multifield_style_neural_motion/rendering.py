from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from PIL import Image

from ..multifield_style import render_layers
from ..multifield_style.model import CategoricalFields, StyleCondition
from ..multifield_style_motion.hashing import (
    array_sha256,
    canonical_json_bytes,
    categorical_sha256,
    sha256_bytes,
)
from ..multifield_style_motion.model import LAYER_NAMES
from ..multifield_style_motion.rendering import chebyshev_ring
from ..neural_rig_bridge import NeuralMotionFrame
from ..neural_rig_bridge.hashing import aligned_fields_hash
from .model import NeuralPresentationFrame


def _composite(layers: list[np.ndarray]) -> np.ndarray:
    image = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    for values in layers:
        image = Image.alpha_composite(image, Image.fromarray(values))
    return np.asarray(image, dtype=np.uint8).copy()


def _apply_presentation_pulse(
    layers: dict[str, np.ndarray],
    palette: Mapping[str, Any],
    emission: np.ndarray,
    pulse: int,
) -> None:
    """Pulse existing emissive support without adding a categorical pixel."""

    if not 0 <= pulse <= 3:
        raise ValueError("Neural motion emission pulse must be in [0, 3]")
    if pulse == 0:
        return
    core = layers["emission_core"]
    palette_levels = palette["effects"]["emission_levels"]
    for source_level in (1, 2, 3):
        active = (core[..., 3] > 0) & (emission == source_level)
        level = min(3, source_level + pulse)
        core[active, :3] = np.asarray(palette_levels[level - 1], dtype=np.uint8)
    for name, gain in (("bloom_r1", 24), ("bloom_r2", 13), ("aura", 10)):
        alpha = layers[name][..., 3]
        active = alpha > 0
        promoted = np.minimum(244, alpha.astype(np.uint16) + pulse * gain)
        alpha[active] = promoted[active].astype(np.uint8)
    layers["composite"] = _composite(
        [
            layers["bloom_r2"],
            layers["bloom_r1"],
            layers["aura"],
            layers["outline"],
            layers["base"],
            layers["emission_core"],
        ]
    )


def _effect_margin(mask: np.ndarray) -> int:
    points = np.argwhere(mask)
    if not len(points):
        return 48
    return int(
        min(
            points[:, 0].min(),
            points[:, 1].min(),
            47 - points[:, 0].max(),
            47 - points[:, 1].max(),
        )
    )


def render_neural_motion_frame(
    frame: NeuralMotionFrame,
    condition: StyleCondition,
    identity_fields_sha256: str,
    expected_palette: Mapping[str, Any],
    expected_palette_sha256: str,
) -> NeuralPresentationFrame:
    fields = frame.fields
    part = fields.part_owner
    material = fields.material
    emission = fields.emission_level
    categorical_before = categorical_sha256(part, material, emission)
    aligned_before = aligned_fields_hash(part, material, emission)
    driver_before = fields.manifest["driver_index_sha256"]
    bound_before = fields.sha256
    motion_before = frame.sha256
    rendered = render_layers(
        CategoricalFields(
            part=part,
            material=material,
            emission=emission,
            aligned_sha256=identity_fields_sha256,
        ),
        condition,
    )
    layers = {
        "base": rendered.base.copy(),
        "outline": rendered.outline.copy(),
        "emission_core": rendered.emission_core.copy(),
        "aura": rendered.aura.copy(),
        "bloom_r1": rendered.bloom_r1.copy(),
        "bloom_r2": rendered.bloom_r2.copy(),
        "composite": rendered.composite.copy(),
    }
    palette = dict(rendered.palette)
    palette_bytes = canonical_json_bytes(palette)
    palette_sha = sha256_bytes(palette_bytes)
    if palette != dict(expected_palette) or palette_sha != expected_palette_sha256:
        raise ValueError("Neural motion palette diverged from the static neural style parent")
    static_support = {name: layers[name][..., 3] > 0 for name in LAYER_NAMES[:-1]}
    static_alpha = {name: layers[name][..., 3].copy() for name in ("aura", "bloom_r1", "bloom_r2")}
    _apply_presentation_pulse(layers, palette, emission, frame.emission_pulse)
    categorical_after = categorical_sha256(part, material, emission)
    aligned_after = aligned_fields_hash(part, material, emission)
    body = (part != 0) & (part != 16)
    emission_source = (part != 0) & (emission > 0)
    expected_outline = chebyshev_ring(body, 1)
    expected_bloom_r1 = chebyshev_ring(emission_source, 1) if emission_source.any() else np.zeros_like(body)
    expected_bloom_r2 = chebyshev_ring(emission_source, 2) if emission_source.any() else np.zeros_like(body)
    pulse_monotonic = all(
        np.all(layers[name][..., 3].astype(np.uint16) >= static_alpha[name].astype(np.uint16))
        for name in static_alpha
    )
    support_unchanged = all(
        np.array_equal(layers[name][..., 3] > 0, support)
        for name, support in static_support.items()
    )
    gates = {
        "categorical_fields_unchanged": categorical_before == categorical_after,
        "aligned_fields_unchanged": aligned_before == aligned_after,
        "bound_frame_authority_unchanged": fields.sha256 == bound_before,
        "motion_frame_authority_unchanged": frame.sha256 == motion_before,
        "driver_authority_unchanged": fields.manifest["driver_index_sha256"] == driver_before,
        "native_rgba_layers": all(
            values.shape == (48, 48, 4) and values.dtype == np.uint8
            for values in layers.values()
        ),
        "categorical_body_alpha_exact": bool(np.array_equal(layers["base"][..., 3] > 0, body)),
        "outline_radius_1_exact": bool(np.array_equal(layers["outline"][..., 3] > 0, expected_outline)),
        "bloom_radius_1_exact": bool(np.array_equal(layers["bloom_r1"][..., 3] > 0, expected_bloom_r1)),
        "bloom_radius_2_exact": bool(np.array_equal(layers["bloom_r2"][..., 3] > 0, expected_bloom_r2)),
        "effect_rings_unclipped": _effect_margin(emission_source) >= 2,
        "pulse_support_unchanged": support_unchanged,
        "pulse_alpha_monotonic": pulse_monotonic,
        "effects_partial_alpha": all(
            not np.any(layers[name][..., 3] == 255)
            for name in ("aura", "bloom_r1", "bloom_r2")
        ),
        "palette_matches_static_parent": True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"Neural motion presentation frame failed gates: {failed}")
    return NeuralPresentationFrame(
        layers=layers,
        palette=palette,
        palette_sha256=palette_sha,
        categorical_sha256=categorical_before,
        aligned_fields_sha256=aligned_before,
        bound_frame_sha256=bound_before,
        motion_frame_sha256=motion_before,
        presentation_sha256=tuple(array_sha256(name, layers[name]) for name in LAYER_NAMES),
        gates=gates,
    )
