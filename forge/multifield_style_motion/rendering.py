from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..multifield_style import render_layers
from .hashing import (
    array_sha256,
    authority_sha256,
    canonical_json_bytes,
    categorical_sha256,
    named_points_sha256,
    sha256_bytes,
)
from .model import (
    IMAGE_SIZE,
    JOINT_NAMES,
    LAYER_NAMES,
    SOCKET_NAMES,
    FrameAudit,
    IdentityStyleFields,
    MotionStyleCondition,
)


def dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    if active.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Motion presentation masks must be native 48px")
    if not 0 <= radius <= 4:
        raise ValueError("Motion presentation dilation radius must be in [0, 4]")
    if radius == 0:
        return active.copy()
    padded = np.pad(active, radius, mode="constant", constant_values=False)
    result = np.zeros_like(active)
    for offset_y in range(2 * radius + 1):
        for offset_x in range(2 * radius + 1):
            result |= padded[
                offset_y : offset_y + IMAGE_SIZE,
                offset_x : offset_x + IMAGE_SIZE,
            ]
    return result


def chebyshev_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        raise ValueError("Ring radius must be positive")
    return dilate_chebyshev(mask, radius) & ~dilate_chebyshev(mask, radius - 1)


def build_condition(source: Mapping[str, Any], family_ordinal: int) -> MotionStyleCondition:
    training = source["training_contract"]
    return MotionStyleCondition(
        sample_id=str(source["id"]),
        ordinal=int(family_ordinal),
        sample_seed=int(source["seed"]),
        morphology_id=int(training["morphology_index"]),
        morphology_name=str(source["family"]),
        subtype_id=int(training["subtype_id"]),
        subtype_name=str(training["subtype_name"]),
        role_id=int(training["role_id"]),
        role_name=str(training["role_name"]),
    )


def _effect_margin(mask: np.ndarray) -> int:
    points = np.argwhere(mask)
    if len(points) == 0:
        return IMAGE_SIZE
    return int(
        min(
            points[:, 0].min(),
            points[:, 1].min(),
            IMAGE_SIZE - 1 - points[:, 0].max(),
            IMAGE_SIZE - 1 - points[:, 1].max(),
        )
    )


def render_motion_frame(
    frame: Any,
    specimen: Any,
    condition: MotionStyleCondition,
    identity_sha256: str,
    *,
    expected_palette_sha256: str | None = None,
) -> FrameAudit:
    training = frame.training_fields(specimen)
    part = np.array(training.part_owner, dtype=np.uint8, copy=True)
    material = np.array(training.material, dtype=np.uint8, copy=True)
    emission = np.array(training.emission_level, dtype=np.uint8, copy=True)
    categorical_before = categorical_sha256(part, material, emission)
    joint_before = named_points_sha256("joints", JOINT_NAMES, frame.joints)
    socket_before = named_points_sha256("sockets", SOCKET_NAMES, frame.sockets)
    authority_before = authority_sha256(
        part,
        material,
        emission,
        JOINT_NAMES,
        frame.joints,
        SOCKET_NAMES,
        frame.sockets,
    )
    proxy = IdentityStyleFields(
        part=part,
        material=material,
        emission=emission,
        aligned_sha256=identity_sha256,
    )
    rendered = render_layers(proxy, condition)  # public presentation API
    categorical_after = categorical_sha256(part, material, emission)
    joint_after = named_points_sha256("joints", JOINT_NAMES, frame.joints)
    socket_after = named_points_sha256("sockets", SOCKET_NAMES, frame.sockets)
    authority_after = authority_sha256(
        part,
        material,
        emission,
        JOINT_NAMES,
        frame.joints,
        SOCKET_NAMES,
        frame.sockets,
    )

    body = (part != 0) & (part != 16)
    categorical_aura = part == 16
    emission_source = (part != 0) & (emission > 0)
    emission_core = body & (emission > 0)
    expected_outline = chebyshev_ring(body, 1)
    expected_bloom_r1 = (
        chebyshev_ring(emission_source, 1)
        if emission_source.any()
        else np.zeros_like(body)
    )
    expected_bloom_r2 = (
        chebyshev_ring(emission_source, 2)
        if emission_source.any()
        else np.zeros_like(body)
    )
    aura_allowed = categorical_aura | chebyshev_ring(body, 1) | chebyshev_ring(body, 2)
    arrays = {
        "base": rendered.base,
        "outline": rendered.outline,
        "emission_core": rendered.emission_core,
        "aura": rendered.aura,
        "bloom_r1": rendered.bloom_r1,
        "bloom_r2": rendered.bloom_r2,
        "composite": rendered.composite,
    }
    palette_payload = canonical_json_bytes(dict(rendered.palette))
    palette_sha = sha256_bytes(palette_payload)
    aura_alpha = rendered.aura[..., 3]
    bloom_alpha = np.concatenate(
        (
            rendered.bloom_r1[..., 3][rendered.bloom_r1[..., 3] > 0],
            rendered.bloom_r2[..., 3][rendered.bloom_r2[..., 3] > 0],
        )
    )
    gates = {
        "categorical_fields_unchanged": categorical_before == categorical_after,
        "rig_authority_unchanged": joint_before == joint_after,
        "socket_authority_unchanged": socket_before == socket_after,
        "combined_authority_unchanged": authority_before == authority_after,
        "native_rgba_layers": all(
            values.shape == (48, 48, 4) and values.dtype == np.uint8
            for values in arrays.values()
        ),
        "categorical_body_alpha_exact": bool(
            np.array_equal(rendered.base[..., 3] > 0, body)
        ),
        "categorical_aura_excluded_from_body": bool(
            not np.any((rendered.base[..., 3] > 0) & categorical_aura)
        ),
        "outline_radius_1_exact": bool(
            np.array_equal(rendered.outline[..., 3] > 0, expected_outline)
        ),
        "emission_core_support_exact": bool(
            np.array_equal(rendered.emission_core[..., 3] > 0, emission_core)
        ),
        "bloom_radius_1_exact": bool(
            np.array_equal(rendered.bloom_r1[..., 3] > 0, expected_bloom_r1)
        ),
        "bloom_radius_2_exact": bool(
            np.array_equal(rendered.bloom_r2[..., 3] > 0, expected_bloom_r2)
        ),
        "effect_rings_unclipped": _effect_margin(emission_source) >= 2,
        "aura_effect_support_bounded": bool(
            np.all(~(aura_alpha > 0) | aura_allowed)
            and not np.any((aura_alpha > 0) & body)
        ),
        "aura_partial_alpha": bool(
            not np.any(aura_alpha == 255)
        ),
        "bloom_partial_alpha": bool(
            bloom_alpha.size == 0 or int(bloom_alpha.max()) < 255
        ),
        "palette_identity_invariant": bool(
            expected_palette_sha256 is None or palette_sha == expected_palette_sha256
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"Motion presentation frame failed gates: {failed}")
    return FrameAudit(
        layers=arrays,
        palette=dict(rendered.palette),
        palette_sha256=palette_sha,
        categorical_sha256=categorical_before,
        joint_sha256=joint_before,
        socket_sha256=socket_before,
        authority_sha256=authority_before,
        presentation_sha256=tuple(
            array_sha256(name, arrays[name]) for name in LAYER_NAMES
        ),
        gates=gates,
    )
