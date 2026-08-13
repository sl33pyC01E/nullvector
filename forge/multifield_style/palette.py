from __future__ import annotations

import hashlib
from typing import Any

from ..morphology.constants import MATERIAL_NAMES
from .color import delta_e_oklab, oklch_to_srgb8, srgb8_to_oklab
from .model import StyleCondition


PALETTE_FORMAT = "nullvector-perceptual-palette-v1"
PALETTE_ALGORITHM = "bounded-oklch-srgb-fixed24-v1"

FAMILY_VOCABULARY: tuple[dict[str, Any], ...] = (
    {"name": "humanoid", "base_hue": 238.0, "chroma": 0.145, "motif": "axial-panel"},
    {"name": "animalian", "base_hue": 33.0, "chroma": 0.165, "motif": "dorsal-stripe"},
    {"name": "plantlike", "base_hue": 142.0, "chroma": 0.155, "motif": "growth-ring"},
    {"name": "anomaly", "base_hue": 306.0, "chroma": 0.185, "motif": "phase-checker"},
    {"name": "machine", "base_hue": 198.0, "chroma": 0.135, "motif": "panel-grid"},
)

ROLE_VOCABULARY: tuple[dict[str, Any], ...] = (
    {"name": "striker", "accent_hue": 17.0, "motif": "hot-edge"},
    {"name": "defender", "accent_hue": 244.0, "motif": "cool-plate"},
    {"name": "scout", "accent_hue": 91.0, "motif": "signal-dash"},
    {"name": "controller", "accent_hue": 285.0, "motif": "phase-node"},
    {"name": "support", "accent_hue": 157.0, "motif": "vital-node"},
    {"name": "artillery", "accent_hue": 49.0, "motif": "muzzle-band"},
    {"name": "harvester", "accent_hue": 119.0, "motif": "resource-mark"},
    {"name": "disruptor", "accent_hue": 330.0, "motif": "fault-line"},
)

# Every material has an independent perceptual identity. The base family hue
# rotates the set as a whole while these offsets and lightness bands retain
# cross-family material readability.
_MATERIAL_HUE_OFFSETS = (180.0, 0.0, 42.0, 84.0, 143.0, 191.0, 235.0, 292.0, 326.0, 18.0)
_MATERIAL_LIGHTNESS = (0.18, 0.61, 0.57, 0.53, 0.67, 0.55, 0.64, 0.72, 0.69, 0.79)
_MATERIAL_CHROMA_SCALE = (0.25, 0.88, 0.95, 0.72, 0.42, 0.92, 0.72, 0.97, 1.05, 1.12)
_RAMP_LIGHTNESS_DELTA = (-0.13, 0.0, 0.105)


def _seed_for(condition: StyleCondition, fields_sha256: str) -> int:
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-palette-seed-v1\0")
    digest.update(str(condition.sample_seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(fields_sha256.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def palette_for_condition(
    condition: StyleCondition,
    fields_sha256: str,
) -> dict[str, Any]:
    family = FAMILY_VOCABULARY[condition.morphology_id]
    role = ROLE_VOCABULARY[condition.role_id]
    seed = _seed_for(condition, fields_sha256)
    subtype_shift = (condition.subtype_id % 4 - 1.5) * 4.0
    seed_shift = float((seed >> 9) % 13) - 6.0
    family_hue = (float(family["base_hue"]) + subtype_shift + seed_shift) % 360.0
    family_chroma = float(family["chroma"])

    materials: dict[str, Any] = {}
    for index, name in enumerate(MATERIAL_NAMES):
        hue = (family_hue + _MATERIAL_HUE_OFFSETS[index]) % 360.0
        lightness = _MATERIAL_LIGHTNESS[index]
        chroma = max(0.018, family_chroma * _MATERIAL_CHROMA_SCALE[index])
        ramps = [
            list(oklch_to_srgb8(lightness + delta, chroma, hue))
            for delta in _RAMP_LIGHTNESS_DELTA
        ]
        materials[name] = {
            "id": index,
            "oklch": [round(lightness, 6), round(chroma, 6), round(hue, 6)],
            "shadow": ramps[0],
            "mid": ramps[1],
            "highlight": ramps[2],
        }

    role_hue = float(role["accent_hue"])
    outline_shadow = list(oklch_to_srgb8(0.235, 0.075, family_hue))
    outline_chroma = list(oklch_to_srgb8(0.455, 0.15, role_hue))
    role_mid = list(oklch_to_srgb8(0.71, 0.19, role_hue))
    role_hot = list(oklch_to_srgb8(0.84, 0.17, role_hue + 9.0))
    emission_levels = [
        list(oklch_to_srgb8(lightness, chroma, role_hue + 22.0))
        for lightness, chroma in ((0.60, 0.16), (0.75, 0.17), (0.89, 0.14))
    ]
    mids = [materials[name]["mid"] for name in MATERIAL_NAMES]
    separations = [
        delta_e_oklab(mids[first], mids[second])
        for first in range(len(mids))
        for second in range(first + 1, len(mids))
    ]
    emission_lightness = [srgb8_to_oklab(color)[0] for color in emission_levels]
    return {
        "format": PALETTE_FORMAT,
        "algorithm": PALETTE_ALGORITHM,
        "family": {
            "id": condition.morphology_id,
            "name": condition.morphology_name,
            "hue": round(family_hue, 6),
            "motif": family["motif"],
        },
        "role": {
            "id": condition.role_id,
            "name": condition.role_name,
            "accent_hue": role_hue,
            "motif": role["motif"],
        },
        "materials": materials,
        "effects": {
            "outline_shadow": outline_shadow,
            "outline_chromatic": outline_chroma,
            "role_accent": role_mid,
            "role_hot": role_hot,
            "emission_levels": emission_levels,
            "aura": list(oklch_to_srgb8(0.72, 0.145, role_hue + 18.0)),
            "bloom": list(oklch_to_srgb8(0.82, 0.13, role_hue + 27.0)),
        },
        "diagnostics": {
            "minimum_material_mid_delta_e_oklab": round(min(separations), 9),
            "emission_oklab_lightness": [round(value, 9) for value in emission_lightness],
        },
    }
