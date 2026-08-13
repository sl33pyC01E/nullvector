from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Final

import numpy as np

from ..map_decorator.catalog import (
    CATALOG_SHA256,
    EMISSION_CLASS_COUNT,
    MAX_DECAL_CLASSES,
    MAX_PROP_CLASSES,
    VARIANT_CLASS_COUNT,
)
from ..map_decorator.contract import FEATURE_CONTRACT_SHA256
from ..map_decorator.features import EncodedFeatures
from ..maps.model import THEMES
from ..map_decorator.hashing import json_sha256


MODEL_CONTRACT_NAME: Final[str] = "nullvector-topology-locked-categorical-refiner"
MODEL_CONTRACT_VERSION: Final[str] = "0.1.0"
CHECKPOINT_FORMAT_VERSION: Final[str] = "1.0.0"
TEACHER_PROJECTION_VERSION: Final[str] = "map-art-legal-projection-v1"
SPLIT_POLICY_VERSION: Final[str] = "canonical-map-identity-sha256-80-10-10-v1"
GLOBAL_CONDITION_DIM: Final[int] = 8
FEATURE_CHANNEL_COUNT: Final[int] = 53
HEAD_CLASS_COUNTS: Final[dict[str, int]] = {
    "variant": VARIANT_CLASS_COUNT,
    "decal": MAX_DECAL_CLASSES,
    "prop": MAX_PROP_CLASSES,
    "emission": EMISSION_CLASS_COUNT,
}
HEAD_NAMES: Final[tuple[str, ...]] = tuple(HEAD_CLASS_COUNTS)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    base_channels: int = 48
    condition_channels: int = 96
    residual_blocks_per_scale: int = 1
    padding_multiple: int = 4

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            for value in (
                self.base_channels,
                self.condition_channels,
                self.residual_blocks_per_scale,
                self.padding_multiple,
            )
        ):
            raise TypeError("ModelConfig integer fields cannot be booleans.")
        if not 4 <= self.base_channels <= 128:
            raise ValueError("base_channels must be in [4, 128].")
        if not 8 <= self.condition_channels <= 256:
            raise ValueError("condition_channels must be in [8, 256].")
        if not 1 <= self.residual_blocks_per_scale <= 4:
            raise ValueError("residual_blocks_per_scale must be in [1, 4].")
        if self.padding_multiple != 4:
            raise ValueError("This two-scale U-Net requires padding_multiple=4.")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def model_contract_manifest() -> dict[str, object]:
    return {
        "contract_name": MODEL_CONTRACT_NAME,
        "contract_version": MODEL_CONTRACT_VERSION,
        "feature_channels": FEATURE_CHANNEL_COUNT,
        "state_encoding": {
            "categorical_one_hot_channels": sum(HEAD_CLASS_COUNTS.values()),
            "masked_indicator_channels": len(HEAD_CLASS_COUNTS),
            "refinement_level_channel": 1,
            "masked_categories_have_zero_one_hot": True,
        },
        "global_condition_dim": GLOBAL_CONDITION_DIM,
        "global_condition_order": [
            "width_div_256",
            "height_div_256",
            "log2_aspect_div_3",
            "sqrt_area_div_256",
            "map_seed_low32_unit",
            "map_seed_high32_unit",
            "feature_seed_low32_unit",
            "feature_seed_high32_unit",
        ],
        "theme_order": list(THEMES),
        "heads": dict(HEAD_CLASS_COUNTS),
        "object_decoding": "single categorical choice over empty + decal nonempty + prop nonempty",
        "emission_decoding": "legality recomputed after selected variant/decal/prop",
        "shape_policy": "right-bottom pad to multiple of four; exact top-left crop; no resampling",
        "supported_height_width": [32, 256],
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "teacher_projection_version": TEACHER_PROJECTION_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
    }


MODEL_CONTRACT_SHA256: Final[str] = json_sha256(model_contract_manifest())


def _seed_pair(seed: int) -> tuple[np.float32, np.float32]:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("Seed conditions must be integers.")
    value = int(seed)
    if not 0 <= value < (1 << 64):
        raise ValueError("Seed conditions must be unsigned 64-bit integers.")
    denominator = np.float64((1 << 32) - 1)
    return (
        np.float32((value & 0xFFFF_FFFF) / denominator),
        np.float32((value >> 32) / denominator),
    )


def global_condition_vector(encoded: EncodedFeatures) -> np.ndarray:
    """Encode only versioned global metadata; never inspect RGB or target fields."""
    conditions = encoded.global_conditions
    required = {
        "map_width",
        "map_height",
        "aspect_ratio",
        "map_public_seed",
        "feature_public_seed",
        "feature_contract_sha256",
        "catalog_sha256",
    }
    missing = sorted(required - set(conditions))
    if missing:
        raise ValueError(f"Encoded global conditions are incomplete: {missing}")
    if conditions["feature_contract_sha256"] != FEATURE_CONTRACT_SHA256:
        raise ValueError("Feature contract hash does not match the model contract.")
    if conditions["catalog_sha256"] != CATALOG_SHA256:
        raise ValueError("Catalog hash does not match the model contract.")
    width = int(conditions["map_width"])
    height = int(conditions["map_height"])
    aspect = float(conditions["aspect_ratio"])
    if not (32 <= width <= 256 and 32 <= height <= 256 and math.isfinite(aspect) and aspect > 0):
        raise ValueError("Global map dimensions/aspect are outside the supported contract.")
    map_low, map_high = _seed_pair(int(conditions["map_public_seed"]))
    feature_low, feature_high = _seed_pair(int(conditions["feature_public_seed"]))
    vector = np.asarray(
        [
            width / 256.0,
            height / 256.0,
            float(np.clip(math.log2(aspect) / 3.0, -1.0, 1.0)),
            math.sqrt(width * height) / 256.0,
            map_low,
            map_high,
            feature_low,
            feature_high,
        ],
        dtype=np.float32,
    )
    if vector.shape != (GLOBAL_CONDITION_DIM,) or not bool(np.isfinite(vector).all()):
        raise RuntimeError("Global condition encoding produced an invalid vector.")
    return np.ascontiguousarray(vector)
