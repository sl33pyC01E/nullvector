from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.contract import HEAD_CLASS_COUNTS, ModelConfig


V3_CONTRACT_NAME: Final[str] = "nullvector-map-decorator-sparse-locator-v3"
V3_CONTRACT_VERSION: Final[str] = "3.0.0"
V3_CHECKPOINT_FORMAT_VERSION: Final[str] = "3.0.0"


@dataclass(frozen=True, slots=True)
class LocatorModelConfig:
    base_channels: int = 48
    condition_channels: int = 96
    residual_blocks_per_scale: int = 1
    padding_multiple: int = 4
    locator_channels: int = 32
    locator_blocks: int = 2
    count_hidden_channels: int = 32
    presence_bias_init: float = -3.0
    count_prior: float = 4.0
    maximum_objects_per_head: int = 256

    def __post_init__(self) -> None:
        ModelConfig(
            base_channels=self.base_channels,
            condition_channels=self.condition_channels,
            residual_blocks_per_scale=self.residual_blocks_per_scale,
            padding_multiple=self.padding_multiple,
        )
        integer_values = (
            self.locator_channels,
            self.locator_blocks,
            self.count_hidden_channels,
            self.maximum_objects_per_head,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise TypeError("Locator architecture dimensions must be integers.")
        if not 4 <= self.locator_channels <= 256 or not 1 <= self.locator_blocks <= 8:
            raise ValueError("Locator tower dimensions are outside their bounded contract.")
        if not 4 <= self.count_hidden_channels <= 256:
            raise ValueError("Count-head width is outside its bounded contract.")
        if not 1 <= self.maximum_objects_per_head <= 4096:
            raise ValueError("Maximum object quota is outside its bounded contract.")
        if isinstance(self.presence_bias_init, bool) or not -12.0 <= self.presence_bias_init <= 0.0:
            raise ValueError("presence_bias_init must be in [-12,0].")
        if isinstance(self.count_prior, bool) or not 0.0 <= self.count_prior <= 256.0:
            raise ValueError("count_prior must be in [0,256].")

    def core_config(self) -> ModelConfig:
        return ModelConfig(
            base_channels=self.base_channels,
            condition_channels=self.condition_channels,
            residual_blocks_per_scale=self.residual_blocks_per_scale,
            padding_multiple=self.padding_multiple,
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocatorLossConfig:
    variant_weight: float = 1.0
    emission_weight: float = 1.0
    object_presence_weight: float = 1.5
    object_ranking_weight: float = 1.0
    object_type_weight: float = 1.0
    object_categorical_weight: float = 0.25
    object_count_weight: float = 1.25
    object_exclusivity_weight: float = 0.25
    hard_negative_ratio: int = 6
    halo_radius: int = 3
    halo_target: float = 0.15
    ranking_margin: float = 1.0

    def __post_init__(self) -> None:
        weights = (
            self.variant_weight,
            self.emission_weight,
            self.object_presence_weight,
            self.object_ranking_weight,
            self.object_type_weight,
            self.object_categorical_weight,
            self.object_count_weight,
            self.object_exclusivity_weight,
        )
        if any(isinstance(value, bool) or not 0.0 <= value <= 100.0 for value in weights):
            raise ValueError("Loss weights must be finite values in [0,100].")
        if self.object_presence_weight <= 0 or self.object_ranking_weight <= 0 or self.object_count_weight <= 0:
            raise ValueError("Presence, ranking, and independent count supervision cannot be disabled.")
        if isinstance(self.hard_negative_ratio, bool) or not 1 <= self.hard_negative_ratio <= 64:
            raise ValueError("hard_negative_ratio must be in [1,64].")
        if isinstance(self.halo_radius, bool) or not 1 <= self.halo_radius <= 8:
            raise ValueError("halo_radius must be in [1,8].")
        if isinstance(self.halo_target, bool) or not 0.0 < self.halo_target < 0.5:
            raise ValueError("halo_target must be in (0,0.5).")
        if isinstance(self.ranking_margin, bool) or not 0.0 < self.ranking_margin <= 8.0:
            raise ValueError("ranking_margin must be in (0,8].")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocatorTrainingConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    corruption_min: float = 0.35
    corruption_max: float = 0.95
    ema_decay: float = 0.999
    seed: int = 0x10CA71E3
    full_mask_stride: int = 2

    def __post_init__(self) -> None:
        numeric = (
            self.learning_rate,
            self.weight_decay,
            self.corruption_min,
            self.corruption_max,
            self.ema_decay,
        )
        if any(isinstance(value, bool) for value in (*numeric, self.seed, self.full_mask_stride)):
            raise TypeError("Training configuration fields cannot be booleans.")
        if not 0 < self.learning_rate <= 0.01 or not 0 <= self.weight_decay <= 1:
            raise ValueError("Optimizer configuration is outside its bounded contract.")
        if not 0 < self.corruption_min <= self.corruption_max <= 1:
            raise ValueError("Corruption curriculum is invalid.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("EMA decay must be in [0,1).")
        if not 0 <= self.seed < (1 << 63):
            raise ValueError("Training seed must be unsigned 63-bit.")
        if not 1 <= self.full_mask_stride <= 16:
            raise ValueError("full_mask_stride must be in [1,16].")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def v3_contract_manifest() -> dict[str, object]:
    return {
        "contract_name": V3_CONTRACT_NAME,
        "contract_version": V3_CONTRACT_VERSION,
        "checkpoint_format_version": V3_CHECKPOINT_FORMAT_VERSION,
        "output_heads": dict(HEAD_CLASS_COUNTS),
        "model": LocatorModelConfig().to_dict(),
        "loss": LocatorLossConfig().to_dict(),
        "training": LocatorTrainingConfig().to_dict(),
        "object_factorization": {
            "presence": "spatial context tower ranks legal cells",
            "type": "foreground-only categorical logits",
            "count": "independent pooled log1p count head",
            "decode": "count quota then stable legal top-k with cross-head exclusion",
        },
        "localization_supervision": {
            "exact_foreground": True,
            "bounded_halo": True,
            "positive_vs_hard-negative_ranking": True,
            "count_decoupled_from_probability_sum": True,
        },
        "authority": {
            "corpus": "map-decorator-production-v1 topology-v2 semantic corpus",
            "foreground_index": "foreground-index-v2",
            "mutates_authority": False,
        },
        "safety": {
            "cpu_foundation_only": True,
            "cuda_calibration_authorized": False,
            "godot_integration_authorized": False,
            "production_claim": False,
        },
    }


V3_CONTRACT_SHA256: Final[str] = json_sha256(v3_contract_manifest())
