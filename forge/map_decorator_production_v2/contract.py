from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.contract import HEAD_CLASS_COUNTS, ModelConfig


V2_CONTRACT_NAME: Final[str] = "nullvector-map-decorator-foreground-factored-v2"
V2_CONTRACT_VERSION: Final[str] = "2.0.0"
V2_CHECKPOINT_FORMAT_VERSION: Final[str] = "2.0.0"
V2_INDEX_FORMAT_VERSION: Final[str] = "2.0.0"
DISK_FLOOR_GIB: Final[float] = 100.0
MAX_INDEX_WORKERS: Final[int] = 2
MAX_PROCESS_ATTEMPTS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class FactoredModelConfig:
    base_channels: int = 48
    condition_channels: int = 96
    residual_blocks_per_scale: int = 1
    padding_multiple: int = 4
    presence_bias_init: float = -4.0

    def __post_init__(self) -> None:
        ModelConfig(
            base_channels=self.base_channels,
            condition_channels=self.condition_channels,
            residual_blocks_per_scale=self.residual_blocks_per_scale,
            padding_multiple=self.padding_multiple,
        )
        if isinstance(self.presence_bias_init, bool) or not -12.0 <= self.presence_bias_init <= 0.0:
            raise ValueError("presence_bias_init must be a finite float in [-12,0].")

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
class FactoredLossConfig:
    variant_weight: float = 1.0
    emission_weight: float = 1.0
    object_presence_weight: float = 2.0
    object_type_weight: float = 1.0
    object_categorical_weight: float = 0.25
    object_count_weight: float = 0.50
    object_exclusivity_weight: float = 0.25
    hard_negative_ratio: int = 4

    def __post_init__(self) -> None:
        numeric = (
            self.variant_weight,
            self.emission_weight,
            self.object_presence_weight,
            self.object_type_weight,
            self.object_categorical_weight,
            self.object_count_weight,
            self.object_exclusivity_weight,
        )
        if any(isinstance(value, bool) or not 0.0 <= value <= 100.0 for value in numeric):
            raise ValueError("Loss weights must be finite floats in [0,100].")
        if self.object_presence_weight <= 0 or self.object_type_weight <= 0:
            raise ValueError("Presence and within-type losses may not be disabled.")
        if isinstance(self.hard_negative_ratio, bool) or not 1 <= self.hard_negative_ratio <= 64:
            raise ValueError("hard_negative_ratio must be an integer in [1,64].")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ForegroundPatchConfig:
    patch_size: int = 48
    batch_size: int = 4
    decal_slots: int = 2
    prop_slots: int = 2
    jitter_radius: int = 8

    def __post_init__(self) -> None:
        values = (self.patch_size, self.batch_size, self.decal_slots, self.prop_slots, self.jitter_radius)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("Foreground patch configuration fields must be integers.")
        if not 32 <= self.patch_size <= 128:
            raise ValueError("patch_size must remain in [32,128].")
        if not 2 <= self.batch_size <= 64:
            raise ValueError("batch_size must remain in [2,64].")
        if self.decal_slots < 1 or self.prop_slots < 1:
            raise ValueError("Every batch must reserve at least one decal and one prop focus slot.")
        if self.decal_slots + self.prop_slots != self.batch_size:
            raise ValueError("Foreground focus quotas must exactly exhaust the batch.")
        if not 0 <= self.jitter_radius <= self.patch_size // 2:
            raise ValueError("jitter_radius exceeds the bounded patch contract.")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class V2TrainingConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    corruption_min: float = 0.35
    corruption_max: float = 0.95
    ema_decay: float = 0.999
    seed: int = 0xDEC0A7E2
    precision: str = "bf16"

    def __post_init__(self) -> None:
        numeric = (
            self.learning_rate,
            self.weight_decay,
            self.corruption_min,
            self.corruption_max,
            self.ema_decay,
        )
        if any(isinstance(value, bool) for value in (*numeric, self.seed)):
            raise TypeError("Training configuration numeric fields cannot be booleans.")
        if not 0 < self.learning_rate <= 0.01:
            raise ValueError("learning_rate must be in (0,0.01].")
        if not 0 <= self.weight_decay <= 1:
            raise ValueError("weight_decay must be in [0,1].")
        if not 0 < self.corruption_min <= self.corruption_max <= 1:
            raise ValueError("Corruption bounds are invalid.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1).")
        if not 0 <= self.seed < (1 << 63):
            raise ValueError("seed must be unsigned 63-bit.")
        if self.precision != "bf16":
            raise ValueError("The eventual CUDA calibration contract is BF16-only.")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


CALIBRATION_GATES: Final[dict[str, object]] = {
    "stage": "pre-production-calibration",
    "required_splits": ["validation", "test"],
    "hard_legality": 1.0,
    "immutable_semantic_changes": 0,
    "source_provenance_failures": 0,
    "heads": {
        "decal": {
            "foreground_macro_iou_min": 0.02,
            "foreground_f1_min": 0.05,
            "rare_class_recall_min": 0.10,
            "foreground_density_ratio": [0.25, 4.0],
            "every_target_foreground_class_predicted": True,
        },
        "prop": {
            "foreground_macro_iou_min": 0.02,
            "foreground_f1_min": 0.05,
            "rare_class_recall_min": 0.10,
            "foreground_density_ratio": [0.25, 4.0],
            "every_target_foreground_class_predicted": True,
        },
    },
    "historical_v1_100step_reference": {
        "held_out": {
            "decal": {"foreground_macro_iou": 0.014706692817661647, "foreground_f1": 0.04732526356891839},
            "prop": {"foreground_macro_iou": 0.008033322283042328, "foreground_f1": 0.016449314142466833},
        },
        "sentinel": {
            "decal": {"foreground_macro_iou": 0.009136864629702934, "foreground_f1": 0.02681841349819218},
            "prop": {"foreground_macro_iou": 0.00688339104972111, "foreground_f1": 0.011783179161280001},
        },
        "note": "v2 minima exceed the best v1 calibration IoU/F1 while density gates prevent recall-by-flooding",
    },
}


PRODUCTION_GATES: Final[dict[str, object]] = {
    "stage": "production",
    "required_splits": ["validation", "test"],
    "hard_legality": 1.0,
    "immutable_semantic_changes": 0,
    "source_provenance_failures": 0,
    "heads": {
        "variant": {
            "foreground_macro_iou_min": 0.06,
            "foreground_f1_min": 0.125,
            "rare_class_recall_min": 0.12,
        },
        "decal": {
            "foreground_macro_iou_min": 0.08,
            "foreground_f1_min": 0.15,
            "rare_class_recall_min": 0.20,
            "foreground_density_ratio": [0.50, 2.0],
            "every_target_foreground_class_predicted": True,
        },
        "prop": {
            "foreground_macro_iou_min": 0.08,
            "foreground_f1_min": 0.15,
            "rare_class_recall_min": 0.20,
            "foreground_density_ratio": [0.50, 2.0],
            "every_target_foreground_class_predicted": True,
        },
        "emission": {
            "foreground_macro_iou_min": 0.55,
            "foreground_f1_min": 0.85,
            "rare_class_recall_min": 0.45,
            "foreground_density_ratio": [0.50, 2.0],
            "every_target_foreground_class_predicted": True,
        },
    },
}


def v2_contract_manifest() -> dict[str, object]:
    return {
        "contract_name": V2_CONTRACT_NAME,
        "contract_version": V2_CONTRACT_VERSION,
        "checkpoint_format_version": V2_CHECKPOINT_FORMAT_VERSION,
        "index_format_version": V2_INDEX_FORMAT_VERSION,
        "output_heads": dict(HEAD_CLASS_COUNTS),
        "object_factorization": {
            "heads": ["decal", "prop"],
            "presence": "learned binary logit",
            "type": "foreground-only categorical distribution",
            "composition": "log P(empty)=logsigmoid(-presence); log P(type)=logsigmoid(presence)+log_softmax(type)",
        },
        "loss": FactoredLossConfig().to_dict(),
        "patch_quota": ForegroundPatchConfig().to_dict(),
        "training": V2TrainingConfig().to_dict(),
        "calibration_gates": CALIBRATION_GATES,
        "production_gates": PRODUCTION_GATES,
        "safety": {
            "disk_floor_gib": DISK_FLOOR_GIB,
            "max_index_workers": MAX_INDEX_WORKERS,
            "max_process_attempts": MAX_PROCESS_ATTEMPTS,
            "no_godot_integration_without_production_gate": True,
        },
    }


V2_CONTRACT_SHA256: Final[str] = json_sha256(v2_contract_manifest())
