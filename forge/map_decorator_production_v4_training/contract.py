from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from ..map_decorator.hashing import json_sha256
from ..map_decorator_production_v4.contract import V4_CONTRACT_SHA256


V4_TRAINING_CONTRACT_NAME: Final[str] = "nullvector-map-decorator-v4-residual-training"
V4_TRAINING_CONTRACT_VERSION: Final[str] = "1.0.0"


@dataclass(frozen=True, slots=True)
class ResidualLossConfig:
    variant_weight: float = 1.0
    emission_weight: float = 1.0
    proposal_presence_weight: float = 2.0
    proposal_type_weight: float = 1.0
    proposal_count_weight: float = 1.0
    residual_regularization_weight: float = 0.05
    extra_proposal_weight: float = 4.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(isinstance(value, bool) or not 0.0 <= float(value) <= 100.0 for value in values.values()):
            raise ValueError("V4 residual loss weights must be finite values in [0,100].")
        if self.proposal_presence_weight <= 0 or self.proposal_type_weight <= 0 or self.proposal_count_weight <= 0:
            raise ValueError("V4 proposal presence/type/count supervision cannot be disabled.")
        if self.extra_proposal_weight < 1.0:
            raise ValueError("Extra-proposal suppression weight cannot be below one.")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResidualTrainingConfig:
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    corruption_min: float = 0.35
    corruption_max: float = 0.95
    ema_decay: float = 0.999
    seed: int = 0x44D3CA11
    full_mask_stride: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate <= 0.01 or not 0 <= self.weight_decay <= 1:
            raise ValueError("V4 residual optimizer configuration is outside its bounded contract.")
        if not 0 < self.corruption_min <= self.corruption_max <= 1:
            raise ValueError("V4 residual corruption curriculum is invalid.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("V4 residual EMA decay must be in [0,1).")
        if isinstance(self.seed, bool) or not 0 <= self.seed < (1 << 63):
            raise ValueError("V4 residual seed must be unsigned 63-bit.")
        if isinstance(self.full_mask_stride, bool) or not 1 <= self.full_mask_stride <= 16:
            raise ValueError("V4 residual full-mask stride must be in [1,16].")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def v4_training_contract_manifest() -> dict[str, object]:
    return {
        "contract_name": V4_TRAINING_CONTRACT_NAME,
        "contract_version": V4_TRAINING_CONTRACT_VERSION,
        "v4_substrate_sha256": V4_CONTRACT_SHA256,
        "loss": ResidualLossConfig().to_dict(),
        "training": ResidualTrainingConfig().to_dict(),
        "authority": {
            "proposal_fields_are_immutable_inputs": True,
            "target_fields_never_generate_proposals": True,
            "off_proposal_object_decode_impossible": True,
            "procedural_baseline_reported_separately": True,
        },
        "acceptance": {
            "zero_legality_topology_or_provenance_failures": True,
            "trained_raw_and_ema_compared_to_untrained_baseline": True,
            "no_object_head_may_regress_below_v4_baseline": True,
            "unchanged_validation_and_test_quality_gates": True,
        },
        "safety": {
            "cpu_foundation_only": True,
            "cuda_calibration_authorized": False,
            "runtime_integration_authorized": False,
        },
    }


V4_TRAINING_CONTRACT_SHA256: Final[str] = json_sha256(v4_training_contract_manifest())
