from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.contract import ModelConfig
from ..map_decorator_production_v2.contract import ForegroundPatchConfig
from ..map_decorator_production_v4.contract import ProposalLocatorConfig
from ..map_decorator_production_v4_training.contract import (
    ResidualLossConfig,
    ResidualTrainingConfig,
    V4_TRAINING_CONTRACT_SHA256,
)


CALIBRATION_FORMAT: Final[str] = "nullvector-map-decorator-v4-cuda-calibration/1.0.0"
SUPERVISOR_FORMAT: Final[str] = "nullvector-map-decorator-v4-calibration-supervisor/1.0.0"
OBJECT_METRICS: Final[tuple[str, ...]] = (
    "foreground_macro_iou",
    "foreground_f1",
    "rare_class_recall",
)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    steps: int = 100
    validation_batch_size: int = 4
    test_batch_size: int = 1
    core: ModelConfig = ModelConfig()
    locator: ProposalLocatorConfig = ProposalLocatorConfig()
    training: ResidualTrainingConfig = ResidualTrainingConfig()
    loss: ResidualLossConfig = ResidualLossConfig()
    patch: ForegroundPatchConfig = ForegroundPatchConfig()

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not 1 <= self.steps <= 1_000:
            raise ValueError("V4 calibration steps must be in [1,1000].")
        if not 1 <= self.validation_batch_size <= 16 or not 1 <= self.test_batch_size <= 4:
            raise ValueError("V4 calibration evaluation batch size is outside its safe bound.")

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "validation_batch_size": self.validation_batch_size,
            "test_batch_size": self.test_batch_size,
            "precision": "bf16",
            "core": self.core.to_dict(),
            "locator": self.locator.to_dict(),
            "training": self.training.to_dict(),
            "loss": self.loss.to_dict(),
            "patch": self.patch.to_dict(),
        }


def calibration_contract_manifest() -> dict[str, object]:
    return {
        "format": "nullvector-map-decorator-v4-calibration-contract/1.0.0",
        "training_contract_sha256": V4_TRAINING_CONTRACT_SHA256,
        "default_config": CalibrationConfig().to_dict(),
        "baseline": {
            "same_initial_model": True,
            "same_full_validation_and_test_splits": True,
            "same_public_proposals": True,
            "reported_before_optimizer_update": True,
        },
        "acceptance": {
            "hard_legality_and_provenance_exact": True,
            "raw_and_ema_each_nonregressing": True,
            "heads": ["decal", "prop"],
            "metrics": list(OBJECT_METRICS),
            "tolerance": 1.0e-7,
            "at_least_one_strict_object_metric_improvement": True,
            "no_runtime_integration": True,
        },
        "execution": {
            "cuda_bf16_only": True,
            "deterministic_algorithms": True,
            "isolated_worker": True,
            "maximum_attempts": 3,
            "disk_floor_gib": 100,
        },
    }


V4_CALIBRATION_CONTRACT_SHA256: Final[str] = json_sha256(calibration_contract_manifest())
