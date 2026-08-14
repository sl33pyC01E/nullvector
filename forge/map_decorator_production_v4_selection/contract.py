from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from ..map_decorator.hashing import json_sha256
from ..map_decorator_production_v4_calibration.contract import V4_CALIBRATION_CONTRACT_SHA256


AUDIT_FORMAT: Final[str] = "nullvector-map-decorator-v4-protected-selection-audit/1.0.0"


@dataclass(frozen=True, slots=True)
class ProtectedSelectionConfig:
    decal_classes: tuple[int, ...] = (2,)
    prop_classes: tuple[int, ...] = ()
    restore_only_into_empty: bool = True
    preserve_cross_head_noncollision: bool = True

    def __post_init__(self) -> None:
        for name, values in (("decal", self.decal_classes), ("prop", self.prop_classes)):
            if not isinstance(values, tuple) or len(values) != len(set(values)):
                raise ValueError(f"Protected {name} classes must be a unique tuple.")
            if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2 for value in values):
                raise ValueError(f"Protected {name} class IDs must be in [1,2].")
        if not self.decal_classes and not self.prop_classes:
            raise ValueError("At least one sparse object class must be protected.")
        if self.restore_only_into_empty is not True or self.preserve_cross_head_noncollision is not True:
            raise ValueError("Protected selection cannot overwrite a neural choice or create a collision.")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["decal_classes"] = list(self.decal_classes)
        value["prop_classes"] = list(self.prop_classes)
        return value


def selection_contract_manifest() -> dict[str, object]:
    return {
        "format": "nullvector-map-decorator-v4-protected-selection-contract/1.0.0",
        "calibration_contract_sha256": V4_CALIBRATION_CONTRACT_SHA256,
        "default_config": ProtectedSelectionConfig().to_dict(),
        "rule": {
            "source": "public proposal channel for a baseline-declared rare class",
            "destination_must_be_empty": True,
            "other_object_head_must_be_empty": True,
            "class_legality_rechecked": True,
            "hard_empty_excluded": True,
            "no_new_off_proposal_cells": True,
        },
        "acceptance": {
            "full_validation_and_test_splits": True,
            "all_object_metrics_nonregressing_vs_procedural_baseline": True,
            "all_object_metrics_nonregressing_vs_trained_ema": True,
            "at_least_one_strict_improvement_vs_each": True,
            "hard_legality_exact": True,
            "runtime_integration_permitted": False,
        },
    }


V4_SELECTION_CONTRACT_SHA256: Final[str] = json_sha256(selection_contract_manifest())
