from __future__ import annotations

from pathlib import Path

from ..config import PROJECT_ROOT
from ..creature_stage_manipulation_v1.contract import CONTROLLER_SHA256, file_sha256


FORMAT = "nullvector-nature-neural-feeding/1.0.0"
CONTROLLER = PROJECT_ROOT / "outputs/creature_stage_neural_grasper_v1/production_v3_physical_feeder/runtime.pt"


def assert_runtime() -> None:
    if file_sha256(Path(CONTROLLER)) != CONTROLLER_SHA256:
        raise ValueError("accepted neural grasper runtime drifted")
