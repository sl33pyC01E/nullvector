from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-creature-stage-manipulation-arena/1.0.0"
CONTROLLER = PROJECT_ROOT / "outputs/creature_stage_neural_grasper_v1/production_v3_physical_feeder/runtime.pt"
CONTROLLER_SHA256 = "cd550ae1d75140555b4f51d4d27271da7012f9dfc3c4b99f85eca5b21e8b3e50"
LIMB_POSE_CONTROLLER = PROJECT_ROOT / "outputs/creature_stage_neural_limb_pose_v1/production_2400_catalog/runtime.pt"
LIMB_POSE_CONTROLLER_SHA256 = "ffeddc23501700d5d98b62b5b0ca6b7352ff75705e14fe17050d4ffc9d01ad78"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_controller() -> None:
    if file_sha256(CONTROLLER) != CONTROLLER_SHA256:
        raise ValueError("accepted neural grasper runtime drifted")


def assert_limb_pose_controller() -> None:
    if file_sha256(LIMB_POSE_CONTROLLER) != LIMB_POSE_CONTROLLER_SHA256:
        raise ValueError("accepted neural limb pose runtime drifted")
