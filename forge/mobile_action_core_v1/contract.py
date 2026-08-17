from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-mobile-action-core-v1/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
TEACHER = PROJECT_ROOT / "outputs/monolithic_world_model_v1/production_002/runtime.pt"
TEACHER_SHA256 = "eab44d442b0b33d82a3ba156a234815169e5e7ac688815d4cbab3253d9dd255f"
LATENT_ROOT = PROJECT_ROOT / "outputs/world_action_natural_v10/corpus_v1_6world"
TRAJECTORY_ROOT = PROJECT_ROOT / "outputs/action_teacher_natural_v4/production_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mobile_action_core_v1/production_001"
SOURCE_FILES = (
    "forge/mobile_action_core_v1/__init__.py",
    "forge/mobile_action_core_v1/__main__.py",
    "forge/mobile_action_core_v1/contract.py",
    "forge/mobile_action_core_v1/data.py",
    "forge/mobile_action_core_v1/training.py",
    "forge/mobile_action_core_v1/promotion.py",
)


@dataclass(frozen=True, slots=True)
class MobileActionConfig:
    width: int = 304
    layers: int = 7
    heads: int = 4
    patch: int = 4


@dataclass(frozen=True, slots=True)
class MobileActionPlan:
    steps: int = 4500
    batch_size: int = 8
    learning_rate: float = 2e-4
    ema_decay: float = .995
    seed: int = 0x4D4F42494C454143


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-mobile-action-core-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def config_dict(value: object) -> dict[str, object]: return asdict(value)
