from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-mobile-frame-decoder-v1/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
TEACHER = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v3/production_v1/runtime.pt"
TEACHER_SHA256 = "03f3e147e1e4007aa01c063cf2cfc8717f169dc4974a7914e78d389a00d0d872"
LATENT_SHARDS = PROJECT_ROOT / "outputs/world_action_natural_v10/corpus_v1_6world/shards"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mobile_frame_decoder_v1/production_001"
SOURCE_FILES = (
    "forge/mobile_frame_decoder_v1/__init__.py",
    "forge/mobile_frame_decoder_v1/__main__.py",
    "forge/mobile_frame_decoder_v1/contract.py",
    "forge/mobile_frame_decoder_v1/model.py",
    "forge/mobile_frame_decoder_v1/training.py",
)


@dataclass(frozen=True, slots=True)
class MobileDecoderConfig:
    latent_channels: int = 48
    widths: tuple[int, int, int, int] = (160, 128, 96, 64)


@dataclass(frozen=True, slots=True)
class MobileDecoderPlan:
    steps: int = 3000
    batch_size: int = 4
    learning_rate: float = 3e-4
    ema_decay: float = .997
    seed: int = 0x4D4F42494C455641


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-mobile-frame-decoder-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def config_dict(value: object) -> dict[str, object]:
    result = asdict(value)
    if "widths" in result: result["widths"] = list(result["widths"])
    return result
