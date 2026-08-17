from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-mobile-neural-ecology-v1/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
DEFAULT_CORPUS = PROJECT_ROOT / "outputs/mobile_ecology_v1/corpus_current.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mobile_ecology_v1/production_001"
SOURCE_FILES = (
    "forge/mobile_ecology_v1/__init__.py", "forge/mobile_ecology_v1/__main__.py",
    "forge/mobile_ecology_v1/contract.py", "forge/mobile_ecology_v1/model.py",
    "forge/mobile_ecology_v1/training.py",
)


@dataclass(frozen=True, slots=True)
class MobileEcologyConfig:
    self_width: int = 96
    token_width: int = 64
    hidden_width: int = 192

    def __post_init__(self) -> None:
        if self.self_width not in (64, 96, 128) or self.token_width not in (48, 64, 80): raise ValueError("mobile ecology encoder width drifted")
        if self.hidden_width not in (128, 160, 192, 224): raise ValueError("mobile ecology trunk width drifted")


@dataclass(frozen=True, slots=True)
class MobileEcologyPlan:
    steps: int = 1800
    batch_size: int = 768
    learning_rate: float = 4e-4
    ema_decay: float = .997
    seed: int = 0x4D4F42494C45434F

    def __post_init__(self) -> None:
        if not 100 <= self.steps <= 20_000 or not 64 <= self.batch_size <= 2048: raise ValueError("mobile ecology schedule drifted")
        if not 1e-5 <= self.learning_rate <= 2e-3 or not .9 <= self.ema_decay < 1: raise ValueError("mobile ecology optimizer drifted")


def canonical(value: object) -> bytes: return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
def config_dict(value: object) -> dict[str, object]: return asdict(value)  # type: ignore[arg-type]
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()
def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-mobile-neural-ecology-v1\0")
    for relative in SOURCE_FILES: digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
