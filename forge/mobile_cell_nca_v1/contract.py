from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..config import PROJECT_ROOT


FORMAT = "nullvector-mobile-cell-nca-v1/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
CORPUS = PROJECT_ROOT / "outputs/cellular_nca/nca_v1/cellular_nca_corpus.npz"
CORPUS_SHA256 = "aa301e740b90cd6ce616340d2c939d7d1f8f97194979641830bfd534dd9ab2b4"
TEACHER = PROJECT_ROOT / "outputs/cellular_nca/nca_causal_v3_selected/runtime.pt"
TEACHER_SHA256 = "0548dda78eac08e088487a4cbf63a8cde010794215ae15b2dc31eb25729f1903"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mobile_cell_nca_v1/production_001"
SOURCE_FILES = (
    "forge/mobile_cell_nca_v1/__init__.py", "forge/mobile_cell_nca_v1/__main__.py",
    "forge/mobile_cell_nca_v1/contract.py", "forge/mobile_cell_nca_v1/model.py",
    "forge/mobile_cell_nca_v1/evaluation.py", "forge/mobile_cell_nca_v1/training.py",
)


@dataclass(frozen=True, slots=True)
class MobileCellNCAConfig:
    width: int = 96
    depth: int = 8
    expansion: int = 2
    max_delta: float = .16

    def __post_init__(self) -> None:
        if self.width % 8 or not 32 <= self.width <= 128:
            raise ValueError("Mobile NCA width contract drifted.")
        if not 4 <= self.depth <= 10 or self.expansion not in (1, 2, 3):
            raise ValueError("Mobile NCA capacity contract drifted.")
        if not .02 <= self.max_delta <= .3:
            raise ValueError("Mobile NCA update contract drifted.")


@dataclass(frozen=True, slots=True)
class MobileCellNCAPlan:
    steps: int = 2600
    batch_size: int = 8
    learning_rate: float = 3e-4
    ema_decay: float = .997
    rollout_steps: int = 4
    seed: int = 0x4D4F42494C454E43

    def __post_init__(self) -> None:
        if not 1 <= self.steps <= 20_000 or not 2 <= self.batch_size <= 16:
            raise ValueError("Mobile NCA training schedule drifted.")
        if not 1e-5 <= self.learning_rate <= 2e-3 or not .9 <= self.ema_decay < 1:
            raise ValueError("Mobile NCA optimizer contract drifted.")
        if self.rollout_steps not in (1, 2, 4, 6, 8):
            raise ValueError("Mobile NCA rollout contract drifted.")


def canonical(value): return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
def file_sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()
def source_sha256():
    digest = hashlib.sha256(b"nullvector-mobile-cell-nca-v1\0")
    for relative in SOURCE_FILES: digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
def config_dict(value): return asdict(value)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"nullvector-mobile-cell-nca-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        array = value.numpy()
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()
