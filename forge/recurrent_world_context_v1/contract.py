from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-recurrent-world-context-v1/1.0.0"
CHECKPOINT_FORMAT = "nullvector-recurrent-world-context-checkpoint-v1/1.0.0"
WORLD_STATE = PROJECT_ROOT / "examples/models/neural_world_state_v1.pt"
WORLD_STATE_SHA256 = "98956a07db713306ab2e772943d95b5999cf94b57062b65b89bc275066e64184"
RECURRENT = PROJECT_ROOT / "outputs/recurrent_action_dit_v2/production_v1/runtime.pt"
RECURRENT_SHA256 = "22c3c9a23a411057bf376bee77b750318f459414f35bd4d87ab6e326b90323ef"
CORPUS = PROJECT_ROOT / "outputs/world_action_cellular_v7/corpus_v1_6world"
SOURCE_FILES = (
    "forge/recurrent_world_context_v1/contract.py",
    "forge/recurrent_world_context_v1/data.py",
    "forge/recurrent_world_context_v1/model.py",
    "forge/recurrent_world_context_v1/training.py",
    "forge/neural_world_state_v1/model.py",
)


@dataclass(frozen=True, slots=True)
class ContextModelConfig:
    input_features: int = 84
    width: int = 256
    output_features: int = 64


@dataclass(frozen=True, slots=True)
class ContextTrainingConfig:
    steps: int = 3000
    batch_size: int = 256
    learning_rate: float = 4e-4
    ema_decay: float = .995
    seed: int = 0x434F4E5445585453


def canonical(value: object) -> bytes: return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-recurrent-world-context-v1\0")
    for relative in SOURCE_FILES: digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def config_dict(value: object) -> dict[str, object]: return asdict(value)
