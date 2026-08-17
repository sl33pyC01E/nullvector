from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-monolithic-world-model-v1/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
WORLD_STATE = PROJECT_ROOT / "examples/models/neural_world_state_v1.pt"
WORLD_STATE_SHA256 = "98956a07db713306ab2e772943d95b5999cf94b57062b65b89bc275066e64184"
CONTEXT_ADAPTER = PROJECT_ROOT / "examples/models/recurrent_world_context_v1.pt"
CONTEXT_ADAPTER_SHA256 = "78c540fcaf402a5db0530cad35acb7f602066f1c1b15d5316d25dd818e869763"
RECURRENT = PROJECT_ROOT / "outputs/recurrent_world_student_v6/production_v1/runtime_calibrated_ramp.pt"
RECURRENT_SHA256 = "1516633d413aa19930dea53d0eb5a526d8528761e4120f4a0e9b70da42489b64"
DECODER = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v3/production_v1/runtime.pt"
DECODER_SHA256 = "03f3e147e1e4007aa01c063cf2cfc8717f169dc4974a7914e78d389a00d0d872"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/monolithic_world_model_v1/production_001"

SOURCE_FILES = (
    "forge/monolithic_world_model_v1/__init__.py",
    "forge/monolithic_world_model_v1/__main__.py",
    "forge/monolithic_world_model_v1/contract.py",
    "forge/monolithic_world_model_v1/model.py",
    "forge/monolithic_world_model_v1/training.py",
    "forge/monolithic_world_model_v1/runtime.py",
    "forge/monolithic_world_model_v1/evaluation.py",
)


@dataclass(frozen=True, slots=True)
class DirectContextConfig:
    embedding_features: int = 8
    width: int = 64
    output_features: int = 64


@dataclass(frozen=True, slots=True)
class DistillationPlan:
    corpus_size: int = 4096
    steps: int = 2500
    batch_size: int = 96
    learning_rate: float = 4e-4
    ema_decay: float = .995
    seed: int = 0x4D4F4E4F574F524C


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-monolithic-world-model-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def config_dict(value: object) -> dict[str, object]:
    return asdict(value)
