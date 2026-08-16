from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT

CHECKPOINT_FORMAT = "nullvector-cellular-temporal-action-checkpoint/7.0.0"
REPORT_FORMAT = "nullvector-cellular-temporal-action-training/7.0.0"
CORPUS_FORMAT = "nullvector-cellular-temporal-action-corpus/7.0.0"
SOURCE_FILES = (
    "forge/action_teacher_v2/contract.py",
    "forge/action_teacher_v2/actor.py",
    "forge/action_teacher_v2/recorder.py",
    "forge/world_action_sparse_v5/contract.py",
    "forge/world_action_sparse_v5/model.py",
    "forge/world_action_cellular_v7/contract.py",
    "forge/world_action_cellular_v7/data.py",
    "forge/world_action_cellular_v7/corpus.py",
    "forge/world_action_cellular_v7/model.py",
    "forge/world_action_cellular_v7/checkpoint.py",
    "forge/world_action_cellular_v7/training.py",
    "forge/world_action_cellular_v7/runtime.py",
)
CORPUS_SOURCE_FILES = (
    "forge/action_teacher_v2/contract.py",
    "forge/action_teacher_v2/recorder.py",
    "forge/world_action_cellular_v7/contract.py",
    "forge/world_action_cellular_v7/data.py",
    "forge/world_action_cellular_v7/corpus.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 512
    layers: int = 8
    heads: int = 8
    patch: int = 4
    spatial_channels: int = 5
    actor_field_channels: int = 8
    gate_bias: float = -4.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 12000
    batch_size: int = 10
    learning_rate: float = 8e-5
    ema_decay: float = 0.999
    validate_every: int = 500
    checkpoint_every: int = 500
    changed_weight: float = 7.0
    actor_state_weight: float = 0.35
    actor_field_weight: float = 0.55
    actor_changed_weight: float = 5.0
    gate_weight: float = 0.35
    gate_positive_weight: float = 2.0
    leakage_weight: float = 0.35
    contrastive_weight: float = 0.22
    contrastive_margin: float = 0.012
    contrastive_batch: int = 6
    input_noise: float = 0.002
    milestone_every: int = 2000
    validation_batch_size: int = 8
    cpu_threads: int = 8
    cuda_memory_fraction: float = 0.85
    max_process_memory_gib: float = 32.0
    target_duty_cycle: float = 0.90
    seed: int = 0x43454C4C554C4152


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value) -> dict:
    return asdict(value)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-temporal-action-v7\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def corpus_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-temporal-action-corpus-v7\0")
    for relative in CORPUS_SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
