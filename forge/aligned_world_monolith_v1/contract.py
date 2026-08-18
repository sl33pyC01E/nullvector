from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT

FORMAT = "nullvector-aligned-world-monolith/1.0.0"
CHECKPOINT_FORMAT = FORMAT + "-checkpoint"
DEFAULT_CORPUS = PROJECT_ROOT / "outputs/action_teacher_viewport_v5/macro_corpus_v1"
DEFAULT_RENDERER = PROJECT_ROOT / "outputs/whole_viewport_latent_v1/production_macro_adapted_vae_v5_rollout8/model.pt"
DEFAULT_VAE = PROJECT_ROOT / "outputs/whole_viewport_raster_vae_v1/production_macro_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/aligned_world_monolith_v1/production_v1"
SOURCE_FILES = (
    "forge/aligned_world_monolith_v1/contract.py",
    "forge/aligned_world_monolith_v1/data.py",
    "forge/aligned_world_monolith_v1/model.py",
    "forge/aligned_world_monolith_v1/training.py",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 128
    blocks: int = 6
    organism_width: int = 192
    organism_slots: int = 64


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 3600
    transition_only_steps: int = 900
    batch_size: int = 5
    rollout_steps: int = 4
    learning_rate: float = 1.5e-4
    renderer_learning_rate: float = 2e-5
    weight_decay: float = 2e-4
    ema_decay: float = .999
    validation_every: int = 300
    checkpoint_every: int = 100
    seed: int = 0x414C49474E454457


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value: object) -> dict:
    return asdict(value)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-aligned-world-monolith-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
