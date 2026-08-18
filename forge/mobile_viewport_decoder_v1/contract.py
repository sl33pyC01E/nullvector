from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT

FORMAT = "nullvector-mobile-viewport-decoder/1.0.0"
CHECKPOINT_FORMAT = "nullvector-mobile-viewport-decoder-checkpoint/1.0.0"
DEFAULT_CORPUS = PROJECT_ROOT / "outputs/action_teacher_viewport_v5/macro_corpus_v1"
DEFAULT_VAE = PROJECT_ROOT / "outputs/whole_viewport_raster_vae_v1/production_macro_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mobile_viewport_decoder_v1/production_v1"
SOURCE_FILES = (
    "forge/mobile_viewport_decoder_v1/contract.py",
    "forge/mobile_viewport_decoder_v1/model.py",
    "forge/mobile_viewport_decoder_v1/training.py",
)


@dataclass(frozen=True)
class ModelConfig:
    latent_channels: int = 48
    widths: tuple[int, int, int, int] = (128, 96, 64, 32)
    residual_blocks: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 4800
    batch_size: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 2e-4
    ema_decay: float = 0.9995
    edge_weight: float = 2.5
    laplacian_weight: float = 1.0
    multiscale_weight: float = 0.5
    validation_every: int = 300
    checkpoint_every: int = 100
    seed: int = 0x4D4F42494C454445


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value): return asdict(value)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-mobile-viewport-decoder-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
