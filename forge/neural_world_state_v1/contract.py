from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT


FORMAT = "nullvector-neural-world-state-v1/1.0.0"
CHECKPOINT_FORMAT = "nullvector-neural-world-state-checkpoint-v1/1.0.0"
GRID_SIZE = 32
TERRAIN_CLASSES = 8
CITY_CLASSES = 8
CONTINUOUS_NAMES = ("elevation", "walkability", "nav_cost", "biomass", "mineral", "moisture", "energy")
CONDITION_NAMES = (
    *(f"theme_{name}" for name in ("arena", "rooms", "caves", "archipelago", "garden", "anomaly")),
    *(f"family_{index}" for index in range(5)),
    "season_sin", "season_cos", "development", "disturbance",
)
SOURCE_FILES = (
    "forge/neural_world_state_v1/contract.py",
    "forge/neural_world_state_v1/data.py",
    "forge/neural_world_state_v1/model.py",
    "forge/neural_world_state_v1/training.py",
)


@dataclass(frozen=True, slots=True)
class WorldStateModelConfig:
    width: int = 64
    latent_channels: int = 20
    global_features: int = 64


@dataclass(frozen=True, slots=True)
class WorldStateTrainingConfig:
    steps: int = 3000
    batch_size: int = 128
    corpus_size: int = 8192
    learning_rate: float = 3e-4
    # This compact codec converges quickly; a faster EMA tracks its continuous
    # ecology heads instead of lagging several thousand updates behind them.
    ema_decay: float = .995
    kl_weight: float = 2e-5
    seed: int = 0x574F524C44535441


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-world-state-v1\0")
    for relative in SOURCE_FILES: digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def config_dict(value: object) -> dict[str, object]: return asdict(value)
