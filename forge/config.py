from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ARCHETYPES = ("dart", "hound", "oracle", "bulwark")
LAYER_NAMES = (
    "hull",
    "armor",
    "left",
    "right",
    "weapon",
    "core",
    "circuit",
    "emission",
)


@dataclass(slots=True)
class ForgeConfig:
    image_size: int = 32
    latent_dim: int = 32
    condition_dim: int = len(ARCHETYPES)
    layer_count: int = len(LAYER_NAMES)
    dataset_size: int = 24_576
    validation_size: int = 2_048
    batch_size: int = 384
    epochs: int = 26
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    beta_max: float = 8.0e-4
    beta_warmup_steps: int = 800
    dice_weight: float = 0.35
    seed: int = 7_193_003
    num_workers: int = 0
    amp: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GAME_GENERATED_DIR = PROJECT_ROOT / "game" / "generated"
