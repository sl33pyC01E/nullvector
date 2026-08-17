from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from ..config import PROJECT_ROOT
from ..nature_world_scale_v1.atlas import BIOMES


FORMAT = "nullvector-neural-city-layout-v1/1.0.0"
CHECKPOINT_FORMAT = "nullvector-neural-city-layout-checkpoint-v1/1.0.0"
GRID_SIZE = 64
CLASSES = ("empty", "road", "wall", "floor", "door", "utility", "garden", "storage")
PURPOSES = ("habitat", "workshop", "clinic", "granary", "observatory", "graft_house", "battery_hall", "shrine", "market")
MASK_TOKEN = len(CLASSES)
CONDITION_NAMES = (
    *(f"family_{index}" for index in range(5)),
    *(f"culture_{index}" for index in range(8)),
    *(f"project_{name}" for name in PURPOSES),
    *(f"biome_{name}" for name in BIOMES),
    *(f"style_{index}" for index in range(16)),
    "population", "wealth", "technology", "building_target",
)
SOURCE_FILES = (
    "forge/neural_city_layout_v1/contract.py",
    "forge/neural_city_layout_v1/teacher.py",
    "forge/neural_city_layout_v1/model.py",
    "forge/neural_city_layout_v1/training.py",
)


@dataclass(frozen=True, slots=True)
class CityCondition:
    family: int
    culture: tuple[float, ...]
    project: str
    biome: str
    style: tuple[float, ...]
    population: int
    wealth: float
    technology: float
    building_target: int

    def vector(self) -> np.ndarray:
        if not 0 <= self.family < 5 or len(self.culture) != 8 or len(self.style) != 16:
            raise ValueError("City condition family/culture drifted.")
        if self.project not in PURPOSES or self.biome not in BIOMES:
            raise ValueError("City condition vocabulary drifted.")
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in (*self.culture, *self.style)):
            raise ValueError("City culture/style is outside [0,1].")
        row = np.zeros(len(CONDITION_NAMES), np.float32)
        row[self.family] = 1
        row[5:13] = self.culture
        row[13 + PURPOSES.index(self.project)] = 1
        row[13 + len(PURPOSES) + BIOMES.index(self.biome)] = 1
        style_start = 13 + len(PURPOSES) + len(BIOMES)
        row[style_start:style_start + 16] = self.style
        row[-4:] = (
            min(1, max(0, self.population) / 32),
            min(1, max(0.0, self.wealth) / 5),
            min(1, max(0.0, self.technology)),
            min(1, max(1, self.building_target) / 12),
        )
        return row


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 64
    levels: int = 3
    blocks_per_level: int = 2
    seed: int = 0x434954594C41594F


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 800
    batch_size: int = 32
    learning_rate: float = 2.5e-4
    ema_decay: float = 0.999
    corpus_size: int = 4096
    seed: int = 0x4349545954524149


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def source_manifest() -> dict[str, str]:
    return {
        relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }


def source_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest())).hexdigest()


def config_dict(value: object) -> dict[str, object]:
    return asdict(value)
