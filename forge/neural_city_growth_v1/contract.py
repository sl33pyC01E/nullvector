from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from ..config import PROJECT_ROOT
from ..neural_city_layout_v1.contract import CONDITION_NAMES as CITY_CONDITION_NAMES, PURPOSES, CityCondition


FORMAT = "nullvector-neural-city-growth-v1/1.0.0"
CHECKPOINT_FORMAT = "nullvector-neural-city-growth-checkpoint-v1/1.0.0"
PATCH_SIZE = 24
ACTIONS = PURPOSES
RESOURCE_NAMES = ("mineral", "biomass", "energy", "knowledge")
GROWTH_CONDITION_NAMES = (
    *CITY_CONDITION_NAMES,
    *(f"action_{name}" for name in ACTIONS),
    *(f"resource_{name}" for name in RESOURCE_NAMES),
    "site_x", "site_y",
    "growth_stage",
)
SITE_X_INDEX = len(GROWTH_CONDITION_NAMES) - 3
SITE_Y_INDEX = len(GROWTH_CONDITION_NAMES) - 2
PURPOSE_COSTS = {
    "habitat": (.45, .35, .10, .05), "workshop": (.65, .12, .25, .15),
    "clinic": (.30, .55, .20, .25), "granary": (.35, .35, .08, .08),
    "observatory": (.50, .08, .55, .65), "graft_house": (.25, .70, .25, .35),
    "battery_hall": (.70, .05, .75, .20), "shrine": (.38, .25, .15, .55),
    "market": (.50, .18, .20, .25),
}
SOURCE_FILES = (
    "forge/neural_city_growth_v1/contract.py",
    "forge/neural_city_growth_v1/teacher.py",
    "forge/neural_city_growth_v1/model.py",
    "forge/neural_city_growth_v1/training.py",
    "forge/neural_city_layout_v1/contract.py",
    "forge/neural_city_layout_v1/teacher.py",
)


@dataclass(frozen=True, slots=True)
class GrowthCondition:
    city: CityCondition
    action: str
    resources: tuple[float, ...]
    site: tuple[float, float]
    stage: int

    def affordable(self) -> bool:
        if self.action not in PURPOSE_COSTS or len(self.resources) != len(RESOURCE_NAMES):
            raise ValueError("Growth condition vocabulary drifted.")
        return all(have + 1e-6 >= need for have, need in zip(self.resources, PURPOSE_COSTS[self.action], strict=True))

    def vector(self) -> np.ndarray:
        if self.action not in ACTIONS or len(self.resources) != len(RESOURCE_NAMES) or len(self.site) != 2:
            raise ValueError("Growth action/resources drifted.")
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in (*self.resources, *self.site)):
            raise ValueError("Growth resources are outside [0,1].")
        row = np.zeros(len(GROWTH_CONDITION_NAMES), np.float32)
        city = self.city.vector(); row[:len(city)] = city
        offset = len(city); row[offset + ACTIONS.index(self.action)] = 1; offset += len(ACTIONS)
        row[offset:offset + len(RESOURCE_NAMES)] = self.resources
        row[-3:-1] = self.site
        row[-1] = min(1, max(0, self.stage) / 12)
        return row


@dataclass(frozen=True, slots=True)
class GrowthModelConfig:
    width: int = 64
    levels: int = 3
    blocks_per_level: int = 2
    seed: int = 0x4349545947524F57


@dataclass(frozen=True, slots=True)
class GrowthTrainingConfig:
    steps: int = 2400
    batch_size: int = 32
    learning_rate: float = 2.5e-4
    ema_decay: float = .999
    corpus_size: int = 8192
    seed: int = 0x47524F5754484E4E


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def source_manifest() -> dict[str, str]:
    return {relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() for relative in SOURCE_FILES}


def source_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest())).hexdigest()


def config_dict(value: object) -> dict[str, object]: return asdict(value)
