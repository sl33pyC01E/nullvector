from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..config import PROJECT_ROOT

CORPUS_FORMAT = "nullvector-neural-macro-patch-corpus/1.0.0"
CHECKPOINT_FORMAT = "nullvector-neural-macro-patch-checkpoint/1.0.0"
REPORT_FORMAT = "nullvector-neural-macro-patch-training/1.0.0"
PATCH_SIZE = 32
WORLD_SIZE = 64
STATE_CHANNELS = (
    "resource_water", "resource_light", "resource_mineral", "resource_charge", "resource_phase",
    "resource_oxygen", "resource_heat", "resource_toxin", "resource_flora", "resource_biomass",
    "population_humanoid", "population_animalian", "population_plantlike", "population_anomaly", "population_machine",
    "organism_energy", "organism_integrity", "colony_influence",
    "building_habitat", "building_workshop", "building_clinic", "building_granary", "building_observatory",
    "building_graft_house", "building_battery_hall", "building_shrine", "building_market",
    "road", "structure", "material_mass", "material_damage", "material_temperature",
)
GLOBAL_FEATURES = 44
SOURCE_FILES = (
    "forge/nature_macro_nn/contract.py", "forge/nature_macro_nn/state.py", "forge/nature_macro_nn/corpus.py",
    "forge/nature_macro_nn/model.py", "forge/nature_macro_nn/training.py", "forge/nature_macro_nn/runtime.py",
)
CORPUS_SOURCE_FILES = SOURCE_FILES[:3]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 96
    blocks: int = 8
    global_width: int = 192


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 2400
    batch_size: int = 24
    learning_rate: float = 2e-4
    ema_decay: float = 0.998
    changed_weight: float = 12.0
    gate_weight: float = 0.18
    global_weight: float = 0.45
    validation_every: int = 400
    seed: int = 0x4D4143524F4E4E


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def config_dict(value) -> dict:
    return asdict(value)


def _source_hash(label: bytes, files) -> str:
    digest = hashlib.sha256(label)
    for relative in files:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def source_sha256() -> str:
    return _source_hash(b"nullvector-neural-macro-patch-v1\0", SOURCE_FILES)


def corpus_source_sha256() -> str:
    return _source_hash(b"nullvector-neural-macro-patch-corpus-v1\0", CORPUS_SOURCE_FILES)
