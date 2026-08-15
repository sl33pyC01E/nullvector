from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


CORPUS_FORMAT: Final[str] = "nullvector-neural-cell-motion-corpus-v1"
MODEL_FORMAT: Final[str] = "nullvector-neural-cell-motion-model-v1"
DEFAULT_MOTION: Final[Path] = PROJECT_ROOT / "outputs/cellular_motion_v2/cellular_motion_manifest.json"
DEFAULT_ANATOMY: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
DEFAULT_CORPUS: Final[Path] = PROJECT_ROOT / "outputs/neural_cell_motion/corpus_v1"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/neural_cell_motion_corpus.schema.json"
GRID_SIZE: Final[int] = 48
FEATURE_CHANNELS: Final[int] = 60
STATE_CHANNELS: Final[int] = 4
MAX_DISPLACEMENT: Final[float] = 6.0
ORGAN_CHANNELS: Final[tuple[str, ...]] = (
    "chassis", "neural", "sensory", "left_appendage", "right_appendage",
    "left_locomotor", "right_locomotor", "auxiliary", "weapon", "emitter", "mouth",
)
FEATURE_GROUPS: Final[dict[str, tuple[int, int]]] = {
    "occupancy": (0, 1), "coordinates": (1, 3), "continuous_cell": (3, 9),
    "emission": (9, 10), "tissue_one_hot": (10, 22), "material_one_hot": (22, 32),
    "part_owner_one_hot": (32, 49), "organ_channel_one_hot": (49, 60),
}
TARGET_NAMES: Final[tuple[str, ...]] = ("delta_x", "delta_y", "motor_activation", "emission_activation")
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/neural_cell_motion/__init__.py", "forge/neural_cell_motion/contract.py",
    "forge/neural_cell_motion/dataset.py", "forge/neural_cell_motion/worker.py",
    "forge/neural_cell_motion/supervisor.py", "shared/schema/neural_cell_motion_corpus.schema.json",
)
MODEL_FILES: Final[tuple[str, ...]] = (
    "forge/neural_cell_motion/model.py", "forge/neural_cell_motion/training.py", "forge/neural_cell_motion/contract.py",
)


@dataclass(frozen=True)
class NeuralCellMotionConfig:
    base_channels: int = 128
    channel_multipliers: tuple[int, int, int] = (1, 2, 3)
    blocks_per_level: int = 2
    condition_dim: int = 384
    attention_heads: int = 8
    dropout: float = 0.05
    static_channels: int = FEATURE_CHANNELS
    state_channels: int = STATE_CHANNELS

    def __post_init__(self) -> None:
        if not 24 <= self.base_channels <= 256 or len(self.channel_multipliers) != 3 or self.channel_multipliers[0] != 1 or any(type(value) is not int or not 1 <= value <= 6 for value in self.channel_multipliers):
            raise ValueError("Neural motion channel configuration drifted.")
        if not 1 <= self.blocks_per_level <= 5 or not 96 <= self.condition_dim <= 1024 or not 1 <= self.attention_heads <= 16 or self.base_channels * self.channel_multipliers[-1] % self.attention_heads:
            raise ValueError("Neural motion depth/attention configuration drifted.")
        if not 0 <= self.dropout <= .25 or self.static_channels != FEATURE_CHANNELS or self.state_channels != STATE_CHANNELS:
            raise ValueError("Neural motion input/output contract drifted.")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self); value["channel_multipliers"] = list(self.channel_multipliers); return value


def _hash_files(domain: bytes, files: tuple[str, ...]) -> str:
    digest = hashlib.sha256(domain)
    for relative in files:
        path = PROJECT_ROOT / relative
        if not path.is_file(): raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def corpus_source_sha256() -> str:
    return _hash_files(b"nullvector-neural-cell-motion-corpus-source-v1\0", SOURCE_FILES)


def model_source_sha256() -> str:
    return _hash_files(b"nullvector-neural-cell-motion-model-source-v1\0", MODEL_FILES)
