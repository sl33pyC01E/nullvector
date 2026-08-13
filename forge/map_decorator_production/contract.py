from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from ..map_decorator.hashing import json_sha256
from ..maps.model import THEMES


CORPUS_FORMAT_VERSION: Final[str] = "1.0.0"
SHARD_FORMAT_VERSION: Final[str] = "1.0.0"
TRAINING_FORMAT_VERSION: Final[str] = "1.0.0"
PRODUCTION_CONTRACT_VERSION: Final[str] = "1.0.0"
PRODUCTION_CONTRACT_NAME: Final[str] = "nullvector-map-decorator-production-slice"
DISK_FLOOR_GIB: Final[float] = 100.0
MAX_WORKERS: Final[int] = 2
MAX_PROCESS_ATTEMPTS: Final[int] = 3
MAIN_IDENTITIES_PER_STRATUM: Final[int] = 16
TRAIN_IDENTITIES_PER_STRATUM: Final[int] = 13
VALIDATION_IDENTITIES_PER_STRATUM: Final[int] = 3
SENTINELS_PER_THEME: Final[int] = 4
FEATURE_SEED_SALT: Final[int] = 0x4645415455524553
GENERATION_SEED_SALT: Final[int] = 0x434F525055534D50
SENTINEL_SEED_SALT: Final[int] = 0x53454E54494E454C


@dataclass(frozen=True, slots=True)
class SizeProfile:
    key: str
    width: int
    height: int
    spawn_count: int

    def __post_init__(self) -> None:
        if not self.key or not self.key.isascii():
            raise ValueError("Size profile keys must be non-empty ASCII strings.")
        if not 32 <= self.width <= 256 or not 32 <= self.height <= 256:
            raise ValueError("Size profiles must remain inside the map contract [32,256].")
        if not 0 <= self.spawn_count <= 256:
            raise ValueError("Size profile spawn_count is outside the map contract.")

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObjectiveBucket:
    key: str
    objective_count: int

    def __post_init__(self) -> None:
        if not self.key or not self.key.isascii():
            raise ValueError("Objective bucket keys must be non-empty ASCII strings.")
        if not 1 <= self.objective_count <= 12:
            raise ValueError("Objective bucket is outside the map contract [1,12].")

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


SIZE_PROFILES: Final[tuple[SizeProfile, ...]] = (
    SizeProfile("square32", 32, 32, 4),
    SizeProfile("square48", 48, 48, 6),
    SizeProfile("square72", 72, 72, 8),
    SizeProfile("square128", 128, 128, 12),
    SizeProfile("wide64x40", 64, 40, 6),
    SizeProfile("tall40x64", 40, 64, 6),
    SizeProfile("wide96x56", 96, 56, 8),
    SizeProfile("tall56x96", 56, 96, 8),
)
OBJECTIVE_BUCKETS: Final[tuple[ObjectiveBucket, ...]] = (
    ObjectiveBucket("single", 1),
    ObjectiveBucket("few", 3),
    ObjectiveBucket("several", 6),
    ObjectiveBucket("dense", 10),
)
SENTINEL_DIMENSIONS: Final[tuple[int, ...]] = (32, 72, 128, 256)


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    global_seed: int = 0x4D41505F50524F44
    identities_per_stratum: int = MAIN_IDENTITIES_PER_STRATUM
    train_per_stratum: int = TRAIN_IDENTITIES_PER_STRATUM
    validation_per_stratum: int = VALIDATION_IDENTITIES_PER_STRATUM
    max_candidates_per_shard: int = 768
    replay_every_sample: bool = True
    compression: str = "npz-stored"

    def __post_init__(self) -> None:
        if isinstance(self.global_seed, bool) or not isinstance(self.global_seed, int):
            raise TypeError("Corpus global_seed must be an integer.")
        if not 0 <= self.global_seed < (1 << 64):
            raise ValueError("Corpus global_seed must be unsigned 64-bit.")
        if self.identities_per_stratum != self.train_per_stratum + self.validation_per_stratum:
            raise ValueError("Train and validation counts must exhaust each stratum.")
        if min(
            self.identities_per_stratum,
            self.train_per_stratum,
            self.validation_per_stratum,
        ) < 1:
            raise ValueError("Every corpus stratum and split must be non-empty.")
        if not 32 <= self.max_candidates_per_shard <= 100_000:
            raise ValueError("Candidate bound must be in [32,100000].")
        if self.compression != "npz-stored":
            raise ValueError("The production corpus requires npz-stored for bounded member streaming.")

    @property
    def main_map_count(self) -> int:
        return (
            len(THEMES)
            * len(SIZE_PROFILES)
            * len(OBJECTIVE_BUCKETS)
            * self.identities_per_stratum
        )

    @property
    def sentinel_count(self) -> int:
        return len(THEMES) * len(SENTINEL_DIMENSIONS)

    def to_dict(self) -> dict[str, int | bool | str]:
        return asdict(self)


def production_contract_manifest() -> dict[str, object]:
    config = CorpusConfig()
    return {
        "contract_name": PRODUCTION_CONTRACT_NAME,
        "contract_version": PRODUCTION_CONTRACT_VERSION,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "shard_format_version": SHARD_FORMAT_VERSION,
        "training_format_version": TRAINING_FORMAT_VERSION,
        "themes": list(THEMES),
        "size_profiles": [profile.to_dict() for profile in SIZE_PROFILES],
        "objective_buckets": [bucket.to_dict() for bucket in OBJECTIVE_BUCKETS],
        "main_balance": {
            "identities_per_stratum": MAIN_IDENTITIES_PER_STRATUM,
            "train_per_stratum": TRAIN_IDENTITIES_PER_STRATUM,
            "validation_per_stratum": VALIDATION_IDENTITIES_PER_STRATUM,
            "main_map_count": config.main_map_count,
        },
        "sentinels": {
            "dimensions": list(SENTINEL_DIMENSIONS),
            "per_theme": SENTINELS_PER_THEME,
            "count": config.sentinel_count,
            "required_split": "test",
        },
        "process_policy": {
            "homogeneous_shards": True,
            "max_workers": MAX_WORKERS,
            "max_attempts": MAX_PROCESS_ATTEMPTS,
            "atomic_directory_publish": True,
            "disk_floor_gib": DISK_FLOOR_GIB,
        },
    }


PRODUCTION_CONTRACT_SHA256: Final[str] = json_sha256(production_contract_manifest())
