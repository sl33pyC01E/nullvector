from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from ..map_topology_neural_prior.contract import (
    FROZEN_CODEC_CHECKPOINT_SHA256,
    FROZEN_CODEC_EMA_SHA256,
    FROZEN_CODEC_SOURCE_SHA256,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CORPUS_FORMAT: Final[str] = "nullvector-neural-map-topology-latent-corpus/1.0.0"
SHARD_FORMAT: Final[str] = "nullvector-neural-map-topology-latent-shard/1.0.0"
PRIOR_SOURCE_SHA256: Final[str] = "76fcbce48e1ce20f5e1f28c20a38cc9c9d8c98be2cedccd221e7f95bb6145e15"
EXPECTED_SHARDS: Final[int] = 216
EXPECTED_SAMPLES: Final[int] = 3_096
MAX_WORKERS: Final[int] = 2
MAX_ATTEMPTS: Final[int] = 3
MAX_SHARD_BYTES: Final[int] = 32 * 1024 * 1024
MAX_MANIFEST_BYTES: Final[int] = 8 * 1024 * 1024
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_topology_neural_prior_corpus/__init__.py",
    "forge/map_topology_neural_prior_corpus/__main__.py",
    "forge/map_topology_neural_prior_corpus/contract.py",
    "forge/map_topology_neural_prior_corpus/shard.py",
    "forge/map_topology_neural_prior_corpus/supervisor.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def corpus_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return sha256_bytes(canonical_json_bytes(source_manifest(root)))


def authority() -> dict[str, str]:
    return {
        "prior_source_sha256": PRIOR_SOURCE_SHA256,
        "codec_checkpoint_sha256": FROZEN_CODEC_CHECKPOINT_SHA256,
        "codec_source_sha256": FROZEN_CODEC_SOURCE_SHA256,
        "codec_ema_sha256": FROZEN_CODEC_EMA_SHA256,
    }

