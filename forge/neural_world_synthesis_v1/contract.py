from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-neural-world-synthesis-v1/1.0.0"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/neural_world_synthesis_v1/build_002"
PRIOR_CHECKPOINT: Final[Path] = PROJECT_ROOT / "outputs/map_topology_neural_prior_v3/scale_aware_0500/checkpoint.pt"
SELECTION_AUDIT: Final[Path] = PROJECT_ROOT / "outputs/map_decorator_production_v4_selection/protected_selection_audit_20260817"
CALIBRATION_ROOT: Final[Path] = PROJECT_ROOT / "outputs/map_decorator_production_v4_calibration/calibration_100step_20260817"
CORPUS_ROOT: Final[Path] = PROJECT_ROOT / "outputs/map_decorator_corpus_v1"
INDEX_ROOT: Final[Path] = PROJECT_ROOT / "outputs/map_decorator_production_v2/foreground_index_v2"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/neural_world_synthesis_v1/__init__.py",
    "forge/neural_world_synthesis_v1/__main__.py",
    "forge/neural_world_synthesis_v1/contract.py",
    "forge/neural_world_synthesis_v1/build.py",
    "forge/neural_world_synthesis_v1/map_pack.py",
    "forge/neural_world_synthesis_v1/decorator.py",
    "shared/schema/neural_world_synthesis_v1.schema.json",
)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def source_sha256(root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest(root))).hexdigest()
