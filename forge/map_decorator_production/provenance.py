from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256


CORPUS_SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_decorator_production/contract.py",
    "forge/map_decorator_production/provenance.py",
    "forge/map_decorator_production/teacher.py",
    "forge/map_decorator_production/corpus.py",
    "forge/map_decorator_production/worker.py",
)
TRAINING_SOURCE_FILES: Final[tuple[str, ...]] = (
    *CORPUS_SOURCE_FILES,
    "forge/map_decorator_production/training.py",
    "forge/map_decorator_production/supervisor.py",
)


def source_manifest(
    scope: str,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, str]:
    if scope == "corpus":
        names = CORPUS_SOURCE_FILES
    elif scope == "training":
        names = TRAINING_SOURCE_FILES
    else:
        raise ValueError("Production source scope must be 'corpus' or 'training'.")
    result: dict[str, str] = {}
    for name in names:
        path = Path(root) / name
        if not path.is_file():
            raise FileNotFoundError(f"Production source is missing: {path}")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def source_sha256(scope: str, *, root: Path = PROJECT_ROOT) -> str:
    return json_sha256(source_manifest(scope, root=root))

