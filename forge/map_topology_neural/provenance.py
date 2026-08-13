from __future__ import annotations

from pathlib import Path

from ..config import PROJECT_ROOT
from .hashing import file_sha256, json_sha256


PACKAGE_ROOT = Path(__file__).resolve().parent
SCHEMA_ROOT = PROJECT_ROOT / "shared" / "schema"
EXTERNAL_DEPENDENCIES = (
    "forge/config.py",
    "forge/safety.py",
    "forge/maps/model.py",
    "forge/maps/render.py",
    "forge/maps/validate.py",
)


def source_manifest() -> dict[str, object]:
    files: dict[str, str] = {}
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        files[path.relative_to(PROJECT_ROOT).as_posix()] = file_sha256(path)
    for path in sorted(SCHEMA_ROOT.glob("map_topology_neural*.json")):
        files[path.relative_to(PROJECT_ROOT).as_posix()] = file_sha256(path)
    for relative in EXTERNAL_DEPENDENCIES:
        path = PROJECT_ROOT / relative
        files[relative] = file_sha256(path)
    if not files:
        raise RuntimeError("Neural topology source manifest cannot be empty.")
    return {
        "format": "nullvector-map-topology-neural-source-v1",
        "files": files,
    }


def source_sha256() -> str:
    return json_sha256(source_manifest())


def compiler_source_sha256() -> str:
    paths = (
        PACKAGE_ROOT / "compiler.py",
        PACKAGE_ROOT / "contract.py",
        PACKAGE_ROOT / "hashing.py",
        PROJECT_ROOT / "forge" / "maps" / "model.py",
        PROJECT_ROOT / "forge" / "maps" / "validate.py",
    )
    return json_sha256(
        {
            "format": "nullvector-map-topology-compiler-source-v1",
            "files": {
                path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
                for path in paths
            },
        }
    )
