from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-living-body-causal-nca-bridge-v1"
DEFAULT_AUTHORITY: Final[Path] = PROJECT_ROOT / "outputs/cellular_nca/nca_causal_v2_rngfix"
CANVAS: Final[int] = 48
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/living_body_nca_v1/__init__.py",
    "forge/living_body_nca_v1/contract.py",
    "forge/living_body_nca_v1/adapter.py",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-living-body-causal-nca-bridge-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
