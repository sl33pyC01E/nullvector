from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_FILES = (
    "__init__.py",
    "__main__.py",
    "authority.py",
    "hashing.py",
    "pilot.py",
    "projection.py",
)


def source_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        path = root / name
        payload = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()
