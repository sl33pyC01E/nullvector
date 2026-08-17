from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-android-port-v1/1.0.0"
TARGET = {
    "device": "Samsung Galaxy S25 Ultra",
    "soc": "Snapdragon 8 Elite for Galaxy",
    "abi": "arm64-v8a",
    "minimum_android_api": 29,
    "target_display_fps": 30,
    "world_context_hz": 15,
    "organism_motion_hz": 30,
    "preferred_precision": "fp16",
    "execution_provider_order": ["QNN-HTP", "NNAPI", "XNNPACK", "CPU"],
}
MONOLITHIC = PROJECT_ROOT / "outputs/monolithic_world_model_v1/production_002/runtime.pt"
MONOLITHIC_SHA256 = "eab44d442b0b33d82a3ba156a234815169e5e7ac688815d4cbab3253d9dd255f"
MOBILE_DECODER = PROJECT_ROOT / "examples/models/mobile_frame_decoder_v1.pt"
MOBILE_DECODER_SHA256 = "dcf4d3808a49f6fd966aa2cd637729abd47b7180f552b9eb21a1bdd038d449d1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/android_port_v1/export_001"
SOURCE_FILES = (
    "forge/android_port_v1/__init__.py",
    "forge/android_port_v1/__main__.py",
    "forge/android_port_v1/contract.py",
    "forge/android_port_v1/export.py",
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-android-port-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
