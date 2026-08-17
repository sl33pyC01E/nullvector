from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT

FORMAT = "nullvector-neural-foundation-v2/1.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/neural_foundation_v2/build_003"
COMPONENTS = {
    "playable_ensemble": ("outputs/playable_neural_runtime_v1/build_001/runtime_manifest.json", None),
    "recurrent_action_dit": ("outputs/recurrent_action_dit_v2/production_v1/report.json", "outputs/recurrent_action_dit_v2/production_v1/runtime.pt"),
    "adapted_world_decoder": ("outputs/world_frame_decoder_adapt_v1/production_v1/report.json", "outputs/world_frame_decoder_adapt_v1/production_v1/runtime.pt"),
    "action_frame_compositor": ("outputs/neural_action_frame_v1/evaluation_v1/report.json", None),
}
SOURCE_FILES = (
    "forge/neural_foundation_v2/__init__.py",
    "forge/neural_foundation_v2/__main__.py",
    "forge/neural_foundation_v2/contract.py",
    "forge/neural_foundation_v2/release.py",
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256():
    digest = hashlib.sha256(b"nullvector-neural-foundation-v2\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
