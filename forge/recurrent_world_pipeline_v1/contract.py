from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT

FORMAT = "nullvector-recurrent-world-pipeline-v1/1.0.0"
RECURRENT = PROJECT_ROOT / "outputs/recurrent_world_student_v6/production_v1/runtime_calibrated_ramp.pt"
RECURRENT_SHA256 = "1516633d413aa19930dea53d0eb5a526d8528761e4120f4a0e9b70da42489b64"
DECODER = PROJECT_ROOT / "outputs/world_frame_rollout_decoder_v3/production_v1/runtime.pt"
DECODER_SHA256 = "03f3e147e1e4007aa01c063cf2cfc8717f169dc4974a7914e78d389a00d0d872"
NATURAL_CORPUS = PROJECT_ROOT / "outputs/world_action_natural_v10/corpus_v1_6world"
NATURAL_CORPUS_SHA256 = "e96b10f80db3e824fdb768dc9e52ac8ff5e7f228cf3b87ba89d1df8d3047662f"
DEFAULT_RELEASE = PROJECT_ROOT / "outputs/recurrent_world_pipeline_v1/release.json"
SOURCE_FILES = (
    "forge/recurrent_world_pipeline_v1/__init__.py",
    "forge/recurrent_world_pipeline_v1/__main__.py",
    "forge/recurrent_world_pipeline_v1/contract.py",
    "forge/recurrent_world_pipeline_v1/runtime.py",
    "forge/recurrent_world_pipeline_v1/release.py",
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
    digest = hashlib.sha256(b"nullvector-recurrent-world-pipeline-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
