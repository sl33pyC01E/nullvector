from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from ..morphology import FACING_NAMES, FAMILIES, MOTION_NAMES
from .hashing import canonical_json_bytes
from .io import verify_artifact
from .model import LAYER_NAMES
from .schema import FAMILY_SCHEMA, validate_schema


FRAME_INDEX_KEYS = {
    "format",
    "family",
    "layer_names",
    "clip_ids",
    "clip_offsets",
    "phases",
    "source_frame_sha256",
    "categorical_sha256",
    "joint_sha256",
    "socket_sha256",
    "authority_sha256",
    "presentation_sha256",
}
MAX_FAMILY_INDEX_BYTES = 16 * 1024 * 1024
MAX_FAMILY_INDEX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024


def strict_json_file(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"JSON must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(payload, allow_nan=False)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Invalid strict UTF-8 JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _verify_frame_index(
    path: Path,
    manifest: Mapping[str, Any],
) -> None:
    if path.stat().st_size > MAX_FAMILY_INDEX_BYTES:
        raise ValueError("Style-motion frame index exceeds its compressed bound")
    expected_members = {f"{name}.npy" for name in FRAME_INDEX_KEYS}
    with zipfile.ZipFile(path, "r") as archive:
        entries = archive.infolist()
        if len(entries) != len(expected_members) or {entry.filename for entry in entries} != expected_members:
            raise ValueError("Style-motion frame index ZIP members mismatch")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Style-motion frame index has duplicate ZIP members")
        total = 0
        for entry in entries:
            if "/" in entry.filename or "\\" in entry.filename:
                raise ValueError("Style-motion frame index has nested ZIP members")
            total += entry.file_size
        if total > MAX_FAMILY_INDEX_UNCOMPRESSED_BYTES:
            raise ValueError("Style-motion frame index exceeds its uncompressed bound")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FRAME_INDEX_KEYS:
            raise ValueError("Style-motion frame index array keys mismatch")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    frame_count = int(manifest["frame_count"])
    clip_count = int(manifest["clip_count"])
    expected_shapes = {
        "format": (1,),
        "family": (1,),
        "layer_names": (len(LAYER_NAMES),),
        "clip_ids": (clip_count,),
        "clip_offsets": (clip_count + 1,),
        "phases": (frame_count,),
        "source_frame_sha256": (frame_count,),
        "categorical_sha256": (frame_count,),
        "joint_sha256": (frame_count,),
        "socket_sha256": (frame_count,),
        "authority_sha256": (frame_count,),
        "presentation_sha256": (frame_count, len(LAYER_NAMES)),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Style-motion frame index {name} shape mismatch")
    if arrays["clip_offsets"].dtype != np.uint32 or arrays["phases"].dtype != np.float32:
        raise ValueError("Style-motion frame index numeric dtypes mismatch")
    for name in FRAME_INDEX_KEYS - {"clip_offsets", "phases"}:
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"Style-motion frame index {name} must be Unicode")
    if arrays["format"].tolist() != ["nullvector-multifield-style-motion-frame-index-v1"]:
        raise ValueError("Style-motion frame index format mismatch")
    if arrays["family"].tolist() != [manifest["family"]]:
        raise ValueError("Style-motion frame index family mismatch")
    if arrays["layer_names"].tolist() != list(LAYER_NAMES):
        raise ValueError("Style-motion frame index layer order mismatch")
    clips = list(manifest["clips"])
    if arrays["clip_ids"].tolist() != [clip["id"] for clip in clips]:
        raise ValueError("Style-motion frame index clip order mismatch")
    offsets = arrays["clip_offsets"].astype(np.int64)
    expected_offsets = [0]
    for clip in clips:
        expected_offsets.append(expected_offsets[-1] + int(clip["frame_count"]))
    if offsets.tolist() != expected_offsets or expected_offsets[-1] != frame_count:
        raise ValueError("Style-motion frame index clip offsets mismatch")
    if not np.all(np.isfinite(arrays["phases"])) or np.any(arrays["phases"] < 0) or np.any(arrays["phases"] > 1):
        raise ValueError("Style-motion frame phases are outside [0, 1]")
    sha_arrays = (
        "source_frame_sha256",
        "categorical_sha256",
        "joint_sha256",
        "socket_sha256",
        "authority_sha256",
    )
    for name in sha_arrays:
        if any(len(value) != 64 or set(value) - set("0123456789abcdef") for value in arrays[name].tolist()):
            raise ValueError(f"Style-motion frame index {name} contains invalid SHA-256 values")
    for row in arrays["presentation_sha256"].tolist():
        if any(len(value) != 64 or set(value) - set("0123456789abcdef") for value in row):
            raise ValueError("Style-motion presentation hashes contain invalid SHA-256 values")


def load_verified_family_manifest(
    output_root: Path,
    family: str,
    *,
    bank: Any | None = None,
    compiler: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown style-motion family: {family!r}")
    root = Path(output_root).resolve()
    path = root / "families" / family / "family_manifest.json"
    manifest = strict_json_file(path)
    validate_schema(manifest, FAMILY_SCHEMA)
    if manifest["family"] != family:
        raise ValueError("Style-motion family manifest family mismatch")
    if compiler is not None and manifest["compiler"] != dict(compiler):
        raise ValueError("Style-motion family compiler provenance mismatch")
    expected_clip_keys = [
        (motion, facing)
        for motion in MOTION_NAMES
        for facing in FACING_NAMES
    ]
    actual_clip_keys = [(clip["motion"], clip["facing"]) for clip in manifest["clips"]]
    if actual_clip_keys != expected_clip_keys:
        raise ValueError("Style-motion family clip matrix order mismatch")
    if bank is not None:
        expected = bank.clips_by_family[family]
        for derived, source in zip(manifest["clips"], expected, strict=True):
            if (
                derived["id"] != source["id"]
                or derived["source_clip_sha256"] != source["clip_sha256"]
                or derived["events"] != source["events"]
                or derived["frame_count"] != source["frame_count"]
                or derived["fps"] != source["fps"]
                or derived["loop"] != source["loop"]
                or derived["start_cell"] != source["start_cell"]
            ):
                raise ValueError(f"Style-motion family source binding mismatch: {source['id']}")
    palette_path = verify_artifact(root, manifest["artifacts"]["palette"])
    palette = strict_json_file(palette_path)
    if canonical_json_bytes(palette) != palette_path.read_bytes():
        raise ValueError("Style-motion palette is not canonical JSON")
    if palette.get("format") != "nullvector-perceptual-palette-v1":
        raise ValueError("Style-motion palette format mismatch")
    index_path = verify_artifact(root, manifest["artifacts"]["frame_index"])
    _verify_frame_index(index_path, manifest)
    for layer_name in LAYER_NAMES:
        verify_artifact(root, manifest["artifacts"]["layers"][layer_name])
    return manifest
