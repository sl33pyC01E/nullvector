from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def array_sha256(label: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-motion-array-v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def categorical_sha256(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-motion-categorical-frame-v1\0")
    for name, values in (
        ("part", part),
        ("material", material),
        ("emission", emission),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def named_points_sha256(
    label: str,
    names: Sequence[str],
    points: Mapping[str, list[int]],
) -> str:
    payload = {name: list(points[name]) for name in names}
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-motion-points-v1\0")
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(payload))
    return digest.hexdigest()


def authority_sha256(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    joint_names: Sequence[str],
    joints: Mapping[str, list[int]],
    socket_names: Sequence[str],
    sockets: Mapping[str, list[int]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-motion-authority-v1\0")
    digest.update(categorical_sha256(part, material, emission).encode("ascii"))
    digest.update(named_points_sha256("joints", joint_names, joints).encode("ascii"))
    digest.update(named_points_sha256("sockets", socket_names, sockets).encode("ascii"))
    return digest.hexdigest()


def identity_style_sha256(
    *,
    source_id: str,
    source_seed: int,
    semantic_sha256: str,
    genome_sha256: str,
    training_arrays_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-motion-identity-v1\0")
    for value in (
        source_id,
        str(source_seed),
        semantic_sha256,
        genome_sha256,
        training_arrays_sha256,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def clip_presentation_sha256(
    *,
    identity_sha256: str,
    source_clip_sha256: str,
    events: list[dict[str, Any]],
    categorical_hashes: Sequence[str],
    authority_hashes: Sequence[str],
    presentation_hashes: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-motion-clip-presentation-v1\0")
    digest.update(identity_sha256.encode("ascii"))
    digest.update(source_clip_sha256.encode("ascii"))
    digest.update(canonical_json_bytes({"events": events}))
    for value in categorical_hashes:
        digest.update(value.encode("ascii"))
    for value in authority_hashes:
        digest.update(value.encode("ascii"))
    digest.update(np.ascontiguousarray(presentation_hashes).tobytes())
    return digest.hexdigest()


def png_bytes(pixels: np.ndarray) -> bytes:
    values = np.asarray(pixels)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] not in (3, 4):
        raise ValueError("PNG pixels must be uint8 RGB or RGBA")
    buffer = BytesIO()
    Image.fromarray(values).save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Encode a deterministic, bounded NPZ with fixed ZIP metadata/order."""

    output = BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
                raise ValueError(f"Unsafe deterministic NPZ member name: {name!r}")
            payload = BytesIO()
            np.lib.format.write_array(
                payload,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def artifact_record_from_bytes(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(payload), "sha256": sha256_bytes(payload)}
