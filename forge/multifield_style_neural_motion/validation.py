from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image

from ..morphology import FACING_NAMES, FAMILIES, MOTION_NAMES
from ..morphology.motion import DEFAULT_FPS, DEFAULT_FRAME_COUNTS, LOOPING_MOTIONS
from ..multifield_style.source import PROJECT_ROOT
from ..multifield_style_motion.hashing import array_sha256, canonical_json_bytes
from ..multifield_style_motion.io import verify_artifact
from ..multifield_style_motion.model import ATLAS_COLUMNS, IMAGE_SIZE, LAYER_NAMES
from ..multifield_style_motion.rendering import chebyshev_ring, dilate_chebyshev
from ..neural_rig_bridge import validate_binding_schema
from ..neural_rig_bridge.hashing import canonical_json_hash
from .schema import IDENTITY_SCHEMA, validate_schema


FRAME_INDEX_FORMAT = "nullvector-multifield-style-neural-motion-frame-index-v1"
MOTION_MANIFESTS_FORMAT = "nullvector-neural-motion-manifest-collection-v1"
FRAME_INDEX_KEYS = frozenset(
    {
        "format",
        "sample_id",
        "family",
        "layer_names",
        "clip_ids",
        "clip_offsets",
        "phases",
        "emission_pulses",
        "motion_frame_sha256",
        "bound_frame_sha256",
        "categorical_sha256",
        "aligned_fields_sha256",
        "driver_index_sha256",
        "joint_sha256",
        "socket_sha256",
        "presentation_sha256",
    }
)
MAX_FRAME_INDEX_BYTES = 8 * 1024 * 1024
MAX_FRAME_INDEX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
HEX = frozenset("0123456789abcdef")


def strict_json_file(path: Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"JSON must be a regular non-symlink file: {path}")
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"JSON exceeds its byte bound: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(payload, allow_nan=False)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Invalid strict UTF-8 JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - HEX)


@lru_cache(maxsize=1)
def _motion_validator() -> Draft202012Validator:
    path = PROJECT_ROOT / "shared" / "schema" / "neural_rig_motion.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_motion_schema(payload: Mapping[str, Any]) -> None:
    errors = sorted(
        _motion_validator().iter_errors(dict(payload)),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(map(str, error.absolute_path)) or "<root>"
        raise ValueError(f"Neural source motion schema failed at {location}: {error.message}")


def _expected_clip_keys() -> list[tuple[str, str]]:
    return [(motion, facing) for motion in MOTION_NAMES for facing in FACING_NAMES]


def _artifact_paths(manifest: Mapping[str, Any]) -> dict[str, str]:
    prefix = f"identities/{manifest['family']}/{manifest['sample_id']}"
    expected = {
        "palette": f"{prefix}/palette.json",
        "binding_manifest": f"{prefix}/binding_manifest.json",
        "motion_manifests": f"{prefix}/motion_manifests.json",
        "frame_index": f"{prefix}/frame_index.npz",
    }
    expected.update({f"layers.{name}": f"{prefix}/{name}.png" for name in LAYER_NAMES})
    return expected


def _load_atlases(root: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    expected_size = (
        int(manifest["layout"]["columns"]) * IMAGE_SIZE,
        int(manifest["layout"]["rows"]) * IMAGE_SIZE,
    )
    atlases: dict[str, np.ndarray] = {}
    for name in LAYER_NAMES:
        path = verify_artifact(root, manifest["artifacts"]["layers"][name])
        try:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "RGBA" or image.size != expected_size:
                    raise ValueError(f"Neural motion {name} atlas format/geometry mismatch")
                image.load()
                atlases[name] = np.asarray(image, dtype=np.uint8).copy()
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError(f"Neural motion {name} atlas is not a bounded RGBA PNG: {error}") from error
    return atlases


def _composite_cells(cells: Mapping[str, np.ndarray]) -> np.ndarray:
    image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
    for name in ("bloom_r2", "bloom_r1", "aura", "outline", "base", "emission_core"):
        image = Image.alpha_composite(image, Image.fromarray(cells[name], mode="RGBA"))
    return np.asarray(image, dtype=np.uint8).copy()


def _verify_frame_index(
    path: Path,
    manifest: Mapping[str, Any],
    source_clips: list[Mapping[str, Any]],
    atlases: Mapping[str, np.ndarray],
) -> None:
    if path.stat().st_size > MAX_FRAME_INDEX_BYTES:
        raise ValueError("Neural motion frame index exceeds its compressed bound")
    expected_members = {f"{name}.npy" for name in FRAME_INDEX_KEYS}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            entries = archive.infolist()
            if len(entries) != len(expected_members) or {entry.filename for entry in entries} != expected_members:
                raise ValueError("Neural motion frame index ZIP members mismatch")
            if len({entry.filename for entry in entries}) != len(entries):
                raise ValueError("Neural motion frame index has duplicate ZIP members")
            if any("/" in entry.filename or "\\" in entry.filename for entry in entries):
                raise ValueError("Neural motion frame index has nested ZIP members")
            if sum(entry.file_size for entry in entries) > MAX_FRAME_INDEX_UNCOMPRESSED_BYTES:
                raise ValueError("Neural motion frame index exceeds its uncompressed bound")
    except zipfile.BadZipFile as error:
        raise ValueError("Neural motion frame index is not a valid NPZ") from error
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FRAME_INDEX_KEYS:
            raise ValueError("Neural motion frame index array keys mismatch")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    frame_count = int(manifest["frame_count"])
    clip_count = int(manifest["clip_count"])
    one_per_frame = (
        "phases",
        "emission_pulses",
        "motion_frame_sha256",
        "bound_frame_sha256",
        "categorical_sha256",
        "aligned_fields_sha256",
        "driver_index_sha256",
        "joint_sha256",
        "socket_sha256",
    )
    expected_shapes = {
        "format": (1,),
        "sample_id": (1,),
        "family": (1,),
        "layer_names": (len(LAYER_NAMES),),
        "clip_ids": (clip_count,),
        "clip_offsets": (clip_count + 1,),
        "presentation_sha256": (frame_count, len(LAYER_NAMES)),
        **{name: (frame_count,) for name in one_per_frame},
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Neural motion frame index {name} shape mismatch")
    if (
        arrays["clip_offsets"].dtype != np.uint32
        or arrays["phases"].dtype != np.float32
        or arrays["emission_pulses"].dtype != np.uint8
    ):
        raise ValueError("Neural motion frame index numeric dtypes mismatch")
    for name in FRAME_INDEX_KEYS - {"clip_offsets", "phases", "emission_pulses"}:
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"Neural motion frame index {name} must be Unicode")
    if arrays["format"].tolist() != [FRAME_INDEX_FORMAT]:
        raise ValueError("Neural motion frame index format mismatch")
    if arrays["sample_id"].tolist() != [manifest["sample_id"]]:
        raise ValueError("Neural motion frame index sample mismatch")
    if arrays["family"].tolist() != [manifest["family"]]:
        raise ValueError("Neural motion frame index family mismatch")
    if arrays["layer_names"].tolist() != list(LAYER_NAMES):
        raise ValueError("Neural motion frame index layer order mismatch")
    clips = list(manifest["clips"])
    if arrays["clip_ids"].tolist() != [clip["id"] for clip in clips]:
        raise ValueError("Neural motion frame index clip order mismatch")
    expected_offsets = [0]
    for clip in clips:
        expected_offsets.append(expected_offsets[-1] + int(clip["frame_count"]))
    if arrays["clip_offsets"].astype(np.int64).tolist() != expected_offsets or expected_offsets[-1] != frame_count:
        raise ValueError("Neural motion frame index clip offsets mismatch")
    if not np.all(np.isfinite(arrays["phases"])) or np.any(arrays["phases"] < 0) or np.any(arrays["phases"] > 1):
        raise ValueError("Neural motion phases are outside [0, 1]")
    if np.any(arrays["emission_pulses"] > 3):
        raise ValueError("Neural motion emission pulses are outside [0, 3]")
    for name in (
        "motion_frame_sha256",
        "bound_frame_sha256",
        "categorical_sha256",
        "aligned_fields_sha256",
        "driver_index_sha256",
        "joint_sha256",
        "socket_sha256",
    ):
        if any(not _is_sha(value) for value in arrays[name].tolist()):
            raise ValueError(f"Neural motion frame index {name} contains invalid SHA-256 values")
    for row in arrays["presentation_sha256"].tolist():
        if any(not _is_sha(value) for value in row):
            raise ValueError("Neural motion presentation hashes contain invalid SHA-256 values")

    source_frames = [frame for clip in source_clips for frame in clip["frames"]]
    if len(source_frames) != frame_count:
        raise ValueError("Neural source motion frame accounting mismatch")
    expected_source = {
        "phases": np.asarray([frame["phase"] for frame in source_frames], dtype=np.float32),
        "emission_pulses": np.asarray([frame["emission_pulse"] for frame in source_frames], dtype=np.uint8),
        "motion_frame_sha256": [frame["motion_frame_sha256"] for frame in source_frames],
        "bound_frame_sha256": [frame["bound_frame_sha256"] for frame in source_frames],
        "aligned_fields_sha256": [frame["raw_fields_sha256"] for frame in source_frames],
        "driver_index_sha256": [frame["driver_index_sha256"] for frame in source_frames],
    }
    for name, expected in expected_source.items():
        actual = arrays[name]
        if isinstance(expected, np.ndarray):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Neural motion frame index {name} source mismatch")
        elif actual.tolist() != expected:
            raise ValueError(f"Neural motion frame index {name} source mismatch")

    for index in range(frame_count):
        row, column = divmod(index, ATLAS_COLUMNS)
        y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
        cells = {
            name: values[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE]
            for name, values in atlases.items()
        }
        expected_hashes = arrays["presentation_sha256"][index].tolist()
        actual_hashes = [array_sha256(name, cells[name]) for name in LAYER_NAMES]
        if actual_hashes != expected_hashes:
            raise ValueError(f"Neural motion atlas/index presentation mismatch at cell {index}")
        body = cells["base"][..., 3] > 0
        outline = cells["outline"][..., 3] > 0
        if not np.array_equal(outline, chebyshev_ring(body, 1)):
            raise ValueError(f"Neural motion outline radius mismatch at cell {index}")
        if np.any(cells["aura"][..., 3][body] > 0):
            raise ValueError(f"Neural motion aura overlaps opaque body at cell {index}")
        if not np.all(np.isin(cells["base"][..., 3], (0, 255))):
            raise ValueError(f"Neural motion base alpha is not categorical at cell {index}")
        if not np.all(np.isin(cells["outline"][..., 3], (0, 255))):
            raise ValueError(f"Neural motion outline alpha is not categorical at cell {index}")
        if not np.all(np.isin(cells["emission_core"][..., 3], (0, 255))):
            raise ValueError(f"Neural motion emission alpha is not categorical at cell {index}")
        if np.any((cells["emission_core"][..., 3] > 0) & ~body):
            raise ValueError(f"Neural motion emission core escapes body at cell {index}")
        bloom_r1 = cells["bloom_r1"][..., 3] > 0
        bloom_r2 = cells["bloom_r2"][..., 3] > 0
        if np.any(bloom_r1 & bloom_r2) or np.any(bloom_r2 & ~dilate_chebyshev(bloom_r1, 1)):
            raise ValueError(f"Neural motion bloom ring adjacency mismatch at cell {index}")
        if any(
            bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
            for mask in (outline, bloom_r1, bloom_r2)
        ):
            raise ValueError(f"Neural motion effect ring is clipped at cell {index}")
        for effect in ("aura", "bloom_r1", "bloom_r2"):
            alpha = cells[effect][..., 3]
            if np.any(alpha == 255):
                raise ValueError(f"Neural motion {effect} contains clipped opacity at cell {index}")
        if not np.array_equal(cells["composite"], _composite_cells(cells)):
            raise ValueError(f"Neural motion composite is not an exact layer composite at cell {index}")


def load_verified_identity_manifest(
    output_root: Path,
    family: str,
    *,
    sample_id: str | None = None,
    compiler: Mapping[str, Any] | None = None,
    generation_manifest_sha256: str | None = None,
    style_manifest_sha256: str | None = None,
    static_palette_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown neural motion family: {family!r}")
    root = Path(output_root).resolve()
    family_root = root / "identities" / family
    if sample_id is None:
        manifests = list(family_root.glob("*/identity_manifest.json"))
        if len(manifests) != 1:
            raise ValueError(f"Neural motion family output count mismatch: {family}")
        path = manifests[0]
    else:
        path = family_root / sample_id / "identity_manifest.json"
    manifest = strict_json_file(path)
    validate_schema(manifest, IDENTITY_SCHEMA)
    if canonical_json_bytes(manifest) != path.read_bytes():
        raise ValueError("Neural motion identity manifest is not canonical JSON")
    if manifest["family"] != family or (sample_id is not None and manifest["sample_id"] != sample_id):
        raise ValueError("Neural motion identity family/sample mismatch")
    if manifest["condition"]["sample_id"] != manifest["sample_id"] or manifest["condition"]["morphology_name"] != family:
        raise ValueError("Neural motion identity condition mismatch")
    if compiler is not None and manifest["compiler"] != dict(compiler):
        raise ValueError("Neural motion identity compiler provenance mismatch")
    if generation_manifest_sha256 is not None and manifest["source"]["generation_manifest_sha256"] != generation_manifest_sha256:
        raise ValueError("Neural motion generation provenance mismatch")
    if style_manifest_sha256 is not None and manifest["source"]["style_manifest_sha256"] != style_manifest_sha256:
        raise ValueError("Neural motion style provenance mismatch")

    expected_paths = _artifact_paths(manifest)
    for key, expected in expected_paths.items():
        if key.startswith("layers."):
            record = manifest["artifacts"]["layers"][key.split(".", 1)[1]]
        else:
            record = manifest["artifacts"][key]
        if record["path"] != expected:
            raise ValueError(f"Neural motion artifact path mismatch: {key}")

    palette_path = verify_artifact(root, manifest["artifacts"]["palette"])
    palette = strict_json_file(palette_path)
    if canonical_json_bytes(palette) != palette_path.read_bytes() or palette.get("format") != "nullvector-perceptual-palette-v1":
        raise ValueError("Neural motion palette is not a canonical perceptual palette")
    if manifest["artifacts"]["palette"]["sha256"] != manifest["source"]["static_palette_sha256"]:
        raise ValueError("Neural motion palette does not match its static parent hash")
    if static_palette_artifact is not None and (
        manifest["artifacts"]["palette"]["bytes"] != static_palette_artifact.get("bytes")
        or manifest["artifacts"]["palette"]["sha256"] != static_palette_artifact.get("sha256")
    ):
        raise ValueError("Neural motion palette artifact diverges from the loaded static parent")

    binding_path = verify_artifact(root, manifest["artifacts"]["binding_manifest"])
    binding = strict_json_file(binding_path)
    if canonical_json_bytes(binding) != binding_path.read_bytes():
        raise ValueError("Neural motion binding manifest is not canonical JSON")
    binding_schema_errors = validate_binding_schema(binding)
    if binding_schema_errors:
        raise ValueError("Neural motion binding schema failed: " + "; ".join(binding_schema_errors[:8]))
    if (
        binding.get("hashes", {}).get("binding_sha256") != manifest["source"]["binding_sha256"]
        or binding.get("id") != manifest["sample_id"]
        or binding.get("condition", {}).get("family") != family
        or binding.get("source", {}).get("raw_fields_sha256") != manifest["source"]["raw_fields_sha256"]
        or binding.get("source", {}).get("binder_source_sha256") != manifest["compiler"]["bridge_source_sha256"]
    ):
        raise ValueError("Neural motion binding artifact provenance mismatch")
    binding_base = {key: value for key, value in binding.items() if key != "hashes"}
    if binding["hashes"]["binding_sha256"] != canonical_json_hash(binding_base):
        raise ValueError("Neural motion binding artifact hash is not canonical")

    expected_keys = _expected_clip_keys()
    clips = list(manifest["clips"])
    actual_keys = [(clip["motion"], clip["facing"]) for clip in clips]
    if actual_keys != expected_keys:
        raise ValueError("Neural motion identity clip matrix order mismatch")
    cursor = 0
    for clip, (motion, facing) in zip(clips, expected_keys, strict=True):
        expected_id = f"{manifest['sample_id']}__{motion}__{facing}"
        if (
            clip["id"] != expected_id
            or clip["start_cell"] != cursor
            or clip["frame_count"] != DEFAULT_FRAME_COUNTS[motion]
            or clip["fps"] != DEFAULT_FPS[motion]
            or clip["loop"] != (motion in LOOPING_MOTIONS)
        ):
            raise ValueError(f"Neural motion identity clip contract mismatch: {expected_id}")
        cursor += int(clip["frame_count"])
    if cursor != manifest["frame_count"]:
        raise ValueError("Neural motion identity frame accounting mismatch")

    motions_path = verify_artifact(root, manifest["artifacts"]["motion_manifests"])
    motions = strict_json_file(motions_path)
    if canonical_json_bytes(motions) != motions_path.read_bytes():
        raise ValueError("Neural motion source manifest collection is not canonical JSON")
    if motions.get("format") != MOTION_MANIFESTS_FORMAT or motions.get("sample_id") != manifest["sample_id"]:
        raise ValueError("Neural motion source manifest collection header mismatch")
    source_clips = motions.get("clips")
    if not isinstance(source_clips, list) or len(source_clips) != len(clips):
        raise ValueError("Neural motion source manifest collection length mismatch")
    for derived, source, (motion, facing) in zip(clips, source_clips, expected_keys, strict=True):
        if not isinstance(source, Mapping):
            raise ValueError("Neural source motion entry must be an object")
        _validate_motion_schema(source)
        source_base = {key: value for key, value in source.items() if key != "hashes"}
        if source["hashes"]["clip_sha256"] != canonical_json_hash(source_base):
            raise ValueError(f"Neural source motion hash is not canonical: {derived['id']}")
        for expected_index, frame in enumerate(source["frames"]):
            if frame["index"] != expected_index:
                raise ValueError(f"Neural source motion frame order mismatch: {derived['id']}")
            frame_base = {key: value for key, value in frame.items() if key != "motion_frame_sha256"}
            if frame["motion_frame_sha256"] != canonical_json_hash(frame_base):
                raise ValueError(f"Neural source motion frame hash is not canonical: {derived['id']}#{expected_index}")
        if (
            source["id"] != derived["id"]
            or source["motion"] != motion
            or source["facing"] != facing
            or source["fps"] != derived["fps"]
            or source["loop"] != derived["loop"]
            or source["frame_count"] != derived["frame_count"]
            or source["events"] != derived["events"]
            or source["hashes"]["clip_sha256"] != derived["source_clip_sha256"]
            or source["binding_sha256"] != manifest["source"]["binding_sha256"]
            or source["source_raw_fields_sha256"] != manifest["source"]["raw_fields_sha256"]
            or source["binder_source_sha256"] != manifest["compiler"]["bridge_source_sha256"]
        ):
            raise ValueError(f"Neural source/derived clip mismatch: {derived['id']}")

    atlases = _load_atlases(root, manifest)
    index_path = verify_artifact(root, manifest["artifacts"]["frame_index"])
    _verify_frame_index(index_path, manifest, source_clips, atlases)
    return manifest
