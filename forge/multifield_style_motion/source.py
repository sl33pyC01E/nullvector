from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import zipfile

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image

from ..morphology import (
    FACING_NAMES,
    FAMILIES,
    LAYER_NAMES as SEMANTIC_LAYER_NAMES,
    MOTION_NAMES,
    MOTION_RENDERER_VERSION,
    RENDERER_VERSION,
    MorphologyGenome,
    render_specimen,
)
from ..morphology.constants import ROLE_NAMES
from .hashing import sha256_file
from .model import JOINT_NAMES, SOCKET_NAMES, LoadedMotionBank


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_INDEX_FORMAT = "nullvector-forge-lab-assets-v1"
SOURCE_BANK_FORMAT = "neural-morphology-motion-bank-v1"
SOURCE_CLIP_FORMAT = "neural-morphology-motion-manifest-v1"
EXPECTED_FRAME_COUNT = 4720
EXPECTED_CLIP_COUNT = 520
SOURCE_CLIP_COUNT = 65
SOURCE_FRAME_COUNT = 590
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
SOURCE_ARCHIVE_KEYS = {
    "layers",
    "tokens",
    "joints",
    "sockets",
    "phases",
    "frame_sha256",
    "clip_offsets",
    "clip_ids",
    "layer_names",
    "joint_names",
    "socket_names",
}
MOTION_SOURCE_FILES = (
    "forge/morphology/__init__.py",
    "forge/morphology/constants.py",
    "forge/morphology/contract.py",
    "forge/morphology/fields.py",
    "forge/morphology/genome.py",
    "forge/morphology/render.py",
    "forge/morphology/motion.py",
    "shared/schema/morphology_manifest.schema.json",
    "shared/schema/morphology_motion_manifest.schema.json",
)
SOURCE_BANK_KEYS = {
    "format",
    "families",
    "motion_names",
    "facing",
    "source_count",
    "sources",
    "clip_count",
    "frame_count",
    "disk_budget",
    "archive",
    "previews",
    "clips",
}
MOTION_SECTION_KEYS = {
    "status",
    "source_manifest",
    "source_manifest_sha256",
    "source_format",
    "renderer",
    "source_morphology_renderer",
    "families",
    "motions",
    "facings",
    "atlases",
    "clips",
    "clip_count",
    "frame_count",
}
ATLAS_ENTRY_KEYS = {
    "family",
    "source_id",
    "source_seed",
    "source_semantic_sha256",
    "source_genome_sha256",
    "source_renderer_version",
    "source_role_id",
    "source_role_name",
    "atlas",
    "atlas_sha256",
    "atlas_size",
    "columns",
    "rows",
    "cell_size",
    "frame_count",
    "palette",
    "genome",
}
CLIP_ENTRY_KEYS = {
    "id",
    "family",
    "motion",
    "facing",
    "fps",
    "loop",
    "frame_count",
    "start_cell",
    "atlas_columns",
    "cell_size",
    "clip_sha256",
    "frame_sha256",
    "events",
    "metrics",
}


def motion_source_hash(project_root: Path = PROJECT_ROOT) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    digest.update(b"nullvector-style-motion-upstream-source-v1\0")
    for relative in MOTION_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Motion source member missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"JSON source must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(payload, allow_nan=False)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Invalid strict UTF-8 JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _safe_source_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Source artifact path must be a string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe source artifact path: {relative!r}")
    if "\\" in relative:
        raise ValueError("Source artifact paths must use POSIX separators")
    target = (root.resolve() / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Source artifact escapes root: {relative}") from error
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"Source artifact must be a regular non-symlink file: {relative}")
    return target


@lru_cache(maxsize=2)
def _schema_validator(filename: str) -> Draft202012Validator:
    path = PROJECT_ROOT / "shared" / "schema" / filename
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_schema(payload: Mapping[str, Any], filename: str, label: str) -> None:
    errors = sorted(
        _schema_validator(filename).iter_errors(payload),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        rendered = []
        for error in errors[:8]:
            location = "/".join(map(str, error.absolute_path)) or "<root>"
            rendered.append(f"{location}: {error.message}")
        raise ValueError(f"{label} schema failure: {'; '.join(rendered)}")


def _validate_source_archive(
    path: Path,
    bank: Mapping[str, Any],
) -> None:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Motion semantic source archive exceeds the compressed bound")
    expected_members = {f"{name}.npy" for name in SOURCE_ARCHIVE_KEYS}
    with zipfile.ZipFile(path, "r") as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or set(names) != expected_members:
            raise ValueError("Motion semantic source archive ZIP members mismatch")
        total = 0
        for entry in entries:
            if PurePosixPath(entry.filename).name != entry.filename:
                raise ValueError("Motion semantic source archive contains unsafe nested members")
            if entry.file_size < 0 or entry.file_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError("Motion semantic source archive member exceeds its bound")
            total += entry.file_size
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("Motion semantic source archive exceeds its total uncompressed bound")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != SOURCE_ARCHIVE_KEYS:
            raise ValueError("Motion semantic source archive array keys mismatch")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
        expected_shapes = {
            "layers": (SOURCE_FRAME_COUNT, 12, 48, 48),
            "tokens": (SOURCE_FRAME_COUNT, 48, 48),
            "joints": (SOURCE_FRAME_COUNT, len(JOINT_NAMES), 2),
            "sockets": (SOURCE_FRAME_COUNT, len(SOCKET_NAMES), 2),
            "phases": (SOURCE_FRAME_COUNT,),
            "frame_sha256": (SOURCE_FRAME_COUNT,),
            "clip_offsets": (SOURCE_CLIP_COUNT + 1,),
            "clip_ids": (SOURCE_CLIP_COUNT,),
            "layer_names": (12,),
            "joint_names": (len(JOINT_NAMES),),
            "socket_names": (len(SOCKET_NAMES),),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(f"Motion semantic source archive {name} shape mismatch")
        for name in ("layers", "tokens", "joints", "sockets"):
            if arrays[name].dtype != np.uint8:
                raise ValueError(f"Motion semantic source archive {name} dtype mismatch")
        if arrays["phases"].dtype != np.float32 or arrays["clip_offsets"].dtype != np.uint32:
            raise ValueError("Motion semantic source archive phase/offset dtype mismatch")
        if arrays["layer_names"].tolist() != list(SEMANTIC_LAYER_NAMES):
            raise ValueError("Motion semantic source layer vocabulary mismatch")
        if arrays["joint_names"].tolist() != list(JOINT_NAMES):
            raise ValueError("Motion semantic source joint vocabulary mismatch")
        if arrays["socket_names"].tolist() != list(SOCKET_NAMES):
            raise ValueError("Motion semantic source socket vocabulary mismatch")
        clips = list(bank["clips"])
        if arrays["clip_ids"].tolist() != [clip["id"] for clip in clips]:
            raise ValueError("Motion semantic source clip order mismatch")
        offsets = arrays["clip_offsets"].astype(np.int64)
        if int(offsets[0]) != 0 or int(offsets[-1]) != SOURCE_FRAME_COUNT or np.any(np.diff(offsets) <= 0):
            raise ValueError("Motion semantic source clip offsets are malformed")
        expected_hashes = [value for clip in clips for value in clip["frame_sha256"]]
        if arrays["frame_sha256"].tolist() != expected_hashes:
            raise ValueError("Motion semantic source frame hashes disagree with the manifest")
        if any(int(offsets[index + 1] - offsets[index]) != clips[index]["frame_count"] for index in range(len(clips))):
            raise ValueError("Motion semantic source offsets disagree with clip frame counts")


def load_motion_bank(asset_index_path: Path) -> LoadedMotionBank:
    asset_index_path = Path(asset_index_path).resolve()
    index = _strict_json(asset_index_path)
    if index.get("format") != ASSET_INDEX_FORMAT:
        raise ValueError("Unsupported ForgeLab asset-index format")
    if index.get("pixel_filter") != "nearest" or index.get("errors") != []:
        raise ValueError("ForgeLab asset index is not a clean nearest-filter bank")
    if index.get("source_root") != "outputs/":
        raise ValueError("ForgeLab asset index source_root drifted from outputs/")
    generator = index.get("generator")
    if not isinstance(generator, dict) or generator.get("module") != "forge.forge_lab_sync" or generator.get("deterministic") is not True:
        raise ValueError("ForgeLab generator contract mismatch")
    forge_lab_source = PROJECT_ROOT / "forge" / "forge_lab_sync.py"
    if generator.get("source_sha256") != sha256_file(forge_lab_source):
        raise ValueError("ForgeLab asset index was produced by stale sync source")
    motion = index.get("motion")
    if not isinstance(motion, dict) or set(motion) != MOTION_SECTION_KEYS:
        raise ValueError("ForgeLab motion section keys mismatch")
    if (
        motion.get("status") != "ready"
        or motion.get("source_format") != SOURCE_BANK_FORMAT
        or motion.get("renderer") != MOTION_RENDERER_VERSION
        or motion.get("source_morphology_renderer") != RENDERER_VERSION
        or motion.get("families") != list(FAMILIES)
        or motion.get("motions") != list(MOTION_NAMES)
        or motion.get("facings") != list(FACING_NAMES)
        or motion.get("clip_count") != EXPECTED_CLIP_COUNT
        or motion.get("frame_count") != EXPECTED_FRAME_COUNT
    ):
        raise ValueError("ForgeLab full motion matrix contract mismatch")

    asset_root = asset_index_path.parent.resolve()
    source_manifest_path = _safe_source_file(asset_root, motion["source_manifest"])
    actual_source_hash = sha256_file(source_manifest_path)
    if actual_source_hash != motion["source_manifest_sha256"]:
        raise ValueError("ForgeLab motion source-manifest hash mismatch")
    source_manifest = _strict_json(source_manifest_path)
    if set(source_manifest) != SOURCE_BANK_KEYS or source_manifest.get("format") != SOURCE_BANK_FORMAT:
        raise ValueError("Canonical north motion source-bank contract mismatch")
    if (
        source_manifest.get("families") != list(FAMILIES)
        or source_manifest.get("motion_names") != list(MOTION_NAMES)
        or source_manifest.get("facing") != "north"
        or source_manifest.get("source_count") != len(FAMILIES)
        or source_manifest.get("clip_count") != SOURCE_CLIP_COUNT
        or source_manifest.get("frame_count") != SOURCE_FRAME_COUNT
    ):
        raise ValueError("Canonical north motion source-bank matrix mismatch")
    original_manifest_path = PROJECT_ROOT / "outputs" / "morphology_motion" / "morphology_motion_manifest.json"
    if sha256_file(original_manifest_path) != actual_source_hash:
        raise ValueError("ForgeLab source-manifest copy disagrees with the immutable output source")

    source_entries = source_manifest.get("sources")
    if not isinstance(source_entries, list) or len(source_entries) != len(FAMILIES):
        raise ValueError("Motion source specimens do not cover all families")
    sources: dict[str, Mapping[str, Any]] = {}
    for expected_family, entry in zip(FAMILIES, source_entries, strict=True):
        if not isinstance(entry, dict):
            raise ValueError("Motion source specimen entries must be objects")
        _validate_schema(entry, "morphology_manifest.schema.json", f"source specimen {expected_family}")
        if entry.get("family") != expected_family or expected_family in sources:
            raise ValueError("Motion source specimen family order/uniqueness mismatch")
        genome = MorphologyGenome.from_dict(dict(entry["genome"]))
        specimen = render_specimen(genome)
        if specimen.manifest["hashes"] != entry["hashes"]:
            raise ValueError(f"Replayed source specimen hashes mismatch for {expected_family}")
        sources[expected_family] = entry

    source_clips = source_manifest.get("clips")
    if not isinstance(source_clips, list) or len(source_clips) != SOURCE_CLIP_COUNT:
        raise ValueError("Canonical north source clips are incomplete")
    expected_source_keys = [
        (family, motion_name, "north")
        for family in FAMILIES
        for motion_name in MOTION_NAMES
    ]
    source_lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for entry, expected_key in zip(source_clips, expected_source_keys, strict=True):
        if not isinstance(entry, dict):
            raise ValueError("Canonical north source clip entries must be objects")
        _validate_schema(entry, "morphology_motion_manifest.schema.json", f"source clip {expected_key}")
        key = (str(entry["family"]), str(entry["motion"]), str(entry["facing"]))
        if key != expected_key:
            raise ValueError("Canonical north source clip order mismatch")
        source_lookup[key] = entry

    archive_record = source_manifest.get("archive")
    if not isinstance(archive_record, dict) or archive_record.get("file") != "morphology_motion_semantics.npz":
        raise ValueError("Canonical motion source archive record is malformed")
    source_archive_path = PROJECT_ROOT / "outputs" / "morphology_motion" / archive_record["file"]
    if not source_archive_path.is_file() or source_archive_path.is_symlink():
        raise ValueError("Canonical motion source archive is missing or unsafe")
    if sha256_file(source_archive_path) != archive_record.get("sha256"):
        raise ValueError("Canonical motion source archive hash mismatch")
    _validate_source_archive(source_archive_path, source_manifest)

    atlas_entries = motion.get("atlases")
    if not isinstance(atlas_entries, list) or len(atlas_entries) != len(FAMILIES):
        raise ValueError("ForgeLab motion atlas family coverage mismatch")
    atlases: dict[str, Mapping[str, Any]] = {}
    for expected_family, entry in zip(FAMILIES, atlas_entries, strict=True):
        if not isinstance(entry, dict) or set(entry) != ATLAS_ENTRY_KEYS:
            raise ValueError(f"ForgeLab atlas entry contract mismatch for {expected_family}")
        if entry.get("family") != expected_family or expected_family in atlases:
            raise ValueError("ForgeLab atlas family order/uniqueness mismatch")
        source = sources[expected_family]
        if (
            entry.get("source_id") != source["id"]
            or entry.get("source_seed") != source["seed"]
            or entry.get("source_semantic_sha256") != source["hashes"]["semantic_sha256"]
            or entry.get("source_genome_sha256") != source["hashes"]["genome_sha256"]
            or entry.get("source_role_id") != source["training_contract"]["role_id"]
            or entry.get("source_role_name") != source["training_contract"]["role_name"]
            or entry.get("columns") != 16
            or entry.get("cell_size") != 48
            or entry.get("frame_count") != EXPECTED_FRAME_COUNT // len(FAMILIES)
        ):
            raise ValueError(f"ForgeLab atlas source/layout mismatch for {expected_family}")
        atlas_path = _safe_source_file(asset_root, entry["atlas"])
        if sha256_file(atlas_path) != entry["atlas_sha256"]:
            raise ValueError(f"ForgeLab source atlas hash mismatch for {expected_family}")
        with Image.open(atlas_path) as image:
            if image.mode != "RGBA" or list(image.size) != entry["atlas_size"]:
                raise ValueError(f"ForgeLab source atlas image contract mismatch for {expected_family}")
        atlases[expected_family] = entry

    clip_entries = motion.get("clips")
    if not isinstance(clip_entries, list) or len(clip_entries) != EXPECTED_CLIP_COUNT:
        raise ValueError("ForgeLab full motion clips are incomplete")
    expected_full_keys = [
        (family, motion_name, facing)
        for family in FAMILIES
        for motion_name in MOTION_NAMES
        for facing in FACING_NAMES
    ]
    clips_by_family: dict[str, list[Mapping[str, Any]]] = {family: [] for family in FAMILIES}
    family_cursor = {family: 0 for family in FAMILIES}
    for entry, expected_key in zip(clip_entries, expected_full_keys, strict=True):
        if not isinstance(entry, dict) or set(entry) != CLIP_ENTRY_KEYS:
            raise ValueError(f"ForgeLab full clip entry contract mismatch: {expected_key}")
        key = (str(entry["family"]), str(entry["motion"]), str(entry["facing"]))
        if key != expected_key:
            raise ValueError("ForgeLab full clip order mismatch")
        family = key[0]
        if entry["start_cell"] != family_cursor[family] or entry["atlas_columns"] != 16 or entry["cell_size"] != 48:
            raise ValueError(f"ForgeLab clip atlas cursor mismatch: {entry['id']}")
        if len(entry["frame_sha256"]) != entry["frame_count"]:
            raise ValueError(f"ForgeLab clip frame hash count mismatch: {entry['id']}")
        if not entry["events"] or any(
            event.get("frame", -1) < 0
            or event.get("frame", -1) >= entry["frame_count"]
            or event.get("socket") not in SOCKET_NAMES
            for event in entry["events"]
        ):
            raise ValueError(f"ForgeLab clip events are malformed: {entry['id']}")
        if key[2] == "north":
            source_entry = source_lookup[key]
            if (
                entry["clip_sha256"] != source_entry["hashes"]["clip_sha256"]
                or entry["frame_sha256"] != source_entry["frame_sha256"]
                or entry["events"] != source_entry["events"]
            ):
                raise ValueError(f"ForgeLab north clip disagrees with canonical source: {entry['id']}")
        family_cursor[family] += int(entry["frame_count"])
        clips_by_family[family].append(entry)
    expected_family_frames = EXPECTED_FRAME_COUNT // len(FAMILIES)
    if any(value != expected_family_frames for value in family_cursor.values()):
        raise ValueError("ForgeLab family frame totals mismatch")

    return LoadedMotionBank(
        asset_root=asset_root,
        asset_index_path=asset_index_path,
        asset_index_sha256=sha256_file(asset_index_path),
        asset_index_bytes=asset_index_path.stat().st_size,
        index=index,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=actual_source_hash,
        source_manifest_bytes=source_manifest_path.stat().st_size,
        source_manifest=source_manifest,
        source_archive_path=source_archive_path,
        source_archive_sha256=sha256_file(source_archive_path),
        source_archive_bytes=source_archive_path.stat().st_size,
        sources=sources,
        atlases=atlases,
        clips_by_family={family: tuple(entries) for family, entries in clips_by_family.items()},
    )
