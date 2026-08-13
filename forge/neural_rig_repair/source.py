from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import zipfile

from jsonschema import Draft202012Validator
import numpy as np

from ..multifield_style_neural_motion.source import load_neural_motion_source
from ..multifield_style_neural_motion.style_parent import load_neural_style_parent
from ..neural_rig_bridge.hashing import aligned_fields_hash
from .constants import (
    MAX_JSON_BYTES,
    MAX_RAW_ARCHIVE_BYTES,
    MAX_RAW_MEMBER_BYTES,
    MAX_RAW_UNCOMPRESSED_BYTES,
    PROJECT_ROOT,
)
from .hashing import sha256_bytes, sha256_file
from .model import RepairSource, RepairSourceSample, readonly_array


RAW_FORMAT = "nullvector-multifield-raw-sample-v1"
RAW_SCHEMA_PATH = PROJECT_ROOT / "shared" / "schema" / "multifield_raw_sample.schema.json"
RAW_ARRAY_SPECS = {
    "part": ((48, 48), np.dtype(np.uint8)),
    "material": ((48, 48), np.dtype(np.uint8)),
    "emission": ((48, 48), np.dtype(np.uint8)),
    "guide": ((8, 48, 48), np.dtype(np.float32)),
    "genes": ((24,), np.dtype(np.float32)),
    "target_part": ((48, 48), np.dtype(np.uint8)),
    "target_material": ((48, 48), np.dtype(np.uint8)),
    "target_emission": ((48, 48), np.dtype(np.uint8)),
    "morphology": ((1,), np.dtype(np.uint8)),
    "subtype": ((1,), np.dtype(np.uint8)),
    "role": ((1,), np.dtype(np.uint8)),
    "source_index": ((1,), np.dtype(np.int64)),
    "corpus_seed": ((1,), np.dtype(np.uint32)),
    "sample_seed": ((1,), np.dtype(np.uint64)),
}
RAW_KEYS = frozenset((*RAW_ARRAY_SPECS, "format"))


def _strict_json(payload: bytes, *, label: str, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if not payload or len(payload) > maximum:
        raise ValueError(f"{label} violates the bounded JSON size")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _safe_path(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} path must be nonempty canonical POSIX text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe {label} path: {relative!r}")
    root = Path(root).resolve()
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"{label} escapes its source root")
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return target


def _verify_record(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} artifact record keys are not exact")
    path = _safe_path(root, record["path"], label=label)
    if type(record["bytes"]) is not int or record["bytes"] < 1:
        raise ValueError(f"{label} artifact byte count is malformed")
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"{label} artifact byte count mismatch")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} artifact SHA-256 mismatch")
    return path


def _preflight_npz(payload: bytes) -> None:
    if not payload or len(payload) > MAX_RAW_ARCHIVE_BYTES:
        raise ValueError("Raw archive violates the bounded container size")
    expected = {f"{name}.npy" for name in RAW_KEYS}
    try:
        with zipfile.ZipFile(BytesIO(payload), "r") as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("Raw archive ZIP member registry is not exact")
            total = 0
            for entry in entries:
                if PurePosixPath(entry.filename).name != entry.filename:
                    raise ValueError("Raw archive contains a nested or unsafe member")
                if entry.file_size < 0 or entry.file_size > MAX_RAW_MEMBER_BYTES:
                    raise ValueError("Raw archive member exceeds the uncompressed bound")
                if entry.compress_size < 0:
                    raise ValueError("Raw archive member has an invalid compressed size")
                total += entry.file_size
            if total > MAX_RAW_UNCOMPRESSED_BYTES:
                raise ValueError("Raw archive exceeds the total uncompressed bound")
    except zipfile.BadZipFile as error:
        raise ValueError(f"Raw archive is not a valid NPZ: {error}") from error


def _load_raw_arrays(payload: bytes) -> dict[str, np.ndarray]:
    _preflight_npz(payload)
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(RAW_KEYS):
                raise ValueError("Raw archive array registry changed after preflight")
            format_values = np.asarray(archive["format"])
            if format_values.shape != (1,) or format_values.dtype.kind not in "US":
                raise ValueError("Raw archive format scalar is malformed")
            if str(format_values[0]) != RAW_FORMAT:
                raise ValueError("Raw archive format is unsupported")
            result = {
                name: np.asarray(archive[name]).copy() for name in RAW_ARRAY_SPECS
            }
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"Raw archive could not be loaded safely: {error}") from error
    for name, (shape, dtype) in RAW_ARRAY_SPECS.items():
        values = result[name]
        if values.shape != shape or values.dtype != dtype:
            raise ValueError(f"Raw archive {name} must be {dtype} {shape}")
    for name in ("guide", "genes"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Raw archive {name} contains non-finite values")
    if bool((result["guide"] < 0.0).any() or (result["guide"] > 1.0).any()):
        raise ValueError("Raw archive guide leaves [0, 1]")
    return result


def _validate_raw_manifest(payload: Mapping[str, Any]) -> None:
    schema = json.loads(RAW_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        rendered = []
        for error in errors[:8]:
            location = "/".join(map(str, error.absolute_path)) or "<root>"
            rendered.append(f"{location}: {error.message}")
        raise ValueError("Raw manifest schema failure: " + "; ".join(rendered))


def _project_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Repair sources must remain inside the project root")
    return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()


def load_repair_source(
    generation_manifest: Path,
    style_manifest: Path,
) -> RepairSource:
    generation_manifest = Path(generation_manifest).resolve()
    style_manifest = Path(style_manifest).resolve()
    for path, label in (
        (generation_manifest, "generation manifest"),
        (style_manifest, "style manifest"),
    ):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError(f"{label} violates the regular bounded file contract")
        _project_relative(path)

    source = load_neural_motion_source(generation_manifest)
    style = load_neural_style_parent(style_manifest, source)
    entries = list(source.bank.manifest["samples"])
    candidates = [
        candidate
        for family in source.candidates_by_family
        for candidate in source.candidates_by_family[family]
    ]
    candidates.sort(key=lambda candidate: candidate.sample.condition.ordinal)
    if len(candidates) != 80 or len(entries) != 80:
        raise ValueError("Repair source must contain the exact 80-sample production bank")

    samples: list[RepairSourceSample] = []
    for ordinal, (candidate, entry) in enumerate(zip(candidates, entries, strict=True)):
        condition = candidate.sample.condition
        condition_record = entry["condition"]
        expected_condition_keys = {
            *condition.as_dict(),
            "grid_mode",
            "source_index",
            "variation",
        }
        if (
            condition.ordinal != ordinal
            or not isinstance(condition_record, Mapping)
            or set(condition_record) != expected_condition_keys
            or any(
                condition_record[key] != value
                for key, value in condition.as_dict().items()
            )
            or condition_record["grid_mode"] != "stratified"
            or condition_record["variation"]
            != int(condition.sample_id.rsplit("_v", maxsplit=1)[1])
        ):
            raise ValueError("Repair source condition registry drifted")
        raw_manifest_record = dict(entry["raw_manifest"])
        raw_manifest_path = _verify_record(
            source.bank.root,
            raw_manifest_record,
            label=f"raw manifest {condition.sample_id}",
        )
        raw_manifest_bytes = raw_manifest_path.read_bytes()
        raw_manifest = _strict_json(
            raw_manifest_bytes,
            label=f"raw manifest {condition.sample_id}",
        )
        _validate_raw_manifest(raw_manifest)
        raw_archive_record = dict(raw_manifest["artifacts"]["fields"])
        raw_archive_path = _verify_record(
            source.bank.root,
            raw_archive_record,
            label=f"raw archive {condition.sample_id}",
        )
        if raw_archive_path != candidate.raw_archive_path:
            raise ValueError("Repair raw archive disagrees with the generation registry")
        raw_archive_bytes = raw_archive_path.read_bytes()
        arrays = _load_raw_arrays(raw_archive_bytes)
        if sha256_bytes(raw_archive_bytes) != raw_archive_record["sha256"]:
            raise ValueError("Repair raw archive bytes changed after verification")
        for name, loaded in (
            ("part", candidate.sample.fields.part),
            ("material", candidate.sample.fields.material),
            ("emission", candidate.sample.fields.emission),
        ):
            if not np.array_equal(arrays[name], loaded):
                raise ValueError(f"Raw {name} differs from compiled immutable fields")
        raw_hash = aligned_fields_hash(
            arrays["part"], arrays["material"], arrays["emission"]
        )
        if (
            raw_hash != entry["raw_fields_sha256"]
            or raw_hash != entry["compiled_fields_sha256"]
            or raw_manifest["raw_fields_sha256"] != raw_hash
            or entry["postprocess"]["changed_pixels"] != 0
            or entry["postprocess"]["changed_fraction"] != 0.0
        ):
            raise ValueError("Repair requires byte-identical raw and compiled categorical fields")
        scalar_contract = {
            "morphology": condition.morphology_id,
            "subtype": condition.subtype_id,
            "role": condition.role_id,
            "source_index": condition_record["source_index"],
            "sample_seed": condition.sample_seed,
        }
        for name, expected in scalar_contract.items():
            if int(arrays[name][0]) != expected:
                raise ValueError(f"Raw scalar {name} disagrees with the condition")
        validation = raw_manifest["validation"]
        if (
            validation["accepted"] is not True
            or validation["hard_valid"] is not True
            or validation["errors"] != []
            or any(value is not True for value in validation["hard_gates"].values())
        ):
            raise ValueError("Repair source is not accepted by all generation hard gates")
        palette_record = style.palette_artifacts.get(condition.sample_id)
        if palette_record is None:
            raise ValueError("Static style parent lacks a repair source identity")
        samples.append(
            RepairSourceSample(
                sample_id=condition.sample_id,
                ordinal=ordinal,
                family=condition.morphology_name,
                family_id=condition.morphology_id,
                subtype_id=condition.subtype_id,
                role_id=condition.role_id,
                corpus_seed=int(arrays["corpus_seed"][0]),
                sample_seed=condition.sample_seed,
                part_owner=readonly_array(arrays["part"], dtype=np.uint8),
                material=readonly_array(arrays["material"], dtype=np.uint8),
                emission_level=readonly_array(arrays["emission"], dtype=np.uint8),
                guide=readonly_array(arrays["guide"], dtype=np.float32),
                genes=readonly_array(arrays["genes"], dtype=np.float32),
                legal_tuples=readonly_array(source.legal_tuples, dtype=np.uint8),
                raw_manifest_path=raw_manifest_path,
                raw_manifest_bytes=len(raw_manifest_bytes),
                raw_manifest_sha256=sha256_bytes(raw_manifest_bytes),
                raw_archive_path=raw_archive_path,
                raw_archive_bytes=len(raw_archive_bytes),
                raw_archive_sha256=sha256_bytes(raw_archive_bytes),
                raw_fields_sha256=raw_hash,
                compiled_fields_sha256=candidate.sample.fields.aligned_sha256,
                static_palette_sha256=str(palette_record["sha256"]),
            )
        )
    return RepairSource(
        generation_manifest_path=generation_manifest,
        generation_manifest_bytes=generation_manifest.stat().st_size,
        generation_manifest_sha256=source.bank.manifest_sha256,
        style_manifest_path=style_manifest,
        style_manifest_bytes=style_manifest.stat().st_size,
        style_manifest_sha256=style.manifest_sha256,
        legal_tuple_fingerprint=source.legal_tuple_fingerprint,
        samples=tuple(samples),
    )
