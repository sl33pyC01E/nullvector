from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import zipfile

from jsonschema import Draft202012Validator
import numpy as np

from .hashing import aligned_fields_hash, is_sha256, sha256_file
from .model import (
    CategoricalFields,
    IMAGE_SIZE,
    LoadedGenerationBank,
    LoadedSourceSample,
    StyleCondition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATION_SCHEMA_PATH = PROJECT_ROOT / "shared" / "schema" / "multifield_generation_bank.schema.json"
GENERATION_FORMAT = "nullvector-multifield-generation-bank-v1"
COMPILED_FIELD_FORMAT = "nullvector-bounded-field-postprocess-v1"
EXPECTED_NPZ_KEYS = {
    "format",
    "part",
    "material",
    "emission",
    "raw_fields_sha256",
    "processed_fields_sha256",
}
MAX_NPZ_BYTES = 2 * 1024 * 1024
MAX_NPZ_MEMBER_BYTES = 512 * 1024
MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024


def _validate_json_schema(payload: Mapping[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)
    schema = json.loads(GENERATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        rendered = []
        for error in errors[:10]:
            location = "/".join(map(str, error.absolute_path)) or "<root>"
            rendered.append(f"{location}: {error.message}")
        suffix = f" (+{len(errors) - 10} more)" if len(errors) > 10 else ""
        raise ValueError("Generation manifest schema failure: " + "; ".join(rendered) + suffix)


def _resolve_artifact(root: Path, record: Mapping[str, Any]) -> Path:
    try:
        relative_text = str(record["path"])
        expected_bytes = int(record["bytes"])
        expected_sha256 = str(record["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed source artifact record: {error}") from error
    pure = PurePosixPath(relative_text)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe source artifact path: {relative_text!r}")
    if "\\" in relative_text:
        raise ValueError("Source artifact paths must use canonical POSIX separators")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Source artifact escapes generation root: {relative_text}") from error
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"Source artifact must be a regular non-symlink file: {relative_text}")
    if expected_bytes <= 0 or target.stat().st_size != expected_bytes:
        raise ValueError(f"Source artifact byte count mismatch: {relative_text}")
    if not is_sha256(expected_sha256) or sha256_file(target) != expected_sha256:
        raise ValueError(f"Source artifact SHA-256 mismatch: {relative_text}")
    return target


def _scalar_string(values: np.ndarray, name: str) -> str:
    if values.shape != (1,) or values.dtype.kind not in "US":
        raise ValueError(f"Compiled field {name} must be a one-element Unicode string")
    return str(values[0])


def _validate_npz_container(path: Path) -> None:
    if path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("Compiled field NPZ exceeds the bounded container size")
    expected_members = {f"{name}.npy" for name in EXPECTED_NPZ_KEYS}
    with zipfile.ZipFile(path, "r") as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Compiled field NPZ contains duplicate ZIP members")
        if set(names) != expected_members:
            raise ValueError(
                f"Compiled field NPZ member mismatch: expected {sorted(expected_members)}, got {sorted(names)}"
            )
        total = 0
        for entry in entries:
            if PurePosixPath(entry.filename).name != entry.filename:
                raise ValueError("Compiled field NPZ contains nested or unsafe ZIP members")
            if entry.file_size < 0 or entry.file_size > MAX_NPZ_MEMBER_BYTES:
                raise ValueError("Compiled field NPZ member exceeds the uncompressed bound")
            total += entry.file_size
        if total > MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("Compiled field NPZ exceeds the total uncompressed bound")


def _load_fields(
    path: Path,
    *,
    expected_compiled_hash: str,
    expected_raw_hash: str,
) -> CategoricalFields:
    _validate_npz_container(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_NPZ_KEYS:
            raise ValueError("Compiled field NPZ array key mismatch")
        format_name = _scalar_string(archive["format"], "format")
        embedded_raw = _scalar_string(archive["raw_fields_sha256"], "raw_fields_sha256")
        embedded_compiled = _scalar_string(
            archive["processed_fields_sha256"], "processed_fields_sha256"
        )
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in ("part", "material", "emission")
        }
    if format_name != COMPILED_FIELD_FORMAT:
        raise ValueError("Unexpected compiled categorical-field format")
    if embedded_raw != expected_raw_hash or embedded_compiled != expected_compiled_hash:
        raise ValueError("Compiled categorical-field embedded hash mismatch")
    for name, values in arrays.items():
        if values.shape != (IMAGE_SIZE, IMAGE_SIZE) or values.dtype != np.uint8:
            raise ValueError(f"Compiled {name} must be uint8 {(IMAGE_SIZE, IMAGE_SIZE)}")
    if int(arrays["part"].max(initial=0)) > 16:
        raise ValueError("Compiled part owner exceeds the 17-token vocabulary")
    if int(arrays["material"].max(initial=0)) > 9:
        raise ValueError("Compiled material exceeds the 10-token vocabulary")
    if int(arrays["emission"].max(initial=0)) > 3:
        raise ValueError("Compiled emission exceeds the 4-token vocabulary")
    background = arrays["part"] == 0
    if np.any(arrays["material"][background] != 0) or np.any(arrays["emission"][background] != 0):
        raise ValueError("Background tuples must be exactly (part=0, material=0, emission=0)")
    actual_hash = aligned_fields_hash(arrays["part"], arrays["material"], arrays["emission"])
    if actual_hash != expected_compiled_hash:
        raise ValueError("Compiled categorical-field aligned hash mismatch")
    return CategoricalFields(
        part=arrays["part"],
        material=arrays["material"],
        emission=arrays["emission"],
        aligned_sha256=actual_hash,
    )


def load_generation_bank(manifest_path: Path) -> LoadedGenerationBank:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Generation manifest must be a regular non-symlink file")
    if manifest_path.name != "generation_manifest.json":
        raise ValueError("Input must be the canonical generation_manifest.json")
    raw_bytes = manifest_path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Generation manifest is not strict UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Generation manifest root must be an object")
    _validate_json_schema(payload)
    if payload.get("format") != GENERATION_FORMAT or payload.get("status") != "ready":
        raise ValueError("Generation bank must be immutable, ready, and format v1")
    entries = payload.get("samples")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Generation bank has no samples")
    if int(payload.get("grid", {}).get("samples", -1)) != len(entries):
        raise ValueError("Generation grid sample count disagrees with sample entries")

    root = manifest_path.parent.resolve()
    loaded: list[LoadedSourceSample] = []
    sample_ids: set[str] = set()
    for expected_ordinal, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("Generation sample entries must be objects")
        condition = StyleCondition.from_mapping(entry.get("condition", {}))
        if condition.ordinal != expected_ordinal:
            raise ValueError("Generation sample ordinals must be dense and ordered")
        if condition.sample_id in sample_ids:
            raise ValueError("Generation sample IDs must be unique")
        sample_ids.add(condition.sample_id)
        raw_validation = entry.get("raw_validation", {})
        if not isinstance(raw_validation, dict) or not raw_validation.get("accepted") or not raw_validation.get("hard_valid"):
            raise ValueError(f"Source sample is not accepted and hard-valid: {condition.sample_id}")
        post_validation = entry.get("postprocess_validation", {})
        if not isinstance(post_validation, dict) or not post_validation.get("valid"):
            raise ValueError(f"Source postprocess is not valid: {condition.sample_id}")
        compiled_hash = str(entry.get("compiled_fields_sha256", ""))
        raw_hash = str(entry.get("raw_fields_sha256", ""))
        if not is_sha256(compiled_hash) or not is_sha256(raw_hash):
            raise ValueError("Source categorical-field hashes are malformed")
        artifacts = entry.get("compiled_artifacts", {})
        if not isinstance(artifacts, dict) or not isinstance(artifacts.get("fields"), dict):
            raise ValueError("Source compiled field artifact is missing")
        fields_record = dict(artifacts["fields"])
        fields_path = _resolve_artifact(root, fields_record)
        fields = _load_fields(
            fields_path,
            expected_compiled_hash=compiled_hash,
            expected_raw_hash=raw_hash,
        )
        loaded.append(
            LoadedSourceSample(
                condition=condition,
                fields=fields,
                raw_fields_sha256=raw_hash,
                fields_artifact=fields_record,
            )
        )
    return LoadedGenerationBank(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest_bytes=manifest_path.stat().st_size,
        manifest=payload,
        samples=tuple(loaded),
    )
