from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .constants import BANK_SCHEMA, PLAN_SCHEMA, PROJECT_ROOT, REPLAY_SCHEMA
from .hashing import sha256_file


SCHEMA_ROOT = PROJECT_ROOT / "shared" / "schema"
SCHEMAS = frozenset((PLAN_SCHEMA, BANK_SCHEMA, REPLAY_SCHEMA))


@lru_cache(maxsize=3)
def _validator(filename: str) -> Draft202012Validator:
    if filename not in SCHEMAS:
        raise ValueError(f"Unsupported neural rig repair schema: {filename!r}")
    schema = json.loads((SCHEMA_ROOT / filename).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_schema(payload: Mapping[str, Any], filename: str) -> None:
    json.dumps(payload, allow_nan=False)
    errors = sorted(
        _validator(filename).iter_errors(payload),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        rendered: list[str] = []
        for error in errors[:12]:
            location = "/".join(map(str, error.absolute_path)) or "<root>"
            rendered.append(f"{location}: {error.message}")
        suffix = f" (+{len(errors) - 12} more)" if len(errors) > 12 else ""
        raise ValueError(f"Manifest failed {filename}: " + "; ".join(rendered) + suffix)


def load_strict_json(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if size < 2 or size > maximum_bytes:
        raise ValueError(f"{label} violates its bounded JSON size")
    payload = path.read_bytes()
    if len(payload) != size:
        raise ValueError(f"{label} changed while it was read")

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
        result = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} root must be an object")
    return result


def load_schema_json(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    schema: str,
) -> dict[str, Any]:
    result = load_strict_json(path, maximum_bytes=maximum_bytes, label=label)
    validate_schema(result, schema)
    return result


def resolve_artifact_record(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    maximum_bytes: int,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} artifact record keys are not exact")
    relative = record["path"]
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} artifact path is not canonical POSIX text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} artifact path is unsafe")
    root = Path(root).resolve()
    path = (root / Path(*pure.parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} artifact is outside its regular-file contract")
    if (
        type(record["bytes"]) is not int
        or record["bytes"] < 1
        or record["bytes"] > maximum_bytes
        or path.stat().st_size != record["bytes"]
    ):
        raise ValueError(f"{label} artifact byte count mismatch")
    if not isinstance(record["sha256"], str) or sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} artifact SHA-256 mismatch")
    return path
