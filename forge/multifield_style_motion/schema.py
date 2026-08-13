from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = PROJECT_ROOT / "shared" / "schema"
FAMILY_SCHEMA = "multifield_style_motion_family.schema.json"
BANK_SCHEMA = "multifield_style_motion_bank.schema.json"
SCHEMA_FILES = frozenset((FAMILY_SCHEMA, BANK_SCHEMA))


@lru_cache(maxsize=2)
def _validator(filename: str) -> Draft202012Validator:
    if filename not in SCHEMA_FILES:
        raise ValueError(f"Unsupported style-motion schema: {filename!r}")
    path = (SCHEMA_ROOT / filename).resolve()
    if not path.is_relative_to(SCHEMA_ROOT.resolve()):
        raise ValueError("Style-motion schema filename escapes the schema directory")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_schema(
    payload: Mapping[str, Any],
    filename: str = BANK_SCHEMA,
) -> None:
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
        raise ValueError(
            f"Manifest failed {filename}: " + "; ".join(rendered) + suffix
        )
