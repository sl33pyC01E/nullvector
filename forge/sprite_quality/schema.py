from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "shared" / "schema" / "sprite_quality_report.schema.json"


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_schema(payload: Mapping[str, Any]) -> None:
    json.dumps(payload, allow_nan=False)
    errors = sorted(
        _validator().iter_errors(payload),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        rendered: list[str] = []
        for error in errors[:12]:
            location = "/".join(map(str, error.absolute_path)) or "<root>"
            rendered.append(f"{location}: {error.message}")
        suffix = f" (+{len(errors) - 12} more)" if len(errors) > 12 else ""
        raise ValueError("Sprite-quality report schema failure: " + "; ".join(rendered) + suffix)
