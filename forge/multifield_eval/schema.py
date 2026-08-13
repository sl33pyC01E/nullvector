from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from ..config import PROJECT_ROOT


SCHEMA_ROOT = PROJECT_ROOT / "shared" / "schema"
GENERATION_BANK_SCHEMA = "multifield_generation_bank.schema.json"
RAW_SAMPLE_SCHEMA = "multifield_raw_sample.schema.json"
BENCHMARK_SCHEMA = "multifield_benchmark.schema.json"
CALIBRATION_SCHEMA = "multifield_reference_calibration.schema.json"


@lru_cache(maxsize=8)
def _validator(filename: str) -> Draft202012Validator:
    path = (SCHEMA_ROOT / filename).resolve()
    if not path.is_relative_to(SCHEMA_ROOT.resolve()):
        raise ValueError("Schema filename escapes the shared schema directory")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_manifest_schema(payload: Mapping[str, Any], filename: str) -> None:
    # JSON Schema's Python number checker accepts some non-finite floats even
    # though strict JSON does not.  Reject those before structural validation.
    json.dumps(payload, allow_nan=False)
    errors = sorted(
        _validator(filename).iter_errors(payload),
        key=lambda error: tuple(map(str, error.absolute_path)),
    )
    if errors:
        rendered = []
        for error in errors[:12]:
            location = "/".join(map(str, error.absolute_path)) or "<root>"
            rendered.append(f"{location}: {error.message}")
        suffix = f" (+{len(errors) - 12} more)" if len(errors) > 12 else ""
        raise ValueError(
            f"Manifest failed {filename}: " + "; ".join(rendered) + suffix
        )
