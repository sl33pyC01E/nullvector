from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


FORMAT = "nullvector-creature-stage-causal-trace-v1"
MAX_TRACE_BYTES = 8 * 1024 * 1024
EXPECTED_STEPS = 240
EXPECTED_DT = 1.0 / 30.0
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "schema"
    / "creature_stage_causal_trace.schema.json"
)


class TraceValidationError(ValueError):
    """Raised when a causal trace fails closed."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _extract_raw_transitions(raw: str) -> str:
    marker = '"transitions"'
    marker_index = raw.find(marker)
    if marker_index < 0:
        raise TraceValidationError("missing transitions member")
    start = raw.find("[", marker_index + len(marker))
    if start < 0:
        raise TraceValidationError("transitions is not an array")
    depth = 0
    in_string = False
    escaped = False
    end: int | None = None
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise TraceValidationError("unterminated transitions array")
    return raw[start:end]


def _minify_json_lexically(raw: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in raw:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            output.append(char)
        elif not char.isspace():
            output.append(char)
    if in_string or escaped:
        raise TraceValidationError("malformed JSON string in transitions")
    return "".join(output)


def _assert_finite(value: Any, path: str = "trace") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise TraceValidationError(f"non-finite numeric value at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
        return
    raise TraceValidationError(f"unsupported value at {path}: {type(value).__name__}")


def _length(vector: list[float]) -> float:
    return math.hypot(float(vector[0]), float(vector[1]))


def validate_trace(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path).resolve()
    size = trace_path.stat().st_size
    if size <= 0 or size > MAX_TRACE_BYTES:
        raise TraceValidationError(
            f"trace size {size} is outside (0, {MAX_TRACE_BYTES}]"
        )
    raw_bytes = trace_path.read_bytes()
    try:
        raw = raw_bytes.decode("utf-8", errors="strict")
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceValidationError(f"invalid UTF-8 JSON: {exc}") from exc

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=str)
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path)
        raise TraceValidationError(f"schema violation at {location or '<root>'}: {first.message}")

    _assert_finite(document)
    transitions: list[dict[str, Any]] = document["transitions"]
    if document["transition_count"] != len(transitions) or len(transitions) != EXPECTED_STEPS:
        raise TraceValidationError("transition count mismatch")

    raw_transition_json = _minify_json_lexically(_extract_raw_transitions(raw))
    transition_sha256 = hashlib.sha256(raw_transition_json.encode("utf-8")).hexdigest()
    if transition_sha256 != document["transition_sha256"]:
        raise TraceValidationError("transition SHA-256 mismatch")

    moved_distance = 0.0
    energy_delta = 0.0
    attack_steps = 0
    utility_steps = 0
    projectile_peak = 0
    for index, transition in enumerate(transitions):
        if transition["step"] != index:
            raise TraceValidationError(f"non-sequential step at transition {index}")
        if not math.isclose(float(transition["dt"]), EXPECTED_DT, rel_tol=0.0, abs_tol=1e-12):
            raise TraceValidationError(f"invalid fixed dt at transition {index}")
        if index and transitions[index - 1]["after"] != transition["before"]:
            raise TraceValidationError(f"broken state chain before transition {index}")

        action = transition["action"]
        move_length = _length(action["move"])
        aim_length = _length(action["aim"])
        if move_length > 1.001:
            raise TraceValidationError(f"move vector exceeds unit disk at transition {index}")
        if not math.isclose(aim_length, 1.0, rel_tol=0.0, abs_tol=2e-5):
            raise TraceValidationError(f"aim vector is not normalized at transition {index}")
        attack_steps += int(float(action["attack"]) > 0.5)
        utility_steps += int(float(action["utility"]) > 0.5)

        before = transition["before"]
        after = transition["after"]
        moved_distance += math.dist(before["position"], after["position"])
        energy_delta += abs(float(after["status"]["energy"]) - float(before["status"]["energy"]))
        projectile_peak = max(projectile_peak, int(after["active_projectiles"]))
        for organ, alive in after["organ_alive"].items():
            if int(alive) > int(after["organ_totals"].get(organ, -1)):
                raise TraceValidationError(
                    f"organ alive count exceeds total for {organ} at transition {index}"
                )

    if moved_distance < 25.0:
        raise TraceValidationError("trajectory contains insufficient locomotion")
    if energy_delta <= 0.001:
        raise TraceValidationError("physiology is static across trajectory")
    if attack_steps < 10 or utility_steps != 3 or projectile_peak < 1:
        raise TraceValidationError("action curriculum coverage is incomplete")

    return {
        "passed": True,
        "format": FORMAT,
        "path": str(trace_path),
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "transition_sha256": transition_sha256,
        "transition_count": len(transitions),
        "moved_distance": moved_distance,
        "energy_delta": energy_delta,
        "attack_steps": attack_steps,
        "utility_steps": utility_steps,
        "projectile_peak": projectile_peak,
    }


def assert_valid_trace(path: str | Path) -> dict[str, Any]:
    return validate_trace(path)
