from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from ..cellular_nca.contract import canonical_json_bytes, source_sha256 as parent_source_sha256
from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-organism-neural-cellular-automaton-causal-v2"
CHECKPOINT_FORMAT: Final[str] = "nullvector-organism-neural-cellular-automaton-causal-checkpoint-v2"
TRAINING_FORMAT: Final[str] = "nullvector-organism-neural-cellular-automaton-causal-training-v2"
TELEMETRY_FORMAT: Final[str] = "nullvector-organism-neural-cellular-automaton-causal-telemetry-v2"
PARENT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_nca/nca_v1"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_nca/nca_causal_v2"
MANIFEST_NAME: Final[str] = "cellular_nca_causal_manifest.json"
TRAINING_NAME: Final[str] = "causal_training_contract.json"
TELEMETRY_NAME: Final[str] = "causal_training_telemetry.json"
VISUAL_NAME: Final[str] = "organ_causality_comparison.png"
SEGMENT_TIMEOUT_SECONDS: Final[int] = 900
REQUIRED_GATES: Final[tuple[str, ...]] = (
    "all_values_finite",
    "all_four_organs_reduce_their_readout",
    "counterfactual_error_improves_over_parent",
    "general_health_mae_below_0_02",
    "general_fluid_mae_below_0_04",
    "general_neural_mae_below_0_06",
    "general_rollout_not_regressed_50_percent",
)
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_nca_causal/__init__.py",
    "forge/cellular_nca_causal/__main__.py",
    "forge/cellular_nca_causal/contract.py",
    "forge/cellular_nca_causal/curriculum.py",
    "forge/cellular_nca_causal/training.py",
    "forge/cellular_nca_causal/evaluation.py",
    "shared/schema/cellular_nca_causal_manifest.schema.json",
)


def causal_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-nca-causal-source-v2\0")
    digest.update(parent_source_sha256().encode("ascii") + b"\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def read_canonical_json(path: Path, *, maximum_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 1 <= path.stat().st_size <= maximum_bytes:
        raise ValueError(f"JSON artifact is missing, linked, empty, or oversized: {path}")
    encoded = path.read_bytes()
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Non-finite JSON constant {token}.")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, dict) or encoded != canonical_json_bytes(value):
        raise ValueError(f"JSON artifact is not a canonical object: {path}")
    return value


__all__ = ["CHECKPOINT_FORMAT", "DEFAULT_OUTPUT", "FORMAT", "MANIFEST_NAME", "PARENT_OUTPUT", "REQUIRED_GATES", "SEGMENT_TIMEOUT_SECONDS", "TELEMETRY_FORMAT", "TELEMETRY_NAME", "TRAINING_FORMAT", "TRAINING_NAME", "VISUAL_NAME", "canonical_json_bytes", "causal_source_sha256", "read_canonical_json", "sha256_bytes", "sha256_file"]
