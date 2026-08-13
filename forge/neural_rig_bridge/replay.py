from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .binding import bind_neural_fields
from .hashing import canonical_json_hash
from .model import BindingRejected, NeuralRigBinding, REPLAY_FORMAT
from .validation import validate_binding


_REPLAY_CHECK_NAMES = (
    "binding_valid",
    "binding_sha256_exact",
    "manifest_exact",
    "owner_masks_exact",
    "driver_index_exact",
    "joints_exact",
    "sockets_exact",
    "rest_fields_exact",
)


def _failed_checks() -> dict[str, bool]:
    return {name: False for name in _REPLAY_CHECK_NAMES}


def replay_binding(
    expected: NeuralRigBinding,
    part_owner: np.ndarray,
    material: np.ndarray,
    emission_level: np.ndarray,
    guide: np.ndarray,
    *,
    genes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Rebind source arrays and compare every derived artifact exactly."""
    checks: dict[str, bool] = {}
    errors: list[str] = []
    expected_errors = validate_binding(expected)
    if expected_errors:
        actual = None
        errors.extend(f"expected binding invalid: {error}" for error in expected_errors)
    else:
        try:
            actual = bind_neural_fields(
                part_owner,
                material,
                emission_level,
                guide,
                family=expected.family_id,
                subtype_id=expected.subtype_id,
                role_id=expected.role_id,
                anatomy=expected.anatomy,
                sample_id=expected.sample_id,
                legal_tuples=expected.legal_tuples,
                genes=expected.genes if genes is None else genes,
                corpus_seed=expected.corpus_seed,
                upstream_hashes=expected.upstream_hashes,
            )
        except (BindingRejected, ValueError, TypeError) as error:
            actual = None
            errors.append(str(error))

    if actual is None:
        checks = _failed_checks()
    else:
        checks = {
            "binding_valid": validate_binding(actual) == [],
            "binding_sha256_exact": actual.sha256 == expected.sha256,
            "manifest_exact": actual.manifest == expected.manifest,
            "owner_masks_exact": np.array_equal(
                actual.owner_masks, expected.owner_masks
            ),
            "driver_index_exact": np.array_equal(
                actual.driver_index, expected.driver_index
            ),
            "joints_exact": actual.joints == expected.joints,
            "sockets_exact": actual.sockets == expected.sockets,
            "rest_fields_exact": all(
                np.array_equal(first, second)
                for first, second in zip(
                    actual.reconstruct_fields(),
                    expected.reconstruct_fields(),
                    strict=True,
                )
            ),
        }
    exact = all(checks.values())
    base: dict[str, Any] = {
        "format": REPLAY_FORMAT,
        "status": "exact" if exact else "mismatch",
        "expected_binding_sha256": (
            expected.sha256 if not expected_errors else None
        ),
        "actual_binding_sha256": actual.sha256 if actual is not None else None,
        "checks": checks,
        "errors": errors,
    }
    base["report_sha256"] = canonical_json_hash(base)
    return base


def assert_exact_replay(report: Mapping[str, Any]) -> None:
    required = {
        "format",
        "status",
        "expected_binding_sha256",
        "actual_binding_sha256",
        "checks",
        "errors",
        "report_sha256",
    }
    failures: list[str] = []
    if not isinstance(report, Mapping) or set(report) != required:
        raise BindingRejected(["binding replay report keys are not exact"])
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(_REPLAY_CHECK_NAMES):
        failures.append("replay checks are not exact")
    elif any(checks[name] is not True for name in _REPLAY_CHECK_NAMES):
        failures.append("one or more replay checks failed")
    if report.get("format") != REPLAY_FORMAT or report.get("status") != "exact":
        failures.append("replay status is not exact")
    expected_sha = report.get("expected_binding_sha256")
    actual_sha = report.get("actual_binding_sha256")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or actual_sha != expected_sha
    ):
        failures.append("replay binding hashes are not one identical lowercase SHA-256")
    if report.get("errors") != []:
        failures.append("replay report contains errors")
    payload = dict(report)
    claimed_hash = payload.pop("report_sha256", None)
    if claimed_hash != canonical_json_hash(payload):
        failures.append("replay report SHA-256 is incorrect")
    if failures:
        raise BindingRejected(failures)
