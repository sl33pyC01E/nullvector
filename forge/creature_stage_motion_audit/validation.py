from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


FORMAT = "nullvector-creature-stage-motion-audit-v1"
MAX_AUDIT_BYTES = 4 * 1024 * 1024
FAMILIES = ["humanoid", "animalian", "plantlike", "anomaly", "machine"]
MORPHOTYPES = [
    ["balanced", "longarm", "sixlimb", "crowned"],
    ["quadruped", "crawler", "longtail", "horned"],
    ["treeform", "rosette", "runner", "twin_stem"],
    ["triad", "cross", "pentad", "halo"],
    ["tracked", "walker", "hover", "crab"],
]
MOTIONS = [
    "idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear",
    "confused", "sleep", "taunt", "attack", "cast", "hit", "death",
]
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "schema"
    / "creature_stage_motion_audit.schema.json"
)


class MotionAuditValidationError(ValueError):
    """Raised when a layered motion audit fails closed."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MotionAuditValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise MotionAuditValidationError(f"non-finite JSON number: {value}")


def validate_motion_audit(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path).resolve()
    size = audit_path.stat().st_size
    if size <= 0 or size > MAX_AUDIT_BYTES:
        raise MotionAuditValidationError(
            f"audit size {size} is outside (0, {MAX_AUDIT_BYTES}]"
        )
    raw_bytes = audit_path.read_bytes()
    try:
        raw = raw_bytes.decode("utf-8", errors="strict")
        document = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MotionAuditValidationError(f"invalid UTF-8 JSON: {exc}") from exc

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=str)
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path)
        raise MotionAuditValidationError(
            f"schema violation at {location or '<root>'}: {first.message}"
        )

    clips: list[dict[str, Any]] = document["clips"]
    expected_keys = [
        (family_id, morphotype_id, motion)
        for family_id in range(5)
        for morphotype_id in range(4)
        for motion in MOTIONS
    ]
    observed_keys: list[tuple[int, int, str]] = []
    pose_hashes: set[str] = set()
    family_hashes: list[set[str]] = [set() for _ in FAMILIES]
    family_minimums = [math.inf] * 5
    family_maximums = [0.0] * 5
    for index, clip in enumerate(clips):
        family_id = int(clip["family_id"])
        morphotype_id = int(clip["morphotype_id"])
        motion = str(clip["motion"])
        observed_keys.append((family_id, morphotype_id, motion))
        if clip["family"] != FAMILIES[family_id]:
            raise MotionAuditValidationError(f"family mismatch at clip {index}")
        if clip["morphotype"] != MORPHOTYPES[family_id][morphotype_id]:
            raise MotionAuditValidationError(f"morphotype mismatch at clip {index}")
        expected_seed = 0x6D0F0000 + family_id * 0x100 + morphotype_id
        if int(clip["seed"]) != expected_seed:
            raise MotionAuditValidationError(f"seed mismatch at clip {index}")
        meaningful = (
            float(clip["maximum_core_displacement"])
            if motion == "idle_breathe"
            else float(clip["maximum_appendage_displacement"])
        )
        if motion == "death":
            meaningful = float(clip["maximum_displacement"])
        if meaningful < 0.025:
            raise MotionAuditValidationError(f"collapsed motion at clip {index}")
        pose_hash = str(clip["pose_sha256"])
        if pose_hash in pose_hashes:
            raise MotionAuditValidationError(f"duplicate pose signature at clip {index}")
        pose_hashes.add(pose_hash)
        family_hashes[family_id].add(pose_hash)
        family_minimums[family_id] = min(family_minimums[family_id], meaningful)
        family_maximums[family_id] = max(
            family_maximums[family_id], float(clip["maximum_displacement"])
        )

    if observed_keys != expected_keys:
        raise MotionAuditValidationError("clip matrix is incomplete, duplicated, or reordered")
    if len(pose_hashes) != 260:
        raise MotionAuditValidationError("pose signature coverage mismatch")
    identity_material = "|".join(
        f"{clip['family_id']}:{clip['morphotype_id']}:{clip['motion']}:{clip['pose_sha256']}"
        for clip in clips
    )
    clip_identity_sha256 = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()
    if clip_identity_sha256 != document["clip_identity_sha256"]:
        raise MotionAuditValidationError("clip identity SHA-256 mismatch")

    for family_id, family in enumerate(FAMILIES):
        summary = document["family_summary"][str(family_id)]
        if summary["family"] != family or len(family_hashes[family_id]) != 52:
            raise MotionAuditValidationError(f"family summary mismatch for {family}")
        if not math.isclose(
            float(summary["minimum_meaningful_displacement"]),
            family_minimums[family_id],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MotionAuditValidationError(f"family minimum mismatch for {family}")
        if not math.isclose(
            float(summary["maximum_displacement"]),
            family_maximums[family_id],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MotionAuditValidationError(f"family maximum mismatch for {family}")

    global_maximum = max(float(clip["maximum_displacement"]) for clip in clips)
    if not math.isclose(
        float(document["maximum_displacement"]),
        global_maximum,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise MotionAuditValidationError("global displacement maximum mismatch")

    return {
        "passed": True,
        "format": FORMAT,
        "path": str(audit_path),
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "clip_identity_sha256": clip_identity_sha256,
        "clip_count": len(clips),
        "unique_pose_signatures": len(pose_hashes),
        "maximum_displacement": global_maximum,
        "minimum_meaningful_displacement": min(family_minimums),
    }


def assert_valid_motion_audit(path: str | Path) -> dict[str, Any]:
    return validate_motion_audit(path)
