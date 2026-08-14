from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Final, Mapping

from jsonschema import Draft202012Validator
import numpy as np

from .cellular_motion import validate_bank
from .cellular_motion.contract import DRIVER_NAMES, MOTION_NAMES
from .config import PROJECT_ROOT
from .map_decorator.hashing import json_sha256
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


FORMAT: Final[str] = "nullvector-cellular-motion-quality-audit-v1"
DEFAULT_SOURCE: Final[Path] = PROJECT_ROOT / "outputs/cellular_motion_v2/cellular_motion_manifest.json"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/cellular_motion_quality_v1/motion_quality_report.json"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/cellular_motion_quality.schema.json"
ACTIVE_THRESHOLD: Final[float] = 0.05
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "body": ("body_bob", "body_sway", "body_squash"),
    "appendage": ("appendage_left", "appendage_right", "auxiliary", "weapon_recoil"),
    "locomotor": ("locomotor_left", "locomotor_right", "propulsion"),
    "expression": ("head_tilt", "sensory_focus", "emission_pulse", "pain_spasm"),
}
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/cellular_motion_quality.py",
    "shared/schema/cellular_motion_quality.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-motion-quality-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _round(value: float) -> float:
    return round(float(value), 7)


def _clip_metrics(clip: Mapping[str, object]) -> dict[str, object]:
    facings = clip["facings"]
    north = np.asarray(facings[0]["frames"], dtype=object)
    drivers = np.asarray([frame["drivers"] for frame in north], dtype=np.float64)
    if drivers.shape[1] != len(DRIVER_NAMES) or not np.isfinite(drivers).all():
        raise ValueError("Motion-quality driver matrix differs")
    for facing in facings[1:]:
        candidate = np.asarray([frame["drivers"] for frame in facing["frames"]], dtype=np.float64)
        if not np.array_equal(candidate, drivers):
            raise ValueError("Facing-specific driver curves are not canonical rotations")
    ranges = np.ptp(drivers, axis=0)
    index = {name: offset for offset, name in enumerate(DRIVER_NAMES)}
    group_excursion = {
        name: _round(max(float(ranges[index[driver]]) for driver in members))
        for name, members in GROUPS.items()
    }
    left = drivers[:, index["locomotor_left"]]
    right = drivers[:, index["locomotor_right"]]
    antiphase = 0.0
    if float(np.std(left)) > 1e-8 and float(np.std(right)) > 1e-8:
        antiphase = float(np.corrcoef(left, right)[0, 1])
    consecutive = np.abs(np.diff(drivers, axis=0))
    frame_zero = drivers[0]
    maximum_pose_excursion = float(np.linalg.norm(drivers - frame_zero, axis=1).max())
    event_excursions = []
    for event in clip["events"]:
        event_frame = int(event["frame"])
        event_excursions.append({
            "name": str(event["name"]), "frame": event_frame,
            "driver_l2_from_first": _round(np.linalg.norm(drivers[event_frame] - frame_zero)),
        })
    payload = canonical_json_bytes([[round(float(value), 7) for value in row] for row in drivers])
    return {
        "motion": clip["motion"], "frame_count": int(clip["frame_count"]), "fps": int(clip["fps"]),
        "loop": bool(clip["loop"]), "active_driver_count": int(np.count_nonzero(ranges > ACTIVE_THRESHOLD)),
        "driver_ranges": [_round(value) for value in ranges], "group_excursion": group_excursion,
        "temporal_energy": _round(float(consecutive.mean()) if len(consecutive) else 0.0),
        "maximum_pose_excursion": _round(maximum_pose_excursion),
        "locomotor_antiphase_correlation": _round(antiphase),
        "event_excursions": event_excursions, "trajectory_sha256": sha256_bytes(payload),
        "loop_endpoint_exact": not bool(clip["loop"]) or bool(np.array_equal(drivers[0], drivers[-1])),
        "facings_share_driver_curve": True,
    }


def audit_motion_bank(manifest_path: Path = DEFAULT_SOURCE) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    validation = validate_bank(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for program in manifest["programs"]:
        for clip in program["clips"]:
            records.append({
                "family": program["family"], "family_id": int(program["family_id"]),
                **_clip_metrics(clip),
            })
    if len(records) != 65:
        raise ValueError("Motion-quality family/motion census differs")

    locomotion = [item for item in records if item["motion"] == "locomote"]
    breathing = [item for item in records if item["motion"] == "idle_breathe"]
    wiggles = [item for item in records if item["motion"] == "idle_wiggle"]
    emotes = [item for item in records if item["motion"] in {"joy", "anger", "fear", "confused", "sleep", "taunt"}]
    actions = [item for item in records if item["motion"] in {"attack", "cast", "hit", "death"}]
    distinct_by_motion = {
        motion: len({item["trajectory_sha256"] for item in records if item["motion"] == motion})
        for motion in MOTION_NAMES
    }
    gates = {
        "all_65_family_motion_programs_audited": len(records) == 65,
        "all_facings_share_exact_driver_curve": all(item["facings_share_driver_curve"] for item in records),
        "all_loop_endpoints_exact": all(item["loop_endpoint_exact"] for item in records),
        "breathing_has_body_and_appendage_cycle": all(item["group_excursion"]["body"] >= 0.25 and item["group_excursion"]["appendage"] >= 0.10 for item in breathing),
        "idle_wiggle_moves_appendages": all(item["group_excursion"]["appendage"] >= 0.45 for item in wiggles),
        "locomotion_has_paired_antiphase_limbs": all(item["active_driver_count"] >= 9 and item["group_excursion"]["appendage"] >= 0.90 and item["group_excursion"]["locomotor"] >= 1.0 and item["locomotor_antiphase_correlation"] <= -0.95 for item in locomotion),
        "emotes_remain_articulated": all(item["active_driver_count"] >= 4 and item["group_excursion"]["appendage"] >= 0.14 for item in emotes),
        "actions_remain_articulated": all(item["active_driver_count"] >= 9 and item["group_excursion"]["appendage"] >= 0.50 and item["maximum_pose_excursion"] >= 0.70 for item in actions),
        "all_programs_have_temporal_energy": all(item["temporal_energy"] >= 0.01 for item in records),
        "all_family_trajectories_are_distinct": all(value == 5 for value in distinct_by_motion.values()),
    }
    report: dict[str, object] = {
        "format": FORMAT, "status": "passed" if all(gates.values()) else "failed",
        "compiler": {"source_sha256": source_sha256(), "python_runtime_required": False},
        "source": {
            "manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest_path), "semantic_sha256": manifest["semantic_sha256"],
            "validation": validation,
        },
        "active_driver_threshold": ACTIVE_THRESHOLD,
        "driver_vocab": list(DRIVER_NAMES), "motion_vocab": list(MOTION_NAMES),
        "driver_groups": {name: list(values) for name, values in GROUPS.items()},
        "family_count": 5, "motion_count": 13, "record_count": len(records),
        "records": records,
        "aggregate": {
            "minimum_active_driver_count": min(item["active_driver_count"] for item in records),
            "minimum_temporal_energy": _round(min(item["temporal_energy"] for item in records)),
            "minimum_action_appendage_excursion": _round(min(item["group_excursion"]["appendage"] for item in actions)),
            "minimum_locomotor_excursion": _round(min(item["group_excursion"]["locomotor"] for item in locomotion)),
            "maximum_locomotor_antiphase_correlation": _round(max(item["locomotor_antiphase_correlation"] for item in locomotion)),
            "distinct_family_trajectories_by_motion": distinct_by_motion,
        },
        "gates": gates,
    }
    report["semantic_sha256"] = json_sha256(report)
    return report


def validate_report(report_path: Path) -> dict[str, object]:
    report_path = Path(report_path).resolve(); raw = report_path.read_bytes(); report = json.loads(raw)
    errors = sorted(Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).iter_errors(report), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError(f"Motion-quality schema validation failed: {errors[0].message}")
    if raw != canonical_json_bytes(report):
        raise ValueError("Motion-quality report is not canonical JSON")
    if report["semantic_sha256"] != json_sha256({key: value for key, value in report.items() if key != "semantic_sha256"}):
        raise ValueError("Motion-quality semantic identity differs")
    source_path = PROJECT_ROOT.joinpath(*Path(report["source"]["manifest"]).parts).resolve()
    expected = audit_motion_bank(source_path)
    if report != expected:
        raise ValueError("Motion-quality exact semantic replay differs")
    if not all(report["gates"].values()):
        raise ValueError("Motion-quality gate failed")
    return {
        "passed": True, "record_count": 65, "semantic_sha256": report["semantic_sha256"],
        "report_sha256": sha256_file(report_path), "source_manifest_sha256": report["source"]["manifest_sha256"],
    }


def build_report(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_motion_bank(source)
    if not all(report["gates"].values()):
        raise ValueError("Motion-quality audit failed")
    payload = canonical_json_bytes(report)
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=len(payload) + 16 * 1024**2)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary_name, output)
    return validate_report(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit cellular motion for semantic action collapse")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build"); build.add_argument("--source", type=Path, default=DEFAULT_SOURCE); build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = subparsers.add_parser("validate"); validate.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    result = build_report(args.source, args.output) if args.command == "build" else validate_report(args.report)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
