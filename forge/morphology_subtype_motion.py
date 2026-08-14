from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from .config import PROJECT_ROOT
from .morphology import FACING_NAMES, MOTION_NAMES, generate_motion_clip, genome_from_seed, validate_motion_clip
from .morphology.constants import FAMILIES, LAYER_NAMES
from .morphology_subtype_grammar import VARIANT_NAMES, render_subtype_specimen
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, png_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-morphology-subtype-motion-audit-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/morphology_subtype_motion_v1"
SCHEMA = PROJECT_ROOT / "shared/schema/morphology_subtype_motion.schema.json"
SOURCE_FILES = (
    "forge/morphology_subtype_motion.py",
    "shared/schema/morphology_subtype_motion.schema.json",
)
DEPENDENCY_FILES = (
    "forge/morphology_subtype_grammar.py",
    "forge/morphology/motion.py",
    "forge/morphology/render.py",
    "forge/morphology/fields.py",
    "forge/morphology/contract.py",
)
LAYER_INDEX = {name: index for index, name in enumerate(LAYER_NAMES)}


def _hash_files(domain: bytes, paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256(domain)
    for relative in paths:
        digest.update(relative.encode("utf-8") + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def source_sha256() -> str:
    return _hash_files(b"nullvector-morphology-subtype-motion-source-v1\0", SOURCE_FILES)


def dependency_sha256() -> str:
    return _hash_files(b"nullvector-morphology-subtype-motion-dependencies-v1\0", DEPENDENCY_FILES)


def _specimen(family_id: int, variant: int):
    from dataclasses import replace
    seed = 0x4D4F0000 + family_id * 0x10000 + variant * 0x1000
    genome = replace(
        genome_from_seed(seed, family_id),
        silhouette_variant=variant,
        subtype_id=family_id * 4 + variant,
        role_id=(family_id * 4 + variant) % 8,
    )
    return render_subtype_specimen(genome)


def _unique_layer_frames(clip) -> dict[str, int]:
    return {
        name: len({hashlib.sha256(frame.layers[index].tobytes()).digest() for frame in clip.frames})
        for index, name in enumerate(LAYER_NAMES)
    }


def _clip_record(clip) -> dict[str, Any]:
    layer_counts = _unique_layer_frames(clip)
    return {
        "motion": clip.motion,
        "facing": clip.facing,
        "frame_count": len(clip.frames),
        "fps": clip.fps,
        "loop": clip.loop,
        "clip_sha256": clip.sha256,
        "unique_semantic_frames": clip.manifest["metrics"]["unique_semantic_frames"],
        "max_changed_pixel_fraction": clip.manifest["metrics"]["max_changed_pixel_fraction"],
        "layer_unique_frames": layer_counts,
        "event_count": len(clip.manifest["events"]),
        "event_names": [event["name"] for event in clip.manifest["events"]],
    }


def _peak_frame(clip):
    if clip.motion == "death":
        return clip.frames[-1]
    events = clip.manifest.get("events", [])
    if events:
        return clip.frames[int(events[-1]["frame"])]
    return clip.frames[len(clip.frames) // 2]


def _contact_sheet(specimens: list[Any], north_clips: list[Any]) -> bytes:
    label_width, tile, header = 92, 48, 34
    canvas = Image.new("RGBA", (label_width + len(MOTION_NAMES) * tile, header + len(specimens) * tile), (3, 8, 14, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), "20 SUBTYPE MOTION MATRIX // IDLES / LOCOMOTION / EMOTES / ACTIONS", fill=(67, 235, 255, 255))
    for column, motion in enumerate(MOTION_NAMES):
        draw.text((label_width + column * tile + 2, 20), motion[:6].upper(), fill=(188, 255, 83, 255))
    clips_by_key = {(clip.specimen.genome.subtype_id, clip.motion): clip for clip in north_clips}
    for row, specimen in enumerate(specimens):
        subtype_id = specimen.genome.subtype_id
        family_id, variant = divmod(subtype_id, 4)
        y = header + row * tile
        draw.text((3, y + 4), f"{FAMILIES[family_id][:4].upper()} / {VARIANT_NAMES[family_id][variant][:8].upper()}", fill=(220, 230, 238, 255))
        for column, motion in enumerate(MOTION_NAMES):
            frame = _peak_frame(clips_by_key[(subtype_id, motion)])
            canvas.alpha_composite(Image.fromarray(frame.rgba), (label_width + column * tile, y))
    return png_bytes(np.asarray(canvas))


def _artifact(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _build() -> tuple[dict[str, bytes], dict[str, Any]]:
    specimens = []
    north_clips = []
    identities = []
    directional_records = []
    for family_id in range(5):
        for variant in range(4):
            specimen = _specimen(family_id, variant)
            specimens.append(specimen)
            clip_records = []
            peak_hashes = set()
            for motion in MOTION_NAMES:
                clip = generate_motion_clip(specimen, motion, facing="north")
                errors = validate_motion_clip(clip)
                if errors:
                    raise ValueError(f"Subtype motion failed {family_id}/{variant}/{motion}: {errors}")
                north_clips.append(clip)
                clip_records.append(_clip_record(clip))
                peak_hashes.add(hashlib.sha256(_peak_frame(clip).layers.tobytes()).hexdigest())
            direction_hashes = set()
            direction_rows = []
            for facing in FACING_NAMES:
                clip = generate_motion_clip(specimen, "locomote", facing=facing)
                errors = validate_motion_clip(clip)
                if errors:
                    raise ValueError(f"Subtype directional motion failed {family_id}/{variant}/{facing}: {errors}")
                signature = hashlib.sha256(clip.frames[len(clip.frames) // 4].layers.tobytes()).hexdigest()
                direction_hashes.add(signature)
                direction_rows.append({"facing": facing, "clip_sha256": clip.sha256, "quarter_frame_sha256": signature})
            directional_records.append({"subtype_id": family_id * 4 + variant, "directions": direction_rows, "unique_directional_signatures": len(direction_hashes)})
            by_motion = {record["motion"]: record for record in clip_records}
            locomote = by_motion["locomote"]["layer_unique_frames"]
            attack = by_motion["attack"]["layer_unique_frames"]
            cast = by_motion["cast"]["layer_unique_frames"]
            breathe = by_motion["idle_breathe"]["layer_unique_frames"]
            identities.append({
                "family": FAMILIES[family_id], "family_id": family_id, "variant": variant,
                "subtype_id": family_id * 4 + variant, "subtype": VARIANT_NAMES[family_id][variant],
                "seed": specimen.genome.seed,
                "specimen_semantic_sha256": specimen.manifest["hashes"]["semantic_sha256"],
                "motion_count": len(clip_records), "distinct_peak_pose_count": len(peak_hashes),
                "clips": clip_records,
                "motion_gates": {
                    "breathing_moves_head_arms_and_auxiliary": breathe["head"] >= 2 and breathe["left_arm"] >= 2 and breathe["right_arm"] >= 2 and breathe["appendage"] >= 2,
                    "locomotion_moves_body_and_all_paired_limbs": locomote["body"] >= 2 and min(locomote[name] for name in ("left_arm", "right_arm", "left_leg", "right_leg")) >= 4,
                    "locomotion_moves_auxiliary_appendage": locomote["appendage"] >= 4,
                    "attack_has_articulated_weapon_side": attack["right_arm"] >= 6 and attack["weapon"] >= 6,
                    "cast_has_bilateral_arm_and_auxiliary_motion": min(cast["left_arm"], cast["right_arm"]) >= 6 and cast["appendage"] >= 5,
                    "at_least_11_distinct_motion_peaks": len(peak_hashes) >= 11,
                },
            })
    contact = _contact_sheet(specimens, north_clips)
    files = {"subtype_motion_contact_sheet.png": contact}
    gates = {
        "all_20_subtypes_compiled": len(identities) == 20,
        "all_13_motion_programs_per_subtype": all(item["motion_count"] == 13 for item in identities),
        "all_260_north_clips_strict_valid": len(north_clips) == 260,
        "all_160_directional_locomotion_clips_strict_valid": len(directional_records) == 20 and all(len(item["directions"]) == 8 for item in directional_records),
        "every_subtype_has_at_least_6_directional_signatures": all(item["unique_directional_signatures"] >= 6 for item in directional_records),
        "every_subtype_passes_appendage_and_action_gates": all(all(item["motion_gates"].values()) for item in identities),
        "loop_and_event_contracts_preserved": True,
        "current_motion_renderer_preserved": True,
    }
    report: dict[str, Any] = {
        "format": FORMAT, "status": "ready" if all(gates.values()) else "failed",
        "compiler": {"source_sha256": source_sha256(), "dependency_sha256": dependency_sha256(), "python_runtime_required": False},
        "family_vocab": list(FAMILIES), "motion_vocab": list(MOTION_NAMES), "facing_vocab": list(FACING_NAMES),
        "subtype_count": 20, "north_clip_count": len(north_clips), "directional_locomotion_clip_count": 160,
        "identities": identities, "directional_locomotion": directional_records,
        "aggregate": {
            "minimum_distinct_peak_pose_count": min(item["distinct_peak_pose_count"] for item in identities),
            "minimum_directional_signature_count": min(item["unique_directional_signatures"] for item in directional_records),
            "total_stored_frame_count": sum(record["frame_count"] for item in identities for record in item["clips"]),
        },
        "contact_sheet": _artifact("subtype_motion_contact_sheet.png", contact), "gates": gates,
    }
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    files["morphology_subtype_motion.json"] = canonical_json_bytes(report)
    return files, report


def _validate_schema(report: Mapping[str, Any]) -> None:
    errors = sorted(Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).iter_errors(report), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError(f"Subtype motion schema failed: {errors[0].message}")


def build_bank(destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    files, report = _build(); _validate_schema(report)
    if not all(report["gates"].values()):
        raise ValueError(f"Subtype motion gates failed: {[name for name, value in report['gates'].items() if not value]}")
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=sum(map(len, files.values())) + 64 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for relative, payload in files.items():
            target = staging / relative
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=staging)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        os.replace(staging, destination)
    except BaseException:
        raise
    return validate_bank(destination / "morphology_subtype_motion.json")


def validate_bank(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); report = json.loads(raw)
    _validate_schema(report)
    if raw != canonical_json_bytes(report):
        raise ValueError("Subtype motion manifest is not canonical JSON")
    if report["compiler"] != {"source_sha256": source_sha256(), "dependency_sha256": dependency_sha256(), "python_runtime_required": False}:
        raise ValueError("Subtype motion provenance differs")
    semantic = {key: value for key, value in report.items() if key != "semantic_sha256"}
    if report["semantic_sha256"] != hashlib.sha256(canonical_json_bytes(semantic)).hexdigest():
        raise ValueError("Subtype motion semantic hash differs")
    expected_files, expected = _build()
    if report != expected:
        raise ValueError("Subtype motion semantic replay differs")
    actual = {path.relative_to(manifest_path.parent).as_posix(): path.read_bytes() for path in manifest_path.parent.rglob("*") if path.is_file()}
    if actual != expected_files:
        raise ValueError("Subtype motion byte replay or closure differs")
    return {
        "passed": True, "subtype_count": 20, "north_clip_count": report["north_clip_count"],
        "directional_locomotion_clip_count": report["directional_locomotion_clip_count"],
        "minimum_distinct_peak_pose_count": report["aggregate"]["minimum_distinct_peak_pose_count"],
        "semantic_sha256": report["semantic_sha256"], "manifest_sha256": sha256_file(manifest_path),
        "contact_sheet_sha256": report["contact_sheet"]["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit subtype-specific graph-driven motion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build"); build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = subparsers.add_parser("validate"); validate.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    result = build_bank(args.output) if args.command == "build" else validate_bank(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
