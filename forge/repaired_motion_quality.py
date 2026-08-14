from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid
from typing import Any, Mapping

import jsonschema
import numpy as np
from PIL import Image, ImageDraw

from .config import PROJECT_ROOT
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .repaired_motion_lab_sync import FACINGS, MOTIONS, validate_runtime
from .safety import require_disk_floor


FORMAT = "nullvector-repaired-neural-motion-quality-v2"
REPORT_NAME = "motion_quality_report.json"
CONTACT_NAME = "motion_dynamics_contact_sheet.png"
DEFAULT_RUNTIME = PROJECT_ROOT / "game/generated/repaired_motion_lab/v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/repaired_motion_quality_v2"
SCHEMA_PATH = PROJECT_ROOT / "shared/schema/repaired_motion_quality.schema.json"
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_CONTACT_BYTES = 32 * 1024 * 1024
LAYOUT = {"cell_size": 48, "columns": 16, "rows": 59, "frame_count": 944}

# Calibrated below the observed all-80 minima, but high enough to reject rigid
# translation and action collapse. Values are translation-compensated XOR pixels
# divided by first-frame occupancy, expressed in parts per million.
ARTICULATION_FLOOR_PPM = {
    "idle_breathe": 30_000,
    "idle_wiggle": 75_000,
    "locomote": 85_000,
    "joy": 170_000,
    "anger": 150_000,
    "fear": 140_000,
    "confused": 140_000,
    "sleep": 65_000,
    "taunt": 170_000,
    "attack": 320_000,
    "cast": 190_000,
    "hit": 320_000,
    "death": 550_000,
}
MIN_UNIQUE_PLAYBACK_FRAMES = 4
MAX_OCCUPANCY_SPIKE_PPM = 1_350_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "format": "nullvector-repaired-motion-quality-source-v1",
                "files": {
                    Path(__file__).relative_to(PROJECT_ROOT).as_posix(): _sha256_file(Path(__file__)),
                    SCHEMA_PATH.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(SCHEMA_PATH),
                },
            }
        )
    )


def _load_json(path: Path, maximum: int = MAX_REPORT_BYTES) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"JSON artifact is missing, unsafe, or oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact root must be an object.")
    return value


def _resolve(root: Path, record: Mapping[str, Any]) -> Path:
    relative = PurePosixPath(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Motion atlas path is not a canonical relative POSIX path.")
    path = root.joinpath(*relative.parts).resolve()
    path.relative_to(root.resolve())
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Motion atlas is missing or unsafe: {relative}")
    if path.stat().st_size != int(record.get("bytes", -1)) or _sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Motion atlas byte/hash identity drifted: {relative}")
    return path


def _crop(array: np.ndarray, cell_index: int) -> np.ndarray:
    cell = LAYOUT["cell_size"]
    x = (int(cell_index) % LAYOUT["columns"]) * cell
    y = (int(cell_index) // LAYOUT["columns"]) * cell
    return array[y : y + cell, x : x + cell]


def _aligned_xor_ppm(reference: np.ndarray, candidate: np.ndarray) -> int:
    reference_points = np.argwhere(reference)
    candidate_points = np.argwhere(candidate)
    if not len(reference_points) or not len(candidate_points):
        return 1_000_000
    reference_y, reference_x = reference_points.mean(axis=0)
    candidate_y, candidate_x = candidate_points.mean(axis=0)
    shift_y = int(np.rint(reference_y - candidate_y))
    shift_x = int(np.rint(reference_x - candidate_x))
    aligned = np.roll(candidate, (shift_y, shift_x), axis=(0, 1))
    if shift_y > 0:
        aligned[:shift_y] = False
    elif shift_y < 0:
        aligned[shift_y:] = False
    if shift_x > 0:
        aligned[:, :shift_x] = False
    elif shift_x < 0:
        aligned[:, shift_x:] = False
    return int(round(float(np.logical_xor(reference, aligned).sum()) * 1_000_000 / max(1, int(reference.sum()))))


def _clip_metrics(base: np.ndarray, clip: Mapping[str, Any]) -> dict[str, Any]:
    alpha = base[..., 3] > 0
    playback_count = int(clip["frame_count"]) - (1 if bool(clip["loop"]) else 0)
    frames = [_crop(alpha, int(clip["start_cell"]) + index) for index in range(playback_count)]
    rgba_frames = [_crop(base, int(clip["start_cell"]) + index) for index in range(playback_count)]
    occupancies = [int(frame.sum()) for frame in frames]
    alpha_frame_hashes = [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in frames]
    rgba_frame_hashes = [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in rgba_frames]
    articulation = [_aligned_xor_ppm(frames[0], frame) for frame in frames]
    median = float(np.median(np.asarray(occupancies, dtype=np.int64)))
    return {
        "sample_id": str(clip["sample_id"]),
        "family": str(clip["family"]),
        "motion": str(clip["motion"]),
        "facing": str(clip["facing"]),
        "playback_frame_count": playback_count,
        "unique_frame_count": len(set(alpha_frame_hashes)),
        "minimum_occupancy": min(occupancies),
        "maximum_occupancy": max(occupancies),
        "occupancy_spike_ppm": int(round(max(occupancies) * 1_000_000 / max(1.0, median))),
        "articulation_peak_ppm": max(articulation),
        "articulation_mean_ppm": int(round(sum(articulation) / len(articulation))),
        "peak_frame_index": int(max(range(len(articulation)), key=lambda index: (articulation[index], -index))),
        "alpha_sequence_sha256": sha256_bytes(canonical_json_bytes(alpha_frame_hashes)),
        "rgba_sequence_sha256": sha256_bytes(canonical_json_bytes(rgba_frame_hashes)),
    }


def _contact_sheet(runtime: Path, catalog: Mapping[str, Any], rows: list[dict[str, Any]]) -> bytes:
    families = ("humanoid", "animalian", "plantlike", "anomaly", "machine")
    representatives = {family: next(item for item in catalog["identities"] if item["family"] == family) for family in families}
    row_lookup = {(item["sample_id"], item["motion"], item["facing"]): item for item in rows}
    margin_x, header, tile_w, tile_h = 76, 22, 98, 58
    canvas = Image.new("RGB", (margin_x + len(MOTIONS) * tile_w, header + len(families) * tile_h), (3, 7, 13))
    draw = ImageDraw.Draw(canvas)
    for column, motion in enumerate(MOTIONS):
        draw.text((margin_x + column * tile_w + 2, 5), motion[:13], fill=(126, 183, 196))
    for row_index, family in enumerate(families):
        identity = representatives[family]
        y = header + row_index * tile_h
        draw.text((4, y + 21), family.upper(), fill=(214, 101, 211))
        base_path = _resolve(runtime, identity["atlases"]["base"])
        with Image.open(base_path) as source_image:
            source = np.asarray(source_image.convert("RGBA"))
        clip_lookup = {(item["motion"], item["facing"]): item for item in identity["clips"]}
        for column, motion in enumerate(MOTIONS):
            clip = clip_lookup[(motion, "north")]
            metric = row_lookup[(identity["sample_id"], motion, "north")]
            first = Image.fromarray(_crop(source, int(clip["start_cell"])))
            peak = Image.fromarray(_crop(source, int(clip["start_cell"]) + int(metric["peak_frame_index"])))
            x = margin_x + column * tile_w
            canvas.paste(first.convert("RGB"), (x, y + 5), first)
            canvas.paste(peak.convert("RGB"), (x + 49, y + 5), peak)
            draw.rectangle((x, y + 5, x + 47, y + 52), outline=(25, 72, 84))
            draw.rectangle((x + 49, y + 5, x + 96, y + 52), outline=(94, 37, 91))
    import io

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    payload = buffer.getvalue()
    if not 0 < len(payload) <= MAX_CONTACT_BYTES:
        raise ValueError("Motion dynamics contact sheet exceeds its strict byte bound.")
    return payload


def _analyze(runtime: Path, *, visually_inspected: bool) -> tuple[dict[str, Any], bytes]:
    if not isinstance(visually_inspected, bool):
        raise TypeError("visually_inspected must be an explicit boolean attestation.")
    runtime = Path(runtime).resolve()
    validation = validate_runtime(runtime)
    catalog_path = runtime / "catalog.json"
    catalog = _load_json(catalog_path, maximum=8 * 1024 * 1024)
    rows: list[dict[str, Any]] = []
    base_hashes: set[str] = set()
    for identity in catalog["identities"]:
        base_record = identity["atlases"]["base"]
        base_path = _resolve(runtime, base_record)
        base_hashes.add(str(base_record["sha256"]))
        with Image.open(base_path) as source_image:
            base = np.asarray(source_image.convert("RGBA"))
        for clip in identity["clips"]:
            rows.append(_clip_metrics(base, {**clip, "sample_id": identity["sample_id"], "family": identity["family"]}))
    if len(rows) != 8_320:
        raise ValueError("Motion quality audit did not cover the full clip matrix.")

    alpha_sequence_groups: dict[tuple[str, str, str], list[str]] = {}
    rgba_sequence_groups: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        alpha_sequence_groups.setdefault((row["motion"], row["facing"], row["alpha_sequence_sha256"]), []).append(row["sample_id"])
        rgba_sequence_groups.setdefault((row["motion"], row["facing"], row["rgba_sequence_sha256"]), []).append(row["sample_id"])
    cross_identity_alpha_duplicates = sum(1 for samples in alpha_sequence_groups.values() if len(set(samples)) > 1)
    cross_identity_rgba_duplicates = sum(1 for samples in rgba_sequence_groups.values() if len(set(samples)) > 1)
    blank = sum(row["minimum_occupancy"] == 0 for row in rows)
    static = sum(row["unique_frame_count"] < MIN_UNIQUE_PLAYBACK_FRAMES for row in rows)
    spikes = sum(row["occupancy_spike_ppm"] > MAX_OCCUPANCY_SPIKE_PPM for row in rows)
    articulation_failures = sum(row["articulation_peak_ppm"] < ARTICULATION_FLOOR_PPM[row["motion"]] for row in rows)

    def aggregate(group: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "clip_count": len(group),
            "minimum_unique_frames": min(item["unique_frame_count"] for item in group),
            "minimum_articulation_peak_ppm": min(item["articulation_peak_ppm"] for item in group),
            "maximum_articulation_peak_ppm": max(item["articulation_peak_ppm"] for item in group),
            "maximum_occupancy_spike_ppm": max(item["occupancy_spike_ppm"] for item in group),
            "blank_clip_count": sum(item["minimum_occupancy"] == 0 for item in group),
            "static_clip_count": sum(item["unique_frame_count"] < MIN_UNIQUE_PLAYBACK_FRAMES for item in group),
        }

    families = {family: aggregate([row for row in rows if row["family"] == family]) for family in ("humanoid", "animalian", "plantlike", "anomaly", "machine")}
    motions = {motion: {**aggregate([row for row in rows if row["motion"] == motion]), "articulation_floor_ppm": ARTICULATION_FLOOR_PPM[motion]} for motion in MOTIONS}
    clip_metrics_sha256 = sha256_bytes(canonical_json_bytes(rows))
    contact = _contact_sheet(runtime, catalog, rows)
    core: dict[str, Any] = {
        "format": FORMAT,
        "status": "ready",
        "authority": {
            "runtime_catalog": "game/generated/repaired_motion_lab/v1/catalog.json",
            "runtime_catalog_sha256": _sha256_file(catalog_path),
            "runtime_bundle_id": catalog["bundle_id"],
        },
        "source_sha256": _source_sha256(),
        "execution": {"device": "cpu", "cuda_used": False, "pixel_authority": "base_atlas_alpha", "global_translation_removed": "integer_centroid_alignment_v1"},
        "counts": {"identity_count": 80, "family_count": 5, "motion_count": 13, "facing_count": 8, "clip_count": len(rows), "frame_count": validation["frame_count"], "unique_base_atlas_count": len(base_hashes)},
        "thresholds": {"minimum_unique_playback_frames": MIN_UNIQUE_PLAYBACK_FRAMES, "maximum_occupancy_spike_ppm": MAX_OCCUPANCY_SPIKE_PPM, "articulation_floor_ppm": ARTICULATION_FLOOR_PPM},
        "summary": {
            "blank_clip_count": blank,
            "static_clip_count": static,
            "occupancy_spike_failure_count": spikes,
            "articulation_failure_count": articulation_failures,
            "cross_identity_duplicate_alpha_sequence_group_count": cross_identity_alpha_duplicates,
            "cross_identity_duplicate_rgba_sequence_group_count": cross_identity_rgba_duplicates,
            "minimum_articulation_peak_ppm": min(row["articulation_peak_ppm"] for row in rows),
            "maximum_articulation_peak_ppm": max(row["articulation_peak_ppm"] for row in rows),
            "minimum_unique_frame_count": min(row["unique_frame_count"] for row in rows),
            "maximum_occupancy_spike_ppm": max(row["occupancy_spike_ppm"] for row in rows),
            "clip_metrics_sha256": clip_metrics_sha256,
        },
        "families": families,
        "motions": motions,
        "contact_sheet": {"path": CONTACT_NAME, "bytes": len(contact), "sha256": hashlib.sha256(contact).hexdigest(), "columns": list(MOTIONS), "rows": ["humanoid", "animalian", "plantlike", "anomaly", "machine"], "cell_content": "first_frame_then_peak_translation_compensated_articulation_frame", "visually_inspected": visually_inspected},
        "gates": {
            "all_80_identity_atlases_unique": len(base_hashes) == 80,
            "all_8320_clips_audited": len(rows) == 8_320,
            "no_blank_playback_clips": blank == 0,
            "no_static_or_near_static_clips": static == 0,
            "no_occupancy_flash_spikes": spikes == 0,
            "all_motion_articulation_floors_met": articulation_failures == 0,
            "no_cross_identity_duplicate_rgba_sequences": cross_identity_rgba_duplicates == 0,
            "all_five_families_balanced": all(value["clip_count"] == 1_664 for value in families.values()),
            "source_runtime_validation_passed": bool(validation["passed"]),
            "contact_sheet_visually_inspected": visually_inspected,
            "cpu_only": True,
        },
    }
    if not all(core["gates"].values()):
        raise ValueError(f"Motion quality gates failed: {[name for name, passed in core['gates'].items() if not passed]}")
    return core, contact


def build_quality_audit(output: Path = DEFAULT_OUTPUT, runtime: Path = DEFAULT_RUNTIME, *, visually_inspected: bool = False) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Motion quality output is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=64 * 1024 * 1024)
    core, contact = _analyze(runtime, visually_inspected=visually_inspected)
    report = {**core, "report_sha256": sha256_bytes(canonical_json_bytes(core))}
    jsonschema.Draft202012Validator(_load_json(SCHEMA_PATH)).validate(report)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / CONTACT_NAME).write_bytes(contact)
        (staging / REPORT_NAME).write_bytes(canonical_json_bytes(report))
        os.replace(staging, output)
    except BaseException:
        # Preserve bounded unique staging as evidence on the unstable host.
        raise
    return validate_quality_audit(output, runtime=runtime)


def validate_quality_audit(output: Path = DEFAULT_OUTPUT, *, runtime: Path = DEFAULT_RUNTIME) -> dict[str, Any]:
    output = Path(output).resolve()
    report = _load_json(output / REPORT_NAME)
    jsonschema.Draft202012Validator(_load_json(SCHEMA_PATH)).validate(report)
    unsigned = dict(report)
    recorded_hash = unsigned.pop("report_sha256", None)
    if sha256_bytes(canonical_json_bytes(unsigned)) != recorded_hash:
        raise ValueError("Motion quality report self-hash drifted.")
    expected, contact = _analyze(runtime, visually_inspected=bool(report["contact_sheet"]["visually_inspected"]))
    if unsigned != expected:
        raise ValueError("Motion quality report failed exact semantic replay.")
    contact_path = output / CONTACT_NAME
    if not contact_path.is_file() or contact_path.read_bytes() != contact:
        raise ValueError("Motion quality contact sheet failed exact byte replay.")
    return {
        "passed": True,
        "report_sha256": recorded_hash,
        "runtime_bundle_id": report["authority"]["runtime_bundle_id"],
        "clip_count": report["counts"]["clip_count"],
        "frame_count": report["counts"]["frame_count"],
        "clip_metrics_sha256": report["summary"]["clip_metrics_sha256"],
        "contact_sheet_sha256": report["contact_sheet"]["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit all-80 repaired neural motion dynamics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    build.add_argument("--visually-inspected", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args(argv)
    result = build_quality_audit(args.output, args.runtime, visually_inspected=args.visually_inspected) if args.command == "build" else validate_quality_audit(args.output, runtime=args.runtime)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
