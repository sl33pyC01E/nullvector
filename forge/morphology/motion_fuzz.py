from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any

from .constants import FAMILIES
from .disk_guard import guard_corpus_destination
from .genome import genome_from_seed
from .motion import FACING_NAMES, MOTION_NAMES, generate_motion_clip, validate_motion_clip
from .motion_preview import HARD_DISK_FLOOR_BYTES
from .render import render_specimen


FUZZ_SEED = 0x6D6F746E
FUZZ_STRIDE = 0x9E3779B1


def fuzz_motion_contract(count: int = 500) -> dict[str, Any]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    hashes: set[str] = set()
    family_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    facing_counts: Counter[str] = Counter()
    total_frames = 0
    changed_fractions: list[float] = []
    unique_frame_counts: list[int] = []
    role_counts: Counter[int] = Counter()
    for index in range(count):
        family_index = index % len(FAMILIES)
        family = FAMILIES[family_index]
        seed = (FUZZ_SEED + index * FUZZ_STRIDE) & 0xFFFFFFFF
        motion = MOTION_NAMES[(index * 7 + index // len(FAMILIES)) % len(MOTION_NAMES)]
        facing = FACING_NAMES[(index * 3 + index // 11) % len(FACING_NAMES)]
        family_counts[family] += 1
        motion_counts[motion] += 1
        facing_counts[facing] += 1
        try:
            specimen = render_specimen(genome_from_seed(seed, family))
            role_counts[specimen.genome.role_id] += 1
            clip = generate_motion_clip(specimen, motion, facing=facing)
            errors = validate_motion_clip(clip)
            if errors:
                failures.append(
                    {
                        "index": index,
                        "seed": seed,
                        "family": family,
                        "motion": motion,
                        "facing": facing,
                        "errors": errors,
                    }
                )
                continue
            hashes.add(clip.sha256)
            total_frames += len(clip.frames)
            changed_fractions.append(
                float(clip.manifest["metrics"]["max_changed_pixel_fraction"])
            )
            unique_frame_counts.append(
                int(clip.manifest["metrics"]["unique_semantic_frames"])
            )
        except Exception as error:  # fuzz reports must retain the failing seed.
            failures.append(
                {
                    "index": index,
                    "seed": seed,
                    "family": family,
                    "motion": motion,
                    "facing": facing,
                    "exception": f"{type(error).__name__}: {error}",
                }
            )
    elapsed = time.perf_counter() - started
    passed = count - len(failures)
    return {
        "format": "neural-morphology-motion-fuzz-v1",
        "requested_clips": count,
        "passed_clips": passed,
        "failed_clips": len(failures),
        "total_frames": total_frames,
        "unique_clip_hashes": len(hashes),
        "family_counts": dict(sorted(family_counts.items())),
        "motion_counts": dict(sorted(motion_counts.items())),
        "facing_counts": dict(sorted(facing_counts.items())),
        "role_counts": {str(key): value for key, value in sorted(role_counts.items())},
        "min_unique_semantic_frames": min(unique_frame_counts, default=0),
        "max_unique_semantic_frames": max(unique_frame_counts, default=0),
        "min_changed_pixel_fraction": round(min(changed_fractions, default=0.0), 7),
        "max_changed_pixel_fraction": round(max(changed_fractions, default=0.0), 7),
        "elapsed_seconds": round(elapsed, 3),
        "clips_per_second": round(count / max(elapsed, 1e-9), 3),
        "failures": failures,
    }


def write_fuzz_report(report: dict[str, Any], destination: Path) -> None:
    destination = Path(destination).resolve()
    guard_corpus_destination(
        destination,
        int(report["requested_clips"]),
        reserve_bytes=HARD_DISK_FLOOR_BYTES,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Fuzz deterministic graph-rig morphology motion contracts."
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "outputs" / "morphology_motion" / "motion_fuzz_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = fuzz_motion_contract(args.count)
    write_fuzz_report(report, args.report)
    print(json.dumps(report, indent=2))
    if report["failed_clips"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
