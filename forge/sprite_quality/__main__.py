from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import compile_sprite_quality_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a source-bound quality audit for the production neural sprite banks."
    )
    parser.add_argument("--static-manifest", type=Path)
    parser.add_argument("--motion-manifest", type=Path)
    parser.add_argument("--motion-replay", type=Path)
    parser.add_argument("--map-art-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_sprite_quality_audit(
        args.output,
        static_manifest_path=args.static_manifest,
        motion_manifest_path=args.motion_manifest,
        motion_replay_path=args.motion_replay,
        map_art_root=args.map_art_root,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "static_samples": report["coverage"]["static_sample_count"],
                "motion_clips": report["motion_quality"]["clip_count"],
                "collapsed_motion_clips": report["motion_quality"][
                    "collapsed_composite_clip_count"
                ],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
