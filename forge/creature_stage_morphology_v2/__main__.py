from __future__ import annotations

import argparse
from pathlib import Path

from .motion_review import build_motion_review
from .review import build_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the morphology-v2 human review gallery")
    parser.add_argument("--output", type=Path, default=Path("outputs/creature_stage_morphology_v2/review_001"))
    parser.add_argument("--motion", action="store_true")
    args = parser.parse_args()
    print(build_motion_review(args.output) if args.motion else build_review(args.output))


if __name__ == "__main__":
    main()
