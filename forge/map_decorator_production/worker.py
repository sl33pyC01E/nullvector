from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import ShardSpec, build_shard, validate_shard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One process-isolated map-decorator shard action")
    parser.add_argument("mode", choices=("build", "validate"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Shard specification root must be an object.")
    spec = ShardSpec.from_dict(payload)
    if args.mode == "build":
        report = build_shard(spec, args.root)
    else:
        report = validate_shard(spec, args.root)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

