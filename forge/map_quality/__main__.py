from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_packs, write_audit_report
from .render import write_quality_showcase


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit persisted topology-v2 map quality.")
    parser.add_argument("packs", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--showcase", type=Path)
    args = parser.parse_args()
    report = audit_packs(tuple(args.packs))
    write_audit_report(report, args.report)
    showcase = (
        write_quality_showcase(tuple(args.packs), args.showcase)
        if args.showcase is not None
        else None
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "map_count": report["map_count"],
                "unique_semantic_count": report["unique_semantic_count"],
                "report_sha256": report["report_sha256"],
                "showcase_sha256": (
                    showcase["manifest_sha256"] if showcase is not None else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
