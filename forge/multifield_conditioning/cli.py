from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .audit import audit_conditioning_bank


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit condition adherence against each generated sprite's exact "
            "held-out reference and the reference classifier ceiling."
        )
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cuda-interventions",
        action="store_true",
        help=(
            "Run an eight-sample same-noise causal sensitivity probe on CUDA "
            "after all immutable CPU audit checks pass."
        ),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_conditioning_bank(
            args.bank,
            benchmark_path=args.benchmark,
            checkpoint_path=args.checkpoint,
            corpus_path=args.corpus,
            output_path=args.output,
            intervention_device="cuda" if args.cuda_interventions else None,
            intervention_precision="bf16",
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    destination = args.output.resolve()
    print(
        json.dumps(
            {
                "status": report["status"],
                "conclusion": report["decision"]["conclusion"],
                "training_change_warranted": report["decision"][
                    "training_change_warranted"
                ],
                "output": str(destination),
                "output_sha256": _sha256(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
