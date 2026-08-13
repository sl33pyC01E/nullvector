from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only neural map-topology foundation and exact replay tooling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract = subparsers.add_parser("contract")
    contract.add_argument("--pretty", action="store_true")
    corpus = subparsers.add_parser("read-corpus-sample")
    corpus.add_argument("--corpus", type=Path, required=True)
    corpus.add_argument("--shard", required=True)
    corpus.add_argument("--index", type=int, default=0)
    build = subparsers.add_parser("build-smoke")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--visually-inspected", action="store_true")
    replay = subparsers.add_parser("replay-smoke")
    replay.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "contract":
        from .contract import CONTRACT_SHA256, contract_manifest

        payload = {"sha256": CONTRACT_SHA256, "contract": contract_manifest()}
        print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    if args.command == "read-corpus-sample":
        from .corpus import TopologyCorpus

        sample = TopologyCorpus(args.corpus).read_sample(args.shard, args.index)
        print(
            json.dumps(
                {
                    "passed": True,
                    "map_id": sample.map_id,
                    "split": sample.split,
                    "topology_sample_sha256": sample.topology_sample_sha256,
                    "raw_topology_sha256": sample.raw.raw_sha256,
                    "member_array_sha256": sample.member_array_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-smoke":
        from .smoke import build_smoke

        print(
            json.dumps(
                build_smoke(
                    args.output,
                    corpus_root=args.corpus,
                    visually_inspected=args.visually_inspected,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "replay-smoke":
        from .smoke import assert_exact_smoke_replay

        print(json.dumps(assert_exact_smoke_replay(args.output), indent=2, sort_keys=True))
        return 0
    raise AssertionError("Unreachable command.")


if __name__ == "__main__":
    raise SystemExit(main())
