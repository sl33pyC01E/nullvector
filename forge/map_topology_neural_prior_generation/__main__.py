from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bank import generate_bank, generate_case, validate_bank, verify_case_result
from .contract import GenerationConfig, canonical_json_bytes


def _config(path: Path | None, args: argparse.Namespace) -> GenerationConfig:
    if path is not None:
        return GenerationConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return GenerationConfig(
        variants_per_condition=args.variants,
        sampling_steps=args.sampling_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        base_seed=args.seed,
        maximum_workers=args.workers,
        maximum_attempts=args.max_attempts,
        worker_timeout_seconds=args.timeout_seconds,
        contact_scale=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded free generation from the frozen neural map prior.")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate"); generate.add_argument("--destination", type=Path, required=True); generate.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1")); generate.add_argument("--variants", type=int, default=2); generate.add_argument("--sampling-steps", type=int, default=8); generate.add_argument("--temperature", type=float, default=.8); generate.add_argument("--top-k", type=int, default=16); generate.add_argument("--seed", type=int, default=0x465245454D415053); generate.add_argument("--workers", type=int, default=2); generate.add_argument("--max-attempts", type=int, default=3); generate.add_argument("--timeout-seconds", type=int, default=900)
    case = sub.add_parser("case"); case.add_argument("--destination", type=Path, required=True); case.add_argument("--corpus", type=Path, required=True); case.add_argument("--config", type=Path, required=True); case.add_argument("--case-id", required=True)
    verify = sub.add_parser("verify-case"); verify.add_argument("--destination", type=Path, required=True); verify.add_argument("--corpus", type=Path, required=True); verify.add_argument("--config", type=Path, required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--destination", type=Path, required=True); validate.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1")); validate.add_argument("--exact-cases", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_bank(args.destination, corpus_root=args.corpus, config=_config(None, args))
    elif args.command == "case":
        result = generate_case(args.destination, corpus_root=args.corpus, config=_config(args.config, args), case_id=args.case_id)
    elif args.command == "verify-case":
        result = verify_case_result(args.destination, corpus_root=args.corpus, config=_config(args.config, args))
    else:
        result = validate_bank(args.destination, corpus_root=args.corpus, exact_cases=args.exact_cases)
    print(canonical_json_bytes(result).decode("utf-8").strip())


if __name__ == "__main__":
    main()
