from __future__ import annotations

import argparse
from pathlib import Path

from .smoke import build_smoke, validate_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural topology prior-v2 foundation")
    parser.add_argument("command", choices=("smoke", "validate"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    parser.add_argument("--latents", type=Path, default=Path("outputs/map_topology_neural_prior_corpus/v1"))
    args = parser.parse_args()
    result = build_smoke(args.output, corpus_root=args.corpus, latent_root=args.latents) if args.command == "smoke" else validate_smoke(args.output, corpus_root=args.corpus, latent_root=args.latents)
    print(result["manifest_sha256"])


if __name__ == "__main__":
    main()
