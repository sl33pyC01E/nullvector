from __future__ import annotations

import argparse
from pathlib import Path

from .contract import DEFAULT_CORPUS, DEFAULT_OUTPUT
from .corpus import build_corpus, validate_corpus
from .training import train


def main():
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="command", required=True)
    build = command.add_parser("build-corpus")
    build.add_argument("--output", type=Path, default=DEFAULT_CORPUS)
    validate = command.add_parser("validate-corpus")
    validate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    fit = command.add_parser("train")
    fit.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    fit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "build-corpus":
        print(build_corpus(args.output))
    elif args.command == "validate-corpus":
        print(validate_corpus(args.corpus))
    else:
        print(train(args.output, corpus=args.corpus))


if __name__ == "__main__":
    main()
