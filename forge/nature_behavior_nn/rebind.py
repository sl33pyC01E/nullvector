from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .contract import CHECKPOINT_FORMAT, ModelConfig, source_sha256
from .corpus import load_corpus
from .model import NeuralNatureBehavior
from .training import _state_hash, evaluate


def rebind_checkpoint(source: Path, corpus_path: Path, output: Path, *, device: str = "cuda") -> dict[str, object]:
    """Rebind weights only when the optimized scaffold yields the exact same corpus."""
    payload = torch.load(source, map_location="cpu", weights_only=True)
    corpus = load_corpus(corpus_path)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("corpus_sha256") != corpus.semantic_sha256:
        raise ValueError("behavior rebind corpus semantics changed")
    for name in ("model", "ema"):
        if _state_hash(payload[name]) != payload[f"{name}_state_sha256"]:
            raise ValueError(f"behavior rebind {name} state drifted")
    target = torch.device(device)
    model = NeuralNatureBehavior(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload[payload["selected"]])
    model.to(target).eval()
    validation = np.flatnonzero(corpus.world_id % 5 == 0)
    replay = evaluate(model, corpus, validation, target)
    previous_source = str(payload["source_sha256"])
    payload["source_sha256"] = source_sha256()
    payload["validation"][payload["selected"]] = replay
    payload["rebind"] = {
        "kind": "semantics-preserving-scaffold-optimization",
        "previous_source_sha256": previous_source,
        "current_source_sha256": source_sha256(),
        "corpus_semantic_sha256": corpus.semantic_sha256,
        "selected_state_sha256": payload[f"{payload['selected']}_state_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, stage)
    os.replace(stage, output)
    report = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_sha256": source_sha256(),
        "corpus_sha256": corpus.semantic_sha256,
        "selected": payload["selected"],
        "selected_state_sha256": payload[f"{payload['selected']}_state_sha256"],
        "validation": replay,
        "rebind": payload["rebind"],
    }
    output.with_suffix(".json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(rebind_checkpoint(args.source, args.corpus, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
