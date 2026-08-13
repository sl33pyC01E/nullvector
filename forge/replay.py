from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .config import ARCHETYPES, CHECKPOINT_DIR, GAME_GENERATED_DIR, OUTPUT_DIR
from .determinism import configure_deterministic_inference
from .provenance import (
    architecture_from_state_dict,
    canonical_state_dict_hash,
    model_from_architecture,
)
from .safety import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay exported neural noise streams and verify exact raw tokens."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=GAME_GENERATED_DIR / "sprite_registry.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument(
        "--sprite",
        action="append",
        default=[],
        help="Sprite id to replay; repeat the flag. Defaults to every sprite.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_DIR / "replay_report.json",
    )
    return parser.parse_args()


def _load_stored_tokens(
    root: Path, manifest: dict, image_size: int
) -> np.ndarray:
    source = manifest.get("source", {})
    name = source.get("raw_tokens")
    if not isinstance(name, str):
        raise ValueError(f"{manifest.get('id')}: source.raw_tokens is missing.")
    root = root.resolve()
    path = (root / name).resolve()
    path.relative_to(root)
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim != 2 or values.shape != (image_size, image_size):
        raise ValueError(f"{manifest.get('id')}: unexpected raw token shape {values.shape}.")
    return values.astype(np.uint8, copy=False)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    configure_deterministic_inference()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    entries = registry.get("sprites", [])
    requested = set(args.sprite)
    if requested:
        entries = [entry for entry in entries if entry.get("id") in requested]
        missing = requested.difference(entry.get("id") for entry in entries)
        if missing:
            raise ValueError(f"Unknown sprite ids: {sorted(missing)}")
    if not entries:
        raise ValueError("No sprites selected for replay.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model"]
    architecture = checkpoint.get("architecture") or architecture_from_state_dict(state)
    model = model_from_architecture(architecture).to(device)
    model.load_state_dict(state)
    model.eval()
    canonical_hash = canonical_state_dict_hash(state)
    if registry.get("model_hash") != canonical_hash[:16]:
        raise ValueError("Registry model hash does not match the replay checkpoint.")

    results = []
    root = args.registry.parent
    for manifest in entries:
        generation = manifest.get("generation", {})
        genome = manifest.get("genome", {})
        archetype_name = manifest.get("archetype")
        archetype = torch.tensor(
            [ARCHETYPES.index(archetype_name)], dtype=torch.long, device=device
        )
        genes = torch.tensor(
            [genome["genes"]], dtype=torch.float32, device=device
        )
        noise_seed = int(generation["noise_seed"])
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        replay = model.sample(
            archetype,
            genes,
            temperature=float(generation["temperature"]),
            generators=[generator],
        )[0].cpu().numpy().astype(np.uint8)
        stored = _load_stored_tokens(root, manifest, int(architecture["image_size"]))
        difference = int(np.count_nonzero(replay != stored))
        results.append(
            {
                "id": manifest["id"],
                "noise_seed": noise_seed,
                "exact": difference == 0,
                "different_pixels": difference,
            }
        )

    report = {
        "passed": all(item["exact"] for item in results),
        "model_hash": canonical_hash,
        "device": str(device),
        "sprites": results,
    }
    write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
