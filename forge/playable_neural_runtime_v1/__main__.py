from __future__ import annotations

import argparse
import json

import numpy as np

from .contract import COMPOSITE, DEFAULT_OUTPUT, ENSEMBLE, file_sha256, source_sha256
from .release import build,validate
from .runtime import PlayableNeuralRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.build:print(json.dumps(build(DEFAULT_OUTPUT),sort_keys=True));return
    if args.validate:print(json.dumps(validate(DEFAULT_OUTPUT),sort_keys=True));return
    runtime = PlayableNeuralRuntime.from_release(device=args.device)
    frame = np.zeros((256, 256, 3), np.uint8)
    latent = runtime.composite.encode(frame)
    decoded = runtime.composite.decode(latent)
    print(json.dumps({
        "passed": bool(decoded.shape == (1, 3, 256, 256)),
        "components": runtime.component_count,
        "source_sha256": source_sha256(),
        "ensemble_sha256": file_sha256(ENSEMBLE),
        "composite_sha256": file_sha256(COMPOSITE),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
