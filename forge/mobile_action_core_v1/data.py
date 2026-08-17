from __future__ import annotations

import json

import numpy as np

from .contract import LATENT_ROOT, TRAJECTORY_ROOT, file_sha256


NAMES = tuple("abcdef")
FIELDS = ("action", "control", "state", "actor_state", "visibility", "memory")


def load_sequences() -> tuple[dict[str, np.ndarray], ...]:
    manifest = json.loads((LATENT_ROOT / "manifest.json").read_text("utf-8")); sequences = []
    if len(manifest.get("records", ())) != 6: raise ValueError("Mobile action corpus record count drifted.")
    for index, suffix in enumerate(NAMES):
        record = manifest["records"][index]; latent_path = LATENT_ROOT / record["latent"]["path"]
        if file_sha256(latent_path) != record["latent"]["sha256"]: raise ValueError("Mobile action latent shard drifted.")
        with np.load(latent_path, allow_pickle=False) as archive: latent = archive["latent"].astype(np.float32)
        trajectory_root = TRAJECTORY_ROOT / f"natural-world-{suffix}"; trajectory_manifest = json.loads((trajectory_root / "manifest.json").read_text("utf-8")); trajectory_path = trajectory_root / trajectory_manifest["artifact"]["path"]
        if file_sha256(trajectory_path) != trajectory_manifest["artifact"]["sha256"]: raise ValueError("Mobile action trajectory artifact drifted.")
        with np.load(trajectory_path, allow_pickle=False) as archive: sequence = {name: archive[name].astype(np.float32) if name != "action" else archive[name].astype(np.int64) for name in FIELDS}
        if len(latent) != len(sequence["action"]): raise ValueError("Mobile action temporal alignment drifted.")
        sequences.append({"latent": latent, **sequence})
    return tuple(sequences)
