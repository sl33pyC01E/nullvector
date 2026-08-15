from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..cellular_motion import validate_bank as validate_motion_bank
from ..cellular_organism.compiler import _load_arrays
from ..cellular_symmetry import validate_bank as validate_anatomy_bank
from ..multifield_style_motion.hashing import canonical_json_bytes
from .contract import DEFAULT_ANATOMY, DEFAULT_MOTION, corpus_source_sha256
from .dataset import (
    SHARD_KEYS, _atomic_bytes, _build_shard, _npz_bytes, _read_canonical_json,
    _selection_plan, _validate_npz_container, array_sha256, sha256_bytes, sha256_file,
)


RESULT_FORMAT = "nullvector-neural-cell-motion-worker-result-v1"
PREFLIGHT_FORMAT = "nullvector-neural-cell-motion-preflight-v1"


def _semantic(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def validate_sources_worker(destination: Path) -> dict[str, Any]:
    motion_path, anatomy_path = DEFAULT_MOTION.resolve(), DEFAULT_ANATOMY.resolve()
    validate_motion_bank(motion_path); validate_anatomy_bank(anatomy_path)
    motion = _read_canonical_json(motion_path, maximum_bytes=64 * 1024 * 1024)
    anatomy = _read_canonical_json(anatomy_path, maximum_bytes=64 * 1024 * 1024)
    selected, totals, production = _selection_plan(anatomy["offspring"], None)
    if not production or len(selected) != 45 or totals != [11, 10, 9, 8, 7]:
        raise ValueError("Neural motion production authority census drifted.")
    result: dict[str, Any] = {
        "format": PREFLIGHT_FORMAT, "status": "passed", "source_sha256": corpus_source_sha256(),
        "motion_manifest_sha256": sha256_file(motion_path), "anatomy_manifest_sha256": sha256_file(anatomy_path),
        "identity_count": len(selected), "source_family_counts": totals,
    }
    result["semantic_sha256"] = _semantic(result)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()): raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True); _atomic_bytes(destination / "preflight.json", canonical_json_bytes(result)); return result


def build_worker_shard(spec_path: Path, destination: Path, *, reuse: Path | None = None) -> dict[str, Any]:
    spec = _read_canonical_json(spec_path, maximum_bytes=128 * 1024)
    required = {"format", "source_sha256", "identities_per_family", "sample_id", "family", "family_id", "family_ordinal", "split", "motion_manifest_sha256", "anatomy_manifest_sha256", "semantic_sha256"}
    if set(spec) != required or spec["format"] != "nullvector-neural-cell-motion-worker-spec-v1" or spec["source_sha256"] != corpus_source_sha256():
        raise ValueError("Neural motion worker spec contract drifted.")
    if spec["semantic_sha256"] != _semantic({key: value for key, value in spec.items() if key != "semantic_sha256"}):
        raise ValueError("Neural motion worker spec semantic identity drifted.")
    motion_path, anatomy_path = DEFAULT_MOTION.resolve(), DEFAULT_ANATOMY.resolve()
    if sha256_file(motion_path) != spec["motion_manifest_sha256"] or sha256_file(anatomy_path) != spec["anatomy_manifest_sha256"]:
        raise ValueError("Neural motion worker source manifest drifted.")
    motion = _read_canonical_json(motion_path, maximum_bytes=64 * 1024 * 1024)
    anatomy = _read_canonical_json(anatomy_path, maximum_bytes=64 * 1024 * 1024)
    selection_scope = spec["identities_per_family"]
    selected, totals, production = _selection_plan(anatomy["offspring"], selection_scope)
    candidates = [(record, ordinal) for record, ordinal in selected if record["sample_id"] == spec["sample_id"]]
    if len(candidates) != 1:
        raise ValueError("Neural motion worker sample is outside its source selection.")
    record, ordinal = candidates[0]; family_id = int(record["family_id"])
    expected_split = ("test" if ordinal == totals[family_id] - 1 else "validation" if ordinal == totals[family_id] - 2 else "train") if production else "smoke"
    if [spec["family"], spec["family_id"], spec["family_ordinal"], spec["split"]] != [record["family"], family_id, ordinal, expected_split]:
        raise ValueError("Neural motion worker sample coordinates drifted.")
    programs = {item["family_id"]: item for item in motion["programs"]}
    expected, _ = _build_shard(record, _load_arrays(anatomy_path.parent / PurePosixPath(record["arrays"]["path"])), programs[family_id])
    reused = reuse is not None
    if reuse is not None:
        reuse = Path(reuse).resolve(); _validate_npz_container(reuse)
        with np.load(reuse, allow_pickle=False) as archive:
            if any(not np.array_equal(archive[key], expected[key]) for key in SHARD_KEYS):
                raise ValueError("Neural motion recovery shard failed exact replay.")
        encoded = reuse.read_bytes()
    else:
        encoded = _npz_bytes(expected)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / f"{record['sample_id']}.npz"; _atomic_bytes(artifact, encoded)
    corpus_record = {
        "sample_id": record["sample_id"], "family": record["family"], "family_id": family_id,
        "family_ordinal": ordinal, "split": expected_split, "source_anatomy_sha256": record["anatomy_sha256"],
        "path": f"shards/{record['sample_id']}.npz", "bytes": len(encoded), "sha256": sha256_bytes(encoded),
        "features_sha256": array_sha256(expected["features"]), "targets_sha256": array_sha256(expected["targets"]),
        "indices_sha256": array_sha256(expected["indices"]), "previous_index_sha256": array_sha256(expected["previous_index"]),
        "sample_count": 944,
    }
    result: dict[str, Any] = {"format": RESULT_FORMAT, "status": "passed", "source_sha256": corpus_source_sha256(), "spec_sha256": spec["semantic_sha256"], "reused": reused, "record": corpus_record}
    result["semantic_sha256"] = _semantic(result); _atomic_bytes(destination / "result.json", canonical_json_bytes(result)); return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one isolated neural-cell-motion shard")
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--spec", type=Path); mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--destination", type=Path, required=True); parser.add_argument("--reuse", type=Path)
    args = parser.parse_args(argv)
    if args.preflight:
        if args.reuse is not None: parser.error("--reuse is invalid with --preflight")
        result = validate_sources_worker(args.destination); summary = {"passed": True, "identity_count": result["identity_count"]}
    else:
        result = build_worker_shard(args.spec, args.destination, reuse=args.reuse); summary = {"passed": True, "sample_id": result["record"]["sample_id"], "reused": result["reused"]}
    print(json.dumps(summary, sort_keys=True)); return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    raise SystemExit(main())
