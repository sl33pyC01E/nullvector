from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..maps.io import ARRAY_NAMES, array_digest
from ..maps.model import MapConfig, MapData
from ..maps.validate import assert_valid
from .contract import canonical_json_bytes, file_sha256


FORMAT = "nullvector-neural-compiled-map-pack-v1/1.0.0"


def _npy_bytes(array: np.ndarray) -> bytes:
    from io import BytesIO
    buffer = BytesIO(); np.save(buffer, np.ascontiguousarray(array), allow_pickle=False); return buffer.getvalue()


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException: temporary.unlink(missing_ok=True); raise


def write_neural_map_pack(data: MapData, output_root: Path) -> Path:
    report = assert_valid(data); final = Path(output_root) / data.map_id
    if final.exists(): raise FileExistsError(final)
    staging = final.parent / f".{final.name}.tmp-{os.getpid()}"; staging.mkdir(parents=True, exist_ok=False)
    try:
        artifacts = {}
        for name, array in data.arrays().items():
            payload = _npy_bytes(array); path = staging / "arrays" / f"{name}.npy"; _atomic(path, payload)
            artifacts[name] = {"path": path.relative_to(staging).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "dtype": array.dtype.str, "shape": list(array.shape)}
        manifest = {
            "format": FORMAT, "map_id": data.map_id, "theme": data.theme, "seed": int(data.seed), "config": data.config.to_dict(),
            "points": {"start": list(data.start), "exit": list(data.exit), "objectives": [list(point) for point in data.objectives], "spawns": [list(point) for point in data.spawns]},
            "repair_count": int(data.repair_count), "semantic_arrays_sha256": array_digest(data.arrays()), "artifacts": artifacts,
            "validation": {"checks": report["checks"], "metrics": report["metrics"]}, "authority": "neural_raw_plus_deterministic_safety_compiler",
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(); _atomic(staging / "manifest.json", canonical_json_bytes(manifest)); os.replace(staging, final); return final
    finally:
        if staging.exists():
            for root, dirs, files in os.walk(staging, topdown=False):
                for name in files: Path(root, name).unlink(missing_ok=True)
                for name in dirs: Path(root, name).rmdir()
            staging.rmdir()


def load_neural_map_pack(pack: Path) -> MapData:
    pack = Path(pack); raw = (pack / "manifest.json").read_bytes(); manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest) or manifest.get("format") != FORMAT: raise ValueError("Neural map manifest header drifted.")
    stored = manifest.pop("manifest_sha256", None); expected = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(); manifest["manifest_sha256"] = stored
    if stored != expected: raise ValueError("Neural map manifest hash drifted.")
    arrays = {}
    for name in ARRAY_NAMES:
        artifact = manifest["artifacts"][name]; path = pack / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]: raise ValueError("Neural map array artifact drifted.")
        array = np.load(path, allow_pickle=False)
        if array.dtype.str != artifact["dtype"] or list(array.shape) != artifact["shape"]: raise ValueError("Neural map array descriptor drifted.")
        arrays[name] = np.ascontiguousarray(array)
    if array_digest(arrays) != manifest["semantic_arrays_sha256"]: raise ValueError("Neural map array semantics drifted.")
    points = manifest["points"]; config = MapConfig(**manifest["config"])
    data = MapData(seed=int(manifest["seed"]), theme=str(manifest["theme"]), config=config, start=tuple(points["start"]), exit=tuple(points["exit"]), objectives=tuple(map(tuple, points["objectives"])), spawns=tuple(map(tuple, points["spawns"])), repair_count=int(manifest["repair_count"]), metadata={"neural_pack_format": FORMAT}, **arrays)
    if data.map_id != manifest["map_id"]: raise ValueError("Neural map identity drifted.")
    assert_valid(data); return data


def validate_neural_map_pack(pack: Path) -> dict[str, object]:
    data = load_neural_map_pack(pack); return {"passed": True, "map_id": data.map_id, "theme": data.theme, "semantic_arrays_sha256": array_digest(data.arrays())}
