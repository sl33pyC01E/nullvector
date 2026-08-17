from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import COMPOSITE, DEFAULT_OUTPUT, ENSEMBLE, FORMAT, canonical, file_sha256, source_sha256
from .runtime import _component_table


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(output: Path = DEFAULT_OUTPUT) -> dict:
    output = Path(output).resolve();destination = output / "runtime_manifest.json"
    if destination.exists():raise FileExistsError(destination)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 20)
    rows = _component_table()
    payload = {
        "format": FORMAT,
        "status": "playable_neural_runtime_ready",
        "source_sha256": source_sha256(),
        "parents": {
            "teacher_ensemble": {"path": str(ENSEMBLE.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": file_sha256(ENSEMBLE)},
            "composite_world": {"path": str(COMPOSITE.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": file_sha256(COMPOSITE)},
        },
        "neural_components": sorted(rows),
        "live_interfaces": ["world_frame_encode", "world_frame_refine", "action_conditioned_future", "actor_state", "cell_raster", "cell_physiology", "locomotion", "behavior", "colony", "society", "timeline", "counterfactual"],
        "integration": {"nature_sim_v2": True, "cells_organs_damage": True, "material_physics": True, "feeding": True, "evolution_and_grafting": True, "deterministic_safety_projection": True},
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    _atomic(destination, canonical(payload))
    return validate(output)


def validate(output: Path = DEFAULT_OUTPUT) -> dict:
    path = Path(output).resolve() / "runtime_manifest.json";raw = path.read_bytes();payload = json.loads(raw)
    if raw != canonical(payload) or payload.get("format") != FORMAT or payload.get("source_sha256") != source_sha256():raise ValueError("playable neural runtime manifest drifted")
    expected = hashlib.sha256(canonical({key:value for key,value in payload.items() if key!="manifest_sha256"})).hexdigest()
    if payload.get("manifest_sha256") != expected:raise ValueError("playable neural runtime manifest hash drifted")
    for record in payload["parents"].values():
        if file_sha256(PROJECT_ROOT/record["path"]) != record["sha256"]:raise ValueError("playable neural runtime parent drifted")
    return {"passed": payload["status"]=="playable_neural_runtime_ready", "components": len(payload["neural_components"]), "manifest_sha256": payload["manifest_sha256"]}
