from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

import numpy as np

from ..cellular_organism.compiler import _load_arrays
from ..cellular_physiology import validate_bank as validate_physiology_bank
from ..cellular_physiology.compiler import _load_overlay
from ..cellular_trauma import validate_bank as validate_trauma_bank
from ..cellular_trauma.compiler import _load_trauma
from ..cellular_symmetry import validate_bank as validate_anatomy_bank
from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import (
    ANATOMY_MANIFEST, BOND_CHANNELS, CANVAS_SIZE, CORPUS_FORMAT, DIRECTION_XY,
    DYNAMIC_CHANNELS, PHYSIOLOGY_MANIFEST, STATIC_CHANNELS, TRAUMA_MANIFEST,
    canonical_json_bytes, sha256_file, source_sha256,
)


CORPUS_NAME = "cellular_nca_corpus.npz"
CORPUS_MANIFEST_NAME = "cellular_nca_corpus.json"


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"nullvector-cellular-nca-array-set-v1\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(memoryview(value.view(np.uint8)))
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _authority() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_anatomy_bank(ANATOMY_MANIFEST)
    validate_physiology_bank(PHYSIOLOGY_MANIFEST)
    validate_trauma_bank(TRAUMA_MANIFEST)
    anatomy = json.loads(ANATOMY_MANIFEST.read_text(encoding="utf-8"))
    physiology = json.loads(PHYSIOLOGY_MANIFEST.read_text(encoding="utf-8"))
    trauma = json.loads(TRAUMA_MANIFEST.read_text(encoding="utf-8"))
    if [item["sample_id"] for item in anatomy["offspring"]] != [item["sample_id"] for item in physiology["identities"]] or [item["sample_id"] for item in anatomy["offspring"]] != [item["sample_id"] for item in trauma["identities"]]:
        raise ValueError("Cellular NCA authority identity order drifted.")
    return anatomy, physiology, trauma


def _scatter(values: np.ndarray, positions: np.ndarray, channels: int) -> np.ndarray:
    result = np.zeros((channels, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    y, x = positions[:, 1].astype(np.intp), positions[:, 0].astype(np.intp)
    if values.ndim == 1:
        result[0, y, x] = values
    else:
        result[:, y, x] = values.T
    return result


def _rasterize(record: Mapping[str, Any], anatomy: Mapping[str, np.ndarray], physiology: Mapping[str, np.ndarray], trauma: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(anatomy["position_xy"]); positions = anatomy["position_xy"].astype(np.int16, copy=False)
    if positions.min() < 0 or positions.max() >= CANVAS_SIZE:
        raise ValueError("Cellular NCA anatomy leaves its native canvas.")
    static = np.zeros((STATIC_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    state = np.zeros((DYNAMIC_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    bonds = np.zeros((BOND_CHANNELS, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    y, x = positions[:, 1].astype(np.intp), positions[:, 0].astype(np.intp)
    static[0, y, x] = 1.0
    tissue = anatomy["tissue"].astype(np.intp)
    static[tissue, y, x] = 1.0  # channels 1..14
    flags = anatomy["cell_flags"].astype(np.uint8)
    for bit in range(8): static[15 + bit, y, x] = ((flags >> bit) & 1).astype(np.float32)
    static[23 + int(record["family_id"]), y, x] = 1.0
    system_role = physiology["system_role"].astype(np.intp)
    for system in range(8):
        for role in range(1, 4): static[28 + system * 3 + role - 1, y, x] = (system_role[system] == role)
        static[52 + system, y, x] = physiology["system_weight"][system]
    heal_class = trauma["heal_class"].astype(np.intp)
    for value in range(1, 7): static[60 + value - 1, y, x] = (heal_class == value)
    static[66, y, x] = trauma["clotting_weight"]
    static[67, y, x] = trauma["scar_bias"]
    static[68, y, x] = trauma["regrowth_weight"]
    continuous = np.stack((
        anatomy["max_health"] / 2.2, anatomy["fluid_capacity"] / 1.07,
        anatomy["nutrient_initial"] / .95, anatomy["energy_initial"] / .85,
        anatomy["mass"] / 1.55, anatomy["stiffness"] / .9,
    ), axis=0).astype(np.float32)
    static[69:75, y, x] = continuous
    degree = np.bincount(anatomy["bond_ab"].ravel(), minlength=count).astype(np.float32)
    conductance_sum = np.zeros(count, dtype=np.float32)
    np.add.at(conductance_sum, anatomy["bond_ab"][:, 0], anatomy["bond_conductance"])
    np.add.at(conductance_sum, anatomy["bond_ab"][:, 1], anatomy["bond_conductance"])
    static[75, y, x] = degree / 8.0; static[76, y, x] = conductance_sum / np.maximum(degree, 1) / .8
    direction_index = {direction: index for index, direction in enumerate(DIRECTION_XY)}
    for edge, (a_raw, b_raw) in enumerate(anatomy["bond_ab"]):
        a, b = int(a_raw), int(b_raw); ax, ay = map(int, positions[a]); bx, by = map(int, positions[b])
        dab, dba = (bx - ax, by - ay), (ax - bx, ay - by)
        if dab not in direction_index or dba not in direction_index: raise ValueError("Cellular NCA encountered a nonlocal anatomy bond.")
        weight = float(anatomy["bond_conductance"][edge]) / .8
        static[77 + direction_index[dab], ay, ax] = weight; static[77 + direction_index[dba], by, bx] = weight
        bonds[direction_index[dab], ay, ax] = 1.0; bonds[direction_index[dba], by, bx] = 1.0
    state[0, y, x] = 1.0
    state[1, y, x] = anatomy["fluid_initial"] / np.maximum(anatomy["fluid_capacity"], 1e-6)
    state[2, y, x] = anatomy["nutrient_initial"] / .95
    state[3, y, x] = anatomy["energy_initial"] / .85
    state[4, y, x] = .88
    state[8, y, x] = np.clip(physiology["system_weight"][3] * .65 + physiology["system_weight"][4] * .25, .05, 1.0)
    state[11, y, x] = 1.0
    return static, state, bonds


def build_corpus(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); require_disk_floor(output, floor_gb=100, planned_bytes=512 * 1024**2); output.mkdir(parents=True, exist_ok=True)
    anatomy_manifest, physiology_manifest, trauma_manifest = _authority()
    physiology_by_id = {item["sample_id"]: item for item in physiology_manifest["identities"]}; trauma_by_id = {item["sample_id"]: item for item in trauma_manifest["identities"]}
    static_values: list[np.ndarray] = []; state_values: list[np.ndarray] = []; bond_values: list[np.ndarray] = []
    sample_ids: list[str] = []; family_ids: list[int] = []
    for record in anatomy_manifest["offspring"]:
        sample_id = str(record["sample_id"]); p_record = physiology_by_id[sample_id]; t_record = trauma_by_id[sample_id]
        anatomy = _load_arrays(ANATOMY_MANIFEST.parent.joinpath(*PurePosixPath(record["arrays"]["path"]).parts))
        physiology = _load_overlay(PHYSIOLOGY_MANIFEST.parent.joinpath(*PurePosixPath(p_record["arrays"]["path"]).parts), len(anatomy["position_xy"]))
        trauma = _load_trauma(TRAUMA_MANIFEST.parent.joinpath(*PurePosixPath(t_record["arrays"]["path"]).parts), len(anatomy["position_xy"]), len(anatomy["bond_ab"]))
        static, state, bonds = _rasterize(record, anatomy, physiology, trauma)
        static_values.append(static); state_values.append(state); bond_values.append(bonds); sample_ids.append(sample_id); family_ids.append(int(record["family_id"]))
    arrays = {
        "static": np.stack(static_values).astype(np.float32), "initial_state": np.stack(state_values).astype(np.float32),
        "live_bonds": np.stack(bond_values).astype(np.float32), "family_id": np.asarray(family_ids, dtype=np.uint8),
        "sample_id": np.asarray(sample_ids, dtype="<U32"),
    }
    semantic = _array_digest(arrays)
    temporary = output / f".{CORPUS_NAME}.tmp-{os.getpid()}"
    try:
        with temporary.open("wb") as handle: np.savez_compressed(handle, **arrays)
        os.replace(temporary, output / CORPUS_NAME)
    finally: temporary.unlink(missing_ok=True)
    manifest = {
        "format": CORPUS_FORMAT, "source_sha256": source_sha256(), "identity_count": 45,
        "static_channels": STATIC_CHANNELS, "dynamic_channels": DYNAMIC_CHANNELS, "bond_channels": BOND_CHANNELS,
        "arrays_semantic_sha256": semantic,
        "authority": {
            "anatomy_manifest": ANATOMY_MANIFEST.relative_to(PROJECT_ROOT).as_posix(), "anatomy_sha256": sha256_file(ANATOMY_MANIFEST),
            "physiology_manifest": PHYSIOLOGY_MANIFEST.relative_to(PROJECT_ROOT).as_posix(), "physiology_sha256": sha256_file(PHYSIOLOGY_MANIFEST),
            "trauma_manifest": TRAUMA_MANIFEST.relative_to(PROJECT_ROOT).as_posix(), "trauma_sha256": sha256_file(TRAUMA_MANIFEST),
        },
        "artifact": {"path": CORPUS_NAME, "bytes": (output / CORPUS_NAME).stat().st_size, "sha256": sha256_file(output / CORPUS_NAME)},
        "gates": {"all_45_identities": len(sample_ids) == 45, "all_five_families": set(family_ids) == set(range(5)), "native_48px": True, "one_cell_per_anatomy_pixel": True, "explicit_eight_direction_bonds": True},
    }
    _atomic_bytes(output / CORPUS_MANIFEST_NAME, canonical_json_bytes(manifest))
    return validate_corpus(output)


def load_corpus(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); encoded = (output / CORPUS_MANIFEST_NAME).read_bytes(); manifest = json.loads(encoded)
    if encoded != canonical_json_bytes(manifest) or manifest.get("format") != CORPUS_FORMAT or manifest.get("source_sha256") != source_sha256() or not all(manifest.get("gates", {}).values()): raise ValueError("Cellular NCA corpus manifest drifted.")
    for key, expected_path in (("anatomy", ANATOMY_MANIFEST), ("physiology", PHYSIOLOGY_MANIFEST), ("trauma", TRAUMA_MANIFEST)):
        if manifest["authority"][f"{key}_sha256"] != sha256_file(expected_path): raise ValueError(f"Cellular NCA {key} authority drifted.")
    artifact = manifest["artifact"]; path = output / artifact["path"]
    if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"] or path.stat().st_size > 512 * 1024**2: raise ValueError("Cellular NCA corpus artifact drifted.")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"static", "initial_state", "live_bonds", "family_id", "sample_id"}: raise ValueError("Cellular NCA corpus member registry drifted.")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    expected = {"static": (np.float32, (45, STATIC_CHANNELS, 48, 48)), "initial_state": (np.float32, (45, DYNAMIC_CHANNELS, 48, 48)), "live_bonds": (np.float32, (45, BOND_CHANNELS, 48, 48)), "family_id": (np.uint8, (45,)), "sample_id": (np.dtype("<U32"), (45,))}
    for name, (dtype, shape) in expected.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape: raise ValueError(f"Cellular NCA {name} tensor contract drifted.")
    if not np.isfinite(arrays["static"]).all() or not np.isfinite(arrays["initial_state"]).all() or np.any((arrays["live_bonds"] < 0) | (arrays["live_bonds"] > 1)): raise ValueError("Cellular NCA corpus values drifted.")
    if _array_digest(arrays) != manifest["arrays_semantic_sha256"]: raise ValueError("Cellular NCA corpus semantic hash drifted.")
    return {"manifest": manifest, "arrays": arrays}


def validate_corpus(output: Path) -> dict[str, Any]:
    loaded = load_corpus(output); arrays = loaded["arrays"]; mask = arrays["static"][:, 0]
    if not np.array_equal(arrays["initial_state"][:, 11], mask) or np.any(arrays["initial_state"][:, :9] * (1 - mask[:, None])): raise ValueError("Cellular NCA initial state escapes anatomy.")
    return {"passed": True, "identity_count": 45, "cell_count": int(mask.sum()), "arrays_semantic_sha256": loaded["manifest"]["arrays_semantic_sha256"], "artifact_sha256": loaded["manifest"]["artifact"]["sha256"]}

