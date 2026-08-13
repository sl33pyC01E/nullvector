from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import uuid
from typing import Any

import jsonschema
import numpy as np

from ..safety import require_disk_floor, write_json_atomic
from .model import (
    GENERATOR_NAME,
    GENERATOR_VERSION,
    HAZARD_NAMES,
    MAP_SCHEMA_VERSION,
    RNG_NAME,
    TERRAIN_NAMES,
    TOPOLOGY_MASK_CAPTURE_POLICY,
    TOPOLOGY_MASK_CONTRACT_NAME,
    TOPOLOGY_MASK_CONTRACT_VERSION,
    TOPOLOGY_MASK_MEANINGS,
    TOPOLOGY_MASK_NAMES,
    MapConfig,
    MapData,
)
from .render import preview_png_bytes
from .validate import SCHEMA_PATH, assert_valid


ARRAY_FILE = "semantics.npz"
PREVIEW_FILE = "preview.png"
MANIFEST_FILE = "manifest.json"
CANONICAL_ARRAY_HASH_ALGORITHM = "sha256-canonical-named-arrays-v1"
ARRAY_NAMES = (
    "terrain",
    "walkability",
    "hazard",
    "elevation",
    "zone",
    "nav_cost",
    *TOPOLOGY_MASK_NAMES,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    """Canonical semantic digest independent of NPZ compression metadata."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        # Match NumPy int64 shape bytes on little-endian production hosts while
        # avoiding thousands of transient ndarray/bytes allocations in long
        # fuzz processes. hashlib consumes the contiguous buffer directly.
        digest.update(struct.pack("<" + "q" * array.ndim, *array.shape))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, planned_bytes=len(payload) + 1024 * 1024)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **{name: arrays[name] for name in ARRAY_NAMES})
    return buffer.getvalue()


def build_manifest(
    data: MapData,
    report: dict[str, Any],
    *,
    array_hash: str,
    arrays_file_hash: str,
    preview_file_hash: str,
    preview_scale: int,
) -> dict[str, Any]:
    arrays = data.arrays()
    topology_masks = {name: arrays[name] for name in TOPOLOGY_MASK_NAMES}
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "map_id": data.map_id,
        "seed": int(data.seed),
        "seed_hex": f"0x{data.seed:016x}",
        "theme": data.theme,
        "dimensions": {"width": data.config.width, "height": data.config.height},
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "rng": RNG_NAME,
            "config": data.config.to_dict(),
            "theme_parameters": data.metadata.get("theme_parameters", {}),
        },
        "semantics": {
            "coordinate_system": "origin top-left; points are [x, y]; arrays are [y, x]",
            "terrain": {str(key): value for key, value in TERRAIN_NAMES.items()},
            "hazard": {str(key): value for key, value in HAZARD_NAMES.items()},
            "arrays": {
                name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                for name, array in arrays.items()
            },
            "nav_cost": "zero means blocked; positive finite values are relative traversal costs",
            "zone": "-1 means blocked; non-negative values partition navigable cells",
            "topology_masks": {
                "contract_name": TOPOLOGY_MASK_CONTRACT_NAME,
                "contract_version": TOPOLOGY_MASK_CONTRACT_VERSION,
                "capture_policy": TOPOLOGY_MASK_CAPTURE_POLICY,
                "hash_algorithm": CANONICAL_ARRAY_HASH_ALGORITHM,
                "combined_sha256": array_digest(topology_masks),
                "members": {
                    name: {
                        "meaning": TOPOLOGY_MASK_MEANINGS[name],
                        "provenance": "authoritative generation-time capture",
                        "hash_scope": "single_named_array",
                        "sha256": array_digest({name: topology_masks[name]}),
                        "cell_count": int((topology_masks[name] != 0).sum()),
                    }
                    for name in TOPOLOGY_MASK_NAMES
                },
            },
        },
        "points": {
            "start": list(data.start),
            "exit": list(data.exit),
            "objectives": [list(point) for point in data.objectives],
            "spawns": [list(point) for point in data.spawns],
        },
        "topology": {
            "required_route_repairs": int(data.repair_count),
            "protected_backbone_segments": int(
                data.metadata.get("protected_backbone_segments", 1 + len(data.objectives))
            ),
            "start_exit_path_length": int(report["metrics"]["start_exit_path_length"]),
            "minimum_start_exit_path_length": data.config.effective_min_separation,
            "invariants": report["checks"],
        },
        "statistics": report["metrics"],
        "semantic_array_hash_algorithm": CANONICAL_ARRAY_HASH_ALGORITHM,
        "semantic_array_sha256": array_hash,
        "artifacts": {
            "arrays": {
                "file": ARRAY_FILE,
                "format": "npz-deflate",
                "sha256": arrays_file_hash,
            },
            "preview": {
                "file": PREVIEW_FILE,
                "format": "png-rgb-nearest-neighbor",
                "scale": preview_scale,
                "sha256": preview_file_hash,
            },
        },
    }


def write_map_pack(
    data: MapData,
    output_root: Path,
    *,
    preview_scale: int = 5,
    skip_existing: bool = False,
) -> Path:
    """Stage and atomically publish a complete map directory."""
    report = assert_valid(data)
    output_root = Path(output_root)
    final = output_root / data.map_id
    if final.exists():
        if not skip_existing:
            raise FileExistsError(f"Map pack already exists: {final}")
        from .validate import validate_pack

        existing = validate_pack(final)
        if not existing["passed"]:
            raise RuntimeError(f"Existing map pack is invalid and was left untouched: {final}")
        return final

    arrays = data.arrays()
    npz_payload = _npz_bytes(arrays)
    preview_payload = preview_png_bytes(data, scale=preview_scale)
    planned_bytes = len(npz_payload) + len(preview_payload) + 2 * 1024 * 1024
    require_disk_floor(output_root, planned_bytes=planned_bytes)
    output_root.mkdir(parents=True, exist_ok=True)

    staging = output_root / f".{data.map_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    arrays_path = staging / ARRAY_FILE
    preview_path = staging / PREVIEW_FILE
    _atomic_bytes(arrays_path, npz_payload)
    _atomic_bytes(preview_path, preview_payload)
    manifest = build_manifest(
        data,
        report,
        array_hash=array_digest(arrays),
        arrays_file_hash=file_sha256(arrays_path),
        preview_file_hash=file_sha256(preview_path),
        preview_scale=preview_scale,
    )
    write_json_atomic(staging / MANIFEST_FILE, manifest)
    # The destination does not exist, so directory publication is a single rename.
    os.replace(staging, final)
    return final


def load_map_pack(path: Path, *, verify_hashes: bool = True) -> MapData:
    path = Path(path)
    manifest_path = path if path.name == MANIFEST_FILE else path / MANIFEST_FILE
    pack_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_schema = manifest.get("schema_version")
    if observed_schema != MAP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported map schema_version {observed_schema!r}; expected "
            f"{MAP_SCHEMA_VERSION!r}. Legacy packs are rejected because authoritative "
            "topology masks cannot be fabricated."
        )
    generator = manifest.get("generator", {})
    if generator.get("name") != GENERATOR_NAME or generator.get("version") != GENERATOR_VERSION:
        raise ValueError(
            "Map generator identity does not match the authoritative reader: "
            f"expected {GENERATOR_NAME} {GENERATOR_VERSION}."
        )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(manifest),
        key=lambda item: list(item.path),
    )
    if schema_errors:
        raise ValueError(
            "Map manifest does not satisfy the v2 schema: "
            + "; ".join(error.message for error in schema_errors[:8])
        )
    artifacts = manifest.get("artifacts", {})
    array_artifact = artifacts.get("arrays", {})
    preview_artifact = artifacts.get("preview", {})
    if array_artifact.get("file") != ARRAY_FILE or preview_artifact.get("file") != PREVIEW_FILE:
        raise ValueError("Map artifact paths must use the fixed v2 pack filenames.")

    arrays_path = pack_dir / ARRAY_FILE
    with np.load(arrays_path, allow_pickle=False) as archive:
        observed = set(archive.files)
        expected = set(ARRAY_NAMES)
        if observed != expected:
            raise ValueError(f"Expected semantic arrays {sorted(expected)}, observed {sorted(observed)}")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in ARRAY_NAMES}

    config_payload = generator["config"]
    config = MapConfig(
        width=int(config_payload["width"]),
        height=int(config_payload["height"]),
        objective_count=int(config_payload["objective_count"]),
        spawn_count=int(config_payload["spawn_count"]),
        min_start_exit_distance=int(config_payload["min_start_exit_distance"]),
        spawn_clearance_start=int(config_payload["spawn_clearance_start"]),
        spawn_clearance_objective=int(config_payload["spawn_clearance_objective"]),
        spawn_clearance_hazard=int(config_payload["spawn_clearance_hazard"]),
    )
    dimensions = manifest.get("dimensions", {})
    if dimensions != {"width": config.width, "height": config.height}:
        raise ValueError("Manifest dimensions do not exactly match generator config dimensions.")
    descriptors = manifest.get("semantics", {}).get("arrays", {})
    expected_shape = [config.height, config.width]
    if set(descriptors) != set(ARRAY_NAMES):
        raise ValueError("Manifest semantic array descriptors do not match the v2 array set.")
    for name in ARRAY_NAMES:
        observed_descriptor = {
            "dtype": str(arrays[name].dtype),
            "shape": list(arrays[name].shape),
        }
        if descriptors[name] != observed_descriptor:
            raise ValueError(
                f"Manifest descriptor for {name} does not match the NPZ member: "
                f"{descriptors[name]!r} != {observed_descriptor!r}."
            )
        if observed_descriptor["shape"] != expected_shape:
            raise ValueError(f"Semantic array {name} does not match map dimensions.")

    topology_contract = manifest.get("semantics", {}).get("topology_masks", {})
    if (
        topology_contract.get("contract_name") != TOPOLOGY_MASK_CONTRACT_NAME
        or topology_contract.get("contract_version") != TOPOLOGY_MASK_CONTRACT_VERSION
        or topology_contract.get("capture_policy") != TOPOLOGY_MASK_CAPTURE_POLICY
        or topology_contract.get("hash_algorithm") != CANONICAL_ARRAY_HASH_ALGORITHM
        or manifest.get("semantic_array_hash_algorithm") != CANONICAL_ARRAY_HASH_ALGORITHM
    ):
        raise ValueError("Manifest topology-mask contract identity is unsupported.")
    members = topology_contract.get("members", {})
    if set(members) != set(TOPOLOGY_MASK_NAMES):
        raise ValueError("Manifest topology-mask members are incomplete or unexpected.")

    if verify_hashes:
        if file_sha256(arrays_path) != array_artifact["sha256"]:
            raise ValueError("Map NPZ hash does not match manifest.")
        if array_digest(arrays) != manifest["semantic_array_sha256"]:
            raise ValueError("Canonical semantic array hash does not match manifest.")
        topology_arrays = {name: arrays[name] for name in TOPOLOGY_MASK_NAMES}
        if array_digest(topology_arrays) != topology_contract.get("combined_sha256"):
            raise ValueError("Combined topology-mask hash does not match manifest.")
        for name in TOPOLOGY_MASK_NAMES:
            member = members[name]
            if member.get("meaning") != TOPOLOGY_MASK_MEANINGS[name]:
                raise ValueError(f"Topology-mask meaning drifted for {name}.")
            if array_digest({name: arrays[name]}) != member.get("sha256"):
                raise ValueError(f"Topology-mask hash does not match manifest for {name}.")
            if int((arrays[name] != 0).sum()) != member.get("cell_count"):
                raise ValueError(f"Topology-mask cell count does not match manifest for {name}.")
        preview_path = pack_dir / PREVIEW_FILE
        if file_sha256(preview_path) != preview_artifact["sha256"]:
            raise ValueError("Map preview hash does not match manifest.")

    points = manifest["points"]
    data = MapData(
        seed=int(manifest["seed"]),
        theme=str(manifest["theme"]),
        config=config,
        terrain=arrays["terrain"],
        walkability=arrays["walkability"],
        hazard=arrays["hazard"],
        elevation=arrays["elevation"],
        zone=arrays["zone"],
        nav_cost=arrays["nav_cost"],
        protected_backbone=arrays["protected_backbone"],
        required_clearance=arrays["required_clearance"],
        decoration_forbidden=arrays["decoration_forbidden"],
        start=tuple(int(value) for value in points["start"]),
        exit=tuple(int(value) for value in points["exit"]),
        objectives=tuple(tuple(int(value) for value in point) for point in points["objectives"]),
        spawns=tuple(tuple(int(value) for value in point) for point in points["spawns"]),
        repair_count=int(manifest["topology"]["required_route_repairs"]),
        metadata={
            "theme_parameters": generator.get("theme_parameters", {}),
            "protected_backbone_segments": int(
                manifest["topology"]["protected_backbone_segments"]
            ),
            "topology_mask_capture": topology_contract.get("capture_policy"),
            "loaded_from": str(pack_dir),
        },
    )
    if manifest.get("seed_hex") != f"0x{data.seed:016x}":
        raise ValueError("Manifest seed_hex does not match seed.")
    if manifest.get("map_id") != data.map_id:
        raise ValueError("Manifest map_id does not match loaded identity.")
    if verify_hashes:
        map_report = assert_valid(data)
        from .generator import generate_map

        replay = generate_map(data.seed, data.theme, data.config)
        point_state = (
            replay.start,
            replay.exit,
            replay.objectives,
            replay.spawns,
            replay.repair_count,
            replay.metadata.get("theme_parameters"),
            replay.metadata.get("protected_backbone_segments"),
            replay.metadata.get("topology_mask_capture"),
        )
        loaded_state = (
            data.start,
            data.exit,
            data.objectives,
            data.spawns,
            data.repair_count,
            data.metadata.get("theme_parameters"),
            data.metadata.get("protected_backbone_segments"),
            data.metadata.get("topology_mask_capture"),
        )
        if point_state != loaded_state:
            raise ValueError("Map points or generator metadata disagree with deterministic replay.")
        replay_arrays = replay.arrays()
        drifted = [
            name for name in ARRAY_NAMES if not np.array_equal(replay_arrays[name], arrays[name])
        ]
        if drifted:
            raise ValueError(
                "Map semantic arrays disagree with deterministic replay: "
                + ", ".join(drifted)
            )
        expected_preview_hash = hashlib.sha256(
            preview_png_bytes(data, scale=int(preview_artifact["scale"]))
        ).hexdigest()
        if expected_preview_hash != preview_artifact["sha256"]:
            raise ValueError("Map preview disagrees with deterministic rendering replay.")
        expected_manifest = build_manifest(
            data,
            map_report,
            array_hash=array_digest(arrays),
            arrays_file_hash=file_sha256(arrays_path),
            preview_file_hash=file_sha256(pack_dir / PREVIEW_FILE),
            preview_scale=int(preview_artifact["scale"]),
        )
        if manifest != expected_manifest:
            raise ValueError("Map manifest disagrees with loaded semantics and artifacts.")
    return data
