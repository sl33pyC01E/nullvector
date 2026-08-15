from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any
import zipfile

from jsonschema import Draft202012Validator
import numpy as np

from ..cellular_motion import validate_bank as validate_motion_bank
from ..cellular_motion.compiler import _channels, _pose_points
from ..cellular_motion.contract import DRIVER_NAMES, FACING_NAMES, MOTION_NAMES
from ..cellular_organism.compiler import _load_arrays
from ..cellular_symmetry import validate_bank as validate_anatomy_bank
from ..config import PROJECT_ROOT
from ..multifield_style_motion.hashing import canonical_json_bytes
from ..safety import require_disk_floor
from .contract import (
    CORPUS_FORMAT, DEFAULT_ANATOMY, DEFAULT_CORPUS, DEFAULT_MOTION, FEATURE_CHANNELS,
    FEATURE_GROUPS, GRID_SIZE, MAX_DISPLACEMENT, ORGAN_CHANNELS, SCHEMA_PATH,
    STATE_CHANNELS, TARGET_NAMES, corpus_source_sha256,
)


SHARD_KEYS = ("features", "targets", "indices", "previous_index")
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SHARD_BYTES = 100_000_000
_SHARD_SPECS = {
    "features": ((FEATURE_CHANNELS, GRID_SIZE, GRID_SIZE), np.dtype(np.float16)),
    "targets": ((944, STATE_CHANNELS, GRID_SIZE, GRID_SIZE), np.dtype(np.float16)),
    "indices": ((944, 4), np.dtype(np.uint8)),
    "previous_index": ((944,), np.dtype(np.uint16)),
}


def sha256_bytes(payload: bytes) -> str: return hashlib.sha256(payload).hexdigest()
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _read_canonical_json(path: Path, *, maximum_bytes: int = MAX_MANIFEST_BYTES) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 1 <= path.stat().st_size <= maximum_bytes:
        raise ValueError(f"Neural motion JSON artifact is missing, linked, empty, or oversized: {path}")
    encoded = path.read_bytes()
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant {token}.")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Neural motion JSON artifact is malformed: {path}") from error
    if not isinstance(value, dict) or encoded != canonical_json_bytes(value):
        raise ValueError(f"Neural motion JSON artifact is not a canonical object: {path}")
    return value


def _safe_relative(root: Path, value: Any, *, suffix: str | None = None) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Neural motion artifact path is not canonical POSIX text.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Neural motion artifact path is unsafe.")
    if suffix is not None and relative.suffix != suffix:
        raise ValueError("Neural motion artifact path has the wrong suffix.")
    resolved_root = Path(root).resolve()
    path = resolved_root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(resolved_root) or path.is_symlink():
        raise ValueError("Neural motion artifact escapes its authority root.")
    return path


def _validate_npz_container(path: Path) -> None:
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 1 <= path.stat().st_size <= MAX_SHARD_BYTES:
        raise ValueError("Neural motion shard is missing, linked, empty, or oversized.")
    expected = {f"{name}.npy" for name in SHARD_KEYS}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if len(members) != len(expected) or {member.filename for member in members} != expected:
                raise ValueError("Neural motion shard ZIP member census drifted.")
            if len({member.filename for member in members}) != len(members):
                raise ValueError("Neural motion shard ZIP contains duplicate members.")
            for member in members:
                if member.is_dir() or member.compress_size > MAX_SHARD_BYTES:
                    raise ValueError("Neural motion shard ZIP member is invalid or oversized.")
                name = member.filename.removesuffix(".npy")
                shape, dtype = _SHARD_SPECS[name]
                maximum_member_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize + 4096
                if not 1 <= member.file_size <= maximum_member_bytes:
                    raise ValueError("Neural motion shard ZIP member exceeds its tensor bound.")
                with archive.open(member, "r") as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        observed_shape, fortran, observed_dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        observed_shape, fortran, observed_dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise ValueError(f"Unsupported neural motion NPY version {version!r}.")
                    expected_payload = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    if tuple(observed_shape) != shape or observed_dtype != dtype or fortran:
                        raise ValueError("Neural motion shard NPY header violates shape/dtype/layout.")
                    if member.file_size != handle.tell() + expected_payload:
                        raise ValueError("Neural motion shard NPY member size disagrees with its header.")
    except zipfile.BadZipFile as error:
        raise ValueError("Neural motion shard is not a valid NPZ container.") from error


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(b"nullvector-array-v1\0"); digest.update(str(contiguous.dtype).encode("ascii") + b"\0"); digest.update(np.asarray(contiguous.shape, dtype="<u8").tobytes()); digest.update(memoryview(contiguous).cast("B")); return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO(); np.savez_compressed(stream, **{key: arrays[key] for key in SHARD_KEYS}); return stream.getvalue()


def _static_features(arrays: dict[str, np.ndarray], record: dict[str, Any]) -> np.ndarray:
    result = np.zeros((FEATURE_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    positions = arrays["position_xy"].astype(np.int64); channels = _channels(record); organ_to_channel = {int(organ_id): index for index, name in enumerate(ORGAN_CHANNELS) for organ_id in channels[name]}
    continuous = ("mass", "stiffness", "max_health", "fluid_initial", "energy_initial", "nutrient_initial")
    scales = (2.2, 1.0, 2.2, 1.1, 1.0, 1.0)
    for cell, (x_value, y_value) in enumerate(positions):
        x, y = int(x_value), int(y_value)
        if not 0 <= x < GRID_SIZE or not 0 <= y < GRID_SIZE: raise ValueError("Neural motion cell escaped the 48px field.")
        if result[0, y, x] != 0: raise ValueError("Neural motion anatomy contains duplicate cell coordinates.")
        result[0, y, x] = 1; result[1, y, x] = (x - 23.5) / 23.5; result[2, y, x] = (y - 23.5) / 23.5
        for offset, (name, scale) in enumerate(zip(continuous, scales, strict=True)): result[3 + offset, y, x] = float(arrays[name][cell]) / scale
        result[9, y, x] = float(arrays["emission"][cell]) / 3.0
        tissue, material, part = int(arrays["tissue"][cell]), int(arrays["material"][cell]), int(arrays["part_owner"][cell])
        if not 1 <= tissue <= 12 or not 0 <= material <= 9 or not 0 <= part <= 16: raise ValueError("Neural motion categorical cell value escaped its vocabulary.")
        result[10 + tissue - 1, y, x] = 1; result[22 + material, y, x] = 1; result[32 + part, y, x] = 1
        organ_channel = organ_to_channel.get(int(arrays["organ_id"][cell]));
        if organ_channel is None: raise ValueError("Neural motion organ is not assigned to a motor channel.")
        result[49 + organ_channel, y, x] = 1
    if not np.isfinite(result).all() or int(result[0].sum()) != len(positions): raise ValueError("Neural motion feature raster drifted.")
    return result


def _directional_target(arrays: dict[str, np.ndarray], record: dict[str, Any], frame: dict[str, Any], facing_index: int) -> np.ndarray:
    count = GRID_SIZE * GRID_SIZE; result = np.zeros((STATE_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32); rest = arrays["position_xy"].astype(np.float64); posed = _pose_points(arrays, record, frame, 0.0); delta = posed - rest
    angle = facing_index * math.pi / 4; cosine, sine = math.cos(angle), math.sin(angle); rotated = np.column_stack((delta[:, 0] * cosine - delta[:, 1] * sine, delta[:, 0] * sine + delta[:, 1] * cosine))
    driver = {name: float(frame["drivers"][index]) for index, name in enumerate(DRIVER_NAMES)}
    for cell, (x_value, y_value) in enumerate(rest.astype(np.int64)):
        x, y = int(x_value), int(y_value); dx, dy = rotated[cell]
        result[0, y, x] = np.clip(dx / MAX_DISPLACEMENT, -1, 1); result[1, y, x] = np.clip(dy / MAX_DISPLACEMENT, -1, 1)
        result[2, y, x] = min(1.0, math.hypot(dx, dy) / MAX_DISPLACEMENT)
        result[3, y, x] = min(1.0, max(0.0, driver["emission_pulse"])) * float(arrays["emission"][cell]) / 3.0
    if not np.isfinite(result).all() or result.shape != (STATE_CHANNELS, GRID_SIZE, GRID_SIZE): raise ValueError("Neural motion target drifted.")
    return result


def _split_for(family_ordinal: int, family_total: int) -> str:
    """Hold out one validation and one test identity inside every family.

    The authoritative breeding bank contains 45 real offspring, but its
    cross-family pair grammar intentionally produces an uneven primary-family
    census (11/10/9/8/7).  Splits therefore belong to each family's actual
    ordinal range, not to a fictional nine-per-family grid.
    """
    if not 3 <= family_total <= 45 or not 0 <= family_ordinal < family_total:
        raise ValueError("Neural motion family split coordinate drifted.")
    return "test" if family_ordinal == family_total - 1 else "validation" if family_ordinal == family_total - 2 else "train"


def _selection_plan(
    source_records: list[dict[str, Any]], identities_per_family: int | None,
) -> tuple[list[tuple[dict[str, Any], int]], list[int], bool]:
    if len(source_records) != 45:
        raise ValueError("Neural motion source must expose all 45 breeding identities.")
    totals = [sum(int(record["family_id"]) == family_id for record in source_records) for family_id in range(5)]
    if any(total < 3 for total in totals) or sum(totals) != len(source_records):
        raise ValueError("Neural motion source family census cannot support held-out splits.")
    if identities_per_family is not None:
        if type(identities_per_family) is not int or not 1 <= identities_per_family <= min(totals):
            raise ValueError("Neural motion balanced-prefix scope drifted.")
    seen = [0] * 5; selected: list[tuple[dict[str, Any], int]] = []
    for record in source_records:
        family_id = int(record["family_id"])
        if not 0 <= family_id < 5:
            raise ValueError("Neural motion source family escaped its vocabulary.")
        ordinal = seen[family_id]; seen[family_id] += 1
        if identities_per_family is None or ordinal < identities_per_family:
            selected.append((record, ordinal))
    expected = len(source_records) if identities_per_family is None else identities_per_family * 5
    if len(selected) != expected or seen != totals:
        raise ValueError("Neural motion identity selection census drifted.")
    return selected, totals, identities_per_family is None


def _build_shard(record: dict[str, Any], arrays: dict[str, np.ndarray], program: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    features = _static_features(arrays, record).astype(np.float16); targets: list[np.ndarray] = []; indices: list[tuple[int, int, int, int]] = []; previous: list[int] = []; samples: list[dict[str, Any]] = []
    for motion_index, clip in enumerate(program["clips"]):
        for facing_index, facing in enumerate(clip["facings"]):
            clip_start = len(targets); frame_count = len(facing["frames"])
            for frame_index, frame in enumerate(facing["frames"]):
                targets.append(_directional_target(arrays, record, frame, facing_index)); indices.append((motion_index, facing_index, frame_index, int(bool(clip["loop"]))))
                previous.append(clip_start + (frame_index - 1 if frame_index else (frame_count - 2 if clip["loop"] and frame_count > 1 else 0)))
                samples.append({"motion": MOTION_NAMES[motion_index], "facing": FACING_NAMES[facing_index], "frame": frame_index, "loop": bool(clip["loop"])})
    payload = {"features": features, "targets": np.asarray(targets, dtype=np.float16), "indices": np.asarray(indices, dtype=np.uint8), "previous_index": np.asarray(previous, dtype=np.uint16)}
    if payload["targets"].shape != (944, STATE_CHANNELS, GRID_SIZE, GRID_SIZE) or payload["indices"].shape != (944, 4) or payload["previous_index"].shape != (944,): raise ValueError("Neural motion shard census drifted.")
    return payload, samples


def _write_corpus_manifest(
    staging: Path, *, motion_path: Path, anatomy_path: Path,
    motion: dict[str, Any], anatomy: dict[str, Any],
    identities_per_family: int | None, records: list[dict[str, Any]],
) -> dict[str, Any]:
    selected, source_family_counts, production_complete = _selection_plan(anatomy["offspring"], identities_per_family)
    expected_ids = [record["sample_id"] for record, _ in selected]
    if [record["sample_id"] for record in records] != expected_ids:
        raise ValueError("Neural motion manifest records do not cover the selected source order.")
    selected_family_counts = [sum(int(record["family_id"]) == family_id for record, _ in selected) for family_id in range(5)]
    split_counts = {name: sum(item["split"] == name for item in records) for name in ("train", "validation", "test", "smoke")}
    manifest: dict[str, Any] = {
        "format": CORPUS_FORMAT, "status": "ready", "source_sha256": corpus_source_sha256(),
        "sources": {
            "motion_manifest": motion_path.relative_to(PROJECT_ROOT).as_posix(),
            "motion_manifest_sha256": sha256_file(motion_path), "motion_semantic_sha256": motion["semantic_sha256"],
            "anatomy_manifest": anatomy_path.relative_to(PROJECT_ROOT).as_posix(),
            "anatomy_manifest_sha256": sha256_file(anatomy_path), "anatomy_semantic_sha256": anatomy["semantic_sha256"],
        },
        "scope": {
            "selection_mode": "all_source_identities" if production_complete else "balanced_prefix_smoke",
            "identities_per_family": identities_per_family, "source_identity_count": len(anatomy["offspring"]),
            "identity_count": len(records), "sample_count": len(records) * 944,
            "production_complete": production_complete, "source_family_counts": source_family_counts,
            "family_counts": selected_family_counts, "split_counts": split_counts,
        },
        "grid_size": GRID_SIZE, "feature_channels": FEATURE_CHANNELS, "state_channels": STATE_CHANNELS,
        "feature_groups": {name: list(value) for name, value in FEATURE_GROUPS.items()},
        "target_names": list(TARGET_NAMES), "organ_channels": list(ORGAN_CHANNELS),
        "motion_vocab": list(MOTION_NAMES), "facing_vocab": list(FACING_NAMES),
        "max_displacement": MAX_DISPLACEMENT, "records": records,
    }
    manifest["semantic_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    _atomic_bytes(Path(staging) / "neural_cell_motion_corpus.json", canonical_json_bytes(manifest))
    return manifest


def build_corpus(output: Path = DEFAULT_CORPUS, *, identities_per_family: int | None = None, motion_path: Path = DEFAULT_MOTION, anatomy_path: Path = DEFAULT_ANATOMY) -> dict[str, Any]:
    output = Path(output).resolve(); motion_path = Path(motion_path).resolve(); anatomy_path = Path(anatomy_path).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3); validate_motion_bank(motion_path); validate_anatomy_bank(anatomy_path)
    motion = json.loads(motion_path.read_text(encoding="utf-8")); anatomy = json.loads(anatomy_path.read_text(encoding="utf-8")); programs = {item["family_id"]: item for item in motion["programs"]}
    selected, source_family_counts, production_complete = _selection_plan(anatomy["offspring"], identities_per_family)
    selected_family_counts = [sum(int(record["family_id"]) == family_id for record, _ in selected) for family_id in range(5)]
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"; staging.mkdir(parents=True); records: list[dict[str, Any]] = []
    try:
        for record, family_ordinal in selected:
            family_id = int(record["family_id"])
            arrays = _load_arrays(anatomy_path.parent / PurePosixPath(record["arrays"]["path"])); payload, _ = _build_shard(record, arrays, programs[family_id]); encoded = _npz_bytes(payload); relative = f"shards/{record['sample_id']}.npz"; destination = staging / relative; _atomic_bytes(destination, encoded)
            split = _split_for(family_ordinal, source_family_counts[family_id]) if production_complete else "smoke"
            records.append({"sample_id": record["sample_id"], "family": record["family"], "family_id": family_id, "family_ordinal": family_ordinal, "split": split, "source_anatomy_sha256": record["anatomy_sha256"], "path": relative, "bytes": len(encoded), "sha256": sha256_bytes(encoded), "features_sha256": array_sha256(payload["features"]), "targets_sha256": array_sha256(payload["targets"]), "indices_sha256": array_sha256(payload["indices"]), "previous_index_sha256": array_sha256(payload["previous_index"]), "sample_count": 944})
        _write_corpus_manifest(staging, motion_path=motion_path, anatomy_path=anatomy_path, motion=motion, anatomy=anatomy, identities_per_family=identities_per_family, records=records)
        os.replace(staging, output)
    finally:
        if staging.exists():
            import shutil; shutil.rmtree(staging)
    return validate_corpus(output, replay=False)


def load_corpus_manifest(output: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    path = Path(output).resolve() / "neural_cell_motion_corpus.json"; manifest = _read_canonical_json(path)
    errors = sorted(Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Neural motion corpus schema validation failed: {errors[0].message}")
    if manifest["semantic_sha256"] != sha256_bytes(canonical_json_bytes({key: value for key, value in manifest.items() if key != "semantic_sha256"})): raise ValueError("Neural motion corpus semantic identity drifted.")
    return manifest


def validate_corpus(output: Path = DEFAULT_CORPUS, *, replay: bool = False, record_ids: set[str] | None = None) -> dict[str, Any]:
    output = Path(output).resolve(); manifest = load_corpus_manifest(output)
    if manifest["source_sha256"] != corpus_source_sha256(): raise ValueError("Neural motion corpus source drifted.")
    if manifest["grid_size"] != GRID_SIZE or manifest["feature_channels"] != FEATURE_CHANNELS or manifest["state_channels"] != STATE_CHANNELS or manifest["feature_groups"] != {name: list(value) for name, value in FEATURE_GROUPS.items()} or manifest["target_names"] != list(TARGET_NAMES) or manifest["organ_channels"] != list(ORGAN_CHANNELS) or manifest["motion_vocab"] != list(MOTION_NAMES) or manifest["facing_vocab"] != list(FACING_NAMES) or manifest["max_displacement"] != MAX_DISPLACEMENT:
        raise ValueError("Neural motion corpus tensor or vocabulary contract drifted.")
    if manifest["scope"]["identity_count"] != len(manifest["records"]) or manifest["scope"]["sample_count"] != sum(item["sample_count"] for item in manifest["records"]): raise ValueError("Neural motion corpus census drifted.")
    source_manifests: dict[str, dict[str, Any]] = {}
    for label in ("motion", "anatomy"):
        relative = manifest["sources"][f"{label}_manifest"]
        source_path = _safe_relative(PROJECT_ROOT, relative, suffix=".json")
        source = _read_canonical_json(source_path, maximum_bytes=64 * 1024 * 1024)
        if sha256_file(source_path) != manifest["sources"][f"{label}_manifest_sha256"] or source.get("semantic_sha256") != manifest["sources"][f"{label}_semantic_sha256"]:
            raise ValueError(f"Neural motion {label} source authority drifted.")
        source_manifests[label] = source
    anatomy_records = source_manifests["anatomy"]["offspring"]
    identities_per_family = manifest["scope"]["identities_per_family"]
    selected, source_family_counts, production_complete = _selection_plan(anatomy_records, identities_per_family)
    if manifest["scope"]["selection_mode"] != ("all_source_identities" if production_complete else "balanced_prefix_smoke") or manifest["scope"]["source_identity_count"] != len(anatomy_records) or manifest["scope"]["production_complete"] is not production_complete:
        raise ValueError("Neural motion corpus selection authority drifted.")
    expected_records = [record for record, _ in selected]; expected_ids = [record["sample_id"] for record in expected_records]
    if [record["sample_id"] for record in manifest["records"]] != expected_ids:
        raise ValueError("Neural motion corpus source identity order or coverage drifted.")
    requested_ids = set(expected_ids) if record_ids is None else set(record_ids)
    if not requested_ids or not requested_ids <= set(expected_ids):
        raise ValueError("Neural motion shard-validation selection drifted.")
    selected_family_counts = [sum(int(record["family_id"]) == family_id for record in expected_records) for family_id in range(5)]
    if manifest["scope"]["source_family_counts"] != source_family_counts or manifest["scope"]["family_counts"] != selected_family_counts:
        raise ValueError("Neural motion corpus family census drifted.")
    expected_by_id = {record["sample_id"]: record for record in expected_records}
    seen_ids: set[str] = set(); seen_coordinates: set[tuple[int, int]] = set()
    for record in manifest["records"]:
        coordinate = (int(record["family_id"]), int(record["family_ordinal"]))
        expected_record_split = _split_for(coordinate[1], source_family_counts[coordinate[0]]) if production_complete else "smoke"
        if record["sample_id"] in seen_ids or coordinate in seen_coordinates or record["split"] != expected_record_split:
            raise ValueError("Neural motion identity, family ordinal, or split census drifted.")
        source_record = expected_by_id.get(record["sample_id"])
        if source_record is None or record["family_id"] != source_record["family_id"] or record["family"] != source_record["family"] or record["source_anatomy_sha256"] != source_record["anatomy_sha256"]:
            raise ValueError("Neural motion record is not bound to its source anatomy.")
        seen_ids.add(record["sample_id"]); seen_coordinates.add(coordinate)
        if record["sample_id"] not in requested_ids:
            continue
        path = _safe_relative(output, record["path"], suffix=".npz")
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]: raise ValueError("Neural motion shard artifact drifted.")
        _validate_npz_container(path)
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != SHARD_KEYS: raise ValueError("Neural motion shard members drifted.")
            arrays = {key: archive[key] for key in SHARD_KEYS}
        if arrays["features"].shape != (FEATURE_CHANNELS, GRID_SIZE, GRID_SIZE) or arrays["features"].dtype != np.float16 or arrays["targets"].shape != (944, STATE_CHANNELS, GRID_SIZE, GRID_SIZE) or arrays["targets"].dtype != np.float16 or arrays["indices"].shape != (944, 4) or arrays["indices"].dtype != np.uint8 or arrays["previous_index"].shape != (944,) or arrays["previous_index"].dtype != np.uint16: raise ValueError("Neural motion shard tensor contract drifted.")
        for key in SHARD_KEYS:
            if array_sha256(arrays[key]) != record[f"{key}_sha256"]: raise ValueError("Neural motion shard semantic hash drifted.")
        if np.any(arrays["previous_index"] >= 944) or not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["targets"]).all(): raise ValueError("Neural motion shard indices or values drifted.")
        occupancy = arrays["features"][0]
        if not np.all((occupancy == 0) | (occupancy == 1)) or np.any(arrays["features"][1:, occupancy == 0]) or np.any(arrays["targets"][:, :, occupancy == 0]):
            raise ValueError("Neural motion shard leaks features or targets outside the cellular chassis.")
        for start, stop in (FEATURE_GROUPS["tissue_one_hot"], FEATURE_GROUPS["material_one_hot"], FEATURE_GROUPS["part_owner_one_hot"], FEATURE_GROUPS["organ_channel_one_hot"]):
            if not np.all(arrays["features"][start:stop].sum(0)[occupancy == 1] == 1):
                raise ValueError("Neural motion categorical feature group is not exactly one-hot.")
    expected_coordinates = {(family, ordinal) for family, total in enumerate(selected_family_counts) for ordinal in range(total)}
    if seen_coordinates != expected_coordinates:
        raise ValueError("Neural motion family coordinate registry drifted.")
    split_counts = {name: sum(item["split"] == name for item in manifest["records"]) for name in ("train", "validation", "test", "smoke")}
    if manifest["scope"]["split_counts"] != split_counts:
        raise ValueError("Neural motion split census drifted.")
    if replay:
        motion_path = _safe_relative(PROJECT_ROOT, manifest["sources"]["motion_manifest"], suffix=".json"); anatomy_path = _safe_relative(PROJECT_ROOT, manifest["sources"]["anatomy_manifest"], suffix=".json"); motion = source_manifests["motion"]; anatomy = source_manifests["anatomy"]; by_id = {item["sample_id"]: item for item in anatomy["offspring"]}; programs = {item["family_id"]: item for item in motion["programs"]}
        for item in manifest["records"]:
            if item["sample_id"] not in requested_ids: continue
            record = by_id[item["sample_id"]]; expected, _ = _build_shard(record, _load_arrays(anatomy_path.parent / PurePosixPath(record["arrays"]["path"])), programs[record["family_id"]])
            with np.load(output / item["path"], allow_pickle=False) as archive:
                if any(not np.array_equal(archive[key], expected[key]) for key in SHARD_KEYS): raise ValueError("Neural motion exact tensor replay drifted.")
    return {"passed": True, "replay": replay, "identity_count": len(manifest["records"]), "validated_record_count": len(requested_ids), "sample_count": manifest["scope"]["sample_count"], "semantic_sha256": manifest["semantic_sha256"]}
