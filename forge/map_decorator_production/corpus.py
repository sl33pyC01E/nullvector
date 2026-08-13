from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

import numpy as np

from ..config import PROJECT_ROOT
from ..map_decorator.catalog import CATALOG_SHA256
from ..map_decorator.contract import FEATURE_CONTRACT_SHA256
from ..map_decorator.hashing import json_sha256, named_arrays_sha256
from ..map_decorator_ml.contract import HEAD_CLASS_COUNTS, HEAD_NAMES, MODEL_CONTRACT_SHA256
from ..map_decorator_ml.dataset import split_for_identity
from ..maps import MapConfig, MapData, generate_map
from ..maps.io import ARRAY_NAMES, file_sha256
from ..maps.model import THEMES
from ..safety import require_disk_floor
from .contract import (
    CORPUS_FORMAT_VERSION,
    DISK_FLOOR_GIB,
    FEATURE_SEED_SALT,
    GENERATION_SEED_SALT,
    MAX_PROCESS_ATTEMPTS,
    MAX_WORKERS,
    OBJECTIVE_BUCKETS,
    PRODUCTION_CONTRACT_SHA256,
    SENTINEL_DIMENSIONS,
    SENTINEL_SEED_SALT,
    SHARD_FORMAT_VERSION,
    SIZE_PROFILES,
    CorpusConfig,
)
from .provenance import source_manifest, source_sha256
from .teacher import (
    SEMANTIC_TEACHER_VERSION,
    build_production_sample,
    full_map_identity_sha256,
    sample_record,
    stacked_sample_arrays,
)


MANIFEST_FILE = "corpus_manifest.json"
VALIDATION_FILE = "corpus_validation.json"
ESTIMATE_FORMAT_VERSION = "1.0.0"
MAX_SHARD_SIDECAR_BYTES = 8 * 1024 * 1024
NPZ_OVERHEAD_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ShardSpec:
    shard_id: str
    kind: str
    ordinal: int
    theme: str
    width: int
    height: int
    objective_bucket: str
    objective_count: int
    spawn_count: int
    required_splits: dict[str, int]
    global_seed: int
    max_candidates: int
    replay_every_sample: bool
    npz_path: str
    sidecar_path: str
    validation_path: str
    estimated_raw_bytes: int

    def __post_init__(self) -> None:
        if self.kind not in {"main", "sentinel"}:
            raise ValueError("Shard kind must be main or sentinel.")
        if self.theme not in THEMES:
            raise ValueError("Shard theme is outside the map contract.")
        if not self.required_splits or any(value < 1 for value in self.required_splits.values()):
            raise ValueError("Shard split requirements must be positive.")
        if self.kind == "main" and set(self.required_splits) != {"train", "validation"}:
            raise ValueError("Main shards require exact train/validation counts.")
        if self.kind == "sentinel" and self.required_splits != {"test": 1}:
            raise ValueError("Sentinel shards require exactly one test identity.")

    @property
    def sample_count(self) -> int:
        return sum(self.required_splits.values())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ShardSpec":
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("Shard specification members are incomplete or unexpected.")
        return cls(**payload)  # type: ignore[arg-type]


def _mix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 64) - 1)


def _candidate_seed(spec: ShardSpec, candidate_index: int) -> int:
    salt = SENTINEL_SEED_SALT if spec.kind == "sentinel" else GENERATION_SEED_SALT
    domain = (spec.ordinal + 1) * 0xD6E8FEB86659FD93
    return _mix64(spec.global_seed ^ salt ^ domain ^ candidate_index)


def _feature_seed(map_seed: int, ordinal: int) -> int:
    return _mix64(map_seed ^ FEATURE_SEED_SALT ^ (ordinal * 0xA5A3564E27F8862D))


def _expected_raw_bytes(
    count: int,
    height: int,
    width: int,
    objective_count: int,
    spawn_count: int,
) -> int:
    cells = count * height * width
    # Features + four targets + 18 legal booleans + hard-empty + nine semantic arrays.
    per_cell = 53 * 4 + 4 + sum(HEAD_CLASS_COUNTS.values()) + 1 + 13
    fixed = count * (
        8 * 4  # global conditions
        + 8  # theme index
        + 8 * 2  # map and feature seeds
        + 2 * 2 * 2  # start and exit
        + objective_count * 2 * 2
        + spawn_count * 2 * 2
        + 2  # repair count
    )
    return cells * per_cell + fixed


def production_specs(config: CorpusConfig = CorpusConfig()) -> tuple[ShardSpec, ...]:
    specs: list[ShardSpec] = []
    ordinal = 0
    for theme_index, theme in enumerate(THEMES):
        for profile_index, profile in enumerate(SIZE_PROFILES):
            for bucket_index, bucket in enumerate(OBJECTIVE_BUCKETS):
                shard_id = f"main-t{theme_index:02d}-p{profile_index:02d}-o{bucket_index:02d}"
                count = config.identities_per_stratum
                specs.append(
                    ShardSpec(
                        shard_id=shard_id,
                        kind="main",
                        ordinal=ordinal,
                        theme=theme,
                        width=profile.width,
                        height=profile.height,
                        objective_bucket=bucket.key,
                        objective_count=bucket.objective_count,
                        spawn_count=profile.spawn_count,
                        required_splits={
                            "train": config.train_per_stratum,
                            "validation": config.validation_per_stratum,
                        },
                        global_seed=config.global_seed,
                        max_candidates=config.max_candidates_per_shard,
                        replay_every_sample=config.replay_every_sample,
                        npz_path=f"shards/main/{shard_id}/fields.npz",
                        sidecar_path=f"shards/main/{shard_id}/shard.json",
                        validation_path=f"validation/shards/{shard_id}.json",
                        estimated_raw_bytes=_expected_raw_bytes(
                            count,
                            profile.height,
                            profile.width,
                            bucket.objective_count,
                            profile.spawn_count,
                        ),
                    )
                )
                ordinal += 1
    sentinel_spawns = (4, 8, 12, 20)
    for theme_index, theme in enumerate(THEMES):
        for size_index, size in enumerate(SENTINEL_DIMENSIONS):
            bucket = OBJECTIVE_BUCKETS[size_index]
            shard_id = f"sentinel-t{theme_index:02d}-s{size_index:02d}"
            specs.append(
                ShardSpec(
                    shard_id=shard_id,
                    kind="sentinel",
                    ordinal=ordinal,
                    theme=theme,
                    width=size,
                    height=size,
                    objective_bucket=bucket.key,
                    objective_count=bucket.objective_count,
                    spawn_count=sentinel_spawns[size_index],
                    required_splits={"test": 1},
                    global_seed=config.global_seed,
                    max_candidates=config.max_candidates_per_shard,
                    replay_every_sample=config.replay_every_sample,
                    npz_path=f"shards/sentinel/{shard_id}/fields.npz",
                    sidecar_path=f"shards/sentinel/{shard_id}/shard.json",
                    validation_path=f"validation/shards/{shard_id}.json",
                    estimated_raw_bytes=_expected_raw_bytes(
                        1,
                        size,
                        size,
                        bucket.objective_count,
                        sentinel_spawns[size_index],
                    ),
                )
            )
            ordinal += 1
    if len({spec.shard_id for spec in specs}) != len(specs):
        raise RuntimeError("Production shard IDs are not unique.")
    return tuple(specs)


def estimate_corpus(config: CorpusConfig = CorpusConfig(), *, output: Path | None = None) -> dict[str, object]:
    specs = production_specs(config)
    raw_bytes = sum(spec.estimated_raw_bytes for spec in specs)
    planned_bytes = int(raw_bytes * 1.5)
    status = require_disk_floor(
        output or PROJECT_ROOT,
        floor_gb=DISK_FLOOR_GIB,
        planned_bytes=planned_bytes,
    )
    per_kind = Counter(spec.kind for spec in specs)
    return {
        "format_version": ESTIMATE_FORMAT_VERSION,
        "corpus_config": config.to_dict(),
        "production_contract_sha256": PRODUCTION_CONTRACT_SHA256,
        "main_map_count": config.main_map_count,
        "sentinel_count": config.sentinel_count,
        "total_map_count": config.main_map_count + config.sentinel_count,
        "shard_count": len(specs),
        "shards_by_kind": dict(sorted(per_kind.items())),
        "raw_bytes": raw_bytes,
        "raw_gib": raw_bytes / 1024**3,
        "planned_bytes_with_50pct_headroom": planned_bytes,
        "planned_gib_with_50pct_headroom": planned_bytes / 1024**3,
        "disk": status.to_dict(),
        "safe_to_build": status.safe,
    }


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray], *, planned_bytes: int) -> None:
    if path.exists():
        raise FileExistsError(f"Shard artifact already exists and will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=planned_bytes)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        # Keep a non-empty crash artifact for diagnosis; an empty descriptor is noise.
        if temporary.exists() and temporary.stat().st_size == 0:
            temporary.unlink()
        raise


def build_shard(spec: ShardSpec, corpus_root: Path) -> dict[str, object]:
    corpus_root = Path(corpus_root).resolve()
    npz_path = corpus_root / spec.npz_path
    sidecar_path = corpus_root / spec.sidecar_path
    if npz_path.parent != sidecar_path.parent:
        raise ValueError("A shard artifact and sidecar must share one atomic directory.")
    shard_dir = npz_path.parent
    if shard_dir.exists():
        if npz_path.is_file() and sidecar_path.is_file():
            recovered = validate_shard(spec, corpus_root)
            return {
                "passed": True,
                "shard_id": spec.shard_id,
                "sample_count": recovered["sample_count"],
                "split_counts": recovered["split_counts"],
                "npz_sha256": recovered["artifact_sha256"],
                "canonical_arrays_sha256": recovered["canonical_arrays_sha256"],
                "recovered_after_atomic_publish": True,
            }
        raise FileExistsError(
            f"Shard {spec.shard_id} directory exists without a complete atomic pair."
        )
    require_disk_floor(
        corpus_root,
        floor_gb=DISK_FLOOR_GIB,
        planned_bytes=int(spec.estimated_raw_bytes * 1.25) + NPZ_OVERHEAD_BYTES,
    )
    config = MapConfig(
        width=spec.width,
        height=spec.height,
        objective_count=spec.objective_count,
        spawn_count=spec.spawn_count,
    )
    remaining = dict(spec.required_splits)
    observed = Counter()
    generation_errors = Counter()
    samples = []
    accepted_candidates: list[int] = []
    accepted_seed_set: set[int] = set()
    for candidate_index in range(spec.max_candidates):
        if not any(remaining.values()):
            break
        seed = _candidate_seed(spec, candidate_index)
        try:
            data = generate_map(seed, spec.theme, config)
        except (RuntimeError, ValueError) as error:
            generation_errors[f"{type(error).__name__}:{error}"] += 1
            continue
        split = split_for_identity(full_map_identity_sha256(data))
        observed[split] += 1
        if remaining.get(split, 0) <= 0:
            continue
        if seed in accepted_seed_set:
            raise RuntimeError("Candidate seed collision occurred inside one shard.")
        feature_seed = _feature_seed(seed, spec.ordinal)
        replay = generate_map(seed, spec.theme, config) if spec.replay_every_sample else None
        sample = build_production_sample(data, feature_seed=feature_seed, replay_data=replay)
        if sample.split != split:
            raise RuntimeError("Accepted sample split changed after complete teacher construction.")
        samples.append(sample)
        accepted_candidates.append(candidate_index)
        accepted_seed_set.add(seed)
        remaining[split] -= 1
    if any(remaining.values()):
        raise RuntimeError(
            f"Bounded candidate selection exhausted for {spec.shard_id}; "
            f"remaining={remaining}, observed={dict(observed)}."
        )
    if len(samples) != spec.sample_count:
        raise RuntimeError("Shard did not produce its exact contracted sample count.")
    records = [sample_record(sample) for sample in samples]
    full_identities = [str(record["full_map_identity_sha256"]) for record in records]
    sample_identities = [str(record["sample_identity_sha256"]) for record in records]
    if len(full_identities) != len(set(full_identities)):
        raise RuntimeError("Shard contains duplicate full-map identities.")
    if len(sample_identities) != len(set(sample_identities)):
        raise RuntimeError("Shard contains duplicate sample identities.")
    arrays = stacked_sample_arrays(samples)
    canonical_hash = named_arrays_sha256(arrays)
    shard_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = shard_dir.parent / f".{spec.shard_id}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=False, exist_ok=False)
    staging_npz = staging_dir / npz_path.name
    staging_sidecar = staging_dir / sidecar_path.name
    _atomic_npz(
        staging_npz,
        arrays,
        planned_bytes=int(spec.estimated_raw_bytes * 1.25) + NPZ_OVERHEAD_BYTES,
    )
    split_counts = Counter(str(record["split"]) for record in records)
    if dict(split_counts) != spec.required_splits:
        raise RuntimeError("Published shard split counts drifted from its contract.")
    sidecar: dict[str, object] = {
        "format_version": SHARD_FORMAT_VERSION,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "shard_spec": spec.to_dict(),
        "production_contract_sha256": PRODUCTION_CONTRACT_SHA256,
        "corpus_source_sha256": source_sha256("corpus"),
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "model_contract_sha256": MODEL_CONTRACT_SHA256,
        "semantic_teacher_version": SEMANTIC_TEACHER_VERSION,
        "sample_count": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "selection": {
            "candidate_bound": spec.max_candidates,
            "candidates_examined": max(accepted_candidates) + 1,
            "accepted_candidate_indices": accepted_candidates,
            "observed_split_counts": dict(sorted(observed.items())),
            "generation_errors": dict(sorted(generation_errors.items())),
        },
        "samples": records,
        "artifact": {
            "file": Path(spec.npz_path).name,
            "sha256": file_sha256(staging_npz),
            "canonical_arrays_sha256": canonical_hash,
            "compressed_bytes": staging_npz.stat().st_size,
            "uncompressed_array_bytes": sum(array.nbytes for array in arrays.values()),
            "members": {
                name: {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "nbytes": array.nbytes,
                }
                for name, array in sorted(arrays.items())
            },
        },
        "replay": {
            "every_sample": spec.replay_every_sample,
            "passed": spec.replay_every_sample,
            "sample_replay_sha256": [str(record["replay_sha256"]) for record in records],
        },
    }
    _atomic_json(staging_sidecar, sidecar)
    os.replace(staging_dir, shard_dir)
    return {
        "passed": True,
        "shard_id": spec.shard_id,
        "sample_count": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "npz_sha256": sidecar["artifact"]["sha256"],  # type: ignore[index]
        "canonical_arrays_sha256": canonical_hash,
    }


def _read_sidecar(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size > MAX_SHARD_SIDECAR_BYTES:
        raise ValueError("Shard sidecar is missing or exceeds its strict size bound.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Shard sidecar is malformed JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Shard sidecar root must be an object.")
    return payload


def load_shard_arrays(
    corpus_root: Path,
    spec: ShardSpec,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    corpus_root = Path(corpus_root).resolve()
    sidecar_path = corpus_root / spec.sidecar_path
    npz_path = corpus_root / spec.npz_path
    sidecar = _read_sidecar(sidecar_path)
    artifact = sidecar.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("members"), dict):
        raise ValueError("Shard artifact descriptor is malformed.")
    if artifact.get("file") != npz_path.name or artifact.get("sha256") != file_sha256(npz_path):
        raise ValueError("Shard NPZ file identity does not match its sidecar.")
    maximum = int(artifact.get("uncompressed_array_bytes", -1)) + NPZ_OVERHEAD_BYTES
    if maximum < NPZ_OVERHEAD_BYTES or npz_path.stat().st_size > maximum:
        raise ValueError("Shard NPZ exceeds its strict compressed-size bound.")
    expected_members = {f"{name}.npy" for name in artifact["members"]}
    try:
        with zipfile.ZipFile(npz_path, "r") as archive:
            members = archive.infolist()
            if {member.filename for member in members} != expected_members:
                raise ValueError("Shard NPZ members are incomplete or unexpected.")
            if any(member.is_dir() or member.file_size > maximum for member in members):
                raise ValueError("Shard NPZ member exceeds its strict size bound.")
            if sum(member.file_size for member in members) > maximum:
                raise ValueError("Shard NPZ expanded size exceeds its strict bound.")
    except zipfile.BadZipFile as error:
        raise ValueError("Shard artifact is not a valid NPZ container.") from error
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(artifact["members"]):
            raise ValueError("Loaded shard members drifted from the sidecar.")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for name, descriptor in artifact["members"].items():
        array = arrays[name]
        expected = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": array.nbytes,
        }
        if descriptor != expected:
            raise ValueError(f"Shard array descriptor drifted for {name!r}.")
    if named_arrays_sha256(arrays) != artifact.get("canonical_arrays_sha256"):
        raise ValueError("Shard canonical array hash does not match its sidecar.")
    return sidecar, arrays


def load_shard_array(corpus_root: Path, spec: ShardSpec, name: str) -> np.ndarray:
    """Load one bounded NPY member without inflating unrelated feature tensors."""
    corpus_root = Path(corpus_root).resolve()
    sidecar = _read_sidecar(corpus_root / spec.sidecar_path)
    artifact = sidecar.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("members"), dict):
        raise ValueError("Shard artifact descriptor is malformed.")
    descriptor = artifact["members"].get(name)
    if not isinstance(descriptor, dict):
        raise KeyError(f"Unknown shard array {name!r}.")
    npz_path = corpus_root / spec.npz_path
    if artifact.get("sha256") != file_sha256(npz_path):
        raise ValueError("Shard NPZ file identity does not match its sidecar.")
    try:
        with zipfile.ZipFile(npz_path, "r") as archive:
            member = archive.getinfo(f"{name}.npy")
            if member.compress_type != zipfile.ZIP_STORED:
                raise ValueError("Production shard members must be uncompressed for bounded streaming.")
            # The exact NPY header length is checked below; this rejects absurd pre-header bounds.
            if member.file_size > int(descriptor["nbytes"]) + 4096:
                raise ValueError("Shard member exceeds its descriptor plus the bounded NPY header.")
            with archive.open(member, "r") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"Unsupported shard NPY version {version!r}.")
                if (
                    fortran
                    or np.dtype(dtype).str != descriptor["dtype"]
                    or list(shape) != descriptor["shape"]
                    or int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
                    != descriptor["nbytes"]
                ):
                    raise ValueError("Shard member header disagrees with its sidecar descriptor.")
                payload_offset = handle.tell()
    except (KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"Shard member {name!r} cannot be opened safely.") from error
    # ZIP_STORED member data is contiguous. Locate the payload using the verified local header.
    with npz_path.open("rb") as raw:
        raw.seek(member.header_offset)
        local = raw.read(30)
        if len(local) != 30 or local[:4] != b"PK\x03\x04":
            raise ValueError("Shard ZIP local header is malformed.")
        filename_length = int.from_bytes(local[26:28], "little")
        extra_length = int.from_bytes(local[28:30], "little")
    data_offset = member.header_offset + 30 + filename_length + extra_length + payload_offset
    array = np.memmap(
        npz_path,
        mode="r",
        dtype=np.dtype(descriptor["dtype"]),
        offset=data_offset,
        shape=tuple(int(value) for value in descriptor["shape"]),
        order="C",
    )
    return array


def _stored_map(arrays: dict[str, np.ndarray], record: dict[str, object], index: int) -> MapData:
    config_payload = record["generator_config"]
    points = record["points"]
    if not isinstance(config_payload, dict) or not isinstance(points, dict):
        raise ValueError("Stored sample map configuration/points are malformed.")
    config = MapConfig(**config_payload)  # type: ignore[arg-type]
    semantic = {name: np.ascontiguousarray(arrays[f"semantic_{name}"][index]) for name in ARRAY_NAMES}
    return MapData(
        seed=int(record["seed"]),
        theme=str(record["theme"]),
        config=config,
        terrain=semantic["terrain"],
        walkability=semantic["walkability"],
        hazard=semantic["hazard"],
        elevation=semantic["elevation"],
        zone=semantic["zone"],
        nav_cost=semantic["nav_cost"],
        protected_backbone=semantic["protected_backbone"],
        required_clearance=semantic["required_clearance"],
        decoration_forbidden=semantic["decoration_forbidden"],
        start=tuple(int(value) for value in points["start"]),  # type: ignore[arg-type]
        exit=tuple(int(value) for value in points["exit"]),  # type: ignore[arg-type]
        objectives=tuple(tuple(int(value) for value in point) for point in points["objectives"]),  # type: ignore[arg-type]
        spawns=tuple(tuple(int(value) for value in point) for point in points["spawns"]),  # type: ignore[arg-type]
        repair_count=int(record["repair_count"]),
        metadata={
            "theme_parameters": record["theme_parameters"],
            "protected_backbone_segments": int(record["protected_backbone_segments"]),
        },
    )


def validate_shard(spec: ShardSpec, corpus_root: Path) -> dict[str, object]:
    sidecar, arrays = load_shard_arrays(corpus_root, spec)
    expected_sidecar_keys = {
        "format_version",
        "corpus_format_version",
        "shard_spec",
        "production_contract_sha256",
        "corpus_source_sha256",
        "feature_contract_sha256",
        "catalog_sha256",
        "model_contract_sha256",
        "semantic_teacher_version",
        "sample_count",
        "split_counts",
        "selection",
        "samples",
        "artifact",
        "replay",
    }
    if set(sidecar) != expected_sidecar_keys:
        raise ValueError("Shard sidecar members are incomplete or unexpected.")
    expected_contracts = {
        "format_version": SHARD_FORMAT_VERSION,
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "shard_spec": spec.to_dict(),
        "production_contract_sha256": PRODUCTION_CONTRACT_SHA256,
        "corpus_source_sha256": source_sha256("corpus"),
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "model_contract_sha256": MODEL_CONTRACT_SHA256,
        "semantic_teacher_version": SEMANTIC_TEACHER_VERSION,
    }
    for name, expected in expected_contracts.items():
        if sidecar.get(name) != expected:
            raise ValueError(f"Shard contract mismatch for {name}.")
    records = sidecar.get("samples")
    if not isinstance(records, list) or len(records) != spec.sample_count:
        raise ValueError("Shard sample records do not match the contracted count.")
    if arrays["features"].shape[0] != spec.sample_count:
        raise ValueError("Shard array sample axis does not match its records.")
    observed_full: list[str] = []
    observed_sample: list[str] = []
    replay_hashes: list[str] = []
    splits = Counter()
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ValueError("Shard sample record is not an object.")
        stored = _stored_map(arrays, raw_record, index)
        generated = generate_map(stored.seed, stored.theme, stored.config)
        replay = generate_map(stored.seed, stored.theme, stored.config)
        rebuilt = build_production_sample(
            generated,
            feature_seed=int(raw_record["feature_seed"]),
            replay_data=replay,
        )
        expected_record = sample_record(rebuilt)
        if expected_record != raw_record:
            raise ValueError(f"Shard sample record failed exact replay at index {index}.")
        for name in ARRAY_NAMES:
            if not np.array_equal(stored.arrays()[name], generated.arrays()[name]):
                raise ValueError(f"Stored semantic array {name} failed replay at index {index}.")
        if not np.array_equal(arrays["features"][index], rebuilt.features):
            raise ValueError(f"Stored features failed replay at index {index}.")
        if not np.array_equal(arrays["hard_empty"][index], rebuilt.hard_empty):
            raise ValueError(f"Stored hard-empty mask failed replay at index {index}.")
        if not np.array_equal(arrays["global_conditions"][index], rebuilt.global_conditions):
            raise ValueError(f"Stored global conditions failed replay at index {index}.")
        for name in HEAD_NAMES:
            if not np.array_equal(arrays[f"target_{name}"][index], rebuilt.targets[name]):
                raise ValueError(f"Stored target {name} failed replay at index {index}.")
            if not np.array_equal(arrays[f"legal_{name}"][index], rebuilt.legal_masks[name]):
                raise ValueError(f"Stored legality {name} failed replay at index {index}.")
        if int(arrays["seeds"][index]) != stored.seed:
            raise ValueError("Stored seed array disagrees with its sample record.")
        if int(arrays["feature_seeds"][index]) != rebuilt.feature_seed:
            raise ValueError("Stored feature seed disagrees with its sample record.")
        split = str(raw_record["split"])
        if split != split_for_identity(rebuilt.full_map_identity_sha256):
            raise ValueError("Shard sample split is inconsistent with its full-map identity.")
        splits[split] += 1
        observed_full.append(rebuilt.full_map_identity_sha256)
        observed_sample.append(rebuilt.sample_identity_sha256)
        replay_hashes.append(rebuilt.replay_sha256)
    if dict(splits) != spec.required_splits:
        raise ValueError("Validated shard split counts drifted from its contract.")
    if len(observed_full) != len(set(observed_full)):
        raise ValueError("Validated shard contains duplicate full-map identities.")
    if len(observed_sample) != len(set(observed_sample)):
        raise ValueError("Validated shard contains duplicate sample identities.")
    report = {
        "passed": True,
        "shard_id": spec.shard_id,
        "sample_count": spec.sample_count,
        "split_counts": dict(sorted(splits.items())),
        "artifact_sha256": sidecar["artifact"]["sha256"],  # type: ignore[index]
        "canonical_arrays_sha256": sidecar["artifact"]["canonical_arrays_sha256"],  # type: ignore[index]
        "full_map_identity_sha256": observed_full,
        "sample_identity_sha256": observed_sample,
        "sample_replay_sha256": replay_hashes,
        "exact_semantic_feature_target_legality_replay": True,
    }
    validation_path = Path(corpus_root) / spec.validation_path
    if validation_path.exists():
        existing = json.loads(validation_path.read_text(encoding="utf-8"))
        if existing != report:
            raise FileExistsError("Existing shard validation report disagrees with replay.")
    else:
        _atomic_json(validation_path, report)
    return report


def _native_exit_label(returncode: int) -> str | None:
    unsigned = returncode & 0xFFFF_FFFF
    return {
        0xC0000005: "windows_access_violation",
        0xC0000409: "windows_stack_buffer_overrun",
        0xC000001D: "windows_illegal_instruction",
    }.get(unsigned)


def _run_jobs(
    specs: tuple[ShardSpec, ...],
    *,
    mode: str,
    corpus_root: Path,
    python: Path,
    max_workers: int,
    max_attempts: int,
) -> list[dict[str, object]]:
    if mode not in {"build", "validate"}:
        raise ValueError("Worker mode must be build or validate.")
    if not 1 <= max_workers <= MAX_WORKERS:
        raise ValueError(f"Worker count must remain in [1,{MAX_WORKERS}].")
    if not 1 <= max_attempts <= MAX_PROCESS_ATTEMPTS:
        raise ValueError(f"Process attempts must remain in [1,{MAX_PROCESS_ATTEMPTS}].")
    pending = deque((spec, 1) for spec in specs)
    active: dict[subprocess.Popen[bytes], tuple[ShardSpec, int, object, object, float, Path, Path]] = {}
    telemetry: list[dict[str, object]] = []
    telemetry_dir = corpus_root / "telemetry" / mode
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = corpus_root / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        spec_path = specs_dir / f"{spec.shard_id}.json"
        if not spec_path.exists():
            _atomic_json(spec_path, spec.to_dict())
    while pending or active:
        while pending and len(active) < max_workers:
            spec, attempt = pending.popleft()
            require_disk_floor(
                corpus_root,
                floor_gb=DISK_FLOOR_GIB,
                planned_bytes=(
                    int(spec.estimated_raw_bytes * 1.25) + NPZ_OVERHEAD_BYTES
                    if mode == "build"
                    else 64 * 1024 * 1024
                ),
            )
            stdout_path = telemetry_dir / f"{spec.shard_id}-attempt{attempt:02d}.stdout.log"
            stderr_path = telemetry_dir / f"{spec.shard_id}-attempt{attempt:02d}.stderr.log"
            stdout_handle = stdout_path.open("xb")
            stderr_handle = stderr_path.open("xb")
            command = [
                str(python),
                "-m",
                "forge.map_decorator_production.worker",
                mode,
                "--spec",
                str(specs_dir / f"{spec.shard_id}.json"),
                "--root",
                str(corpus_root),
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            active[process] = (
                spec,
                attempt,
                stdout_handle,
                stderr_handle,
                time.perf_counter(),
                stdout_path,
                stderr_path,
            )
        completed = [process for process in active if process.poll() is not None]
        if not completed:
            time.sleep(0.10)
            continue
        for process in completed:
            spec, attempt, stdout_handle, stderr_handle, started, stdout_path, stderr_path = active.pop(
                process
            )
            stdout_handle.close()  # type: ignore[union-attr]
            stderr_handle.close()  # type: ignore[union-attr]
            returncode = int(process.returncode or 0)
            record: dict[str, object] = {
                "mode": mode,
                "shard_id": spec.shard_id,
                "attempt": attempt,
                "pid": process.pid,
                "returncode": returncode,
                "returncode_unsigned_hex": f"0x{returncode & 0xFFFF_FFFF:08x}",
                "native_failure": _native_exit_label(returncode),
                "elapsed_seconds": time.perf_counter() - started,
                "stdout": stdout_path.relative_to(corpus_root).as_posix(),
                "stdout_sha256": file_sha256(stdout_path),
                "stderr": stderr_path.relative_to(corpus_root).as_posix(),
                "stderr_sha256": file_sha256(stderr_path),
                "passed": returncode == 0,
            }
            telemetry.append(record)
            if returncode != 0:
                if attempt >= max_attempts:
                    _atomic_json(corpus_root / f"{mode}_telemetry.json", telemetry)
                    raise RuntimeError(
                        f"Shard {spec.shard_id} failed {mode} after {attempt} attempts; "
                        f"last exit={record['returncode_unsigned_hex']}."
                    )
                pending.append((spec, attempt + 1))
    _atomic_json(corpus_root / f"{mode}_telemetry.json", telemetry)
    return telemetry


def _aggregate_manifest(
    root: Path,
    specs: tuple[ShardSpec, ...],
    config: CorpusConfig,
    estimate: dict[str, object],
    build_telemetry: list[dict[str, object]],
    validation_telemetry: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    shard_entries: list[dict[str, object]] = []
    full_ids: list[str] = []
    sample_ids: list[str] = []
    main_ids: set[str] = set()
    sentinel_ids: set[str] = set()
    split_counts = Counter()
    balance = Counter()
    for spec in specs:
        sidecar = _read_sidecar(root / spec.sidecar_path)
        validation = json.loads((root / spec.validation_path).read_text(encoding="utf-8"))
        if not validation.get("passed"):
            raise RuntimeError(f"Shard validation did not pass for {spec.shard_id}.")
        records = sidecar["samples"]
        if not isinstance(records, list):
            raise TypeError("Shard samples are malformed during aggregation.")
        for record in records:
            identity = str(record["full_map_identity_sha256"])
            sample_identity = str(record["sample_identity_sha256"])
            full_ids.append(identity)
            sample_ids.append(sample_identity)
            split_counts[str(record["split"])] += 1
            if spec.kind == "main":
                main_ids.add(identity)
                balance[(spec.theme, spec.width, spec.height, spec.objective_bucket)] += 1
            else:
                sentinel_ids.add(identity)
        shard_entries.append(
            {
                "shard_id": spec.shard_id,
                "kind": spec.kind,
                "spec": spec.to_dict(),
                "sidecar": spec.sidecar_path,
                "sidecar_sha256": file_sha256(root / spec.sidecar_path),
                "artifact": spec.npz_path,
                "artifact_sha256": sidecar["artifact"]["sha256"],
                "canonical_arrays_sha256": sidecar["artifact"]["canonical_arrays_sha256"],
                "validation": spec.validation_path,
                "validation_sha256": file_sha256(root / spec.validation_path),
                "sample_identity_sha256": [record["sample_identity_sha256"] for record in records],
                "full_map_identity_sha256": [record["full_map_identity_sha256"] for record in records],
            }
        )
    duplicate_full = len(full_ids) - len(set(full_ids))
    duplicate_sample = len(sample_ids) - len(set(sample_ids))
    leakage = sorted(main_ids & sentinel_ids)
    expected_balance = config.identities_per_stratum
    bad_balance = {
        "|".join(map(str, key)): value
        for key, value in sorted(balance.items())
        if value != expected_balance
    }
    expected_splits = {
        "train": len(THEMES)
        * len(SIZE_PROFILES)
        * len(OBJECTIVE_BUCKETS)
        * config.train_per_stratum,
        "validation": len(THEMES)
        * len(SIZE_PROFILES)
        * len(OBJECTIVE_BUCKETS)
        * config.validation_per_stratum,
        "test": config.sentinel_count,
    }
    failures: list[str] = []
    if duplicate_full:
        failures.append("duplicate_full_map_identity")
    if duplicate_sample:
        failures.append("duplicate_sample_identity")
    if leakage:
        failures.append("main_sentinel_identity_leakage")
    if bad_balance or len(balance) != len(THEMES) * len(SIZE_PROFILES) * len(OBJECTIVE_BUCKETS):
        failures.append("stratum_balance")
    if dict(split_counts) != expected_splits:
        failures.append("split_counts")
    identity_payload = {
        "format_version": CORPUS_FORMAT_VERSION,
        "production_contract_sha256": PRODUCTION_CONTRACT_SHA256,
        "corpus_source_sha256": source_sha256("corpus"),
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "model_contract_sha256": MODEL_CONTRACT_SHA256,
        "semantic_teacher_version": SEMANTIC_TEACHER_VERSION,
        "config": config.to_dict(),
        "shards": [
            {
                "shard_id": entry["shard_id"],
                "kind": entry["kind"],
                "canonical_arrays_sha256": entry["canonical_arrays_sha256"],
                "sample_identity_sha256": entry["sample_identity_sha256"],
            }
            for entry in shard_entries
        ],
    }
    corpus_hash = json_sha256(identity_payload)
    validation_report: dict[str, object] = {
        "passed": not failures,
        "failures": failures,
        "corpus_sha256": corpus_hash,
        "main_map_count": len(main_ids),
        "sentinel_count": len(sentinel_ids),
        "total_map_count": len(full_ids),
        "shard_count": len(shard_entries),
        "split_counts": dict(sorted(split_counts.items())),
        "expected_split_counts": expected_splits,
        "duplicate_full_map_identity_count": duplicate_full,
        "duplicate_sample_identity_count": duplicate_sample,
        "main_sentinel_identity_overlap": leakage,
        "stratum_count": len(balance),
        "bad_stratum_balance": bad_balance,
        "all_shards_exact_replay": True,
        "max_build_workers": MAX_WORKERS,
        "max_process_attempts": MAX_PROCESS_ATTEMPTS,
    }
    if failures:
        raise RuntimeError(f"Global corpus validation failed: {failures}")
    manifest: dict[str, object] = {
        "format_version": CORPUS_FORMAT_VERSION,
        "corpus_sha256": corpus_hash,
        "identity": identity_payload,
        "production_contract_sha256": PRODUCTION_CONTRACT_SHA256,
        "corpus_source_manifest": source_manifest("corpus"),
        "corpus_source_sha256": source_sha256("corpus"),
        "config": config.to_dict(),
        "estimate": estimate,
        "counts": {
            "main_maps": len(main_ids),
            "sentinels": len(sentinel_ids),
            "total_maps": len(full_ids),
            "shards": len(shard_entries),
            "splits": dict(sorted(split_counts.items())),
        },
        "shards": shard_entries,
        "telemetry": {
            "build_attempt_count": len(build_telemetry),
            "validation_attempt_count": len(validation_telemetry),
            "build_native_failures": sum(bool(item["native_failure"]) for item in build_telemetry),
            "validation_native_failures": sum(
                bool(item["native_failure"]) for item in validation_telemetry
            ),
            "build_file": "build_telemetry.json",
            "validation_file": "validate_telemetry.json",
        },
        "validation": VALIDATION_FILE,
        "validation_sha256": None,
    }
    return manifest, validation_report


def build_corpus(
    output: Path,
    *,
    config: CorpusConfig = CorpusConfig(),
    python: Path = Path(sys.executable),
    max_workers: int = MAX_WORKERS,
) -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Production corpus output already exists: {output}")
    estimate = estimate_corpus(config, output=output)
    if not estimate["safe_to_build"]:
        raise RuntimeError("Production corpus estimate violates the disk safety floor.")
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    _atomic_json(staging / "estimate.json", estimate)
    specs = production_specs(config)
    try:
        build_telemetry = _run_jobs(
            specs,
            mode="build",
            corpus_root=staging,
            python=Path(python),
            max_workers=max_workers,
            max_attempts=MAX_PROCESS_ATTEMPTS,
        )
        validation_telemetry = _run_jobs(
            specs,
            mode="validate",
            corpus_root=staging,
            python=Path(python),
            max_workers=max_workers,
            max_attempts=MAX_PROCESS_ATTEMPTS,
        )
        manifest, validation = _aggregate_manifest(
            staging,
            specs,
            config,
            estimate,
            build_telemetry,
            validation_telemetry,
        )
        _atomic_json(staging / VALIDATION_FILE, validation)
        manifest["validation_sha256"] = file_sha256(staging / VALIDATION_FILE)
        _atomic_json(staging / MANIFEST_FILE, manifest)
        os.replace(staging, output)
    except BaseException:
        # The unique staging tree, logs, and crash remnants are diagnostic evidence.
        raise
    return validate_corpus(output, verify_shards=False)


def validate_corpus(path: Path, *, verify_shards: bool = False) -> dict[str, object]:
    path = Path(path).resolve()
    manifest_path = path / MANIFEST_FILE
    validation_path = path / VALIDATION_FILE
    if not manifest_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Production corpus manifest/validation is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if manifest.get("format_version") != CORPUS_FORMAT_VERSION:
        failures.append("format_version")
    if manifest.get("production_contract_sha256") != PRODUCTION_CONTRACT_SHA256:
        failures.append("production_contract_sha256")
    if manifest.get("corpus_source_sha256") != source_sha256("corpus"):
        failures.append("corpus_source_sha256")
    if manifest.get("validation_sha256") != file_sha256(validation_path):
        failures.append("validation_sha256")
    if not validation.get("passed") or manifest.get("corpus_sha256") != validation.get(
        "corpus_sha256"
    ):
        failures.append("recorded_validation")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or json_sha256(identity) != manifest.get("corpus_sha256"):
        failures.append("corpus_identity")
    shard_results: list[dict[str, object]] = []
    if verify_shards and not failures:
        for entry in manifest["shards"]:
            spec = ShardSpec.from_dict(entry["spec"])
            shard_results.append(validate_shard(spec, path))
    report = {
        "passed": not failures,
        "failures": failures,
        "corpus_sha256": manifest.get("corpus_sha256"),
        "counts": manifest.get("counts"),
        "validation": validation,
        "fresh_shard_validation_count": len(shard_results),
    }
    if failures:
        raise ValueError(f"Production corpus failed closed: {failures}")
    return report


def _parser() -> object:
    import argparse

    parser = argparse.ArgumentParser(description="Process-isolated production map-decorator corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    estimate = subparsers.add_parser("estimate")
    estimate.add_argument("--output", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, default=Path("outputs/map_decorator_corpus_v1"))
    build.add_argument("--python", type=Path, default=Path(sys.executable))
    build.add_argument("--workers", type=int, default=MAX_WORKERS)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--corpus", type=Path, required=True)
    validate.add_argument("--verify-shards", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)  # type: ignore[union-attr]
    if args.command == "estimate":
        report = estimate_corpus(output=args.output)
    elif args.command == "build":
        report = build_corpus(args.output, python=args.python, max_workers=args.workers)
    else:
        report = validate_corpus(args.corpus, verify_shards=args.verify_shards)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
