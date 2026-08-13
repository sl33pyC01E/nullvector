from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from ..maps.model import THEMES, MapConfig, Point
from .compiler import RawTopology, make_raw_topology
from .hashing import array_sha256, file_sha256, json_sha256, require_sha256


FROZEN_CORPUS_SHA256 = "16ed5f3b1a661e2bfc2abe9e16c39e9b8caaecba81f50fed6658cc4f73cffab8"
FROZEN_CORPUS_MANIFEST_FILE_SHA256 = "fd5ee2e88725262f23ef1943e34aad7f19c1b0886100f43298f93226de2ccbaf"
CORPUS_FORMAT_VERSION = "1.0.0"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_VALIDATION_BYTES = 2 * 1024 * 1024
MAX_SIDECAR_BYTES = 8 * 1024 * 1024
MAX_SHARD_BYTES = 128 * 1024 * 1024
MAX_MEMBER_HEADER_BYTES = 4096
MAX_SHARDS = 512
MAX_SAMPLES_PER_SHARD = 64

ROOT_KEYS = {
    "config",
    "corpus_sha256",
    "corpus_source_manifest",
    "corpus_source_sha256",
    "counts",
    "estimate",
    "format_version",
    "identity",
    "production_contract_sha256",
    "shards",
    "telemetry",
    "validation",
    "validation_sha256",
}
SHARD_ENTRY_KEYS = {
    "shard_id",
    "kind",
    "spec",
    "sidecar",
    "sidecar_sha256",
    "artifact",
    "artifact_sha256",
    "canonical_arrays_sha256",
    "validation",
    "validation_sha256",
    "sample_identity_sha256",
    "full_map_identity_sha256",
}
SIDECAR_KEYS = {
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
SPEC_KEYS = {
    "shard_id",
    "kind",
    "ordinal",
    "theme",
    "width",
    "height",
    "objective_bucket",
    "objective_count",
    "spawn_count",
    "required_splits",
    "global_seed",
    "max_candidates",
    "replay_every_sample",
    "npz_path",
    "sidecar_path",
    "validation_path",
    "estimated_raw_bytes",
}
SAMPLE_KEYS = {
    "map_id",
    "seed",
    "feature_seed",
    "theme",
    "width",
    "height",
    "objective_count",
    "spawn_count",
    "generator_config",
    "split",
    "full_map_identity_sha256",
    "sample_identity_sha256",
    "source_semantic_sha256",
    "topology_masks_sha256",
    "feature_tensor_sha256",
    "target_fields_sha256",
    "legal_masks_sha256",
    "replay_sha256",
    "points",
    "repair_count",
    "theme_parameters",
    "protected_backbone_segments",
}
FULL_MEMBER_NAMES = {
    "exit",
    "feature_seeds",
    "features",
    "global_conditions",
    "hard_empty",
    "legal_decal",
    "legal_emission",
    "legal_prop",
    "legal_variant",
    "objectives",
    "repair_count",
    "seeds",
    "semantic_decoration_forbidden",
    "semantic_elevation",
    "semantic_hazard",
    "semantic_nav_cost",
    "semantic_protected_backbone",
    "semantic_required_clearance",
    "semantic_terrain",
    "semantic_walkability",
    "semantic_zone",
    "spawns",
    "start",
    "target_decal",
    "target_emission",
    "target_prop",
    "target_variant",
    "theme_index",
}
TOPOLOGY_MEMBER_NAMES = (
    "semantic_terrain",
    "semantic_hazard",
    "semantic_elevation",
    "seeds",
    "theme_index",
    "start",
    "exit",
    "objectives",
    "spawns",
)
SAFE_DTYPES = {
    np.dtype(np.bool_).str,
    np.dtype(np.uint8).str,
    np.dtype(np.int8).str,
    np.dtype(np.int16).str,
    np.dtype(np.uint64).str,
    np.dtype(np.int64).str,
    np.dtype(np.float32).str,
}


@dataclass(frozen=True, slots=True)
class TopologyCorpusSample:
    raw: RawTopology
    seed: int
    theme: str
    config: MapConfig
    start: Point
    exit: Point
    objectives: tuple[Point, ...]
    spawns: tuple[Point, ...]
    split: str
    map_id: str
    full_map_identity_sha256: str
    sample_identity_sha256: str
    topology_sample_sha256: str
    corpus_sha256: str
    corpus_manifest_file_sha256: str
    shard_id: str
    shard_artifact_sha256: str
    member_array_sha256: dict[str, str]


def _read_json(path: Path, maximum: int, label: str) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size > maximum:
        raise ValueError(f"{label} is missing, empty, or exceeds its strict byte bound.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object.")
    return payload


def _safe_relative(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty canonical POSIX relative path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} contains an absolute or traversing path.")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the corpus root.") from error
    return target


def _split_for_identity(identity: str) -> str:
    require_sha256(identity, "full_map_identity_sha256")
    bucket = int(identity[:16], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _descriptor_shape(descriptor: object, name: str) -> tuple[np.dtype, tuple[int, ...], int]:
    if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "nbytes"}:
        raise ValueError(f"Shard member descriptor is malformed for {name!r}.")
    dtype_text = descriptor.get("dtype")
    shape_value = descriptor.get("shape")
    nbytes_value = descriptor.get("nbytes")
    if not isinstance(dtype_text, str) or dtype_text not in SAFE_DTYPES:
        raise ValueError(f"Shard member {name!r} uses an unsafe or unsupported dtype.")
    if (
        not isinstance(shape_value, list)
        or not 1 <= len(shape_value) <= 4
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape_value)
    ):
        raise ValueError(f"Shard member {name!r} has an invalid shape.")
    shape = tuple(shape_value)
    dtype = np.dtype(dtype_text)
    calculated = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if not isinstance(nbytes_value, int) or isinstance(nbytes_value, bool) or nbytes_value != calculated:
        raise ValueError(f"Shard member {name!r} byte count disagrees with dtype/shape.")
    return dtype, shape, calculated


def _validate_member_table(
    members: object,
    *,
    sample_count: int,
    width: int,
    height: int,
    objective_count: int,
    spawn_count: int,
) -> dict[str, dict[str, object]]:
    if not isinstance(members, dict) or set(members) != FULL_MEMBER_NAMES:
        raise ValueError("Shard member table is incomplete or contains unexpected members.")
    for name, descriptor in members.items():
        _, shape, _ = _descriptor_shape(descriptor, name)
        if shape[0] != sample_count:
            raise ValueError(f"Shard member {name!r} sample axis drifted.")
    exact = {
        "semantic_terrain": (np.dtype(np.uint8).str, (sample_count, height, width)),
        "semantic_hazard": (np.dtype(np.uint8).str, (sample_count, height, width)),
        "semantic_elevation": (np.dtype(np.int8).str, (sample_count, height, width)),
        "seeds": (np.dtype(np.uint64).str, (sample_count,)),
        "theme_index": (np.dtype(np.int64).str, (sample_count,)),
        "start": (np.dtype(np.int16).str, (sample_count, 2)),
        "exit": (np.dtype(np.int16).str, (sample_count, 2)),
        "objectives": (np.dtype(np.int16).str, (sample_count, objective_count, 2)),
        "spawns": (np.dtype(np.int16).str, (sample_count, spawn_count, 2)),
    }
    for name, (dtype, shape) in exact.items():
        descriptor = members[name]
        if descriptor["dtype"] != dtype or tuple(descriptor["shape"]) != shape:
            raise ValueError(f"Topology member descriptor drifted for {name!r}.")
    return members  # type: ignore[return-value]


def _read_member_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    descriptor: dict[str, object],
) -> tuple[np.dtype, tuple[int, ...], int]:
    if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
        raise ValueError("Production shard members must be uncompressed ZIP_STORED entries.")
    if info.flag_bits & 0x1:
        raise ValueError("Encrypted shard members are forbidden.")
    expected_dtype, expected_shape, expected_nbytes = _descriptor_shape(descriptor, info.filename)
    if info.file_size > expected_nbytes + MAX_MEMBER_HEADER_BYTES:
        raise ValueError("Shard member exceeds its array bytes plus bounded NPY header.")
    with archive.open(info, "r") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY member version {version!r}.")
        header_bytes = handle.tell()
    if (
        fortran
        or np.dtype(dtype) != expected_dtype
        or tuple(shape) != expected_shape
        or header_bytes <= 0
        or header_bytes > MAX_MEMBER_HEADER_BYTES
        or info.file_size != header_bytes + expected_nbytes
    ):
        raise ValueError("NPY member header disagrees with its bounded sidecar descriptor.")
    return expected_dtype, expected_shape, header_bytes


def _load_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    descriptor: dict[str, object],
) -> np.ndarray:
    dtype, shape, header_bytes = _read_member_header(archive, info, descriptor)
    with archive.open(info, "r") as handle:
        prefix = handle.read(header_bytes)
        if len(prefix) != header_bytes:
            raise ValueError("NPY member header could not be reread exactly.")
        expected_nbytes = int(descriptor["nbytes"])
        payload = handle.read(expected_nbytes)
        if len(payload) != expected_nbytes or handle.read(1):
            raise ValueError("NPY member payload length disagrees with its descriptor.")
    # zipfile verifies CRC when the member reaches EOF. Copy detaches from immutable bytes.
    array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy(order="C")
    return np.ascontiguousarray(array)


class TopologyCorpus:
    def __init__(
        self,
        root: Path,
        *,
        expected_corpus_sha256: str = FROZEN_CORPUS_SHA256,
        expected_manifest_file_sha256: str = FROZEN_CORPUS_MANIFEST_FILE_SHA256,
    ) -> None:
        self.root = Path(root).resolve()
        require_sha256(expected_corpus_sha256, "expected_corpus_sha256")
        require_sha256(expected_manifest_file_sha256, "expected_manifest_file_sha256")
        manifest_path = self.root / "corpus_manifest.json"
        if file_sha256(manifest_path) != expected_manifest_file_sha256:
            raise ValueError("Corpus manifest file does not match the pinned frozen artifact SHA-256.")
        manifest = _read_json(manifest_path, MAX_MANIFEST_BYTES, "corpus manifest")
        if set(manifest) != ROOT_KEYS:
            raise ValueError("Corpus manifest members are incomplete or unexpected.")
        if manifest.get("format_version") != CORPUS_FORMAT_VERSION:
            raise ValueError("Corpus format version is unsupported.")
        if manifest.get("corpus_sha256") != expected_corpus_sha256:
            raise ValueError("Corpus identity does not match the pinned frozen corpus SHA-256.")
        identity = manifest.get("identity")
        if not isinstance(identity, dict) or json_sha256(identity) != expected_corpus_sha256:
            raise ValueError("Corpus identity payload does not reproduce its pinned SHA-256.")
        validation_path = _safe_relative(self.root, manifest.get("validation"), "corpus validation path")
        if file_sha256(validation_path) != require_sha256(manifest.get("validation_sha256"), "validation_sha256"):
            raise ValueError("Corpus validation artifact SHA-256 drifted.")
        validation = _read_json(validation_path, MAX_VALIDATION_BYTES, "corpus validation")
        if not validation.get("passed") or validation.get("corpus_sha256") != expected_corpus_sha256:
            raise ValueError("Corpus recorded validation is not a passing report for this identity.")
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not 1 <= len(shards) <= MAX_SHARDS:
            raise ValueError("Corpus shard list is missing or exceeds its strict bound.")
        entries: dict[str, dict[str, object]] = {}
        seen_paths: set[str] = set()
        full_ids: list[str] = []
        sample_ids: list[str] = []
        for entry in shards:
            if not isinstance(entry, dict) or set(entry) != SHARD_ENTRY_KEYS:
                raise ValueError("Corpus shard entry is malformed.")
            shard_id = entry.get("shard_id")
            if not isinstance(shard_id, str) or not shard_id or shard_id in entries:
                raise ValueError("Corpus shard IDs must be unique non-empty strings.")
            spec = entry.get("spec")
            if not isinstance(spec, dict) or set(spec) != SPEC_KEYS or spec.get("shard_id") != shard_id:
                raise ValueError("Corpus shard specification is malformed or mismatched.")
            if spec.get("theme") not in THEMES:
                raise ValueError("Corpus shard theme is outside the topology contract.")
            for label in ("sidecar", "artifact", "validation"):
                value = entry.get(label)
                _safe_relative(self.root, value, f"shard {label}")
                assert isinstance(value, str)
                if value in seen_paths:
                    raise ValueError("Corpus reuses an artifact path across shards.")
                seen_paths.add(value)
            for label in (
                "sidecar_sha256",
                "artifact_sha256",
                "canonical_arrays_sha256",
                "validation_sha256",
            ):
                require_sha256(entry.get(label), f"shard.{label}")
            entry_full = entry.get("full_map_identity_sha256")
            entry_sample = entry.get("sample_identity_sha256")
            expected_count = sum(int(value) for value in spec["required_splits"].values())
            if (
                not isinstance(entry_full, list)
                or not isinstance(entry_sample, list)
                or len(entry_full) != expected_count
                or len(entry_sample) != expected_count
                or not 1 <= expected_count <= MAX_SAMPLES_PER_SHARD
            ):
                raise ValueError("Corpus shard identity census disagrees with its sample count.")
            full_ids.extend(require_sha256(value, "full_map_identity_sha256") for value in entry_full)
            sample_ids.extend(require_sha256(value, "sample_identity_sha256") for value in entry_sample)
            entries[shard_id] = entry
        if len(full_ids) != len(set(full_ids)) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Corpus root manifest contains duplicate map/sample identities.")
        counts = manifest.get("counts")
        if not isinstance(counts, dict) or counts.get("shards") != len(entries) or counts.get("total_maps") != len(full_ids):
            raise ValueError("Corpus root counts disagree with its shard identity census.")
        self.manifest = manifest
        self.corpus_sha256 = expected_corpus_sha256
        self.manifest_file_sha256 = expected_manifest_file_sha256
        self._entries = entries

    @property
    def shard_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def find_shard(
        self,
        *,
        theme: str,
        width: int | None = None,
        height: int | None = None,
        objective_count: int | None = None,
        kind: str = "main",
    ) -> str:
        matches = []
        for shard_id, entry in self._entries.items():
            spec = entry["spec"]
            assert isinstance(spec, dict)
            if entry["kind"] != kind or spec["theme"] != theme:
                continue
            if width is not None and spec["width"] != width:
                continue
            if height is not None and spec["height"] != height:
                continue
            if objective_count is not None and spec["objective_count"] != objective_count:
                continue
            matches.append(shard_id)
        if not matches:
            raise KeyError("No corpus shard matches the requested topology stratum.")
        return sorted(matches)[0]

    def read_sample(
        self,
        shard_id: str,
        sample_index: int,
        *,
        expected_split: str | None = None,
    ) -> TopologyCorpusSample:
        if shard_id not in self._entries:
            raise KeyError(f"Unknown corpus shard {shard_id!r}.")
        entry = self._entries[shard_id]
        spec = entry["spec"]
        assert isinstance(spec, dict)
        sidecar_path = _safe_relative(self.root, entry["sidecar"], "shard sidecar")
        if file_sha256(sidecar_path) != entry["sidecar_sha256"]:
            raise ValueError("Shard sidecar SHA-256 disagrees with the frozen root manifest.")
        sidecar = _read_json(sidecar_path, MAX_SIDECAR_BYTES, "shard sidecar")
        if set(sidecar) != SIDECAR_KEYS or sidecar.get("shard_spec") != spec:
            raise ValueError("Shard sidecar contract or exact specification drifted.")
        identity = self.manifest["identity"]
        assert isinstance(identity, dict)
        expected_sidecar_contract = {
            "format_version": "1.0.0",
            "corpus_format_version": CORPUS_FORMAT_VERSION,
            "production_contract_sha256": identity["production_contract_sha256"],
            "corpus_source_sha256": identity["corpus_source_sha256"],
            "feature_contract_sha256": identity["feature_contract_sha256"],
            "catalog_sha256": identity["catalog_sha256"],
            "model_contract_sha256": identity["model_contract_sha256"],
            "semantic_teacher_version": identity["semantic_teacher_version"],
        }
        for name, expected in expected_sidecar_contract.items():
            if sidecar.get(name) != expected:
                raise ValueError(f"Shard sidecar contract drifted for {name}.")
        records = sidecar.get("samples")
        sample_count = sidecar.get("sample_count")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or not isinstance(records, list)
            or len(records) != sample_count
            or not 0 <= sample_index < sample_count
        ):
            raise ValueError("Shard sample records/count/index are inconsistent.")
        root_full = entry["full_map_identity_sha256"]
        root_sample = entry["sample_identity_sha256"]
        if sample_count != sum(int(value) for value in spec["required_splits"].values()):
            raise ValueError("Shard sidecar sample count drifted from its exact split contract.")
        split_counts: dict[str, int] = {}
        for raw_record in records:
            if not isinstance(raw_record, dict) or set(raw_record) != SAMPLE_KEYS:
                raise ValueError("Shard sample record census contains a malformed record.")
            record_split = raw_record.get("split")
            if not isinstance(record_split, str):
                raise ValueError("Shard sample record split is malformed.")
            if record_split != _split_for_identity(str(raw_record.get("full_map_identity_sha256"))):
                raise ValueError("Shard sample split disagrees with its full-map identity.")
            split_counts[record_split] = split_counts.get(record_split, 0) + 1
            for hash_name in (
                "full_map_identity_sha256",
                "sample_identity_sha256",
                "source_semantic_sha256",
                "topology_masks_sha256",
                "feature_tensor_sha256",
                "target_fields_sha256",
                "legal_masks_sha256",
                "replay_sha256",
            ):
                require_sha256(raw_record.get(hash_name), f"sample.{hash_name}")
        if split_counts != spec["required_splits"] or sidecar.get("split_counts") != spec["required_splits"]:
            raise ValueError("Shard sidecar split census drifted from its exact specification.")
        selection = sidecar.get("selection")
        if not isinstance(selection, dict) or set(selection) != {
            "candidate_bound",
            "candidates_examined",
            "accepted_candidate_indices",
            "observed_split_counts",
            "generation_errors",
        }:
            raise ValueError("Shard bounded-selection record is malformed.")
        accepted = selection.get("accepted_candidate_indices")
        examined = selection.get("candidates_examined")
        if (
            selection.get("candidate_bound") != spec["max_candidates"]
            or not isinstance(examined, int)
            or isinstance(examined, bool)
            or not 1 <= examined <= int(spec["max_candidates"])
            or not isinstance(accepted, list)
            or len(accepted) != sample_count
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < examined for value in accepted)
            or accepted != sorted(set(accepted))
            or not isinstance(selection.get("observed_split_counts"), dict)
            or not isinstance(selection.get("generation_errors"), dict)
        ):
            raise ValueError("Shard bounded-selection values violate their strict contract.")
        if [record.get("full_map_identity_sha256") for record in records if isinstance(record, dict)] != root_full:
            raise ValueError("Shard full-map identity order disagrees with root manifest.")
        if [record.get("sample_identity_sha256") for record in records if isinstance(record, dict)] != root_sample:
            raise ValueError("Shard sample identity order disagrees with root manifest.")
        record = records[sample_index]
        assert isinstance(record, dict)
        replay = sidecar.get("replay")
        if (
            not isinstance(replay, dict)
            or set(replay) != {"every_sample", "passed", "sample_replay_sha256"}
            or replay.get("every_sample") is not True
            or replay.get("passed") is not True
            or replay.get("sample_replay_sha256") != [item["replay_sha256"] for item in records]
        ):
            raise ValueError("Shard sidecar exact-replay census drifted.")
        split = record.get("split")
        full_identity = require_sha256(record.get("full_map_identity_sha256"), "full_map_identity_sha256")
        if split != _split_for_identity(full_identity):
            raise ValueError("Selected sample split disagrees with its full-map identity.")
        if expected_split is not None and split != expected_split:
            raise ValueError(f"Selected sample belongs to {split}, not {expected_split}.")
        artifact = sidecar.get("artifact")
        if not isinstance(artifact, dict) or set(artifact) != {
            "file", "sha256", "canonical_arrays_sha256", "compressed_bytes", "uncompressed_array_bytes", "members"
        }:
            raise ValueError("Shard artifact descriptor is malformed.")
        artifact_path = _safe_relative(self.root, entry["artifact"], "shard artifact")
        if artifact_path.parent != sidecar_path.parent or artifact.get("file") != artifact_path.name:
            raise ValueError("Shard sidecar/artifact are not the contracted atomic pair.")
        artifact_size = artifact_path.stat().st_size if artifact_path.is_file() else -1
        if not 0 < artifact_size <= MAX_SHARD_BYTES or artifact.get("compressed_bytes") != artifact_size:
            raise ValueError("Shard artifact is missing or exceeds its strict compressed-size bound.")
        actual_artifact_hash = file_sha256(artifact_path)
        if actual_artifact_hash != entry["artifact_sha256"] or actual_artifact_hash != artifact.get("sha256"):
            raise ValueError("Shard artifact SHA-256 disagrees with root/sidecar identity.")
        if artifact.get("canonical_arrays_sha256") != entry["canonical_arrays_sha256"]:
            raise ValueError("Shard canonical-array identity disagrees with root manifest.")
        width, height = int(spec["width"]), int(spec["height"])
        objective_count, spawn_count = int(spec["objective_count"]), int(spec["spawn_count"])
        members = _validate_member_table(
            artifact.get("members"),
            sample_count=sample_count,
            width=width,
            height=height,
            objective_count=objective_count,
            spawn_count=spawn_count,
        )
        descriptor_array_bytes = sum(int(member["nbytes"]) for member in members.values())
        if (
            descriptor_array_bytes != artifact.get("uncompressed_array_bytes")
            or not 0 < descriptor_array_bytes <= MAX_SHARD_BYTES
        ):
            raise ValueError("Shard member table byte census drifted or exceeds its strict bound.")
        arrays: dict[str, np.ndarray] = {}
        try:
            with zipfile.ZipFile(artifact_path, "r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                expected_names = {f"{name}.npy" for name in FULL_MEMBER_NAMES}
                if len(names) != len(set(names)) or set(names) != expected_names:
                    raise ValueError("Shard ZIP member census has duplicates, omissions, or extras.")
                if any(
                    PurePosixPath(name).is_absolute()
                    or len(PurePosixPath(name).parts) != 1
                    or "\\" in name
                    or "\0" in name
                    for name in names
                ):
                    raise ValueError("Shard ZIP contains an unsafe member path.")
                info_by_name = {info.filename[:-4]: info for info in infos}
                expanded = 0
                for name in sorted(FULL_MEMBER_NAMES):
                    info = info_by_name[name]
                    _read_member_header(archive, info, members[name])
                    expanded += info.file_size
                if expanded > MAX_SHARD_BYTES:
                    raise ValueError("Shard expanded NPY members exceed the strict byte bound.")
                for name in TOPOLOGY_MEMBER_NAMES:
                    arrays[name] = _load_member(archive, info_by_name[name], members[name])
        except zipfile.BadZipFile as error:
            raise ValueError("Shard artifact is not a valid bounded NPZ container.") from error
        terrain = np.ascontiguousarray(arrays["semantic_terrain"][sample_index], dtype=np.uint8)
        hazard = np.ascontiguousarray(arrays["semantic_hazard"][sample_index], dtype=np.uint8)
        elevation = np.ascontiguousarray(arrays["semantic_elevation"][sample_index], dtype=np.int8)
        raw = make_raw_topology(terrain, hazard, elevation, shape=(height, width))
        if int(arrays["seeds"][sample_index]) != int(record["seed"]):
            raise ValueError("Direct seed member disagrees with the selected sample record.")
        theme = str(record["theme"])
        if theme != spec["theme"] or int(arrays["theme_index"][sample_index]) != THEMES.index(theme):
            raise ValueError("Direct theme member disagrees with shard/sample identity.")
        points = record.get("points")
        if not isinstance(points, dict) or set(points) != {"start", "exit", "objectives", "spawns"}:
            raise ValueError("Selected sample points are malformed.")
        direct_points = {
            "start": arrays["start"][sample_index].astype(int).tolist(),
            "exit": arrays["exit"][sample_index].astype(int).tolist(),
            "objectives": arrays["objectives"][sample_index].astype(int).tolist(),
            "spawns": arrays["spawns"][sample_index].astype(int).tolist(),
        }
        if direct_points != points:
            raise ValueError("Direct point members disagree with selected sample record.")
        generator_config = record.get("generator_config")
        if not isinstance(generator_config, dict):
            raise ValueError("Selected sample generator config is malformed.")
        config = MapConfig(**generator_config)
        if config.width != width or config.height != height:
            raise ValueError("Selected sample config disagrees with shard shape.")
        if (
            record.get("theme") != spec["theme"]
            or record.get("width") != width
            or record.get("height") != height
            or record.get("objective_count") != objective_count
            or record.get("spawn_count") != spawn_count
            or record.get("map_id") != f"{theme}-{int(record['seed']):016x}-{width}x{height}"
        ):
            raise ValueError("Selected sample record identity disagrees with its shard specification.")
        start = tuple(direct_points["start"])
        exit_point = tuple(direct_points["exit"])
        objectives = tuple(tuple(point) for point in direct_points["objectives"])
        spawns = tuple(tuple(point) for point in direct_points["spawns"])
        topology_identity = {
            "format": "nullvector-frozen-topology-corpus-sample-v1",
            "corpus_sha256": self.corpus_sha256,
            "corpus_manifest_file_sha256": self.manifest_file_sha256,
            "shard_id": shard_id,
            "shard_artifact_sha256": actual_artifact_hash,
            "sample_index": sample_index,
            "full_map_identity_sha256": full_identity,
            "sample_identity_sha256": record["sample_identity_sha256"],
            "split": split,
            "raw_topology_sha256": raw.raw_sha256,
            "points": direct_points,
            "config": config.to_dict(),
        }
        member_hashes = {name: array_sha256(arrays[name]) for name in TOPOLOGY_MEMBER_NAMES}
        return TopologyCorpusSample(
            raw=raw,
            seed=int(record["seed"]),
            theme=theme,
            config=config,
            start=start,  # type: ignore[arg-type]
            exit=exit_point,  # type: ignore[arg-type]
            objectives=objectives,  # type: ignore[arg-type]
            spawns=spawns,  # type: ignore[arg-type]
            split=str(split),
            map_id=str(record["map_id"]),
            full_map_identity_sha256=full_identity,
            sample_identity_sha256=require_sha256(record["sample_identity_sha256"], "sample_identity_sha256"),
            topology_sample_sha256=json_sha256(topology_identity),
            corpus_sha256=self.corpus_sha256,
            corpus_manifest_file_sha256=self.manifest_file_sha256,
            shard_id=shard_id,
            shard_artifact_sha256=actual_artifact_hash,
            member_array_sha256=member_hashes,
        )
