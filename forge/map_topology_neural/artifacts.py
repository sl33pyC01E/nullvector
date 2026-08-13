from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import uuid
import zipfile

import jsonschema
import numpy as np

from ..config import PROJECT_ROOT
from ..maps.model import THEMES, MapConfig, MapData, Point
from ..maps.validate import assert_valid
from ..safety import require_disk_floor
from .compiler import (
    COMPILER_NAME,
    COMPILER_VERSION,
    LEDGER_FORMAT,
    CompileResult,
    RawTopology,
    assert_exact_compiler_replay,
    compile_topology,
    make_raw_topology,
)
from .contract import CONTRACT_SHA256
from .hashing import array_sha256, file_sha256, json_sha256, named_arrays_sha256, require_sha256
from .provenance import compiler_source_sha256, source_sha256


RAW_SCHEMA_VERSION = "1.0.0"
COMPILED_SCHEMA_VERSION = "1.0.0"
RAW_MANIFEST = "raw_manifest.json"
RAW_ARRAYS = "raw_topology.npz"
COMPILED_MANIFEST = "compiled_manifest.json"
COMPILED_ARRAYS = "compiled_topology.npz"
LEDGER_FILE = "edit_ledger.json"
REPORT_FILE = "compile_report.json"
MAX_RAW_ARRAY_BYTES = 2 * 1024 * 1024
MAX_COMPILED_ARRAY_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_LEDGER_BYTES = 256 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
ARRAY_NAMES = (
    "terrain",
    "walkability",
    "hazard",
    "elevation",
    "zone",
    "nav_cost",
    "protected_backbone",
    "required_clearance",
    "decoration_forbidden",
)


@dataclass(frozen=True, slots=True)
class RawArtifact:
    path: Path
    raw: RawTopology
    seed: int
    theme: str
    config: MapConfig
    start: Point
    exit: Point
    objectives: tuple[Point, ...]
    spawns: tuple[Point, ...]
    manifest: dict[str, object]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CompiledArtifact:
    path: Path
    result: CompileResult
    manifest: dict[str, object]
    manifest_sha256: str


def _schema(name: str) -> dict[str, object]:
    path = PROJECT_ROOT / "shared" / "schema" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Schema {name} root must be an object.")
    return payload


def _validate_schema(payload: dict[str, object], name: str) -> None:
    jsonschema.Draft202012Validator(_schema(name)).validate(payload)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name in sorted(arrays):
            if not name or not name.replace("_", "").isalnum():
                raise ValueError("NPZ array names must be simple identifiers.")
            contiguous = np.ascontiguousarray(arrays[name])
            member = BytesIO()
            np.lib.format.write_array(member, contiguous, allow_pickle=False, version=(2, 0))
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


def _array_descriptor(arrays: dict[str, np.ndarray], filename: str, payload: bytes) -> dict[str, object]:
    return {
        "file": filename,
        "bytes": len(payload),
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "canonical_arrays_sha256": named_arrays_sha256(arrays),
        "members": {
            name: {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": array.nbytes,
                "array_sha256": array_sha256(array),
            }
            for name, array in sorted(arrays.items())
        },
    }


def _points_payload(
    start: Point,
    exit_point: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
) -> dict[str, object]:
    return {
        "start": list(start),
        "exit": list(exit_point),
        "objectives": [list(point) for point in objectives],
        "spawns": [list(point) for point in spawns],
    }


def _validate_conditioning_points(
    config: MapConfig,
    start: Point,
    exit_point: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
) -> None:
    if len(objectives) != config.objective_count or len(spawns) != config.spawn_count:
        raise ValueError("Artifact point counts disagree with MapConfig.")
    required = (start, exit_point, *objectives)
    if len(set(required)) != len(required) or len(set(spawns)) != len(spawns):
        raise ValueError("Artifact required/spawn point groups contain duplicates.")
    for point in (*required, *spawns):
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in point)
        ):
            raise TypeError("Artifact points must be exact integer (x, y) tuples.")
        if not 0 <= point[0] < config.width or not 0 <= point[1] < config.height:
            raise ValueError("Artifact point lies outside MapConfig dimensions.")


def write_raw_artifact(
    path: Path,
    *,
    raw: RawTopology,
    seed: int,
    theme: str,
    config: MapConfig,
    start: Point,
    exit: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
    provenance: dict[str, object],
    proposal_source: str,
) -> RawArtifact:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Raw neural topology artifact publication is immutable.")
    if raw.terrain.shape != (config.height, config.width):
        raise ValueError("Raw artifact shape disagrees with MapConfig.")
    if (
        theme not in THEMES
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 1 << 64
    ):
        raise ValueError("Raw artifact theme/seed is outside the topology contract.")
    _validate_conditioning_points(config, start, exit, objectives, spawns)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=MAX_RAW_ARRAY_BYTES + MAX_MANIFEST_BYTES)
    staging = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    arrays = raw.arrays()
    payload = deterministic_npz_bytes(arrays)
    if len(payload) > MAX_RAW_ARRAY_BYTES:
        raise ValueError("Raw topology arrays exceed their strict artifact byte bound.")
    descriptor = _array_descriptor(arrays, RAW_ARRAYS, payload)
    identity = {
        "schema_version": RAW_SCHEMA_VERSION,
        "artifact_type": "raw_neural_topology",
        "authority": "proposal_only_never_runtime",
        "proposal_source": proposal_source,
        "source_sha256": source_sha256(),
        "tensor_contract_sha256": CONTRACT_SHA256,
        "seed": int(seed),
        "theme": theme,
        "config": config.to_dict(),
        "points": _points_payload(start, exit, objectives, spawns),
        "raw_topology_sha256": raw.raw_sha256,
        "arrays_sha256": descriptor["sha256"],
        "provenance": provenance,
    }
    manifest: dict[str, object] = {
        **identity,
        "raw_identity_sha256": json_sha256(identity),
        "arrays": descriptor,
    }
    _validate_schema(manifest, "map_topology_neural_raw.schema.json")
    _atomic_bytes(staging / RAW_ARRAYS, payload)
    _atomic_json(staging / RAW_MANIFEST, manifest)
    os.replace(staging, path)
    return load_raw_artifact(path)


def _bounded_npz_load(
    path: Path,
    descriptor: dict[str, object],
    *,
    maximum_bytes: int,
) -> dict[str, np.ndarray]:
    if not path.is_file() or not 0 < path.stat().st_size <= maximum_bytes:
        raise ValueError("Topology array artifact is missing or exceeds its strict byte bound.")
    if descriptor.get("file") != path.name or descriptor.get("bytes") != path.stat().st_size:
        raise ValueError("Topology array descriptor does not match the artifact.")
    if descriptor.get("sha256") != file_sha256(path):
        raise ValueError("Topology array artifact SHA-256 drifted.")
    members = descriptor.get("members")
    if not isinstance(members, dict) or not members:
        raise ValueError("Topology array member descriptor is malformed.")
    arrays: dict[str, np.ndarray] = {}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            expected = {f"{name}.npy" for name in members}
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("Topology NPZ has duplicate, missing, or unexpected members.")
            if any(
                info.compress_type != zipfile.ZIP_STORED
                or info.compress_size != info.file_size
                or info.flag_bits & 1
                or PurePosixPath(info.filename).is_absolute()
                or len(PurePosixPath(info.filename).parts) != 1
                or info.file_size > maximum_bytes
                for info in infos
            ):
                raise ValueError("Topology NPZ member violates storage/path/size policy.")
            if sum(info.file_size for info in infos) > maximum_bytes:
                raise ValueError("Topology NPZ expanded size exceeds its strict bound.")
            for name in sorted(members):
                member_descriptor = members[name]
                if not isinstance(member_descriptor, dict) or set(member_descriptor) != {
                    "dtype", "shape", "nbytes", "array_sha256"
                }:
                    raise ValueError("Topology NPZ member descriptor is malformed.")
                with archive.open(f"{name}.npy", "r") as handle:
                    array = np.lib.format.read_array(handle, allow_pickle=False, max_header_size=4096)
                    if handle.read(1):
                        raise ValueError("Topology NPY member carries trailing bytes.")
                array = np.ascontiguousarray(array)
                if (
                    array.dtype.str != member_descriptor["dtype"]
                    or list(array.shape) != member_descriptor["shape"]
                    or array.nbytes != member_descriptor["nbytes"]
                    or array_sha256(array) != member_descriptor["array_sha256"]
                ):
                    raise ValueError(f"Topology array member {name!r} drifted.")
                arrays[name] = array
    except zipfile.BadZipFile as error:
        raise ValueError("Topology array artifact is not a valid NPZ container.") from error
    if named_arrays_sha256(arrays) != descriptor.get("canonical_arrays_sha256"):
        raise ValueError("Topology canonical array identity drifted.")
    if deterministic_npz_bytes(arrays) != path.read_bytes():
        raise ValueError("Topology NPZ bytes are not the canonical deterministic encoding.")
    return arrays


def load_raw_artifact(path: Path) -> RawArtifact:
    path = Path(path).resolve()
    manifest_path = path / RAW_MANIFEST
    if not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("Raw topology manifest is missing or exceeds its strict byte bound.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Raw topology manifest root must be an object.")
    _validate_schema(manifest, "map_topology_neural_raw.schema.json")
    if manifest.get("source_sha256") != source_sha256() or manifest.get("tensor_contract_sha256") != CONTRACT_SHA256:
        raise ValueError("Raw topology source/tensor contract provenance drifted.")
    descriptor = manifest.get("arrays")
    if not isinstance(descriptor, dict):
        raise ValueError("Raw topology arrays descriptor is malformed.")
    arrays = _bounded_npz_load(path / RAW_ARRAYS, descriptor, maximum_bytes=MAX_RAW_ARRAY_BYTES)
    if set(arrays) != {"terrain", "hazard", "elevation"}:
        raise ValueError("Raw topology artifact contains a non-authoritative field.")
    config_payload = manifest.get("config")
    points = manifest.get("points")
    if not isinstance(config_payload, dict) or not isinstance(points, dict):
        raise ValueError("Raw topology config/points are malformed.")
    config = MapConfig(**config_payload)
    raw = make_raw_topology(
        arrays["terrain"], arrays["hazard"], arrays["elevation"], shape=(config.height, config.width)
    )
    if raw.raw_sha256 != manifest.get("raw_topology_sha256"):
        raise ValueError("Raw topology semantic identity drifted.")
    identity = {key: value for key, value in manifest.items() if key not in {"raw_identity_sha256", "arrays"}}
    if json_sha256(identity) != manifest.get("raw_identity_sha256"):
        raise ValueError("Raw topology identity payload drifted.")
    start = tuple(points["start"])  # type: ignore[assignment]
    exit_point = tuple(points["exit"])  # type: ignore[assignment]
    objectives = tuple(tuple(point) for point in points["objectives"])
    spawns = tuple(tuple(point) for point in points["spawns"])
    _validate_conditioning_points(config, start, exit_point, objectives, spawns)  # type: ignore[arg-type]
    return RawArtifact(
        path=path,
        raw=raw,
        seed=int(manifest["seed"]),
        theme=str(manifest["theme"]),
        config=config,
        start=start,  # type: ignore[arg-type]
        exit=exit_point,  # type: ignore[arg-type]
        objectives=objectives,  # type: ignore[arg-type]
        spawns=spawns,  # type: ignore[arg-type]
        manifest=manifest,
        manifest_sha256=file_sha256(manifest_path),
    )


def write_compiled_artifact(
    path: Path,
    *,
    raw_artifact: RawArtifact,
    result: CompileResult,
) -> CompiledArtifact:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Compiled topology artifact publication is immutable.")
    if result.raw_sha256 != raw_artifact.raw.raw_sha256:
        raise ValueError("Compiled result does not bind the supplied raw artifact.")
    assert_exact_compiler_replay(result, raw_artifact.raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(
        path.parent,
        floor_gb=100.0,
        planned_bytes=MAX_COMPILED_ARRAY_BYTES + MAX_MANIFEST_BYTES + MAX_REPORT_BYTES + MAX_LEDGER_BYTES,
    )
    staging = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    arrays = result.data.arrays()
    arrays_payload = deterministic_npz_bytes(arrays)
    if len(arrays_payload) > MAX_COMPILED_ARRAY_BYTES:
        raise ValueError("Compiled topology arrays exceed their strict byte bound.")
    arrays_descriptor = _array_descriptor(arrays, COMPILED_ARRAYS, arrays_payload)
    ledger_payload = {"format": LEDGER_FORMAT, "entries": list(result.ledger)}
    ledger_bytes = json.dumps(ledger_payload, indent=2).encode("utf-8")
    report_bytes = json.dumps(result.report, indent=2).encode("utf-8")
    if len(ledger_bytes) > MAX_LEDGER_BYTES or len(report_bytes) > MAX_REPORT_BYTES:
        raise ValueError("Compiled ledger/report exceeds its strict byte bound.")
    identity = {
        "schema_version": COMPILED_SCHEMA_VERSION,
        "artifact_type": "compiled_topology",
        "authority": "deterministically_repaired_candidate_not_map_pack_v2",
        "source_sha256": source_sha256(),
        "tensor_contract_sha256": CONTRACT_SHA256,
        "compiler": {"name": COMPILER_NAME, "version": COMPILER_VERSION},
        "compiler_source_sha256": compiler_source_sha256(),
        "seed": int(result.data.seed),
        "theme": result.data.theme,
        "config": result.data.config.to_dict(),
        "points": _points_payload(
            result.data.start, result.data.exit, result.data.objectives, result.data.spawns
        ),
        "raw_manifest_sha256": raw_artifact.manifest_sha256,
        "raw_identity_sha256": raw_artifact.manifest["raw_identity_sha256"],
        "raw_topology_sha256": result.raw_sha256,
        "compiled_arrays_sha256": result.compiled_arrays_sha256,
        "ledger_sha256": result.ledger_sha256,
        "report_sha256": json_sha256(result.report),
    }
    manifest: dict[str, object] = {
        **identity,
        "compiled_identity_sha256": json_sha256(identity),
        "raw_manifest": {"path": "../raw/raw_manifest.json", "sha256": raw_artifact.manifest_sha256},
        "arrays": arrays_descriptor,
        "ledger": {
            "file": LEDGER_FILE,
            "bytes": len(ledger_bytes),
            "sha256": __import__("hashlib").sha256(ledger_bytes).hexdigest(),
            "entry_count": len(result.ledger),
            "ledger_sha256": result.ledger_sha256,
        },
        "report": {
            "file": REPORT_FILE,
            "bytes": len(report_bytes),
            "sha256": __import__("hashlib").sha256(report_bytes).hexdigest(),
            "report_sha256": json_sha256(result.report),
        },
        "gates": {
            "authoritative_assert_valid": True,
            "exact_compiler_replay": True,
            "raw_never_runtime": True,
            "published_as_map_pack_v2": False,
        },
    }
    _validate_schema(manifest, "map_topology_neural_compiled.schema.json")
    _atomic_bytes(staging / COMPILED_ARRAYS, arrays_payload)
    _atomic_bytes(staging / LEDGER_FILE, ledger_bytes)
    _atomic_bytes(staging / REPORT_FILE, report_bytes)
    _atomic_json(staging / COMPILED_MANIFEST, manifest)
    os.replace(staging, path)
    return load_compiled_artifact(path, raw_artifact=raw_artifact, exact_replay=True)


def _read_bound_json(path: Path, maximum: int, label: str) -> tuple[dict[str, object], bytes]:
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"{label} is missing or exceeds its strict byte bound.")
    payload_bytes = path.read_bytes()
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object.")
    return payload, payload_bytes


def load_compiled_artifact(
    path: Path,
    *,
    raw_artifact: RawArtifact,
    exact_replay: bool = True,
) -> CompiledArtifact:
    path = Path(path).resolve()
    manifest, _ = _read_bound_json(path / COMPILED_MANIFEST, MAX_MANIFEST_BYTES, "compiled manifest")
    _validate_schema(manifest, "map_topology_neural_compiled.schema.json")
    if (
        manifest.get("source_sha256") != source_sha256()
        or manifest.get("compiler_source_sha256") != compiler_source_sha256()
        or manifest.get("tensor_contract_sha256") != CONTRACT_SHA256
    ):
        raise ValueError("Compiled topology source/compiler/contract provenance drifted.")
    if (
        manifest.get("raw_manifest_sha256") != raw_artifact.manifest_sha256
        or manifest.get("raw_identity_sha256") != raw_artifact.manifest.get("raw_identity_sha256")
        or manifest.get("raw_topology_sha256") != raw_artifact.raw.raw_sha256
    ):
        raise ValueError("Compiled topology raw-artifact provenance drifted.")
    descriptor = manifest.get("arrays")
    if not isinstance(descriptor, dict):
        raise ValueError("Compiled topology array descriptor is malformed.")
    arrays = _bounded_npz_load(path / COMPILED_ARRAYS, descriptor, maximum_bytes=MAX_COMPILED_ARRAY_BYTES)
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Compiled topology array census drifted.")
    config_payload = manifest.get("config")
    points = manifest.get("points")
    if not isinstance(config_payload, dict) or not isinstance(points, dict):
        raise ValueError("Compiled topology config/points are malformed.")
    config = MapConfig(**config_payload)
    data = MapData(
        seed=int(manifest["seed"]),
        theme=str(manifest["theme"]),
        config=config,
        terrain=np.ascontiguousarray(arrays["terrain"], dtype=np.uint8),
        walkability=np.ascontiguousarray(arrays["walkability"], dtype=np.uint8),
        hazard=np.ascontiguousarray(arrays["hazard"], dtype=np.uint8),
        elevation=np.ascontiguousarray(arrays["elevation"], dtype=np.int8),
        zone=np.ascontiguousarray(arrays["zone"], dtype=np.int16),
        nav_cost=np.ascontiguousarray(arrays["nav_cost"], dtype=np.float32),
        protected_backbone=np.ascontiguousarray(arrays["protected_backbone"], dtype=np.uint8),
        required_clearance=np.ascontiguousarray(arrays["required_clearance"], dtype=np.uint8),
        decoration_forbidden=np.ascontiguousarray(arrays["decoration_forbidden"], dtype=np.uint8),
        start=tuple(points["start"]),  # type: ignore[arg-type]
        exit=tuple(points["exit"]),  # type: ignore[arg-type]
        objectives=tuple(tuple(point) for point in points["objectives"]),  # type: ignore[arg-type]
        spawns=tuple(tuple(point) for point in points["spawns"]),  # type: ignore[arg-type]
        repair_count=0,
        metadata={
            "compiler": COMPILER_NAME,
            "compiler_version": COMPILER_VERSION,
            "compiler_seed": int(manifest["seed"]),
            "raw_topology_sha256": raw_artifact.raw.raw_sha256,
            "protected_backbone_segments": 1 + len(points["objectives"]) + len(points["spawns"]),
            "topology_mask_capture": "captured at compiler mutation sites; never reconstructed",
        },
    )
    if named_arrays_sha256(data.arrays()) != manifest.get("compiled_arrays_sha256"):
        raise ValueError("Compiled semantic array identity drifted.")
    assert_valid(data)
    ledger_descriptor = manifest.get("ledger")
    report_descriptor = manifest.get("report")
    if not isinstance(ledger_descriptor, dict) or not isinstance(report_descriptor, dict):
        raise ValueError("Compiled ledger/report descriptor is malformed.")
    ledger_payload, ledger_bytes = _read_bound_json(path / LEDGER_FILE, MAX_LEDGER_BYTES, "edit ledger")
    report, report_bytes = _read_bound_json(path / REPORT_FILE, MAX_REPORT_BYTES, "compile report")
    if (
        ledger_descriptor.get("bytes") != len(ledger_bytes)
        or ledger_descriptor.get("sha256") != __import__("hashlib").sha256(ledger_bytes).hexdigest()
        or report_descriptor.get("bytes") != len(report_bytes)
        or report_descriptor.get("sha256") != __import__("hashlib").sha256(report_bytes).hexdigest()
    ):
        raise ValueError("Compiled ledger/report file identity drifted.")
    entries = ledger_payload.get("entries")
    if ledger_payload.get("format") != LEDGER_FORMAT or not isinstance(entries, list):
        raise ValueError("Compiled edit ledger payload is malformed.")
    expected_entry_keys = {"sequence", "phase", "field", "x", "y", "before", "after", "reason"}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != expected_entry_keys or entry.get("sequence") != index:
            raise ValueError("Compiled edit ledger ordering or member contract drifted.")
        if entry.get("field") not in {
            "terrain", "hazard", "elevation", "protected_backbone", "required_clearance"
        }:
            raise ValueError("Compiled edit ledger contains an unknown mutable field.")
        if (
            isinstance(entry.get("x"), bool)
            or not isinstance(entry.get("x"), int)
            or isinstance(entry.get("y"), bool)
            or not isinstance(entry.get("y"), int)
            or not 0 <= entry["x"] < config.width
            or not 0 <= entry["y"] < config.height
            or isinstance(entry.get("before"), bool)
            or not isinstance(entry.get("before"), int)
            or isinstance(entry.get("after"), bool)
            or not isinstance(entry.get("after"), int)
            or entry["before"] == entry["after"]
            or not isinstance(entry.get("phase"), str)
            or not entry["phase"]
            or not isinstance(entry.get("reason"), str)
            or not entry["reason"]
        ):
            raise ValueError("Compiled edit ledger entry violates coordinate/value/text bounds.")
    if (
        len(entries) != ledger_descriptor.get("entry_count")
        or json_sha256(ledger_payload) != ledger_descriptor.get("ledger_sha256")
        or json_sha256(ledger_payload) != manifest.get("ledger_sha256")
        or json_sha256(report) != report_descriptor.get("report_sha256")
        or json_sha256(report) != manifest.get("report_sha256")
    ):
        raise ValueError("Compiled ledger/report semantic identity drifted.")
    identity = {
        key: manifest[key]
        for key in (
            "schema_version", "artifact_type", "authority", "source_sha256",
            "tensor_contract_sha256", "compiler", "compiler_source_sha256", "seed",
            "theme", "config", "points", "raw_manifest_sha256", "raw_identity_sha256",
            "raw_topology_sha256", "compiled_arrays_sha256", "ledger_sha256", "report_sha256",
        )
    }
    if json_sha256(identity) != manifest.get("compiled_identity_sha256"):
        raise ValueError("Compiled topology identity payload drifted.")
    result = CompileResult(
        data=data,
        ledger=tuple(entries),
        ledger_sha256=str(manifest["ledger_sha256"]),
        raw_sha256=str(manifest["raw_topology_sha256"]),
        compiler_source_sha256=str(manifest["compiler_source_sha256"]),
        compiled_arrays_sha256=str(manifest["compiled_arrays_sha256"]),
        report=report,
    )
    if exact_replay:
        replay = compile_topology(
            raw_artifact.raw,
            seed=result.data.seed,
            theme=result.data.theme,
            config=result.data.config,
            start=result.data.start,
            exit=result.data.exit,
            objectives=result.data.objectives,
            spawns=result.data.spawns,
        )
        if (
            replay.ledger != result.ledger
            or replay.report != result.report
            or replay.compiled_arrays_sha256 != result.compiled_arrays_sha256
        ):
            raise ValueError("Compiled artifact failed source-bound exact compiler replay.")
        for name, array in result.data.arrays().items():
            if not np.array_equal(array, replay.data.arrays()[name]):
                raise ValueError(f"Compiled artifact replay array drifted: {name}.")
        # Preserve the compiler's meaningful repair count after exact replay.
        result.data.repair_count = replay.data.repair_count
    return CompiledArtifact(
        path=path,
        result=result,
        manifest=manifest,
        manifest_sha256=file_sha256(path / COMPILED_MANIFEST),
    )
