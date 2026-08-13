from __future__ import annotations

import argparse
from collections import Counter, deque
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
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_production.corpus import MANIFEST_FILE as CORPUS_MANIFEST_FILE, ShardSpec
from ..safety import require_disk_floor
from .contract import (
    DISK_FLOOR_GIB,
    MAX_INDEX_WORKERS,
    MAX_PROCESS_ATTEMPTS,
    V2_CONTRACT_SHA256,
    V2_INDEX_FORMAT_VERSION,
)
from .patches import ForegroundSampleStat


INDEX_MANIFEST_FILE = "foreground_index_manifest.json"
INDEX_VALIDATION_FILE = "foreground_index_validation.json"
INDEX_SOURCE_FILES = (
    "forge/map_decorator_production_v2/contract.py",
    "forge/map_decorator_production_v2/index.py",
    "forge/map_decorator_production_v2/patches.py",
)


def _source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in INDEX_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _source_sha256() -> str:
    return json_sha256(_source_manifest())


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(
        path.parent,
        floor_gb=DISK_FLOOR_GIB,
        planned_bytes=len(encoded) + 1024 * 1024,
    )
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


def _target_memmap(
    npz_path: Path,
    archive: zipfile.ZipFile,
    members: dict[str, object],
    name: str,
) -> np.memmap:
    descriptor = members.get(name)
    if not isinstance(descriptor, dict):
        raise ValueError(f"Target member {name!r} is absent from the corpus sidecar.")
    member = archive.getinfo(f"{name}.npy")
    if member.compress_type != zipfile.ZIP_STORED:
        raise ValueError("Foreground index requires the frozen ZIP_STORED corpus contract.")
    if member.file_size > int(descriptor["nbytes"]) + 4096:
        raise ValueError("Target member exceeds its bounded descriptor.")
    with archive.open(member, "r") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported target NPY version {version!r}.")
        if (
            fortran
            or np.dtype(dtype).str != descriptor["dtype"]
            or list(shape) != descriptor["shape"]
            or int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
            != int(descriptor["nbytes"])
        ):
            raise ValueError("Target NPY header drifted from the frozen corpus sidecar.")
        payload_offset = handle.tell()
    with npz_path.open("rb") as raw:
        raw.seek(member.header_offset)
        local = raw.read(30)
        if len(local) != 30 or local[:4] != b"PK\x03\x04":
            raise ValueError("Target ZIP local header is malformed.")
        filename_length = int.from_bytes(local[26:28], "little")
        extra_length = int.from_bytes(local[28:30], "little")
    data_offset = member.header_offset + 30 + filename_length + extra_length + payload_offset
    return np.memmap(
        npz_path,
        mode="r",
        dtype=np.dtype(descriptor["dtype"]),
        offset=data_offset,
        shape=tuple(int(value) for value in descriptor["shape"]),
        order="C",
    )


def _index_shard_path(output: Path, shard_id: str) -> Path:
    return output / "shards" / shard_id / "counts.json"


def _validate_shard_index(
    path: Path,
    *,
    corpus_sha256: str,
    entry: dict[str, object],
    shard_index: int,
) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Foreground shard index is absent or exceeds its bound.")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format_version": V2_INDEX_FORMAT_VERSION,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "index_source_sha256": _source_sha256(),
        "corpus_sha256": corpus_sha256,
        "shard_index": shard_index,
        "shard_id": entry["shard_id"],
        "input_artifact_sha256": entry["artifact_sha256"],
        "input_sidecar_sha256": entry["sidecar_sha256"],
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(f"Foreground shard index drifted for {name!r}.")
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != len(entry["sample_identity_sha256"]):
        raise ValueError("Foreground shard index sample count drifted.")
    if report.get("samples_sha256") != json_sha256(samples):
        raise ValueError("Foreground shard sample hash does not match.")
    expected_ids = list(entry["sample_identity_sha256"])
    if [sample["sample_identity_sha256"] for sample in samples] != expected_ids:
        raise ValueError("Foreground shard sample identities drifted from the corpus manifest.")
    return report


def build_index_shard(corpus: Path, output: Path, shard_index: int) -> dict[str, object]:
    corpus = Path(corpus).resolve()
    output = Path(output).resolve()
    manifest_path = corpus / CORPUS_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest["shards"]
    if not 0 <= shard_index < len(shards):
        raise ValueError("Corpus shard index is out of range.")
    entry = shards[shard_index]
    spec = ShardSpec.from_dict(entry["spec"])
    target_path = _index_shard_path(output, spec.shard_id)
    if target_path.exists():
        recovered = _validate_shard_index(
            target_path,
            corpus_sha256=manifest["corpus_sha256"],
            entry=entry,
            shard_index=shard_index,
        )
        return {"passed": True, "recovered": True, "shard_id": spec.shard_id, "samples": len(recovered["samples"])}
    require_disk_floor(output, floor_gb=DISK_FLOOR_GIB, planned_bytes=64 * 1024 * 1024)
    sidecar_path = corpus / spec.sidecar_path
    npz_path = corpus / spec.npz_path
    if file_sha256(sidecar_path) != entry["sidecar_sha256"]:
        raise ValueError("Input corpus sidecar hash drifted from its manifest.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    artifact = sidecar.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("members"), dict):
        raise ValueError("Input corpus artifact descriptor is malformed.")
    if file_sha256(npz_path) != entry["artifact_sha256"] or artifact["sha256"] != entry["artifact_sha256"]:
        raise ValueError("Input corpus artifact hash drifted from its manifest/sidecar.")
    records = sidecar.get("samples")
    if not isinstance(records, list) or len(records) != spec.sample_count:
        raise ValueError("Input corpus sample records are malformed.")
    with zipfile.ZipFile(npz_path, "r") as archive:
        arrays = {
            name: _target_memmap(npz_path, archive, artifact["members"], f"target_{name}")
            for name in ("decal", "prop", "emission")
        }
    samples: list[dict[str, object]] = []
    for sample_index, record in enumerate(records):
        counts: dict[str, list[int]] = {}
        for name, classes in (("decal", 3), ("prop", 3), ("emission", 4)):
            values = arrays[name][sample_index]
            class_counts = np.bincount(values.reshape(-1), minlength=classes)
            if len(class_counts) != classes or int(class_counts.sum()) != spec.width * spec.height:
                raise ValueError("Target count extraction violated the categorical/shape contract.")
            counts[name] = [int(value) for value in class_counts]
        samples.append(
            {
                "shard_index": shard_index,
                "sample_index": sample_index,
                "split": record["split"],
                "map_id": record["map_id"],
                "sample_identity_sha256": record["sample_identity_sha256"],
                "full_map_identity_sha256": record["full_map_identity_sha256"],
                "width": int(record["width"]),
                "height": int(record["height"]),
                "class_counts": counts,
                "decal_count": sum(counts["decal"][1:]),
                "prop_count": sum(counts["prop"][1:]),
                "emission_count": sum(counts["emission"][1:]),
            }
        )
    payload: dict[str, object] = {
        "format_version": V2_INDEX_FORMAT_VERSION,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "index_source_sha256": _source_sha256(),
        "corpus_sha256": manifest["corpus_sha256"],
        "shard_index": shard_index,
        "shard_id": spec.shard_id,
        "input_artifact_sha256": entry["artifact_sha256"],
        "input_sidecar_sha256": entry["sidecar_sha256"],
        "samples": samples,
        "samples_sha256": json_sha256(samples),
    }
    staging = target_path.parent.parent / f".{spec.shard_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_json(staging / "counts.json", payload)
        require_disk_floor(output, floor_gb=DISK_FLOOR_GIB, planned_bytes=64 * 1024 * 1024)
        target_path.parent.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target_path.parent)
    except BaseException:
        raise
    validated = _validate_shard_index(
        target_path,
        corpus_sha256=manifest["corpus_sha256"],
        entry=entry,
        shard_index=shard_index,
    )
    return {"passed": True, "recovered": False, "shard_id": spec.shard_id, "samples": len(validated["samples"])}


def _native_label(returncode: int) -> str | None:
    return {
        0xC0000005: "windows_access_violation",
        0xC0000409: "windows_stack_buffer_overrun",
        0xC000001D: "windows_illegal_instruction",
    }.get(returncode & 0xFFFF_FFFF)


def _run_workers(
    corpus: Path,
    output: Path,
    *,
    shard_count: int,
    python: Path,
    max_workers: int,
) -> list[dict[str, object]]:
    if not 1 <= max_workers <= MAX_INDEX_WORKERS:
        raise ValueError(f"Index workers must stay in [1,{MAX_INDEX_WORKERS}].")
    pending = deque((index, 1) for index in range(shard_count))
    active: dict[subprocess.Popen[bytes], tuple[int, int, object, object, float, Path, Path]] = {}
    telemetry: list[dict[str, object]] = []
    logs = output / "telemetry"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and len(active) < max_workers:
            shard_index, attempt = pending.popleft()
            require_disk_floor(output, floor_gb=DISK_FLOOR_GIB, planned_bytes=64 * 1024 * 1024)
            label = f"shard-{shard_index:03d}-attempt{attempt:02d}"
            stdout_path = logs / f"{label}.stdout.log"
            stderr_path = logs / f"{label}.stderr.log"
            stdout = stdout_path.open("xb")
            stderr = stderr_path.open("xb")
            process = subprocess.Popen(
                [
                    str(python),
                    "-m",
                    "forge.map_decorator_production_v2.index",
                    "worker",
                    "--corpus",
                    str(corpus),
                    "--output",
                    str(output),
                    "--shard-index",
                    str(shard_index),
                ],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            active[process] = (
                shard_index,
                attempt,
                stdout,
                stderr,
                time.perf_counter(),
                stdout_path,
                stderr_path,
            )
        completed = [process for process in active if process.poll() is not None]
        if not completed:
            time.sleep(0.10)
            continue
        for process in completed:
            shard_index, attempt, stdout, stderr, started, stdout_path, stderr_path = active.pop(process)
            stdout.close()  # type: ignore[union-attr]
            stderr.close()  # type: ignore[union-attr]
            returncode = int(process.returncode or 0)
            record = {
                "shard_index": shard_index,
                "attempt": attempt,
                "pid": process.pid,
                "returncode": returncode,
                "returncode_unsigned_hex": f"0x{returncode & 0xFFFF_FFFF:08x}",
                "native_failure": _native_label(returncode),
                "elapsed_seconds": time.perf_counter() - started,
                "stdout": stdout_path.relative_to(output).as_posix(),
                "stdout_sha256": file_sha256(stdout_path),
                "stderr": stderr_path.relative_to(output).as_posix(),
                "stderr_sha256": file_sha256(stderr_path),
                "passed": returncode == 0,
            }
            telemetry.append(record)
            if returncode != 0:
                if attempt >= MAX_PROCESS_ATTEMPTS:
                    _atomic_json(output / "build_telemetry.json", telemetry)
                    raise RuntimeError(f"Foreground index shard {shard_index} exhausted its retry budget.")
                pending.append((shard_index, attempt + 1))
    _atomic_json(output / "build_telemetry.json", telemetry)
    return telemetry


def _aggregate(corpus: Path, output: Path, telemetry: list[dict[str, object]]) -> dict[str, object]:
    corpus_manifest_path = corpus / CORPUS_MANIFEST_FILE
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    all_samples: list[dict[str, object]] = []
    shard_entries: list[dict[str, object]] = []
    for shard_index, entry in enumerate(corpus_manifest["shards"]):
        path = _index_shard_path(output, entry["shard_id"])
        report = _validate_shard_index(
            path,
            corpus_sha256=corpus_manifest["corpus_sha256"],
            entry=entry,
            shard_index=shard_index,
        )
        all_samples.extend(report["samples"])
        shard_entries.append(
            {
                "shard_index": shard_index,
                "shard_id": entry["shard_id"],
                "file": path.relative_to(output).as_posix(),
                "file_sha256": file_sha256(path),
                "samples_sha256": report["samples_sha256"],
                "sample_count": len(report["samples"]),
            }
        )
    sample_ids = [sample["sample_identity_sha256"] for sample in all_samples]
    full_ids = [sample["full_map_identity_sha256"] for sample in all_samples]
    split_counts = Counter(str(sample["split"]) for sample in all_samples)
    expected_splits = corpus_manifest["counts"]["splits"]
    failures: list[str] = []
    if len(sample_ids) != len(set(sample_ids)):
        failures.append("duplicate_sample_identity")
    if len(full_ids) != len(set(full_ids)):
        failures.append("duplicate_full_map_identity")
    if dict(sorted(split_counts.items())) != expected_splits:
        failures.append("split_counts")
    for head in ("decal", "prop"):
        if not any(int(sample[f"{head}_count"]) > 0 for sample in all_samples if sample["split"] == "train"):
            failures.append(f"empty_train_{head}_pool")
        if not any(int(sample[f"{head}_count"]) > 0 for sample in all_samples if sample["split"] == "validation"):
            failures.append(f"empty_validation_{head}_pool")
        if not any(int(sample[f"{head}_count"]) > 0 for sample in all_samples if sample["split"] == "test"):
            failures.append(f"empty_test_{head}_pool")
    validation: dict[str, object] = {
        "passed": not failures,
        "failures": failures,
        "corpus_sha256": corpus_manifest["corpus_sha256"],
        "sample_count": len(all_samples),
        "shard_count": len(shard_entries),
        "split_counts": dict(sorted(split_counts.items())),
        "duplicate_sample_identity_count": len(sample_ids) - len(set(sample_ids)),
        "duplicate_full_map_identity_count": len(full_ids) - len(set(full_ids)),
        "foreground_map_counts": {
            split: {
                head: sum(
                    int(sample[f"{head}_count"]) > 0
                    for sample in all_samples
                    if sample["split"] == split
                )
                for head in ("decal", "prop", "emission")
            }
            for split in ("train", "validation", "test")
        },
        "foreground_cell_counts": {
            split: {
                head: sum(
                    int(sample[f"{head}_count"])
                    for sample in all_samples
                    if sample["split"] == split
                )
                for head in ("decal", "prop", "emission")
            }
            for split in ("train", "validation", "test")
        },
    }
    if failures:
        raise RuntimeError(f"Foreground index aggregation failed: {failures}")
    _atomic_json(output / INDEX_VALIDATION_FILE, validation)
    identity = {
        "format_version": V2_INDEX_FORMAT_VERSION,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "index_source_sha256": _source_sha256(),
        "corpus_sha256": corpus_manifest["corpus_sha256"],
        "shards": [
            {
                "shard_id": entry["shard_id"],
                "samples_sha256": entry["samples_sha256"],
            }
            for entry in shard_entries
        ],
    }
    manifest: dict[str, object] = {
        **identity,
        "foreground_index_sha256": json_sha256(identity),
        "index_source_manifest": _source_manifest(),
        "corpus_manifest": str(corpus_manifest_path),
        "corpus_manifest_sha256": file_sha256(corpus_manifest_path),
        "shards": shard_entries,
        "validation": INDEX_VALIDATION_FILE,
        "validation_sha256": file_sha256(output / INDEX_VALIDATION_FILE),
        "telemetry": {
            "attempt_count": len(telemetry),
            "retry_count": sum(int(item["attempt"]) - 1 for item in telemetry),
            "native_failure_count": sum(item["native_failure"] is not None for item in telemetry),
            "max_workers": MAX_INDEX_WORKERS,
            "max_attempts": MAX_PROCESS_ATTEMPTS,
        },
    }
    _atomic_json(output / INDEX_MANIFEST_FILE, manifest)
    return manifest


def build_foreground_index(
    corpus: Path,
    output: Path,
    *,
    python: Path = Path(sys.executable),
    max_workers: int = MAX_INDEX_WORKERS,
) -> dict[str, object]:
    corpus = Path(corpus).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Foreground index output already exists: {output}")
    require_disk_floor(output.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=1024**3)
    corpus_manifest = json.loads((corpus / CORPUS_MANIFEST_FILE).read_text(encoding="utf-8"))
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    telemetry = _run_workers(
        corpus,
        staging,
        shard_count=len(corpus_manifest["shards"]),
        python=Path(python),
        max_workers=max_workers,
    )
    manifest = _aggregate(corpus, staging, telemetry)
    require_disk_floor(output.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=1024**3)
    os.replace(staging, output)
    return manifest


def validate_foreground_index(corpus: Path, index: Path) -> dict[str, object]:
    corpus = Path(corpus).resolve()
    index = Path(index).resolve()
    manifest = json.loads((index / INDEX_MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest["v2_contract_sha256"] != V2_CONTRACT_SHA256:
        raise ValueError("Foreground index v2 contract hash drifted.")
    if manifest["index_source_sha256"] != _source_sha256():
        raise ValueError("Foreground index source hash drifted.")
    validation = json.loads((index / INDEX_VALIDATION_FILE).read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise ValueError("Foreground index validation is not passing.")
    corpus_manifest = json.loads((corpus / CORPUS_MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest["corpus_sha256"] != corpus_manifest["corpus_sha256"]:
        raise ValueError("Foreground index references a different corpus.")
    if file_sha256(corpus / CORPUS_MANIFEST_FILE) != manifest["corpus_manifest_sha256"]:
        raise ValueError("Foreground index corpus manifest hash drifted.")
    for entry in manifest["shards"]:
        path = index / entry["file"]
        if file_sha256(path) != entry["file_sha256"]:
            raise ValueError("Foreground index shard file hash drifted.")
    return validation


def load_foreground_stats(index: Path, *, split: str) -> tuple[ForegroundSampleStat, ...]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown foreground index split.")
    index = Path(index).resolve()
    manifest = json.loads((index / INDEX_MANIFEST_FILE).read_text(encoding="utf-8"))
    stats: list[ForegroundSampleStat] = []
    for shard in manifest["shards"]:
        report = json.loads((index / shard["file"]).read_text(encoding="utf-8"))
        for sample in report["samples"]:
            if sample["split"] == split:
                stats.append(
                    ForegroundSampleStat(
                        shard_index=int(sample["shard_index"]),
                        sample_index=int(sample["sample_index"]),
                        split=split,
                        map_id=str(sample["map_id"]),
                        sample_identity_sha256=str(sample["sample_identity_sha256"]),
                        full_map_identity_sha256=str(sample["full_map_identity_sha256"]),
                        decal_count=int(sample["decal_count"]),
                        prop_count=int(sample["prop_count"]),
                    )
                )
    identities = [stat.sample_identity_sha256 for stat in stats]
    if not stats or len(identities) != len(set(identities)):
        raise ValueError("Loaded foreground stats are empty or duplicated.")
    return tuple(stats)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process-isolated v2 foreground density index")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--corpus", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    build = sub.add_parser("build")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workers", type=int, default=MAX_INDEX_WORKERS)
    validate = sub.add_parser("validate")
    validate.add_argument("--corpus", type=Path, required=True)
    validate.add_argument("--index", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        report = build_index_shard(args.corpus, args.output, args.shard_index)
    elif args.command == "build":
        report = build_foreground_index(args.corpus, args.output, max_workers=args.workers)
    else:
        report = validate_foreground_index(args.corpus, args.index)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
