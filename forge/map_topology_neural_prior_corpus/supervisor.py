from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Final

from ..map_topology_neural.corpus import FROZEN_CORPUS_MANIFEST_FILE_SHA256, FROZEN_CORPUS_SHA256
from ..map_topology_neural_production.dataset import TopologyProductionDataset
from ..safety import require_disk_floor
from .contract import (
    CORPUS_FORMAT,
    EXPECTED_SAMPLES,
    EXPECTED_SHARDS,
    MAX_ATTEMPTS,
    MAX_MANIFEST_BYTES,
    MAX_WORKERS,
    authority,
    canonical_json_bytes,
    corpus_source_sha256,
    sha256_bytes,
    sha256_file,
    source_manifest,
)
from .shard import MANIFEST_NAME as SHARD_MANIFEST_NAME, validate_shard


ROOT_MANIFEST: Final[str] = "latent_corpus_manifest.json"


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _registry(corpus_root: Path) -> tuple[TopologyProductionDataset, tuple[str, ...]]:
    dataset = TopologyProductionDataset(corpus_root)
    shard_ids = tuple(sorted({ref.shard_id for ref in dataset.refs}))
    if len(shard_ids) != EXPECTED_SHARDS or len(dataset.refs) != EXPECTED_SAMPLES:
        raise ValueError("Latent corpus source shard/sample census drifted.")
    return dataset, shard_ids


def _attempt(
    *,
    mode: str,
    corpus_root: Path,
    shard_root: Path,
    shard_id: str,
    attempts_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        command = [
            sys.executable, "-m", "forge.map_topology_neural_prior_corpus", "worker",
            "--mode", mode, "--corpus", str(corpus_root), "--destination", str(shard_root),
            "--shard-id", shard_id,
        ]
        environment = dict(os.environ)
        environment.update({
            "CUDA_VISIBLE_DEVICES": "-1", "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })
        timed_out = False
        try:
            completed = subprocess.run(
                command, cwd=Path(__file__).resolve().parents[2], env=environment,
                capture_output=True, text=True, timeout=timeout_seconds, check=False,
            )
            return_code = int(completed.returncode)
            stdout = completed.stdout[-32_768:]
            stderr = completed.stderr[-32_768:]
        except subprocess.TimeoutExpired as error:
            timed_out = True
            return_code = -1
            stdout = str(error.stdout or "")[-32_768:]
            stderr = str(error.stderr or "")[-32_768:]
        unsigned_code = return_code & 0xFFFFFFFF
        record = {
            "mode": mode,
            "shard_id": shard_id,
            "attempt": attempt,
            "return_code": return_code,
            "unsigned_return_code": unsigned_code,
            "access_violation": unsigned_code == 0xC0000005,
            "timed_out": timed_out,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout": stdout,
            "stderr": stderr,
            "passed": return_code == 0,
        }
        records.append(record)
        _atomic_json(attempts_root / f"{mode}-{shard_id}-attempt{attempt:02d}.json", record)
        if return_code == 0:
            return {"shard_id": shard_id, "mode": mode, "attempts": records, "passed": True}
    return {"shard_id": shard_id, "mode": mode, "attempts": records, "passed": False}


def _phase(
    *,
    mode: str,
    corpus_root: Path,
    shards_root: Path,
    shard_ids: tuple[str, ...],
    attempts_root: Path,
    workers: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"latent-{mode}") as executor:
        futures = {
            executor.submit(
                _attempt, mode=mode, corpus_root=corpus_root,
                shard_root=shards_root / shard_id, shard_id=shard_id,
                attempts_root=attempts_root, timeout_seconds=timeout_seconds,
            ): shard_id
            for shard_id in shard_ids
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if not result["passed"]:
                raise RuntimeError(f"Latent corpus {mode} shard {result['shard_id']} exhausted retries.")
    return sorted(results, key=lambda item: item["shard_id"])


def _read_root(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("Latent corpus root manifest is missing or oversized.")
    encoded = path.read_bytes()
    manifest = json.loads(encoded)
    if not isinstance(manifest, dict) or encoded != canonical_json_bytes(manifest):
        raise ValueError("Latent corpus root manifest is not canonical JSON.")
    stored = manifest.pop("manifest_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(manifest)):
        raise ValueError("Latent corpus root manifest self-hash failed.")
    manifest["manifest_sha256"] = stored
    return manifest


def build_corpus(
    corpus_root: Path,
    output: Path,
    *,
    workers: int = MAX_WORKERS,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError("Latent corpus workers must be one or two.")
    if type(timeout_seconds) is not int or not 60 <= timeout_seconds <= 900:
        raise ValueError("Latent corpus worker timeout must be in [60,900] seconds.")
    corpus_root = Path(corpus_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Latent corpus publication is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=2 * 1024**3)
    dataset, shard_ids = _registry(corpus_root)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    shards_root = staging / "shards"
    attempts_root = staging / "attempts"
    shards_root.mkdir(parents=True, exist_ok=False)
    attempts_root.mkdir(parents=True, exist_ok=False)
    try:
        build_results = _phase(
            mode="build", corpus_root=corpus_root, shards_root=shards_root,
            shard_ids=shard_ids, attempts_root=attempts_root, workers=workers,
            timeout_seconds=timeout_seconds,
        )
        validation_results = _phase(
            mode="validate", corpus_root=corpus_root, shards_root=shards_root,
            shard_ids=shard_ids, attempts_root=attempts_root, workers=workers,
            timeout_seconds=timeout_seconds,
        )
        shard_records: list[dict[str, Any]] = []
        for shard_id in shard_ids:
            path = shards_root / shard_id / SHARD_MANIFEST_NAME
            manifest = validate_shard(corpus_root, shards_root / shard_id, replay_source=False)
            shard_records.append({
                "shard_id": shard_id,
                "sample_count": manifest["sample_count"],
                "theme": manifest["theme"],
                "shape": manifest["shape"],
                "manifest": f"shards/{shard_id}/{SHARD_MANIFEST_NAME}",
                "manifest_file_sha256": sha256_file(path),
                "manifest_sha256": manifest["manifest_sha256"],
                "arrays_sha256": manifest["arrays"]["sha256"],
                "semantic_sha256": manifest["arrays"]["semantic_sha256"],
            })
        all_attempts = [attempt for result in (*build_results, *validation_results) for attempt in result["attempts"]]
        split_counts = {name: len(dataset.refs_by_split[name]) for name in ("train", "validation", "test")}
        theme_counts = {theme: sum(ref.theme == theme for ref in dataset.refs) for theme in sorted({ref.theme for ref in dataset.refs})}
        shape_counts: dict[str, int] = {}
        for ref in dataset.refs:
            key = f"{ref.width}x{ref.height}"
            shape_counts[key] = shape_counts.get(key, 0) + 1
        identity = sha256_bytes(canonical_json_bytes([
            {key: record[key] for key in ("shard_id", "manifest_sha256", "arrays_sha256", "semantic_sha256")}
            for record in shard_records
        ]))
        manifest: dict[str, Any] = {
            "format": CORPUS_FORMAT,
            "status": "passed",
            "source_sha256": corpus_source_sha256(),
            "source_manifest": source_manifest(),
            "authority": authority(),
            "corpus_sha256": FROZEN_CORPUS_SHA256,
            "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
            "latent_corpus_identity_sha256": identity,
            "census": {
                "shards": len(shard_records), "samples": sum(record["sample_count"] for record in shard_records),
                "splits": split_counts, "themes": theme_counts, "shapes": dict(sorted(shape_counts.items())),
            },
            "shards": shard_records,
            "telemetry": {
                "workers": workers, "max_attempts": MAX_ATTEMPTS, "timeout_seconds": timeout_seconds,
                "attempts": len(all_attempts), "retries": sum(len(result["attempts"]) - 1 for result in (*build_results, *validation_results)),
                "access_violations": sum(bool(item["access_violation"]) for item in all_attempts),
                "timeouts": sum(bool(item["timed_out"]) for item in all_attempts),
                "build_shards": len(build_results), "validated_shards": len(validation_results),
            },
            "gates": {
                "all_source_maps_encoded": len(shard_records) == EXPECTED_SHARDS and sum(record["sample_count"] for record in shard_records) == EXPECTED_SAMPLES,
                "split_census_exact": split_counts == {"train": 2496, "validation": 576, "test": 24},
                "fresh_process_source_replay": len(validation_results) == EXPECTED_SHARDS,
                "codec_frozen": True,
                "raw_latents_only": True,
                "runtime_integration_disabled": True,
                "disk_floor_preserved": True,
            },
            "claim_boundary": {
                "training_started": False,
                "generative_quality_claim": False,
                "compiled_maps_published": False,
                "godot_integration": False,
            },
        }
        if not all(manifest["gates"].values()):
            raise RuntimeError("Latent corpus root gates failed before publication.")
        require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=0)
        manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        _atomic_json(staging / ROOT_MANIFEST, manifest)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            diagnostic = output.parent / f"{staging.name}.failed"
            if diagnostic.exists():
                diagnostic = output.parent / f"{staging.name}.failed-{time.time_ns()}"
            os.replace(staging, diagnostic)
        raise
    return validate_corpus(corpus_root, output)


def validate_corpus(corpus_root: Path, output: Path) -> dict[str, Any]:
    corpus_root = Path(corpus_root).resolve()
    output = Path(output).resolve()
    manifest = _read_root(output / ROOT_MANIFEST)
    expected_keys = {
        "format", "status", "source_sha256", "source_manifest", "authority",
        "corpus_sha256", "corpus_manifest_file_sha256", "latent_corpus_identity_sha256",
        "census", "shards", "telemetry", "gates", "claim_boundary", "manifest_sha256",
    }
    if set(manifest) != expected_keys or manifest["format"] != CORPUS_FORMAT or manifest["status"] != "passed":
        raise ValueError("Latent corpus root format/status/census failed.")
    if manifest["source_sha256"] != corpus_source_sha256() or manifest["source_manifest"] != source_manifest() or manifest["authority"] != authority():
        raise ValueError("Latent corpus root source/authority drifted.")
    if manifest["corpus_sha256"] != FROZEN_CORPUS_SHA256 or manifest["corpus_manifest_file_sha256"] != FROZEN_CORPUS_MANIFEST_FILE_SHA256:
        raise ValueError("Latent corpus root source corpus drifted.")
    if manifest["claim_boundary"] != {"training_started": False, "generative_quality_claim": False, "compiled_maps_published": False, "godot_integration": False}:
        raise ValueError("Latent corpus root claim boundary drifted.")
    gate_keys = {"all_source_maps_encoded", "split_census_exact", "fresh_process_source_replay", "codec_frozen", "raw_latents_only", "runtime_integration_disabled", "disk_floor_preserved"}
    if not isinstance(manifest["gates"], dict) or set(manifest["gates"]) != gate_keys or not all(value is True for value in manifest["gates"].values()):
        raise ValueError("Latent corpus root gates drifted.")
    dataset, shard_ids = _registry(corpus_root)
    if len(manifest["shards"]) != EXPECTED_SHARDS or [item["shard_id"] for item in manifest["shards"]] != list(shard_ids):
        raise ValueError("Latent corpus root shard registry drifted.")
    records: list[dict[str, Any]] = []
    for record in manifest["shards"]:
        shard_root = output / "shards" / record["shard_id"]
        shard_manifest_path = output / record["manifest"]
        if sha256_file(shard_manifest_path) != record["manifest_file_sha256"]:
            raise ValueError("Latent corpus shard manifest file identity failed.")
        shard_manifest = validate_shard(corpus_root, shard_root, replay_source=False)
        expected_record = {
            "shard_id": record["shard_id"], "sample_count": shard_manifest["sample_count"],
            "theme": shard_manifest["theme"], "shape": shard_manifest["shape"],
            "manifest": record["manifest"], "manifest_file_sha256": record["manifest_file_sha256"],
            "manifest_sha256": shard_manifest["manifest_sha256"],
            "arrays_sha256": shard_manifest["arrays"]["sha256"],
            "semantic_sha256": shard_manifest["arrays"]["semantic_sha256"],
        }
        if record != expected_record:
            raise ValueError("Latent corpus shard root record drifted.")
        records.append(record)
    identity = sha256_bytes(canonical_json_bytes([
        {key: record[key] for key in ("shard_id", "manifest_sha256", "arrays_sha256", "semantic_sha256")}
        for record in records
    ]))
    if identity != manifest["latent_corpus_identity_sha256"]:
        raise ValueError("Latent corpus aggregate identity failed.")
    expected_census = {
        "shards": EXPECTED_SHARDS, "samples": EXPECTED_SAMPLES,
        "splits": {name: len(dataset.refs_by_split[name]) for name in ("train", "validation", "test")},
        "themes": {theme: sum(ref.theme == theme for ref in dataset.refs) for theme in sorted({ref.theme for ref in dataset.refs})},
        "shapes": {},
    }
    for ref in dataset.refs:
        key = f"{ref.width}x{ref.height}"
        expected_census["shapes"][key] = expected_census["shapes"].get(key, 0) + 1
    expected_census["shapes"] = dict(sorted(expected_census["shapes"].items()))
    if manifest["census"] != expected_census:
        raise ValueError("Latent corpus census drifted.")
    return manifest

