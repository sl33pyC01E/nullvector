from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from ..morphology import FACING_NAMES, FAMILIES, MOTION_NAMES
from ..multifield_style.source import PROJECT_ROOT
from ..multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from ..multifield_style_motion.io import verify_artifact, write_exact, write_json_exact
from ..multifield_style_motion.model import LAYER_NAMES
from .compiler import build_contract, compiler_source_hash
from .family import compile_family_identity_payload
from .schema import BANK_SCHEMA, validate_schema
from .showcase import compile_showcase
from .sharding import MOTION_SHARDS, aggregate_family_shards
from .source import compute_binding_census, load_neural_motion_source
from .style_parent import load_neural_style_parent
from .validation import load_verified_identity_manifest


REPLAY_FORMAT = "nullvector-multifield-style-neural-motion-replay-v1"


def replay_family(
    generation_manifest: Path,
    style_manifest: Path,
    output_root: Path,
    family: str,
) -> dict[str, Any]:
    source = load_neural_motion_source(generation_manifest)
    style_parent = load_neural_style_parent(style_manifest, source)
    contract = build_contract(source, style_parent)
    stored = json.loads((Path(output_root) / "build_contract.json").read_text(encoding="utf-8"))
    if stored != contract:
        raise ValueError("Neural motion replay contract mismatch")
    payload = compile_family_identity_payload(source, style_parent, family, contract)
    count = 0; byte_count = 0
    for relative, expected in payload.file_payloads.items():
        path = Path(output_root) / Path(*relative.split("/"))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Neural motion replay artifact missing: {relative}")
        actual = path.read_bytes()
        if actual != expected:
            raise ValueError(f"Neural motion replay mismatch {relative}: expected={sha256_bytes(expected)} actual={sha256_bytes(actual)}")
        count += 1; byte_count += len(actual)
    return {"family": family, "sample_id": payload.sample_id, "exact": True, "artifact_count": count, "bytes_compared": byte_count}


def _shard_worker(generation: Path, style: Path, root: Path, family: str, shard_index: int) -> Mapping[str, Any]:
    command = [sys.executable, "-m", "forge.multifield_style_neural_motion", "shard-worker", str(generation), str(style), str(root), family, str(shard_index)]
    failures = []
    for attempt in range(2):
        environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = ""
        try:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, timeout=55)
        except subprocess.TimeoutExpired as error:
            failures.append(f"attempt {attempt + 1} timed out after {error.timeout}s")
            continue
        if completed.returncode == 0:
            try:
                payload = json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])
            except (IndexError, ValueError):
                payload = {}
            if (
                payload.get("exact") is True
                and payload.get("family") == family
                and payload.get("shard_index") == shard_index
            ):
                return payload
        failures.append((completed.stderr or completed.stdout)[-2000:])
    raise RuntimeError(f"Neural motion replay shard failed: {family}/{shard_index}: {' | '.join(failures)}")


def _replay_family_from_shards(
    generation: Path,
    style: Path,
    source: Any,
    style_parent: Any,
    contract: Mapping[str, Any],
    stored_root: Path,
    replay_root: Path,
    family: str,
) -> dict[str, Any]:
    for shard_index in range(len(MOTION_SHARDS)):
        _shard_worker(generation, style, replay_root, family, shard_index)
    payload = aggregate_family_shards(replay_root, source, style_parent, family, contract)
    count = 0
    byte_count = 0
    for relative, expected in payload.file_payloads.items():
        path = stored_root / Path(*relative.split("/"))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Neural motion replay artifact missing: {relative}")
        actual = path.read_bytes()
        if actual != expected:
            raise ValueError(
                f"Neural motion replay mismatch {relative}: "
                f"expected={sha256_bytes(expected)} actual={sha256_bytes(actual)}"
            )
        count += 1
        byte_count += len(actual)
    return {
        "family": family,
        "sample_id": payload.sample_id,
        "exact": True,
        "artifact_count": count,
        "bytes_compared": byte_count,
        "shard_count": len(MOTION_SHARDS),
    }


def replay_neural_motion_style_bank(manifest_path: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve(); root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); validate_schema(manifest, BANK_SCHEMA)
    if canonical_json_bytes(manifest) != manifest_path.read_bytes() or manifest["compiler"]["source_sha256"] != compiler_source_hash():
        raise ValueError("Neural motion bank manifest/compiler is not exact")
    generation = PROJECT_ROOT / Path(*manifest["parent"]["generation_manifest_path"].split("/"))
    style = PROJECT_ROOT / Path(*manifest["parent"]["style_manifest_path"].split("/"))
    source = load_neural_motion_source(generation); style_parent = load_neural_style_parent(style, source)
    contract = build_contract(source, style_parent)
    if manifest["compiler"] != contract["compiler"]:
        raise ValueError("Neural motion bank provenance drifted")
    if manifest["source_census"] != compute_binding_census(source):
        raise ValueError("Neural motion bank all-80 binding census drifted")
    if (
        manifest["matrix"]["families"] != list(FAMILIES)
        or manifest["matrix"]["motions"] != list(MOTION_NAMES)
        or manifest["matrix"]["facings"] != list(FACING_NAMES)
        or manifest["matrix"]["layers"] != list(LAYER_NAMES)
    ):
        raise ValueError("Neural motion bank matrix order drifted")
    records = list(manifest["identities"])
    if [record["family"] for record in records] != list(FAMILIES):
        raise ValueError("Neural motion bank identity family order mismatch")
    if len({record["sample_id"] for record in records}) != len(FAMILIES):
        raise ValueError("Neural motion bank identity sample IDs are not unique")
    identity_manifests = {}
    for record in records:
        path = verify_artifact(root, record["manifest"])
        expected_path = root / "identities" / record["family"] / record["sample_id"] / "identity_manifest.json"
        if path != expected_path:
            raise ValueError("Neural motion identity manifest path mismatch")
        payload = load_verified_identity_manifest(
            root,
            record["family"],
            sample_id=record["sample_id"],
            compiler=contract["compiler"],
            generation_manifest_sha256=source.bank.manifest_sha256,
            style_manifest_sha256=style_parent.manifest_sha256,
            static_palette_artifact=style_parent.palette_artifacts[record["sample_id"]],
        )
        if (
            payload["source"]["binding_sha256"] != record["binding_sha256"]
            or payload["source"]["raw_fields_sha256"] != record["raw_fields_sha256"]
            or payload["source"]["static_palette_sha256"] != record["static_palette_sha256"]
        ):
            raise ValueError("Neural motion identity summary provenance mismatch")
        identity_manifests[record["family"]] = payload
    work_root = PROJECT_ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neural-motion-replay-", dir=work_root) as temporary:
        replay_root = Path(temporary).resolve()
        write_exact(replay_root / "build_contract.json", canonical_json_bytes(contract))
        results = [
            _replay_family_from_shards(
                generation,
                style,
                source,
                style_parent,
                contract,
                root,
                replay_root,
                family,
            )
            for family in FAMILIES
        ]
    showcase = compile_showcase(root, identity_manifests, ffmpeg_executable=Path(manifest["showcase"]["ffmpeg"]["path"]))
    comparisons = {
        "contact": (manifest["contact_sheet"]["artifact"], showcase.contact_png),
        "poster": (manifest["showcase"]["poster"], showcase.poster_png),
        "video": (manifest["showcase"]["artifact"], showcase.video_mp4),
    }
    for name, (record, expected) in comparisons.items():
        path = verify_artifact(root, record)
        if path.read_bytes() != expected:
            raise ValueError(f"Neural showcase exact replay mismatch: {name}")
    if list(showcase.frame_sha256) != manifest["showcase"]["frame_sha256"]:
        raise ValueError("Neural showcase frame hashes mismatch")
    if dict(showcase.ffmpeg) != manifest["showcase"]["ffmpeg"]:
        raise ValueError("Neural showcase ffmpeg provenance mismatch")
    if dict(showcase.encoding) != manifest["showcase"]["encoding"]:
        raise ValueError("Neural showcase encoding contract mismatch")
    report = {
        "format": REPLAY_FORMAT, "status": "passed", "neural_output": True,
        "manifest": {"path": manifest_path.relative_to(PROJECT_ROOT).as_posix(), "bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path)},
        "compiler_source_sha256": compiler_source_hash(), "identity_results": results,
        "identity_count": 5, "clip_count": 520, "frame_count": 4720,
        "artifact_count_compared": sum(r["artifact_count"] for r in results) + 3,
        "bytes_compared": sum(r["bytes_compared"] for r in results) + len(showcase.contact_png) + len(showcase.poster_png) + len(showcase.video_mp4),
        "exact_identity_replay": True, "exact_showcase_replay": True, "all_gates_passed": True,
    }
    if report_path is not None:
        write_json_exact(Path(report_path).resolve(), report)
    return report
