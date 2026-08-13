from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from ..morphology import FAMILIES
from .compiler import BANK_FORMAT, build_contract, compiler_source_hash
from .family import compile_family_payload
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .io import require_disk_floor, verify_artifact, write_json_exact
from .schema import BANK_SCHEMA, validate_schema
from .showcase import compile_showcase, showcase_artifacts
from .source import PROJECT_ROOT, load_motion_bank
from .validation import load_verified_family_manifest, strict_json_file


REPLAY_FORMAT = "nullvector-multifield-style-motion-replay-v1"


def replay_family_payload(
    asset_index: Path,
    output_root: Path,
    family: str,
) -> dict[str, Any]:
    bank = load_motion_bank(asset_index)
    output_root = Path(output_root).resolve()
    contract = strict_json_file(output_root / "build_contract.json")
    if contract != build_contract(bank):
        raise ValueError("Style-motion replay build contract mismatch")
    payload = compile_family_payload(bank, family, contract)
    compared = 0
    compared_bytes = 0
    for relative, expected in payload.file_payloads.items():
        path = output_root / Path(*relative.split("/"))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Style-motion replay artifact missing/unsafe: {relative}")
        actual = path.read_bytes()
        if actual != expected:
            raise ValueError(
                f"Style-motion exact replay mismatch: {relative}; "
                f"expected={sha256_bytes(expected)} actual={sha256_bytes(actual)}"
            )
        compared += 1
        compared_bytes += len(actual)
    return {
        "family": family,
        "exact": True,
        "artifact_count": compared,
        "bytes_compared": compared_bytes,
    }


def _run_replay_worker(
    asset_index: Path,
    output_root: Path,
    family: str,
    *,
    attempts: int = 2,
) -> Mapping[str, Any]:
    command = [
        sys.executable,
        "-m",
        "forge.multifield_style_motion",
        "family-replay-worker",
        str(asset_index),
        str(output_root),
        family,
    ]
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=55,
        )
        if completed.returncode == 0:
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError(f"Empty replay worker response for {family}")
            payload = json.loads(lines[-1])
            if not isinstance(payload, dict) or payload.get("exact") is not True:
                raise RuntimeError(f"Malformed replay worker response for {family}")
            return payload
        failures.append(
            f"attempt {attempt} rc={completed.returncode}: "
            + (completed.stderr or completed.stdout)[-2000:]
        )
    raise RuntimeError(f"Style-motion replay worker failed for {family}: {' | '.join(failures)}")


def replay_motion_style_bank(
    manifest_path: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    output_root = manifest_path.parent
    manifest = strict_json_file(manifest_path)
    validate_schema(manifest, BANK_SCHEMA)
    if manifest.get("format") != BANK_FORMAT:
        raise ValueError("Unsupported style-motion bank manifest")
    if canonical_json_bytes(manifest) != manifest_path.read_bytes():
        raise ValueError("Style-motion bank manifest is not canonical JSON")
    if manifest["compiler"]["source_sha256"] != compiler_source_hash():
        raise ValueError("Style-motion bank compiler source hash is stale")
    asset_index = PROJECT_ROOT / Path(*manifest["parent"]["asset_index_path"].split("/"))
    bank = load_motion_bank(asset_index)
    contract = build_contract(bank)
    if manifest["compiler"] != contract["compiler"]:
        raise ValueError("Style-motion bank compiler provenance mismatch")
    parent_checks = (
        manifest["parent"]["asset_index_sha256"] == bank.asset_index_sha256,
        manifest["parent"]["asset_index_bytes"] == bank.asset_index_bytes,
        manifest["parent"]["source_manifest_sha256"] == bank.source_manifest_sha256,
        manifest["parent"]["source_manifest_bytes"] == bank.source_manifest_bytes,
        manifest["parent"]["source_archive_sha256"] == bank.source_archive_sha256,
        manifest["parent"]["source_archive_bytes"] == bank.source_archive_bytes,
    )
    if not all(parent_checks):
        raise ValueError("Style-motion bank parent artifact binding mismatch")
    family_records = {record["family"]: record for record in manifest["families"]}
    if list(family_records) != list(FAMILIES):
        raise ValueError("Style-motion bank family ordering mismatch")
    family_manifests: dict[str, Mapping[str, Any]] = {}
    for family in FAMILIES:
        record = family_records[family]
        path = verify_artifact(output_root, record["manifest"])
        expected_path = output_root / "families" / family / "family_manifest.json"
        if path != expected_path.resolve():
            raise ValueError("Style-motion family manifest path contract mismatch")
        family_manifest = load_verified_family_manifest(
            output_root,
            family,
            bank=bank,
            compiler=contract["compiler"],
        )
        if (
            record["style_identity_sha256"] != family_manifest["identity"]["style_identity_sha256"]
            or record["palette_sha256"] != family_manifest["identity"]["palette_sha256"]
            or record["clip_count"] != family_manifest["clip_count"]
            or record["frame_count"] != family_manifest["frame_count"]
        ):
            raise ValueError(f"Style-motion family summary mismatch: {family}")
        family_manifests[family] = family_manifest
    family_results = [
        dict(_run_replay_worker(asset_index, output_root, family))
        for family in FAMILIES
    ]
    showcase = compile_showcase(
        output_root,
        family_manifests,
        ffmpeg_executable=Path(manifest["showcase"]["ffmpeg"]["path"]),
    )
    visual = showcase_artifacts(showcase)
    for key, bank_record in (
        ("contact_sheet", manifest["contact_sheet"]["artifact"]),
        ("poster", manifest["showcase"]["poster"]),
        ("video", manifest["showcase"]["artifact"]),
    ):
        verify_artifact(output_root, bank_record)
        if visual[key] != bank_record:
            raise ValueError(f"Style-motion showcase artifact replay mismatch: {key}")
    if list(showcase.frame_sha256) != manifest["showcase"]["frame_sha256"]:
        raise ValueError("Style-motion showcase frame replay mismatch")
    if dict(showcase.ffmpeg) != manifest["showcase"]["ffmpeg"]:
        raise ValueError("Style-motion ffmpeg provenance drifted")
    if dict(showcase.encoding) != manifest["showcase"]["encoding"]:
        raise ValueError("Style-motion ffmpeg encoding contract drifted")
    require_disk_floor(output_root)
    report = {
        "format": REPLAY_FORMAT,
        "status": "passed",
        "manifest": {
            "path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "compiler_source_sha256": compiler_source_hash(),
        "family_results": family_results,
        "family_count": len(family_results),
        "clip_count": sum(result["clip_count"] for result in family_manifests.values()),
        "frame_count": sum(result["frame_count"] for result in family_manifests.values()),
        "artifact_count_compared": sum(result["artifact_count"] for result in family_results) + 3,
        "bytes_compared": sum(result["bytes_compared"] for result in family_results)
        + len(showcase.contact_png)
        + len(showcase.poster_png)
        + len(showcase.video_mp4),
        "exact_family_replay": True,
        "exact_showcase_replay": True,
        "all_gates_passed": True,
    }
    if report_path is not None:
        write_json_exact(Path(report_path).resolve(), report)
    return report
