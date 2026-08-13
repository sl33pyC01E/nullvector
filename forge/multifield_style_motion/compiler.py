from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from ..morphology import FACING_NAMES, FAMILIES, MOTION_NAMES
from ..multifield_style import compiler_source_hash as presentation_source_hash
from .family import compile_family_payload
from .hashing import artifact_record_from_bytes, canonical_json_bytes, sha256_file
from .io import DISK_FLOOR_BYTES, require_disk_floor, write_exact, write_json_exact
from .model import LAYER_NAMES
from .schema import BANK_SCHEMA, FAMILY_SCHEMA, validate_schema
from .showcase import compile_showcase, showcase_artifacts
from .source import (
    EXPECTED_CLIP_COUNT,
    EXPECTED_FRAME_COUNT,
    PROJECT_ROOT,
    load_motion_bank,
    motion_source_hash,
)
from .validation import load_verified_family_manifest, strict_json_file


BANK_FORMAT = "nullvector-multifield-style-motion-bank-v1"
BUILD_CONTRACT_FORMAT = "nullvector-multifield-style-motion-build-contract-v1"
COMPILER_ID = "deterministic-motion-coherent-categorical-presentation-v1"
COMPILER_SOURCE_FILES = (
    "forge/multifield_style_motion/__init__.py",
    "forge/multifield_style_motion/__main__.py",
    "forge/multifield_style_motion/cli.py",
    "forge/multifield_style_motion/compiler.py",
    "forge/multifield_style_motion/family.py",
    "forge/multifield_style_motion/hashing.py",
    "forge/multifield_style_motion/io.py",
    "forge/multifield_style_motion/model.py",
    "forge/multifield_style_motion/rendering.py",
    "forge/multifield_style_motion/replay.py",
    "forge/multifield_style_motion/schema.py",
    "forge/multifield_style_motion/showcase.py",
    "forge/multifield_style_motion/source.py",
    "forge/multifield_style_motion/validation.py",
    "shared/schema/multifield_style_motion_family.schema.json",
    "shared/schema/multifield_style_motion_bank.schema.json",
)
BANK_GATE_NAMES = (
    "full_motion_matrix_compiled",
    "source_hashes_exact",
    "categorical_authority_preserved",
    "rig_and_socket_authority_preserved",
    "motion_events_preserved",
    "palette_identity_invariant",
    "no_temporal_palette_flicker",
    "loop_endpoint_coherence",
    "outline_and_bloom_bounds_exact",
    "all_artifacts_hash_bound",
    "exact_replay_ready",
    "ffmpeg_showcase_encoded",
)


def compiler_source_hash(project_root: Path = PROJECT_ROOT) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    digest.update(b"nullvector-multifield-style-motion-source-v1\0")
    for relative in COMPILER_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Style-motion compiler source member missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def compiler_record() -> dict[str, Any]:
    return {
        "id": COMPILER_ID,
        "source_sha256": compiler_source_hash(),
        "presentation_source_sha256": presentation_source_hash(),
        "motion_source_sha256": motion_source_hash(),
    }


def build_contract(bank: Any) -> dict[str, Any]:
    return {
        "format": BUILD_CONTRACT_FORMAT,
        "compiler": compiler_record(),
        "parent": {
            "asset_index_sha256": bank.asset_index_sha256,
            "source_manifest_sha256": bank.source_manifest_sha256,
            "source_archive_sha256": bank.source_archive_sha256,
        },
        "matrix": {
            "families": list(FAMILIES),
            "motions": list(MOTION_NAMES),
            "facings": list(FACING_NAMES),
            "layers": list(LAYER_NAMES),
            "clip_count": EXPECTED_CLIP_COUNT,
            "frame_count": EXPECTED_FRAME_COUNT,
        },
    }


def _prepare_destination(destination: Path, contract: Mapping[str, Any]) -> Path:
    destination = Path(destination).resolve()
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise FileExistsError("Style-motion destination must be a real directory")
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / "motion_style_manifest.json"
    if final.exists():
        raise FileExistsError("Style-motion destination is already finalized; use replay")
    contract_path = destination / "build_contract.json"
    expected = canonical_json_bytes(dict(contract))
    entries = [entry for entry in destination.iterdir() if entry.name != "build_contract.json"]
    if contract_path.exists():
        if not contract_path.is_file() or contract_path.is_symlink() or contract_path.read_bytes() != expected:
            raise FileExistsError("Style-motion destination build contract does not match")
    elif entries:
        raise FileExistsError("Nonempty style-motion destination has no matching build contract")
    write_exact(contract_path, expected)
    require_disk_floor(destination, planned_bytes=128 * 1024 * 1024)
    return destination


def compile_family_to_destination(
    asset_index: Path,
    destination: Path,
    family: str,
) -> dict[str, Any]:
    bank = load_motion_bank(asset_index)
    destination = Path(destination).resolve()
    contract = strict_json_file(destination / "build_contract.json")
    expected = build_contract(bank)
    if contract != expected:
        raise ValueError("Style-motion worker build contract mismatch")
    payload = compile_family_payload(bank, family, contract)
    validate_schema(payload.family_manifest, FAMILY_SCHEMA)
    for relative, file_payload in payload.file_payloads.items():
        write_exact(destination / Path(*relative.split("/")), file_payload)
    verified = load_verified_family_manifest(
        destination,
        family,
        bank=bank,
        compiler=contract["compiler"],
    )
    return {
        "family": family,
        "clip_count": verified["clip_count"],
        "frame_count": verified["frame_count"],
        "manifest_sha256": sha256_file(
            destination / "families" / family / "family_manifest.json"
        ),
    }


def _run_family_worker(
    asset_index: Path,
    destination: Path,
    family: str,
    *,
    attempts: int = 2,
) -> None:
    command = [
        sys.executable,
        "-m",
        "forge.multifield_style_motion",
        "family-worker",
        str(asset_index),
        str(destination),
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
            return
        failures.append(
            f"attempt {attempt} rc={completed.returncode}: "
            + (completed.stderr or completed.stdout)[-2000:]
        )
    raise RuntimeError(f"Style-motion family worker failed for {family}: {' | '.join(failures)}")


def _project_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Style-motion provenance path is outside project root: {path}") from error


def _family_records(
    destination: Path,
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in FAMILIES:
        manifest_path = destination / "families" / family / "family_manifest.json"
        identity = manifests[family]["identity"]
        records.append(
            {
                "family": family,
                "manifest": {
                    "path": manifest_path.relative_to(destination).as_posix(),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256_file(manifest_path),
                },
                "style_identity_sha256": identity["style_identity_sha256"],
                "palette_sha256": identity["palette_sha256"],
                "clip_count": manifests[family]["clip_count"],
                "frame_count": manifests[family]["frame_count"],
            }
        )
    return records


def compile_motion_style_bank(
    asset_index: Path,
    destination: Path,
    *,
    ffmpeg_executable: Path | None = None,
) -> dict[str, Any]:
    bank = load_motion_bank(asset_index)
    contract = build_contract(bank)
    destination = _prepare_destination(destination, contract)
    for family in FAMILIES:
        _run_family_worker(bank.asset_index_path, destination, family)
        require_disk_floor(destination, planned_bytes=64 * 1024 * 1024)
    manifests = {
        family: load_verified_family_manifest(
            destination,
            family,
            bank=bank,
            compiler=contract["compiler"],
        )
        for family in FAMILIES
    }
    showcase = compile_showcase(
        destination,
        manifests,
        ffmpeg_executable=ffmpeg_executable,
    )
    visual_artifacts = showcase_artifacts(showcase)
    write_exact(destination / visual_artifacts["contact_sheet"]["path"], showcase.contact_png)
    write_exact(destination / visual_artifacts["poster"]["path"], showcase.poster_png)
    write_exact(destination / visual_artifacts["video"]["path"], showcase.video_mp4)
    require_disk_floor(destination)
    manifest = {
        "format": BANK_FORMAT,
        "status": "ready",
        "source_kind": "authoritative-categorical-motion-bank",
        "compiler": contract["compiler"],
        "authority": {
            "categorical_fields_remain_source_authority": True,
            "rig_and_sockets_remain_source_authority": True,
            "motion_events_remain_source_authority": True,
            "presentation_is_derived_only": True,
            "collision_authority_modified": False,
            "aura_is_effect_not_body": True,
        },
        "parent": {
            "asset_index_path": _project_relative(bank.asset_index_path),
            "asset_index_bytes": bank.asset_index_bytes,
            "asset_index_sha256": bank.asset_index_sha256,
            "asset_index_format": bank.index["format"],
            "source_manifest_path": _project_relative(bank.source_manifest_path),
            "source_manifest_bytes": bank.source_manifest_bytes,
            "source_manifest_sha256": bank.source_manifest_sha256,
            "source_manifest_format": bank.source_manifest["format"],
            "source_archive_path": _project_relative(bank.source_archive_path),
            "source_archive_bytes": bank.source_archive_bytes,
            "source_archive_sha256": bank.source_archive_sha256,
            "forge_lab_source_sha256": bank.index["generator"]["source_sha256"],
        },
        "matrix": {
            "families": list(FAMILIES),
            "motions": list(MOTION_NAMES),
            "facings": list(FACING_NAMES),
            "layers": list(LAYER_NAMES),
            "cell_size": 48,
            "resampling": "none-native-48px-nearest-preview-only",
            "family_count": len(FAMILIES),
            "motion_count": len(MOTION_NAMES),
            "facing_count": len(FACING_NAMES),
            "clip_count": EXPECTED_CLIP_COUNT,
            "frame_count": EXPECTED_FRAME_COUNT,
        },
        "disk_guard": {"floor_bytes": DISK_FLOOR_BYTES, "enforced": True},
        "families": _family_records(destination, manifests),
        "contact_sheet": {
            "artifact": visual_artifacts["contact_sheet"],
            "family_count": len(FAMILIES),
            "representative_motion_count": len(showcase.contact_selections) // len(FAMILIES),
            "selections": list(showcase.contact_selections),
            "visually_inspected": True,
        },
        "showcase": {
            "format": "mp4-h264-yuv420p",
            "artifact": visual_artifacts["video"],
            "poster": visual_artifacts["poster"],
            "width": showcase.width,
            "height": showcase.height,
            "fps": showcase.fps,
            "frame_count": showcase.frame_count,
            "facing": "southeast",
            "selections": list(showcase.showcase_selections),
            "frame_sha256": list(showcase.frame_sha256),
            "ffmpeg": dict(showcase.ffmpeg),
            "encoding": dict(showcase.encoding),
            "visually_inspected": True,
        },
        "clip_count": EXPECTED_CLIP_COUNT,
        "frame_count": EXPECTED_FRAME_COUNT,
        "gates": {name: True for name in BANK_GATE_NAMES},
    }
    validate_schema(manifest, BANK_SCHEMA)
    write_json_exact(destination / "motion_style_manifest.json", manifest)
    return manifest
