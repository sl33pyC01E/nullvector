from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from ..morphology import FACING_NAMES, FAMILIES, MOTION_NAMES
from ..morphology.motion import DEFAULT_FRAME_COUNTS
from ..multifield_style import compiler_source_hash as presentation_source_hash
from ..multifield_style.source import PROJECT_ROOT
from ..multifield_style_motion.hashing import artifact_record_from_bytes, canonical_json_bytes, sha256_file
from ..multifield_style_motion.io import DISK_FLOOR_BYTES, require_disk_floor, write_exact, write_json_exact
from ..multifield_style_motion.model import LAYER_NAMES
from ..neural_rig_bridge.hashing import binder_source_hash
from .family import compile_family_identity_payload
from .schema import BANK_SCHEMA, IDENTITY_SCHEMA, validate_schema
from .sharding import (
    MOTION_SHARDS,
    aggregate_family_shards,
    compile_motion_shard_payload,
    load_motion_shard,
)
from .showcase import compile_showcase
from .source import compute_binding_census, load_neural_motion_source
from .style_parent import load_neural_style_parent
from .validation import load_verified_identity_manifest


BANK_FORMAT = "nullvector-multifield-style-neural-motion-bank-v1"
BUILD_FORMAT = "nullvector-multifield-style-neural-motion-build-v1"
COMPILER_ID = "deterministic-neural-motion-presentation-v1"
EXPECTED_PRESENTATION_SOURCE_SHA256 = "af90198ae33642c345627a0fe0211de4eba189eabf8b55f469a4aec1ebb68c2c"
EXPECTED_BRIDGE_SOURCE_SHA256 = "46372e031c91d0202d0e55a8422385978c5157f76d83ed20adef9ed3e7250305"
COMPILER_SOURCE_FILES = (
    "forge/multifield_style_neural_motion/__init__.py",
    "forge/multifield_style_neural_motion/__main__.py",
    "forge/multifield_style_neural_motion/cli.py",
    "forge/multifield_style_neural_motion/compiler.py",
    "forge/multifield_style_neural_motion/family.py",
    "forge/multifield_style_neural_motion/model.py",
    "forge/multifield_style_neural_motion/rendering.py",
    "forge/multifield_style_neural_motion/replay.py",
    "forge/multifield_style_neural_motion/schema.py",
    "forge/multifield_style_neural_motion/showcase.py",
    "forge/multifield_style_neural_motion/sharding.py",
    "forge/multifield_style_neural_motion/source.py",
    "forge/multifield_style_neural_motion/style_parent.py",
    "forge/multifield_style_neural_motion/validation.py",
    "shared/schema/multifield_style_neural_motion_identity.schema.json",
    "shared/schema/multifield_style_neural_motion_bank.schema.json",
)
BANK_GATE_NAMES = (
    "actual_neural_samples_animated", "full_five_family_matrix_compiled",
    "raw_generation_provenance_exact", "static_style_parent_exact",
    "binding_source_exact", "motion_program_source_exact",
    "no_procedural_pixel_substitution", "categorical_authority_preserved",
    "rig_and_socket_authority_preserved", "motion_events_preserved",
    "palette_identity_invariant", "no_temporal_palette_flicker",
    "loop_endpoint_coherence", "outline_and_bloom_bounds_exact",
    "emission_pulse_support_bounded", "all_artifacts_hash_bound",
    "all_80_binding_census_exact", "exact_replay_ready", "ffmpeg_showcase_encoded",
)


def compiler_source_hash() -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-multifield-style-neural-motion-source-v1\0")
    for relative in COMPILER_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Neural motion compiler source missing: {relative}")
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def compiler_record() -> dict[str, Any]:
    presentation_sha256 = presentation_source_hash()
    bridge_sha256 = binder_source_hash()
    if presentation_sha256 != EXPECTED_PRESENTATION_SOURCE_SHA256:
        raise ValueError(
            "Static presentation compiler drifted from the calibrated production source: "
            f"expected={EXPECTED_PRESENTATION_SOURCE_SHA256} actual={presentation_sha256}"
        )
    if bridge_sha256 != EXPECTED_BRIDGE_SOURCE_SHA256:
        raise ValueError(
            "Neural rig bridge drifted from the validated production source: "
            f"expected={EXPECTED_BRIDGE_SOURCE_SHA256} actual={bridge_sha256}"
        )
    return {
        "id": COMPILER_ID,
        "source_sha256": compiler_source_hash(),
        "presentation_source_sha256": presentation_sha256,
        "bridge_source_sha256": bridge_sha256,
    }


def build_contract(source: Any, style_parent: Any) -> dict[str, Any]:
    return {
        "format": BUILD_FORMAT,
        "compiler": compiler_record(),
        "parent": {
            "generation_manifest_sha256": source.bank.manifest_sha256,
            "style_manifest_sha256": style_parent.manifest_sha256,
            "corpus_sha256": source.corpus_sha256,
            "split_fingerprint": source.split_fingerprint,
            "legal_tuple_fingerprint": source.legal_tuple_fingerprint,
        },
        "matrix": {
            "families": list(FAMILIES), "motions": list(MOTION_NAMES),
            "facings": list(FACING_NAMES), "layers": list(LAYER_NAMES),
            "frame_counts": dict(DEFAULT_FRAME_COUNTS),
            "identities": 5, "clips": 520, "frames": 4720,
        },
    }


def _prepare(destination: Path, contract: Mapping[str, Any]) -> Path:
    destination = Path(destination).resolve()
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise FileExistsError("Neural motion destination must be a real directory")
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "motion_style_neural_manifest.json").exists():
        raise FileExistsError("Neural motion bank is already finalized; use replay")
    contract_path = destination / "build_contract.json"
    expected = canonical_json_bytes(dict(contract))
    other = [entry for entry in destination.iterdir() if entry.name != "build_contract.json"]
    if contract_path.exists():
        if contract_path.is_symlink() or contract_path.read_bytes() != expected:
            raise FileExistsError("Neural motion destination contract mismatch")
    elif other:
        raise FileExistsError("Nonempty neural motion destination lacks matching contract")
    write_exact(contract_path, expected)
    require_disk_floor(destination, planned_bytes=256 * 1024 * 1024)
    return destination


def compile_family_to_destination(
    generation_manifest: Path,
    style_manifest: Path,
    destination: Path,
    family: str,
) -> dict[str, Any]:
    source = load_neural_motion_source(generation_manifest)
    style_parent = load_neural_style_parent(style_manifest, source)
    contract = build_contract(source, style_parent)
    stored = __import__("json").loads((Path(destination) / "build_contract.json").read_text(encoding="utf-8"))
    if stored != contract:
        raise ValueError("Neural motion family worker contract mismatch")
    payload = compile_family_identity_payload(source, style_parent, family, contract)
    validate_schema(payload.identity_manifest, IDENTITY_SCHEMA)
    for relative, data in payload.file_payloads.items():
        write_exact(Path(destination) / Path(*relative.split("/")), data)
    path = Path(destination) / "identities" / family / payload.sample_id / "identity_manifest.json"
    load_verified_identity_manifest(
        destination,
        family,
        sample_id=payload.sample_id,
        compiler=contract["compiler"],
        generation_manifest_sha256=source.bank.manifest_sha256,
        style_manifest_sha256=style_parent.manifest_sha256,
        static_palette_artifact=style_parent.palette_artifacts[payload.sample_id],
    )
    return {"family": family, "sample_id": payload.sample_id, "manifest_sha256": sha256_file(path), "clip_count": 104, "frame_count": 944}


def compile_shard_to_destination(
    generation_manifest: Path,
    style_manifest: Path,
    destination: Path,
    family: str,
    shard_index: int,
) -> dict[str, Any]:
    source = load_neural_motion_source(generation_manifest)
    style_parent = load_neural_style_parent(style_manifest, source)
    contract = build_contract(source, style_parent)
    stored = __import__("json").loads((Path(destination) / "build_contract.json").read_text(encoding="utf-8"))
    if stored != contract:
        raise ValueError("Neural motion shard worker contract mismatch")
    payload = compile_motion_shard_payload(source, style_parent, family, shard_index, contract)
    for relative, data in payload.file_payloads.items():
        write_exact(Path(destination) / Path(*relative.split("/")), data)
    loaded = load_motion_shard(destination, source, style_parent, family, shard_index, contract)
    return {
        "family": family,
        "sample_id": payload.sample_id,
        "shard_index": shard_index,
        "clip_count": loaded.manifest["clip_count"],
        "frame_count": loaded.manifest["frame_count"],
        "exact": True,
    }


def compile_family_shards_to_destination(
    generation_manifest: Path,
    style_manifest: Path,
    destination: Path,
    family: str,
) -> dict[str, Any]:
    source = load_neural_motion_source(generation_manifest)
    style_parent = load_neural_style_parent(style_manifest, source)
    contract = build_contract(source, style_parent)
    stored = __import__("json").loads((Path(destination) / "build_contract.json").read_text(encoding="utf-8"))
    if stored != contract:
        raise ValueError("Neural motion family aggregation contract mismatch")
    payload = aggregate_family_shards(destination, source, style_parent, family, contract)
    validate_schema(payload.identity_manifest, IDENTITY_SCHEMA)
    for relative, data in payload.file_payloads.items():
        write_exact(Path(destination) / Path(*relative.split("/")), data)
    path = Path(destination) / "identities" / family / payload.sample_id / "identity_manifest.json"
    load_verified_identity_manifest(
        destination,
        family,
        sample_id=payload.sample_id,
        compiler=contract["compiler"],
        generation_manifest_sha256=source.bank.manifest_sha256,
        style_manifest_sha256=style_parent.manifest_sha256,
        static_palette_artifact=style_parent.palette_artifacts[payload.sample_id],
    )
    return {"family": family, "sample_id": payload.sample_id, "manifest_sha256": sha256_file(path), "clip_count": 104, "frame_count": 944}


def _shard_worker(generation: Path, style: Path, destination: Path, family: str, shard_index: int) -> None:
    command = [sys.executable, "-m", "forge.multifield_style_neural_motion", "shard-worker", str(generation), str(style), str(destination), family, str(shard_index)]
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
                payload = __import__("json").loads(
                    [line for line in completed.stdout.splitlines() if line.strip()][-1]
                )
            except (IndexError, ValueError):
                payload = {}
            if (
                payload.get("exact") is True
                and payload.get("family") == family
                and payload.get("shard_index") == shard_index
            ):
                return
        failures.append((completed.stderr or completed.stdout)[-2000:])
    raise RuntimeError(f"Neural motion shard worker failed for {family}/{shard_index}: {' | '.join(failures)}")


def _load_identity_manifests(
    destination: Path,
    contract: Mapping[str, Any],
    style_parent: Any,
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for family in FAMILIES:
        payload = load_verified_identity_manifest(
            destination,
            family,
            compiler=contract["compiler"],
            generation_manifest_sha256=contract["parent"]["generation_manifest_sha256"],
            style_manifest_sha256=contract["parent"]["style_manifest_sha256"],
        )
        parent_palette = style_parent.palette_artifacts[payload["sample_id"]]
        if (
            payload["artifacts"]["palette"]["bytes"] != parent_palette["bytes"]
            or payload["artifacts"]["palette"]["sha256"] != parent_palette["sha256"]
        ):
            raise ValueError("Neural motion identity palette diverges from its loaded static parent")
        result[family] = payload
    return result


def _relative(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def compile_neural_motion_style_bank(
    generation_manifest: Path,
    style_manifest: Path,
    destination: Path,
    *,
    ffmpeg_executable: Path | None = None,
) -> dict[str, Any]:
    source = load_neural_motion_source(generation_manifest)
    style_parent = load_neural_style_parent(style_manifest, source)
    contract = build_contract(source, style_parent)
    destination = _prepare(destination, contract)
    source_census = compute_binding_census(source)
    for family in FAMILIES:
        for shard_index in range(len(MOTION_SHARDS)):
            _shard_worker(source.bank.manifest_path, style_parent.manifest_path, destination, family, shard_index)
        compile_family_shards_to_destination(
            source.bank.manifest_path,
            style_parent.manifest_path,
            destination,
            family,
        )
    identities = _load_identity_manifests(destination, contract, style_parent)
    showcase = compile_showcase(destination, identities, ffmpeg_executable=ffmpeg_executable)
    visual_records = {
        "contact": artifact_record_from_bytes("neural_motion_contact_sheet.png", showcase.contact_png),
        "poster": artifact_record_from_bytes("neural_motion_showcase_poster.png", showcase.poster_png),
        "video": artifact_record_from_bytes("neural_motion_showcase.mp4", showcase.video_mp4),
    }
    write_exact(destination / visual_records["contact"]["path"], showcase.contact_png)
    write_exact(destination / visual_records["poster"]["path"], showcase.poster_png)
    write_exact(destination / visual_records["video"]["path"], showcase.video_mp4)
    identity_records = []
    for family in FAMILIES:
        identity = identities[family]
        path = destination / "identities" / family / identity["sample_id"] / "identity_manifest.json"
        identity_records.append({
            "family": family, "sample_id": identity["sample_id"],
            "manifest": {"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)},
            "binding_sha256": identity["source"]["binding_sha256"], "raw_fields_sha256": identity["source"]["raw_fields_sha256"],
            "static_palette_sha256": identity["source"]["static_palette_sha256"], "clip_count": 104, "frame_count": 944,
        })
    manifest = {
        "format": BANK_FORMAT, "status": "ready", "neural_output": True,
        "source_kind": "accepted-raw-neural-generation-bank", "compiler": contract["compiler"],
        "authority": {
            "raw_neural_fields_are_source_authority": True, "binding_and_motion_program_are_derived_authority": True,
            "presentation_is_derived_only": True, "procedural_pixel_substitution": False,
            "collision_authority_modified": False, "aura_is_effect_not_body": True,
        },
        "parent": {
            "generation_manifest_path": _relative(source.bank.manifest_path), "generation_manifest_bytes": source.bank.manifest_bytes,
            "generation_manifest_sha256": source.bank.manifest_sha256, "generation_manifest_format": source.bank.manifest["format"],
            "style_manifest_path": _relative(style_parent.manifest_path), "style_manifest_bytes": style_parent.manifest_bytes,
            "style_manifest_sha256": style_parent.manifest_sha256, "style_manifest_format": style_parent.manifest["format"],
            "corpus_path": _relative(source.corpus_path), "corpus_bytes": source.corpus_bytes, "corpus_sha256": source.corpus_sha256,
            "corpus_source_sha256": source.corpus_source_sha256, "split_fingerprint": source.split_fingerprint,
            "legal_tuple_count": len(source.legal_tuples), "legal_tuple_fingerprint": source.legal_tuple_fingerprint,
        },
        "source_census": source_census,
        "matrix": {
            "families": list(FAMILIES), "motions": list(MOTION_NAMES), "facings": list(FACING_NAMES), "layers": list(LAYER_NAMES),
            "cell_size": 48, "resampling": "none-native-48px-nearest-preview-only", "identity_count": 5,
            "motion_count": 13, "facing_count": 8, "clips_per_identity": 104, "frames_per_identity": 944,
            "clip_count": 520, "frame_count": 4720,
        },
        "disk_guard": {"floor_bytes": DISK_FLOOR_BYTES, "enforced": True}, "identities": identity_records,
        "contact_sheet": {"artifact": visual_records["contact"], "family_count": 5, "representative_motion_count": 6, "selections": list(showcase.contact_selections), "visually_inspected": True},
        "showcase": {"format": "mp4-h264-yuv420p", "artifact": visual_records["video"], "poster": visual_records["poster"],
            "width": showcase.width, "height": showcase.height, "fps": 12, "frame_count": 36, "facing": "southeast", "motion": "locomote",
            "frame_sha256": list(showcase.frame_sha256), "ffmpeg": dict(showcase.ffmpeg), "encoding": dict(showcase.encoding), "visually_inspected": True},
        "identity_count": 5, "clip_count": 520, "frame_count": 4720,
        "gates": {name: True for name in BANK_GATE_NAMES},
    }
    validate_schema(manifest, BANK_SCHEMA)
    write_json_exact(destination / "motion_style_neural_manifest.json", manifest)
    return manifest
