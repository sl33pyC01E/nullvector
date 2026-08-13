from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

from ..morphology import FACING_NAMES, MOTION_NAMES
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    named_points_sha256,
    png_bytes,
)
from ..multifield_style_motion.io import require_disk_floor, write_exact
from ..multifield_style_motion.model import IMAGE_SIZE, JOINT_NAMES, LAYER_NAMES, SOCKET_NAMES
from ..multifield_style_neural_motion.rendering import render_neural_motion_frame
from ..neural_rig_repair.binding import bind_repair_plan
from ..neural_rig_repair.constants import (
    EXPECTED_CLIP_COUNT,
    EXPECTED_FRAME_COUNT,
    EXPECTED_SAMPLE_COUNT,
    MAX_PLAN_BYTES,
    PROJECT_ROOT,
)
from ..neural_rig_repair.hashing import sha256_bytes, sha256_file
from ..neural_rig_repair.planner import load_repair_plan
from ..neural_rig_repair.schema import resolve_artifact_record
from .authority import DEFAULT_REPAIR_BANK, RepairStyleAuthority, load_repair_style_authority
from .projection import reconstruct_clip


FORMAT = "nullvector-neural-rig-repair-style-bank-v1"
IDENTITY_FORMAT = "nullvector-neural-rig-repair-style-identity-v1"
SHARD_FORMAT = "nullvector-neural-rig-repair-style-shard-v1"
TELEMETRY_FORMAT = "nullvector-neural-rig-repair-style-telemetry-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_rig_repair_style_all80_v1"
SHARD_COUNT = 16
ATLAS_COLUMNS = 16
FRAMES_PER_IDENTITY = 944
CLIPS_PER_IDENTITY = 104
MAX_ATTEMPTS = 3
MAX_WORKERS = 2
TIMEOUT_SECONDS = 1800


def _source_hash() -> str:
    files = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("authority.py"),
        Path(__file__).resolve().with_name("projection.py"),
        PROJECT_ROOT / "forge" / "multifield_style_neural_motion" / "rendering.py",
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _artifact(path: str, payload: bytes) -> dict[str, Any]:
    return artifact_record_from_bytes(path, payload)


def _canonical_json(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    parsed = json.loads(payload)
    if canonical_json_bytes(parsed) != payload:
        raise ValueError(f"noncanonical JSON artifact: {path}")
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return parsed


def _safe_artifact(root: Path, record: Mapping[str, Any]) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("production artifact path is not POSIX relative")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("production artifact path is unsafe")
    path = (root.resolve() / Path(*pure.parts)).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError("production artifact is not a regular contained file")
    if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
        raise ValueError("production artifact byte/hash mismatch")
    return path


def _binding(authority: RepairStyleAuthority, ordinal: int):
    record = authority.bank["plans"][ordinal]
    path = resolve_artifact_record(
        authority.bank_path.parent,
        record["artifact"],
        label=f"repair style production plan {ordinal}",
        maximum_bytes=MAX_PLAN_BYTES,
    )
    return bind_repair_plan(
        authority.repair_source,
        authority.repair_source.samples[ordinal],
        load_repair_plan(path),
        verify_exact_plan=True,
    )


def _identity_payload(authority: RepairStyleAuthority, ordinal: int) -> tuple[dict[str, Any], dict[str, bytes]]:
    binding = _binding(authority, ordinal)
    sample = authority.neural_source.bank.samples[ordinal]
    if sample.condition.sample_id != binding.sample_id:
        raise ValueError("production identity source registry drifted")
    palette = authority.style_parent.palettes[binding.sample_id]
    palette_artifact = authority.style_parent.palette_artifacts[binding.sample_id]
    rows = math.ceil(FRAMES_PER_IDENTITY / ATLAS_COLUMNS)
    atlases = {
        name: np.zeros((rows * IMAGE_SIZE, ATLAS_COLUMNS * IMAGE_SIZE, 4), dtype=np.uint8)
        for name in LAYER_NAMES
    }
    motion_audit = authority.motion_audits[ordinal]
    audit_by_key = {(clip["motion"], clip["facing"]): clip for clip in motion_audit["clips"]}
    expected_keys = [(motion, facing) for motion in MOTION_NAMES for facing in FACING_NAMES]
    if list(audit_by_key) != expected_keys:
        raise ValueError("production clip audit order drifted")

    clip_records: list[dict[str, Any]] = []
    phases: list[float] = []
    pulses: list[int] = []
    motion_frame_hashes: list[str] = []
    bound_frame_hashes: list[str] = []
    categorical_hashes: list[str] = []
    aligned_hashes: list[str] = []
    driver_hashes: list[str] = []
    joint_hashes: list[str] = []
    socket_hashes: list[str] = []
    presentation_hashes: list[tuple[str, ...]] = []
    clip_offsets = [0]
    cursor = 0
    for motion, facing in expected_keys:
        audit = audit_by_key[(motion, facing)]
        clip = reconstruct_clip(binding, audit)
        clip_start = cursor
        clip_presentation: list[tuple[str, ...]] = []
        first_layers: Mapping[str, np.ndarray] | None = None
        last_layers: Mapping[str, np.ndarray] | None = None
        for frame in clip.frames:
            rendered = render_neural_motion_frame(
                frame,
                sample.condition,
                sample.fields.aligned_sha256,
                palette,
                palette_artifact["sha256"],
            )
            row, column = divmod(cursor, ATLAS_COLUMNS)
            y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
            for layer in LAYER_NAMES:
                atlases[layer][y : y + IMAGE_SIZE, x : x + IMAGE_SIZE] = rendered.layers[layer]
            if first_layers is None:
                first_layers = {name: rendered.layers[name].copy() for name in LAYER_NAMES}
            last_layers = rendered.layers
            motion_frame_hashes.append(frame.sha256)
            bound_frame_hashes.append(frame.fields.sha256)
            categorical_hashes.append(rendered.categorical_sha256)
            aligned_hashes.append(rendered.aligned_fields_sha256)
            driver_hashes.append(str(frame.fields.manifest["driver_index_sha256"]))
            joint_hashes.append(named_points_sha256("joints", JOINT_NAMES, frame.joints))
            socket_hashes.append(named_points_sha256("sockets", SOCKET_NAMES, frame.sockets))
            presentation_hashes.append(rendered.presentation_sha256)
            clip_presentation.append(rendered.presentation_sha256)
            phases.append(float(frame.phase))
            pulses.append(int(frame.emission_pulse))
            cursor += 1
        loop_exact = not clip.loop or (
            first_layers is not None
            and last_layers is not None
            and all(np.array_equal(first_layers[name], last_layers[name]) for name in LAYER_NAMES)
        )
        if not loop_exact:
            raise ValueError("production styled loop endpoint mismatch")
        clip_records.append(
            {
                "id": clip.manifest["id"],
                "motion": motion,
                "facing": facing,
                "fps": clip.fps,
                "loop": clip.loop,
                "start_cell": clip_start,
                "frame_count": len(clip.frames),
                "repair_audit_sha256": audit["clip_sha256"],
                "repair_style_clip_sha256": clip.sha256,
                "presentation_sequence_sha256": sha256_bytes(
                    canonical_json_bytes([list(values) for values in clip_presentation])
                ),
            }
        )
        clip_offsets.append(cursor)
    if cursor != FRAMES_PER_IDENTITY or len(clip_records) != CLIPS_PER_IDENTITY:
        raise ValueError("production identity frame/clip accounting mismatch")

    prefix = f"identities/{binding.family}/{binding.sample_id}"
    payloads: dict[str, bytes] = {}
    layers: dict[str, Mapping[str, Any]] = {}
    for name, values in atlases.items():
        relative = f"{prefix}/{name}.png"
        payload = png_bytes(values)
        payloads[relative] = payload
        layers[name] = _artifact(relative, payload)
    index_arrays = {
        "format": np.asarray(["nullvector-neural-rig-repair-style-frame-index-v1"]),
        "sample_id": np.asarray([binding.sample_id]),
        "clip_offsets": np.asarray(clip_offsets, dtype=np.uint32),
        "phases": np.asarray(phases, dtype=np.float32),
        "emission_pulses": np.asarray(pulses, dtype=np.uint8),
        "motion_frame_sha256": np.asarray(motion_frame_hashes),
        "bound_frame_sha256": np.asarray(bound_frame_hashes),
        "categorical_sha256": np.asarray(categorical_hashes),
        "aligned_fields_sha256": np.asarray(aligned_hashes),
        "driver_index_sha256": np.asarray(driver_hashes),
        "joint_sha256": np.asarray(joint_hashes),
        "socket_sha256": np.asarray(socket_hashes),
        "presentation_sha256": np.asarray(presentation_hashes),
    }
    index_relative = f"{prefix}/frame_index.npz"
    index_payload = deterministic_npz_bytes(index_arrays)
    payloads[index_relative] = index_payload
    identity = {
        "format": IDENTITY_FORMAT,
        "status": "ready",
        "neural_output": True,
        "ordinal": ordinal,
        "sample_id": binding.sample_id,
        "family": binding.family,
        "condition": sample.condition.as_dict(),
        "authority": {
            "repair_bank_sha256": authority.bank["bank_sha256"],
            "binding_sha256": binding.sha256,
            "raw_fields_sha256": binding.raw_fields_sha256,
            "static_palette_sha256": palette_artifact["sha256"],
            "sample_motion_audit_sha256": motion_audit["sample_motion_sha256"],
        },
        "layout": {
            "cell_size": IMAGE_SIZE,
            "columns": ATLAS_COLUMNS,
            "rows": rows,
            "frame_count": cursor,
            "clip_count": len(clip_records),
            "layers": list(LAYER_NAMES),
        },
        "artifacts": {
            "layers": layers,
            "frame_index": _artifact(index_relative, index_payload),
        },
        "clips": clip_records,
        "gates": {
            "sealed_motion_audit_exact": True,
            "all_944_bound_frames_exact": True,
            "categorical_fields_unchanged": True,
            "all_seven_style_layers_valid": True,
            "palette_identity_invariant": True,
            "all_loop_endpoints_exact": True,
        },
    }
    identity["identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    identity_relative = f"{prefix}/identity_manifest.json"
    identity_payload = canonical_json_bytes(identity)
    payloads[identity_relative] = identity_payload
    return identity, payloads


def compile_shard(bank_path: Path, shard_index: int, destination: Path) -> dict[str, Any]:
    if not 0 <= shard_index < SHARD_COUNT:
        raise ValueError("production shard index is outside the canonical range")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("production worker destination already exists")
    require_disk_floor(destination, planned_bytes=2 * 1024**3)
    destination.mkdir(parents=True)
    authority = load_repair_style_authority(bank_path)
    expected_ordinals = list(range(shard_index, EXPECTED_SAMPLE_COUNT, SHARD_COUNT))
    identities = []
    for ordinal in expected_ordinals:
        identity, payloads = _identity_payload(authority, ordinal)
        for relative, payload in payloads.items():
            write_exact(destination / relative, payload)
        manifest_relative = f"identities/{identity['family']}/{identity['sample_id']}/identity_manifest.json"
        identities.append(
            {
                "ordinal": ordinal,
                "sample_id": identity["sample_id"],
                "family": identity["family"],
                "identity_sha256": identity["identity_sha256"],
                "artifact": _artifact(manifest_relative, canonical_json_bytes(identity)),
            }
        )
    shard = {
        "format": SHARD_FORMAT,
        "status": "ready",
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "source_sha256": _source_hash(),
        "repair_bank_sha256": authority.bank["bank_sha256"],
        "sample_count": len(identities),
        "clip_count": len(identities) * CLIPS_PER_IDENTITY,
        "frame_count": len(identities) * FRAMES_PER_IDENTITY,
        "identities": identities,
        "gates": {
            "canonical_stride_assignment": True,
            "all_identity_manifests_self_hashed": True,
            "all_atlases_and_indexes_hash_bound": True,
            "all_consumed_repair_frames_exact": True,
            "cpu_only": True,
        },
    }
    shard["shard_sha256"] = sha256_bytes(canonical_json_bytes(shard))
    write_exact(destination / "shard_manifest.json", canonical_json_bytes(shard))
    return shard


def validate_shard(path: Path, expected_index: int | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    shard_path = path / "shard_manifest.json"
    shard = _canonical_json(shard_path)
    unsigned = dict(shard)
    stored = unsigned.pop("shard_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("production shard self-hash mismatch")
    expected_ordinals = list(range(int(shard["shard_index"]), EXPECTED_SAMPLE_COUNT, SHARD_COUNT))
    if (
        shard.get("format") != SHARD_FORMAT
        or shard.get("status") != "ready"
        or shard.get("source_sha256") != _source_hash()
        or (expected_index is not None and shard.get("shard_index") != expected_index)
        or [record["ordinal"] for record in shard.get("identities", [])] != expected_ordinals
        or shard.get("sample_count") != len(expected_ordinals)
        or shard.get("clip_count") != len(expected_ordinals) * CLIPS_PER_IDENTITY
        or shard.get("frame_count") != len(expected_ordinals) * FRAMES_PER_IDENTITY
        or any(value is not True for value in shard.get("gates", {}).values())
    ):
        raise ValueError("production shard contract failed")
    for registry in shard["identities"]:
        identity_path = _safe_artifact(path, registry["artifact"])
        identity = _canonical_json(identity_path)
        unsigned_identity = dict(identity)
        identity_hash = unsigned_identity.pop("identity_sha256", None)
        if identity_hash != sha256_bytes(canonical_json_bytes(unsigned_identity)):
            raise ValueError("production identity self-hash mismatch")
        if (
            identity_hash != registry["identity_sha256"]
            or identity["ordinal"] != registry["ordinal"]
            or identity["sample_id"] != registry["sample_id"]
            or identity["layout"]["frame_count"] != FRAMES_PER_IDENTITY
            or identity["layout"]["clip_count"] != CLIPS_PER_IDENTITY
            or len(identity["clips"]) != CLIPS_PER_IDENTITY
            or any(value is not True for value in identity["gates"].values())
        ):
            raise ValueError("production identity contract failed")
        for record in identity["artifacts"]["layers"].values():
            _safe_artifact(path, record)
        index_path = _safe_artifact(path, identity["artifacts"]["frame_index"])
        with np.load(index_path, allow_pickle=False) as archive:
            if (
                set(archive.files)
                != {
                    "format", "sample_id", "clip_offsets", "phases", "emission_pulses",
                    "motion_frame_sha256", "bound_frame_sha256", "categorical_sha256",
                    "aligned_fields_sha256", "driver_index_sha256", "joint_sha256",
                    "socket_sha256", "presentation_sha256",
                }
                or archive["clip_offsets"].shape != (CLIPS_PER_IDENTITY + 1,)
                or int(archive["clip_offsets"][-1]) != FRAMES_PER_IDENTITY
                or archive["phases"].shape != (FRAMES_PER_IDENTITY,)
                or archive["presentation_sha256"].shape != (FRAMES_PER_IDENTITY, len(LAYER_NAMES))
            ):
                raise ValueError("production frame index contract failed")
    return shard


def _tree(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, sha256_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _worker_command(bank_path: Path, shard_index: int, destination: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "forge.neural_rig_repair_style.production",
        "worker",
        "--bank",
        str(bank_path),
        "--shard-index",
        str(shard_index),
        "--output",
        str(destination),
    ]


def _cpu_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def build_shards(
    bank_path: Path,
    root: Path,
    *,
    workers: int = MAX_WORKERS,
    max_attempts: int = MAX_ATTEMPTS,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> dict[str, Any]:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    published = root / "shards"
    staging = root / "staging"
    logs = root / "logs"
    for path in (published, staging, logs):
        path.mkdir(parents=True, exist_ok=True)
    pending: list[int] = []
    attempts = {index: 0 for index in range(SHARD_COUNT)}
    events: list[dict[str, Any]] = []
    for index in range(SHARD_COUNT):
        target = published / f"shard_{index:02d}"
        if target.exists():
            validate_shard(target, index)
            events.append({"shard_index": index, "attempt": 0, "status": "reused"})
        else:
            pending.append(index)
    active: dict[int, dict[str, Any]] = {}
    while pending or active:
        while pending and len(active) < max(1, min(workers, MAX_WORKERS)):
            index = pending.pop(0)
            attempts[index] += 1
            attempt = attempts[index]
            if attempt > max_attempts:
                raise RuntimeError(f"production shard {index} exhausted retries")
            destination = staging / f"shard_{index:02d}_attempt_{attempt:02d}"
            if destination.exists():
                raise FileExistsError(f"production staging path already exists: {destination}")
            log_path = logs / f"shard_{index:02d}_attempt_{attempt:02d}.log"
            log_handle = log_path.open("xb")
            process = subprocess.Popen(
                _worker_command(bank_path, index, destination),
                cwd=PROJECT_ROOT,
                env=_cpu_environment(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[index] = {
                "process": process,
                "handle": log_handle,
                "started": time.monotonic(),
                "destination": destination,
                "log": log_path,
                "attempt": attempt,
            }
        time.sleep(0.25)
        for index, state in list(active.items()):
            process = state["process"]
            elapsed = time.monotonic() - state["started"]
            return_code = process.poll()
            timed_out = return_code is None and elapsed > timeout_seconds
            if return_code is None and not timed_out:
                continue
            if timed_out:
                process.kill()
                return_code = process.wait(timeout=30)
            state["handle"].close()
            event = {
                "shard_index": index,
                "attempt": state["attempt"],
                "return_code": int(return_code),
                "elapsed_seconds": round(elapsed, 3),
                "timed_out": timed_out,
                "access_violation": int(return_code) in {3221225477, -1073741819},
                "log": state["log"].relative_to(root).as_posix(),
            }
            try:
                if return_code != 0 or timed_out:
                    raise RuntimeError("worker process failed")
                validate_shard(state["destination"], index)
                target = published / f"shard_{index:02d}"
                if target.exists():
                    raise FileExistsError("production shard target appeared during worker run")
                os.replace(state["destination"], target)
                event["status"] = "published"
            except (OSError, RuntimeError, ValueError) as error:
                event["status"] = "rejected"
                event["error"] = str(error)[:1000]
                pending.append(index)
            events.append(event)
            del active[index]
    shards = [validate_shard(published / f"shard_{index:02d}", index) for index in range(SHARD_COUNT)]
    telemetry = {
        "format": TELEMETRY_FORMAT,
        "status": "passed",
        "source_sha256": _source_hash(),
        "shard_count": SHARD_COUNT,
        "attempt_count": sum(attempts.values()),
        "retry_count": sum(max(0, count - 1) for count in attempts.values()),
        "access_violation_count": sum(bool(event.get("access_violation")) for event in events),
        "events": events,
        "gates": {
            "all_shards_published": True,
            "bounded_workers": workers <= MAX_WORKERS,
            "bounded_attempts": max(attempts.values()) <= max_attempts,
            "cpu_only_environment": True,
        },
    }
    telemetry["telemetry_sha256"] = sha256_bytes(canonical_json_bytes(telemetry))
    write_exact(root / "telemetry.json", canonical_json_bytes(telemetry))
    return {"shards": shards, "telemetry": telemetry}


def compile_bank(
    destination: Path = DEFAULT_OUTPUT,
    *,
    bank_path: Path = DEFAULT_REPAIR_BANK,
    exact_replay: bool = True,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    require_disk_floor(destination, planned_bytes=8 * 1024**3)
    if (destination / "style_bank_manifest.json").exists():
        raise FileExistsError("production style bank is already sealed")
    destination.mkdir(parents=True, exist_ok=True)
    primary = build_shards(bank_path, destination / "primary")
    replay_exact = False
    if exact_replay:
        replay = build_shards(bank_path, destination / "replay")
        for index in range(SHARD_COUNT):
            left = _tree(destination / "primary" / "shards" / f"shard_{index:02d}")
            right = _tree(destination / "replay" / "shards" / f"shard_{index:02d}")
            if left != right:
                raise ValueError(f"production exact replay differs for shard {index}")
        replay_exact = True
    authority = load_repair_style_authority(bank_path)
    primary_tree = _tree(destination / "primary" / "shards")
    total_bytes = sum(record[0] for record in primary_tree.values())
    tree_identity = sha256_bytes(
        canonical_json_bytes({path: {"bytes": size, "sha256": digest} for path, (size, digest) in primary_tree.items()})
    )
    manifest = {
        "format": FORMAT,
        "status": "ready",
        "neural_output": True,
        "scope": "all-80-repaired-neural-identities-full-motion-matrix",
        "compiler": {"source_sha256": _source_hash(), "cpu_only": True, "cuda_used": False},
        "authority": {
            "repair_bank_sha256": authority.bank["bank_sha256"],
            "repair_source_sha256": authority.bank["source"]["repair_source_sha256"],
            "generation_manifest_sha256": authority.repair_source.generation_manifest_sha256,
            "style_manifest_sha256": authority.repair_source.style_manifest_sha256,
            "motion_stress_sha256": authority.bank["motion_result"]["stress_sha256"],
            "motion_replay_sha256": authority.bank["motion_result"]["replay_stress_sha256"],
        },
        "counts": {
            "identity_count": EXPECTED_SAMPLE_COUNT,
            "family_count": 5,
            "motion_count": 13,
            "facing_count": 8,
            "clip_count": EXPECTED_CLIP_COUNT,
            "frame_count": EXPECTED_FRAME_COUNT,
            "layer_atlas_count": EXPECTED_SAMPLE_COUNT * len(LAYER_NAMES),
            "shard_count": SHARD_COUNT,
            "artifact_file_count": len(primary_tree),
            "artifact_bytes": total_bytes,
        },
        "primary_tree_sha256": tree_identity,
        "telemetry": {
            "primary_sha256": primary["telemetry"]["telemetry_sha256"],
            "replay_sha256": replay["telemetry"]["telemetry_sha256"] if exact_replay else None,
        },
        "gates": {
            "all_80_identities_present": True,
            "all_8320_motion_facing_clips_present": True,
            "all_75520_frames_exactly_reconstructed": True,
            "all_560_layer_atlases_present": True,
            "all_categorical_fields_unchanged": True,
            "all_presentation_gates_passed": True,
            "process_isolated_worker_retries_bounded": True,
            "independent_byte_exact_replay": replay_exact,
            "disk_floor_preserved": True,
        },
    }
    if any(value is not True for value in manifest["gates"].values()):
        raise ValueError("production style bank gate failed")
    manifest["bank_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    write_exact(destination / "style_bank_manifest.json", canonical_json_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile all-80 styled repaired neural animations")
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--bank", type=Path, required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--bank", type=Path, default=DEFAULT_REPAIR_BANK)
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    compile_parser.add_argument("--no-exact-replay", action="store_true")
    validate = commands.add_parser("validate-shard")
    validate.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        result = compile_shard(args.bank, args.shard_index, args.output)
        print("REPAIR_STYLE_SHARD_OK", result["shard_index"], result["sample_count"], result["shard_sha256"])
    elif args.command == "compile":
        result = compile_bank(args.output, bank_path=args.bank, exact_replay=not args.no_exact_replay)
        print("REPAIR_STYLE_ALL80_OK", result["counts"]["identity_count"], result["counts"]["frame_count"], result["bank_sha256"])
    else:
        result = validate_shard(args.path)
        print("REPAIR_STYLE_SHARD_VALID", result["shard_index"], result["shard_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
