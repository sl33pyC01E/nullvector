from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .disk_guard import guard_corpus_destination
from .genome import MorphologyGenome
from .motion import generate_motion_clip
from .motion_preview import HARD_DISK_FLOOR_BYTES
from .render import render_specimen


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_motion_bank(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "neural-morphology-motion-bank-v1":
        raise ValueError("Unsupported motion bank manifest format")
    archive_path = manifest_path.parent / payload["archive"]["file"]
    archive_sha256 = _sha256_file(archive_path)
    archive_hash_matches = archive_sha256 == payload["archive"]["sha256"]

    source_errors: list[dict[str, object]] = []
    specimens = {}
    for source in payload["sources"]:
        genome = MorphologyGenome.from_dict(source["genome"])
        specimen = render_specimen(genome)
        source_id = source["id"]
        specimens[source_id] = specimen
        if specimen.manifest != source:
            source_errors.append(
                {
                    "source_id": source_id,
                    "expected_semantic_sha256": source["hashes"]["semantic_sha256"],
                    "actual_semantic_sha256": specimen.manifest["hashes"]["semantic_sha256"],
                }
            )

    clip_manifest_mismatches: list[str] = []
    layer_mismatch_frames = 0
    token_mismatch_frames = 0
    joint_mismatch_frames = 0
    socket_mismatch_frames = 0
    hash_mismatch_frames = 0
    clips_replayed = 0
    frames_replayed = 0
    with np.load(archive_path, allow_pickle=False) as archive:
        offsets = archive["clip_offsets"]
        clip_ids = archive["clip_ids"].tolist()
        layer_names = archive["layer_names"].tolist()
        joint_names = archive["joint_names"].tolist()
        socket_names = archive["socket_names"].tolist()
        archive_shape_errors: list[str] = []
        if len(offsets) != len(payload["clips"]) + 1:
            archive_shape_errors.append("clip_offsets length is inconsistent")
        if clip_ids != [clip["id"] for clip in payload["clips"]]:
            archive_shape_errors.append("clip_ids order is inconsistent")
        if layer_names != payload["clips"][0]["layer_names"]:
            archive_shape_errors.append("layer_names order is inconsistent")

        for clip_index, expected in enumerate(payload["clips"]):
            specimen = specimens[expected["source_id"]]
            clip = generate_motion_clip(
                specimen,
                expected["motion"],
                facing=expected["facing"],
                frame_count=expected["frame_count"],
                fps=expected["fps"],
            )
            if clip.manifest != expected:
                clip_manifest_mismatches.append(expected["id"])
            start = int(offsets[clip_index])
            end = int(offsets[clip_index + 1])
            layers = np.stack([frame.layers for frame in clip.frames])
            tokens = np.stack([frame.tokens for frame in clip.frames])
            joints = np.asarray(
                [[frame.joints[name] for name in joint_names] for frame in clip.frames],
                dtype=np.uint8,
            )
            sockets = np.asarray(
                [[frame.sockets[name] for name in socket_names] for frame in clip.frames],
                dtype=np.uint8,
            )
            frame_hashes = np.asarray([frame.sha256 for frame in clip.frames])
            archived_layers = archive["layers"][start:end]
            archived_tokens = archive["tokens"][start:end]
            archived_joints = archive["joints"][start:end]
            archived_sockets = archive["sockets"][start:end]
            archived_hashes = archive["frame_sha256"][start:end]
            if archived_layers.shape != layers.shape:
                archive_shape_errors.append(
                    f"{expected['id']}: archived layers shape is inconsistent"
                )
                layer_mismatch_frames += len(clip.frames)
            else:
                layer_mismatch_frames += int(
                    np.any(archived_layers != layers, axis=(1, 2, 3)).sum()
                )
            if archived_tokens.shape != tokens.shape:
                archive_shape_errors.append(
                    f"{expected['id']}: archived tokens shape is inconsistent"
                )
                token_mismatch_frames += len(clip.frames)
            else:
                token_mismatch_frames += int(
                    np.any(archived_tokens != tokens, axis=(1, 2)).sum()
                )
            if archived_joints.shape != joints.shape:
                archive_shape_errors.append(
                    f"{expected['id']}: archived joints shape is inconsistent"
                )
                joint_mismatch_frames += len(clip.frames)
            else:
                joint_mismatch_frames += int(
                    np.any(archived_joints != joints, axis=(1, 2)).sum()
                )
            if archived_sockets.shape != sockets.shape:
                archive_shape_errors.append(
                    f"{expected['id']}: archived sockets shape is inconsistent"
                )
                socket_mismatch_frames += len(clip.frames)
            else:
                socket_mismatch_frames += int(
                    np.any(archived_sockets != sockets, axis=(1, 2)).sum()
                )
            if archived_hashes.shape != frame_hashes.shape:
                archive_shape_errors.append(
                    f"{expected['id']}: archived frame-hash shape is inconsistent"
                )
                hash_mismatch_frames += len(clip.frames)
            else:
                hash_mismatch_frames += int(
                    np.count_nonzero(archived_hashes != frame_hashes)
                )
            clips_replayed += 1
            frames_replayed += len(clip.frames)

    valid = not any(
        (
            not archive_hash_matches,
            source_errors,
            archive_shape_errors,
            clip_manifest_mismatches,
            layer_mismatch_frames,
            token_mismatch_frames,
            joint_mismatch_frames,
            socket_mismatch_frames,
            hash_mismatch_frames,
        )
    )
    return {
        "format": "neural-morphology-motion-replay-v1",
        "valid": valid,
        "manifest": str(manifest_path),
        "archive": str(archive_path),
        "archive_hash_matches": archive_hash_matches,
        "sources_replayed": len(specimens),
        "clips_replayed": clips_replayed,
        "frames_replayed": frames_replayed,
        "source_errors": source_errors,
        "archive_shape_errors": archive_shape_errors,
        "clip_manifest_mismatches": clip_manifest_mismatches,
        "layer_mismatch_frames": layer_mismatch_frames,
        "token_mismatch_frames": token_mismatch_frames,
        "joint_mismatch_frames": joint_mismatch_frames,
        "socket_mismatch_frames": socket_mismatch_frames,
        "hash_mismatch_frames": hash_mismatch_frames,
    }


def write_replay_report(report: dict[str, Any], destination: Path) -> None:
    destination = Path(destination).resolve()
    guard_corpus_destination(
        destination,
        int(report["frames_replayed"]),
        reserve_bytes=HARD_DISK_FLOOR_BYTES,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Replay and byte-compare a morphology motion semantic bank."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root
        / "outputs"
        / "morphology_motion"
        / "morphology_motion_manifest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root
        / "outputs"
        / "morphology_motion"
        / "motion_replay_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = replay_motion_bank(args.manifest)
    write_replay_report(report, args.report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
