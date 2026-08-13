from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from ..morphology import FACING_NAMES, FAMILIES, MOTION_NAMES
from ..morphology.motion import DEFAULT_FRAME_COUNTS
from ..multifield_style_motion.hashing import (
    array_sha256,
    artifact_record_from_bytes,
    canonical_json_bytes,
    clip_presentation_sha256,
    deterministic_npz_bytes,
    named_points_sha256,
)
from ..multifield_style_motion.io import verify_artifact
from ..multifield_style_motion.model import (
    ATLAS_COLUMNS,
    IMAGE_SIZE,
    JOINT_NAMES,
    LAYER_NAMES,
    SOCKET_NAMES,
)
from ..neural_rig_bridge import compile_neural_motion_clip
from ..neural_rig_bridge.hashing import binder_source_hash, canonical_json_hash
from .family import CompiledIdentityFrames, finalize_identity_payload
from .model import NeuralIdentityPayload, NeuralMotionSource, NeuralStyleParent
from .rendering import render_neural_motion_frame
from .source import bind_candidate


SHARD_FORMAT = "nullvector-multifield-style-neural-motion-shard-v1"
SHARD_ARRAY_FORMAT = "nullvector-multifield-style-neural-motion-shard-arrays-v1"
MOTION_SHARDS = (
    ("idle_breathe", "idle_wiggle", "locomote"),
    ("joy", "anger", "fear"),
    ("confused", "sleep", "taunt", "attack"),
    ("cast", "hit", "death"),
)
if tuple(motion for shard in MOTION_SHARDS for motion in shard) != tuple(MOTION_NAMES):
    raise RuntimeError("Neural motion shard partition drifted from the motion vocabulary")

HASH_VECTOR_NAMES = (
    "motion_frame_sha256",
    "bound_frame_sha256",
    "categorical_sha256",
    "aligned_fields_sha256",
    "driver_index_sha256",
    "joint_sha256",
    "socket_sha256",
)
SHARD_ARRAY_KEYS = frozenset(
    {
        "format",
        "phases",
        "emission_pulses",
        "presentation_sha256",
        *HASH_VECTOR_NAMES,
        *(f"layer_{name}" for name in LAYER_NAMES),
    }
)
MAX_SHARD_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_SHARD_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class MotionShardPayload:
    family: str
    sample_id: str
    shard_index: int
    manifest: Mapping[str, Any]
    file_payloads: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class LoadedMotionShard:
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


def _shard_frame_count(shard_index: int) -> int:
    return sum(DEFAULT_FRAME_COUNTS[motion] * len(FACING_NAMES) for motion in MOTION_SHARDS[shard_index])


def _shard_start(shard_index: int) -> int:
    return sum(_shard_frame_count(index) for index in range(shard_index))


def _paths(family: str, sample_id: str, shard_index: int) -> tuple[str, str]:
    prefix = f"_build_shards/{family}/{sample_id}/shard_{shard_index:02d}"
    return f"{prefix}/arrays.npz", f"{prefix}/shard_manifest.json"


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - HEX)


def compile_motion_shard_payload(
    source: NeuralMotionSource,
    style_parent: NeuralStyleParent,
    family: str,
    shard_index: int,
    build_contract: Mapping[str, Any],
) -> MotionShardPayload:
    if family not in FAMILIES:
        raise ValueError(f"Unknown neural motion shard family: {family!r}")
    if not 0 <= shard_index < len(MOTION_SHARDS):
        raise ValueError("Neural motion shard index is outside the canonical partition")
    if binder_source_hash() != build_contract["compiler"]["bridge_source_sha256"]:
        raise ValueError("Neural bridge source changed during shard compilation")
    candidate = source.candidates_by_family[family][0]
    selected = bind_candidate(source, candidate)
    binding = selected.binding
    sample = candidate.sample
    sample_id = binding.sample_id
    palette = style_parent.palettes[sample_id]
    palette_artifact = style_parent.palette_artifacts[sample_id]
    global_cursor = _shard_start(shard_index)
    expected_frames = _shard_frame_count(shard_index)
    layer_frames: dict[str, list[np.ndarray]] = {name: [] for name in LAYER_NAMES}
    vectors: dict[str, list[str]] = {name: [] for name in HASH_VECTOR_NAMES}
    presentations: list[tuple[str, ...]] = []
    phases: list[float] = []
    pulses: list[int] = []
    clip_records: list[dict[str, Any]] = []
    source_clips: list[dict[str, Any]] = []
    for motion in MOTION_SHARDS[shard_index]:
        for facing in FACING_NAMES:
            clip = compile_neural_motion_clip(binding, motion, facing=facing)
            if (
                clip.manifest["metrics"]["all_source_tuples_preserved"] is not True
                or clip.manifest["metrics"]["procedural_pixel_substitution"] is not False
                or clip.manifest["binder_source_sha256"] != build_contract["compiler"]["bridge_source_sha256"]
            ):
                raise ValueError(f"Neural shard source authority failed: {clip.manifest['id']}")
            clip_start = global_cursor
            clip_layers: list[Mapping[str, np.ndarray]] = []
            clip_categorical: list[str] = []
            clip_motion_authority: list[str] = []
            clip_presentations: list[tuple[str, ...]] = []
            for frame in clip.frames:
                rendered = render_neural_motion_frame(
                    frame,
                    sample.condition,
                    sample.fields.aligned_sha256,
                    palette,
                    palette_artifact["sha256"],
                )
                for name in LAYER_NAMES:
                    layer_frames[name].append(rendered.layers[name])
                vectors["motion_frame_sha256"].append(frame.sha256)
                vectors["bound_frame_sha256"].append(frame.fields.sha256)
                vectors["categorical_sha256"].append(rendered.categorical_sha256)
                vectors["aligned_fields_sha256"].append(rendered.aligned_fields_sha256)
                vectors["driver_index_sha256"].append(str(frame.fields.manifest["driver_index_sha256"]))
                vectors["joint_sha256"].append(named_points_sha256("joints", JOINT_NAMES, frame.joints))
                vectors["socket_sha256"].append(named_points_sha256("sockets", SOCKET_NAMES, frame.sockets))
                presentations.append(rendered.presentation_sha256)
                phases.append(float(frame.phase))
                pulses.append(int(frame.emission_pulse))
                clip_layers.append(rendered.layers)
                clip_categorical.append(rendered.categorical_sha256)
                clip_motion_authority.append(frame.sha256)
                clip_presentations.append(rendered.presentation_sha256)
                global_cursor += 1
            loop_exact = True
            if clip.loop:
                loop_exact = (
                    clip_categorical[0] == clip_categorical[-1]
                    and clip.frames[0].fields.sha256 == clip.frames[-1].fields.sha256
                    and clip_presentations[0] == clip_presentations[-1]
                    and clip.frames[0].joints == clip.frames[-1].joints
                    and clip.frames[0].sockets == clip.frames[-1].sockets
                    and all(
                        np.array_equal(clip_layers[0][name], clip_layers[-1][name])
                        for name in LAYER_NAMES
                    )
                )
            if not loop_exact:
                raise ValueError(f"Neural shard loop endpoint mismatch: {clip.manifest['id']}")
            derived_sha = clip_presentation_sha256(
                identity_sha256=sample.fields.aligned_sha256,
                source_clip_sha256=clip.sha256,
                events=list(clip.manifest["events"]),
                categorical_hashes=clip_categorical,
                authority_hashes=clip_motion_authority,
                presentation_hashes=np.asarray(clip_presentations, dtype="<U64"),
            )
            clip_records.append(
                {
                    "id": clip.manifest["id"],
                    "motion": clip.motion,
                    "facing": clip.facing,
                    "fps": clip.fps,
                    "loop": clip.loop,
                    "frame_count": len(clip.frames),
                    "start_cell": clip_start,
                    "source_clip_sha256": clip.sha256,
                    "derived_clip_sha256": derived_sha,
                    "events": list(clip.manifest["events"]),
                    "gates": {
                        "source_motion_valid": True,
                        "events_preserved": True,
                        "palette_identity_invariant": True,
                        "loop_endpoints_exact": loop_exact,
                        "categorical_and_anchor_authority_unchanged": True,
                        "outline_and_bloom_radius_exact": True,
                    },
                }
            )
            source_clips.append(dict(clip.manifest))
    if len(phases) != expected_frames or global_cursor != _shard_start(shard_index) + expected_frames:
        raise RuntimeError("Neural motion shard frame accounting failed")
    arrays: dict[str, np.ndarray] = {
        "format": np.asarray([SHARD_ARRAY_FORMAT]),
        "phases": np.asarray(phases, dtype=np.float32),
        "emission_pulses": np.asarray(pulses, dtype=np.uint8),
        "presentation_sha256": np.asarray(presentations),
    }
    arrays.update({name: np.asarray(values) for name, values in vectors.items()})
    arrays.update(
        {
            f"layer_{name}": np.ascontiguousarray(np.stack(values), dtype=np.uint8)
            for name, values in layer_frames.items()
        }
    )
    arrays_bytes = deterministic_npz_bytes(arrays)
    arrays_relative, manifest_relative = _paths(family, sample_id, shard_index)
    manifest = {
        "format": SHARD_FORMAT,
        "family": family,
        "sample_id": sample_id,
        "condition": sample.condition.as_dict(),
        "compiler": dict(build_contract["compiler"]),
        "parent": {
            "generation_manifest_sha256": source.bank.manifest_sha256,
            "style_manifest_sha256": style_parent.manifest_sha256,
        },
        "selection": {"candidate_ordinal_within_family": 0},
        "shard_index": shard_index,
        "motions": list(MOTION_SHARDS[shard_index]),
        "global_start_cell": _shard_start(shard_index),
        "clip_count": len(clip_records),
        "frame_count": expected_frames,
        "source": {
            "raw_fields_sha256": sample.raw_fields_sha256,
            "compiled_fields_sha256": sample.fields.aligned_sha256,
            "binding_sha256": binding.sha256,
            "static_palette_sha256": palette_artifact["sha256"],
        },
        "arrays": artifact_record_from_bytes(arrays_relative, arrays_bytes),
        "clips": clip_records,
        "source_clips": source_clips,
        "gates": {
            "validating_motion_constructor_used": True,
            "categorical_authority_preserved": True,
            "rig_socket_authority_preserved": True,
            "palette_identity_invariant": True,
            "loop_endpoints_exact": True,
            "presentation_bounds_exact": True,
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    return MotionShardPayload(
        family=family,
        sample_id=sample_id,
        shard_index=shard_index,
        manifest=manifest,
        file_payloads={arrays_relative: arrays_bytes, manifest_relative: manifest_bytes},
    )


def _load_arrays(path: Path, frame_count: int) -> dict[str, np.ndarray]:
    if path.stat().st_size > MAX_SHARD_COMPRESSED_BYTES:
        raise ValueError("Neural motion shard exceeds its compressed bound")
    expected_members = {f"{name}.npy" for name in SHARD_ARRAY_KEYS}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            entries = archive.infolist()
            if len(entries) != len(expected_members) or {entry.filename for entry in entries} != expected_members:
                raise ValueError("Neural motion shard ZIP members mismatch")
            if len({entry.filename for entry in entries}) != len(entries):
                raise ValueError("Neural motion shard has duplicate ZIP members")
            if any("/" in entry.filename or "\\" in entry.filename for entry in entries):
                raise ValueError("Neural motion shard has nested ZIP members")
            if sum(entry.file_size for entry in entries) > MAX_SHARD_UNCOMPRESSED_BYTES:
                raise ValueError("Neural motion shard exceeds its uncompressed bound")
    except zipfile.BadZipFile as error:
        raise ValueError("Neural motion shard is not a valid NPZ") from error
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != SHARD_ARRAY_KEYS:
            raise ValueError("Neural motion shard array keys mismatch")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if arrays["format"].shape != (1,) or arrays["format"].dtype.kind != "U" or arrays["format"].tolist() != [SHARD_ARRAY_FORMAT]:
        raise ValueError("Neural motion shard array format mismatch")
    if arrays["phases"].shape != (frame_count,) or arrays["phases"].dtype != np.float32:
        raise ValueError("Neural motion shard phase vector mismatch")
    if arrays["emission_pulses"].shape != (frame_count,) or arrays["emission_pulses"].dtype != np.uint8:
        raise ValueError("Neural motion shard pulse vector mismatch")
    if not np.all(np.isfinite(arrays["phases"])) or np.any(arrays["phases"] < 0) or np.any(arrays["phases"] > 1) or np.any(arrays["emission_pulses"] > 3):
        raise ValueError("Neural motion shard numeric values are out of bounds")
    for name in HASH_VECTOR_NAMES:
        if arrays[name].shape != (frame_count,) or arrays[name].dtype.kind != "U" or any(not _is_sha(value) for value in arrays[name].tolist()):
            raise ValueError(f"Neural motion shard {name} vector mismatch")
    if arrays["presentation_sha256"].shape != (frame_count, len(LAYER_NAMES)) or arrays["presentation_sha256"].dtype.kind != "U":
        raise ValueError("Neural motion shard presentation hash matrix mismatch")
    if any(not _is_sha(value) for row in arrays["presentation_sha256"].tolist() for value in row):
        raise ValueError("Neural motion shard presentation hash is invalid")
    for layer_index, name in enumerate(LAYER_NAMES):
        values = arrays[f"layer_{name}"]
        if values.shape != (frame_count, IMAGE_SIZE, IMAGE_SIZE, 4) or values.dtype != np.uint8:
            raise ValueError(f"Neural motion shard {name} layer tensor mismatch")
        for frame_index, frame in enumerate(values):
            if array_sha256(name, frame) != arrays["presentation_sha256"][frame_index, layer_index]:
                raise ValueError(f"Neural motion shard {name} hash mismatch at frame {frame_index}")
    return arrays


def load_motion_shard(
    root: Path,
    source: NeuralMotionSource,
    style_parent: NeuralStyleParent,
    family: str,
    shard_index: int,
    build_contract: Mapping[str, Any],
) -> LoadedMotionShard:
    candidate = source.candidates_by_family[family][0]
    selected = bind_candidate(source, candidate)
    sample_id = selected.binding.sample_id
    arrays_relative, manifest_relative = _paths(family, sample_id, shard_index)
    path = Path(root).resolve() / Path(*manifest_relative.split("/"))
    if not path.is_file() or path.is_symlink():
        raise ValueError("Neural motion shard manifest is missing or unsafe")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or canonical_json_bytes(manifest) != path.read_bytes():
            raise ValueError("Neural motion shard manifest is not canonical JSON")
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Neural motion shard manifest is invalid JSON") from error
    expected_frames = _shard_frame_count(shard_index)
    expected_clip_count = len(MOTION_SHARDS[shard_index]) * len(FACING_NAMES)
    if (
        set(manifest) != {
            "format", "family", "sample_id", "condition", "compiler", "parent", "selection",
            "shard_index", "motions", "global_start_cell", "clip_count", "frame_count", "source",
            "arrays", "clips", "source_clips", "gates",
        }
        or manifest["format"] != SHARD_FORMAT
        or manifest["family"] != family
        or manifest["sample_id"] != sample_id
        or manifest["condition"] != candidate.sample.condition.as_dict()
        or manifest["compiler"] != dict(build_contract["compiler"])
        or manifest["parent"] != {
            "generation_manifest_sha256": source.bank.manifest_sha256,
            "style_manifest_sha256": style_parent.manifest_sha256,
        }
        or manifest["selection"] != {"candidate_ordinal_within_family": 0}
        or manifest["shard_index"] != shard_index
        or manifest["motions"] != list(MOTION_SHARDS[shard_index])
        or manifest["global_start_cell"] != _shard_start(shard_index)
        or manifest["clip_count"] != expected_clip_count
        or manifest["frame_count"] != expected_frames
        or manifest.get("gates") != {
            "validating_motion_constructor_used": True,
            "categorical_authority_preserved": True,
            "rig_socket_authority_preserved": True,
            "palette_identity_invariant": True,
            "loop_endpoints_exact": True,
            "presentation_bounds_exact": True,
        }
    ):
        raise ValueError("Neural motion shard manifest contract mismatch")
    if manifest["arrays"]["path"] != arrays_relative:
        raise ValueError("Neural motion shard artifact path mismatch")
    arrays_path = verify_artifact(root, manifest["arrays"])
    arrays = _load_arrays(arrays_path, expected_frames)
    expected_keys = [(motion, facing) for motion in MOTION_SHARDS[shard_index] for facing in FACING_NAMES]
    clips = manifest.get("clips")
    source_clips = manifest.get("source_clips")
    if not isinstance(clips, list) or not isinstance(source_clips, list) or len(clips) != expected_clip_count or len(source_clips) != expected_clip_count:
        raise ValueError("Neural motion shard clip vectors mismatch")
    if [(clip.get("motion"), clip.get("facing")) for clip in clips] != expected_keys:
        raise ValueError("Neural motion shard clip order mismatch")
    cursor = _shard_start(shard_index)
    source_frames: list[Mapping[str, Any]] = []
    for clip, source_clip, (motion, facing) in zip(clips, source_clips, expected_keys, strict=True):
        source_base = {key: value for key, value in source_clip.items() if key != "hashes"}
        if source_clip.get("hashes", {}).get("clip_sha256") != canonical_json_hash(source_base):
            raise ValueError("Neural motion shard source clip hash is not canonical")
        for expected_index, frame in enumerate(source_clip.get("frames", [])):
            frame_base = {key: value for key, value in frame.items() if key != "motion_frame_sha256"}
            if frame.get("index") != expected_index or frame.get("motion_frame_sha256") != canonical_json_hash(frame_base):
                raise ValueError("Neural motion shard source frame hash is not canonical")
        if (
            clip["start_cell"] != cursor
            or clip["id"] != f"{sample_id}__{motion}__{facing}"
            or clip["source_clip_sha256"] != source_clip.get("hashes", {}).get("clip_sha256")
            or clip["events"] != source_clip.get("events")
            or clip["frame_count"] != source_clip.get("frame_count")
        ):
            raise ValueError("Neural motion shard source/derived clip mismatch")
        cursor += int(clip["frame_count"])
        source_frames.extend(source_clip["frames"])
    if cursor != _shard_start(shard_index) + expected_frames or len(source_frames) != expected_frames:
        raise ValueError("Neural motion shard clip frame accounting mismatch")
    comparisons = {
        "phases": np.asarray([frame["phase"] for frame in source_frames], dtype=np.float32),
        "emission_pulses": np.asarray([frame["emission_pulse"] for frame in source_frames], dtype=np.uint8),
        "motion_frame_sha256": np.asarray([frame["motion_frame_sha256"] for frame in source_frames]),
        "bound_frame_sha256": np.asarray([frame["bound_frame_sha256"] for frame in source_frames]),
        "aligned_fields_sha256": np.asarray([frame["raw_fields_sha256"] for frame in source_frames]),
        "driver_index_sha256": np.asarray([frame["driver_index_sha256"] for frame in source_frames]),
    }
    if any(not np.array_equal(arrays[name], expected) for name, expected in comparisons.items()):
        raise ValueError("Neural motion shard arrays disagree with source motion frames")
    source_record = manifest["source"]
    if source_record != {
        "raw_fields_sha256": candidate.sample.raw_fields_sha256,
        "compiled_fields_sha256": candidate.sample.fields.aligned_sha256,
        "binding_sha256": selected.binding.sha256,
        "static_palette_sha256": style_parent.palette_artifacts[sample_id]["sha256"],
    }:
        raise ValueError("Neural motion shard identity source mismatch")
    return LoadedMotionShard(manifest=manifest, arrays=arrays)


def aggregate_family_shards(
    root: Path,
    source: NeuralMotionSource,
    style_parent: NeuralStyleParent,
    family: str,
    build_contract: Mapping[str, Any],
) -> NeuralIdentityPayload:
    selected = bind_candidate(source, source.candidates_by_family[family][0])
    shards = [
        load_motion_shard(root, source, style_parent, family, shard_index, build_contract)
        for shard_index in range(len(MOTION_SHARDS))
    ]
    frame_count = sum(int(shard.manifest["frame_count"]) for shard in shards)
    if frame_count != 944:
        raise ValueError("Neural motion shard family frame total mismatch")
    rows = math.ceil(frame_count / ATLAS_COLUMNS)
    atlases = {
        name: np.zeros((rows * IMAGE_SIZE, ATLAS_COLUMNS * IMAGE_SIZE, 4), dtype=np.uint8)
        for name in LAYER_NAMES
    }
    frame_cursor = 0
    for shard in shards:
        count = int(shard.manifest["frame_count"])
        if int(shard.manifest["global_start_cell"]) != frame_cursor:
            raise ValueError("Neural motion shards are not contiguous")
        for local_index in range(count):
            row, column = divmod(frame_cursor + local_index, ATLAS_COLUMNS)
            y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
            for name in LAYER_NAMES:
                atlases[name][y : y + IMAGE_SIZE, x : x + IMAGE_SIZE] = shard.arrays[f"layer_{name}"][local_index]
        frame_cursor += count
    def vector(name: str) -> tuple[Any, ...]:
        return tuple(value for shard in shards for value in shard.arrays[name].tolist())

    compiled = CompiledIdentityFrames(
        atlases=atlases,
        source_motion_frame_hashes=vector("motion_frame_sha256"),
        bound_frame_hashes=vector("bound_frame_sha256"),
        categorical_hashes=vector("categorical_sha256"),
        aligned_field_hashes=vector("aligned_fields_sha256"),
        driver_hashes=vector("driver_index_sha256"),
        joint_hashes=vector("joint_sha256"),
        socket_hashes=vector("socket_sha256"),
        presentation_hashes=vector("presentation_sha256"),
        phases=tuple(float(value) for value in vector("phases")),
        emission_pulses=tuple(int(value) for value in vector("emission_pulses")),
        clip_records=tuple(record for shard in shards for record in shard.manifest["clips"]),
        source_clip_manifests=tuple(record for shard in shards for record in shard.manifest["source_clips"]),
    )
    return finalize_identity_payload(source, style_parent, selected, build_contract, [], compiled)
