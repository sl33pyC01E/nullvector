from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ..morphology import (
    FACING_NAMES,
    FAMILIES,
    MOTION_NAMES,
    MorphologyGenome,
    assert_valid_motion_clip,
    generate_motion_clip,
    render_specimen,
)
from .hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    clip_presentation_sha256,
    deterministic_npz_bytes,
    identity_style_sha256,
    png_bytes,
)
from .model import ATLAS_COLUMNS, IMAGE_SIZE, LAYER_NAMES, FamilyPayload
from .rendering import build_condition, render_motion_frame


FAMILY_MANIFEST_FORMAT = "nullvector-multifield-style-motion-family-v1"
FRAME_INDEX_FORMAT = "nullvector-multifield-style-motion-frame-index-v1"
FAMILY_GATE_NAMES = (
    "source_clip_hashes_exact",
    "source_frame_hashes_exact",
    "motion_events_preserved",
    "categorical_fields_unchanged",
    "categorical_body_alpha_exact",
    "rig_authority_unchanged",
    "socket_authority_unchanged",
    "palette_identity_invariant",
    "no_temporal_palette_flicker",
    "loop_endpoints_exact",
    "outline_radius_1_exact",
    "bloom_radius_1_exact",
    "bloom_radius_2_exact",
    "effect_rings_unclipped",
)


def _artifact(relative: str, payload: bytes) -> dict[str, Any]:
    return artifact_record_from_bytes(relative, payload)


def compile_family_payload(
    bank: Any,
    family: str,
    build_contract: Mapping[str, Any],
) -> FamilyPayload:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}")
    source = bank.sources[family]
    source_atlas = bank.atlases[family]
    expected_clips = bank.clips_by_family[family]
    expected_clip_keys = [
        (motion, facing)
        for motion in MOTION_NAMES
        for facing in FACING_NAMES
    ]
    if [(entry["motion"], entry["facing"]) for entry in expected_clips] != expected_clip_keys:
        raise ValueError(f"Full clip ordering mismatch for {family}")
    genome = MorphologyGenome.from_dict(dict(source["genome"]))
    specimen = render_specimen(genome)
    if specimen.manifest["hashes"] != source["hashes"]:
        raise ValueError(f"Replayed source specimen hashes mismatch for {family}")
    family_ordinal = list(FAMILIES).index(family)
    condition = build_condition(source, family_ordinal)
    identity_sha = identity_style_sha256(
        source_id=str(source["id"]),
        source_seed=int(source["seed"]),
        semantic_sha256=str(source["hashes"]["semantic_sha256"]),
        genome_sha256=str(source["hashes"]["genome_sha256"]),
        training_arrays_sha256=str(source["hashes"]["training_arrays_sha256"]),
    )
    frame_count = sum(int(entry["frame_count"]) for entry in expected_clips)
    if frame_count != int(source_atlas["frame_count"]):
        raise ValueError(f"Family frame total disagrees with source atlas for {family}")
    rows = math.ceil(frame_count / ATLAS_COLUMNS)
    if rows != int(source_atlas["rows"]):
        raise ValueError(f"Family atlas row count disagrees with source atlas for {family}")
    atlases = {
        name: np.zeros(
            (rows * IMAGE_SIZE, ATLAS_COLUMNS * IMAGE_SIZE, 4),
            dtype=np.uint8,
        )
        for name in LAYER_NAMES
    }

    source_frame_hashes: list[str] = []
    categorical_hashes: list[str] = []
    joint_hashes: list[str] = []
    socket_hashes: list[str] = []
    authority_hashes: list[str] = []
    presentation_hashes: list[tuple[str, ...]] = []
    phases: list[float] = []
    clip_offsets = [0]
    clip_ids: list[str] = []
    clips: list[dict[str, Any]] = []
    palette: Mapping[str, Any] | None = None
    palette_sha: str | None = None
    cursor = 0
    aggregate_gates = {name: True for name in FAMILY_GATE_NAMES}
    for expected in expected_clips:
        clip = generate_motion_clip(
            specimen,
            str(expected["motion"]),
            facing=str(expected["facing"]),
        )
        assert_valid_motion_clip(clip)
        source_exact = (
            clip.manifest["id"] == expected["id"]
            and clip.sha256 == expected["clip_sha256"]
            and len(clip.frames) == expected["frame_count"]
            and clip.fps == expected["fps"]
            and clip.loop == expected["loop"]
            and clip.manifest["events"] == expected["events"]
            and clip.manifest["metrics"] == expected["metrics"]
        )
        if not source_exact:
            raise ValueError(f"Replayed source clip contract mismatch: {expected['id']}")
        frame_hashes = [frame.sha256 for frame in clip.frames]
        if frame_hashes != expected["frame_sha256"]:
            raise ValueError(f"Replayed source frame hashes mismatch: {expected['id']}")

        clip_start = cursor
        clip_layer_frames: list[dict[str, np.ndarray]] = []
        clip_categorical: list[str] = []
        clip_authority: list[str] = []
        clip_presentation: list[tuple[str, ...]] = []
        for frame in clip.frames:
            audit = render_motion_frame(
                frame,
                specimen,
                condition,
                identity_sha,
                expected_palette_sha256=palette_sha,
            )
            if palette is None:
                palette = audit.palette
                palette_sha = audit.palette_sha256
            elif audit.palette_sha256 != palette_sha or audit.palette != palette:
                raise ValueError(f"Temporal palette flicker detected in {clip.manifest['id']}")
            row, column = divmod(cursor, ATLAS_COLUMNS)
            y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
            for layer_name in LAYER_NAMES:
                atlases[layer_name][y : y + IMAGE_SIZE, x : x + IMAGE_SIZE] = audit.layers[layer_name]
            source_frame_hashes.append(frame.sha256)
            categorical_hashes.append(audit.categorical_sha256)
            joint_hashes.append(audit.joint_sha256)
            socket_hashes.append(audit.socket_sha256)
            authority_hashes.append(audit.authority_sha256)
            presentation_hashes.append(audit.presentation_sha256)
            phases.append(float(frame.phase))
            clip_layer_frames.append(dict(audit.layers))
            clip_categorical.append(audit.categorical_sha256)
            clip_authority.append(audit.authority_sha256)
            clip_presentation.append(audit.presentation_sha256)
            cursor += 1

        loop_exact = True
        if clip.loop:
            loop_exact = (
                clip_categorical[0] == clip_categorical[-1]
                and clip_authority[0] == clip_authority[-1]
                and clip_presentation[0] == clip_presentation[-1]
                and all(
                    np.array_equal(
                        clip_layer_frames[0][layer_name],
                        clip_layer_frames[-1][layer_name],
                    )
                    for layer_name in LAYER_NAMES
                )
            )
        if not loop_exact:
            raise ValueError(f"Styled loop endpoint mismatch: {clip.manifest['id']}")
        presentation_matrix = np.asarray(clip_presentation, dtype="<U64")
        derived_clip_hash = clip_presentation_sha256(
            identity_sha256=identity_sha,
            source_clip_sha256=clip.sha256,
            events=list(clip.manifest["events"]),
            categorical_hashes=clip_categorical,
            authority_hashes=clip_authority,
            presentation_hashes=presentation_matrix,
        )
        clips.append(
            {
                "id": clip.manifest["id"],
                "motion": clip.motion,
                "facing": clip.facing,
                "fps": clip.fps,
                "loop": clip.loop,
                "frame_count": len(clip.frames),
                "start_cell": clip_start,
                "source_clip_sha256": clip.sha256,
                "derived_clip_sha256": derived_clip_hash,
                "events": list(clip.manifest["events"]),
                "gates": {
                    "events_preserved": True,
                    "palette_identity_invariant": True,
                    "loop_endpoints_exact": loop_exact,
                    "categorical_and_anchor_authority_unchanged": True,
                    "outline_and_bloom_radius_exact": True,
                },
            }
        )
        clip_ids.append(clip.manifest["id"])
        clip_offsets.append(cursor)

    if cursor != frame_count or palette is None or palette_sha is None:
        raise RuntimeError(f"Family compilation frame/palette accounting failed for {family}")
    index_arrays = {
        "format": np.asarray([FRAME_INDEX_FORMAT]),
        "family": np.asarray([family]),
        "layer_names": np.asarray(LAYER_NAMES),
        "clip_ids": np.asarray(clip_ids),
        "clip_offsets": np.asarray(clip_offsets, dtype=np.uint32),
        "phases": np.asarray(phases, dtype=np.float32),
        "source_frame_sha256": np.asarray(source_frame_hashes),
        "categorical_sha256": np.asarray(categorical_hashes),
        "joint_sha256": np.asarray(joint_hashes),
        "socket_sha256": np.asarray(socket_hashes),
        "authority_sha256": np.asarray(authority_hashes),
        "presentation_sha256": np.asarray(presentation_hashes),
    }
    family_prefix = f"families/{family}"
    file_payloads: dict[str, bytes] = {}
    layer_artifacts: dict[str, dict[str, Any]] = {}
    for layer_name in LAYER_NAMES:
        relative = f"{family_prefix}/{layer_name}.png"
        payload = png_bytes(atlases[layer_name])
        file_payloads[relative] = payload
        layer_artifacts[layer_name] = _artifact(relative, payload)
    palette_relative = f"{family_prefix}/palette.json"
    palette_payload = canonical_json_bytes(dict(palette))
    if palette_sha != _artifact(palette_relative, palette_payload)["sha256"]:
        raise RuntimeError("Family palette hash accounting drifted")
    file_payloads[palette_relative] = palette_payload
    index_relative = f"{family_prefix}/frame_index.npz"
    index_payload = deterministic_npz_bytes(index_arrays)
    file_payloads[index_relative] = index_payload

    family_manifest = {
        "format": FAMILY_MANIFEST_FORMAT,
        "status": "ready",
        "family": family,
        "compiler": {
            "id": build_contract["compiler"]["id"],
            "source_sha256": build_contract["compiler"]["source_sha256"],
            "presentation_source_sha256": build_contract["compiler"]["presentation_source_sha256"],
            "motion_source_sha256": build_contract["compiler"]["motion_source_sha256"],
        },
        "identity": {
            "style_identity_sha256": identity_sha,
            "source_id": source["id"],
            "source_seed": source["seed"],
            "morphology_id": condition.morphology_id,
            "subtype_id": condition.subtype_id,
            "role_id": condition.role_id,
            "role_name": condition.role_name,
            "palette_sha256": palette_sha,
        },
        "authority": {
            "categorical_fields_are_source_authority": True,
            "rig_and_sockets_are_source_authority": True,
            "presentation_is_derived_only": True,
            "atlas_alpha_is_not_collision_authority": True,
        },
        "source": {
            "asset_index_sha256": bank.asset_index_sha256,
            "source_manifest_sha256": bank.source_manifest_sha256,
            "source_archive_sha256": bank.source_archive_sha256,
            "source_atlas_sha256": source_atlas["atlas_sha256"],
            "source_semantic_sha256": source["hashes"]["semantic_sha256"],
            "source_genome_sha256": source["hashes"]["genome_sha256"],
            "source_training_arrays_sha256": source["hashes"]["training_arrays_sha256"],
        },
        "layout": {
            "cell_size": IMAGE_SIZE,
            "columns": ATLAS_COLUMNS,
            "rows": rows,
            "frame_count": frame_count,
            "layer_order": list(LAYER_NAMES),
        },
        "artifacts": {
            "palette": _artifact(palette_relative, palette_payload),
            "frame_index": _artifact(index_relative, index_payload),
            "layers": layer_artifacts,
        },
        "clip_count": len(clips),
        "frame_count": frame_count,
        "gates": aggregate_gates,
        "clips": clips,
    }
    manifest_relative = f"{family_prefix}/family_manifest.json"
    manifest_payload = canonical_json_bytes(family_manifest)
    file_payloads[manifest_relative] = manifest_payload
    return FamilyPayload(
        family=family,
        file_payloads=file_payloads,
        family_manifest=family_manifest,
        frame_count=frame_count,
        clip_count=len(clips),
    )
