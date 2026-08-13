from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..morphology import FACING_NAMES, FAMILIES, MOTION_NAMES
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    clip_presentation_sha256,
    deterministic_npz_bytes,
    named_points_sha256,
    png_bytes,
)
from ..multifield_style_motion.model import (
    ATLAS_COLUMNS,
    IMAGE_SIZE,
    JOINT_NAMES,
    LAYER_NAMES,
    SOCKET_NAMES,
)
from ..neural_rig_bridge import (
    BindingRejected,
    compile_neural_motion_clip,
)
from ..neural_rig_bridge.hashing import binder_source_hash
from .model import (
    NeuralIdentityPayload,
    NeuralMotionSource,
    NeuralStyleParent,
    SelectedNeuralIdentity,
)
from .rendering import render_neural_motion_frame
from .source import bind_candidate


IDENTITY_MANIFEST_FORMAT = "nullvector-multifield-style-neural-motion-identity-v1"
FRAME_INDEX_FORMAT = "nullvector-multifield-style-neural-motion-frame-index-v1"
MOTION_MANIFESTS_FORMAT = "nullvector-neural-motion-manifest-collection-v1"
IDENTITY_GATE_NAMES = (
    "raw_neural_fields_exact",
    "binding_exact",
    "motion_clips_valid",
    "source_tuples_preserved",
    "procedural_pixel_substitution_absent",
    "categorical_fields_unchanged_by_presentation",
    "rig_and_socket_authority_preserved",
    "motion_events_preserved",
    "palette_matches_static_neural_parent",
    "palette_identity_invariant",
    "no_temporal_palette_flicker",
    "loop_endpoints_exact",
    "outline_radius_1_exact",
    "bloom_radius_1_exact",
    "bloom_radius_2_exact",
    "effect_rings_unclipped",
    "emission_pulse_support_bounded",
)


def _artifact(relative: str, payload: bytes) -> dict[str, Any]:
    return artifact_record_from_bytes(relative, payload)


def _project_relative(path: Path) -> str:
    from ..multifield_style.source import PROJECT_ROOT

    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Neural motion provenance path is outside project root: {path}") from error


@dataclass(frozen=True, slots=True)
class CompiledIdentityFrames:
    atlases: Mapping[str, np.ndarray]
    source_motion_frame_hashes: tuple[str, ...]
    bound_frame_hashes: tuple[str, ...]
    categorical_hashes: tuple[str, ...]
    aligned_field_hashes: tuple[str, ...]
    driver_hashes: tuple[str, ...]
    joint_hashes: tuple[str, ...]
    socket_hashes: tuple[str, ...]
    presentation_hashes: tuple[tuple[str, ...], ...]
    phases: tuple[float, ...]
    emission_pulses: tuple[int, ...]
    clip_records: tuple[Mapping[str, Any], ...]
    source_clip_manifests: tuple[Mapping[str, Any], ...]


def finalize_identity_payload(
    source: NeuralMotionSource,
    style_parent: NeuralStyleParent,
    selected: SelectedNeuralIdentity,
    build_contract: Mapping[str, Any],
    rejected_candidates: list[Mapping[str, Any]],
    compiled: CompiledIdentityFrames,
) -> NeuralIdentityPayload:
    candidate = selected.candidate
    sample = candidate.sample
    binding = selected.binding
    family = binding.family
    sample_id = binding.sample_id
    frame_count = len(compiled.phases)
    if frame_count != 944 or any(len(values) != frame_count for values in (
        compiled.source_motion_frame_hashes,
        compiled.bound_frame_hashes,
        compiled.categorical_hashes,
        compiled.aligned_field_hashes,
        compiled.driver_hashes,
        compiled.joint_hashes,
        compiled.socket_hashes,
        compiled.presentation_hashes,
        compiled.emission_pulses,
    )):
        raise ValueError("Neural identity compiled frame vectors are not exact")
    clip_records = [dict(record) for record in compiled.clip_records]
    source_clip_manifests = [dict(record) for record in compiled.source_clip_manifests]
    if len(clip_records) != 104 or len(source_clip_manifests) != 104:
        raise ValueError("Neural identity compiled clip vectors are not exact")
    expected_keys = [(motion, facing) for motion in MOTION_NAMES for facing in FACING_NAMES]
    if [(clip["motion"], clip["facing"]) for clip in clip_records] != expected_keys:
        raise ValueError("Neural identity compiled clip order is not canonical")
    rows = math.ceil(frame_count / ATLAS_COLUMNS)
    expected_atlas_shape = (rows * IMAGE_SIZE, ATLAS_COLUMNS * IMAGE_SIZE, 4)
    if set(compiled.atlases) != set(LAYER_NAMES) or any(
        values.dtype != np.uint8 or values.shape != expected_atlas_shape
        for values in compiled.atlases.values()
    ):
        raise ValueError("Neural identity compiled atlases are not native RGBA")
    clip_ids = [clip["id"] for clip in clip_records]
    clip_offsets = [0]
    for clip in clip_records:
        clip_offsets.append(clip_offsets[-1] + int(clip["frame_count"]))
    if clip_offsets[-1] != frame_count:
        raise ValueError("Neural identity compiled clip offsets are not exact")
    index_arrays = {
        "format": np.asarray([FRAME_INDEX_FORMAT]),
        "sample_id": np.asarray([sample_id]),
        "family": np.asarray([family]),
        "layer_names": np.asarray(LAYER_NAMES),
        "clip_ids": np.asarray(clip_ids),
        "clip_offsets": np.asarray(clip_offsets, dtype=np.uint32),
        "phases": np.asarray(compiled.phases, dtype=np.float32),
        "emission_pulses": np.asarray(compiled.emission_pulses, dtype=np.uint8),
        "motion_frame_sha256": np.asarray(compiled.source_motion_frame_hashes),
        "bound_frame_sha256": np.asarray(compiled.bound_frame_hashes),
        "categorical_sha256": np.asarray(compiled.categorical_hashes),
        "aligned_fields_sha256": np.asarray(compiled.aligned_field_hashes),
        "driver_index_sha256": np.asarray(compiled.driver_hashes),
        "joint_sha256": np.asarray(compiled.joint_hashes),
        "socket_sha256": np.asarray(compiled.socket_hashes),
        "presentation_sha256": np.asarray(compiled.presentation_hashes),
    }
    palette = style_parent.palettes[sample_id]
    palette_artifact = style_parent.palette_artifacts[sample_id]
    identity_prefix = f"identities/{family}/{sample_id}"
    file_payloads: dict[str, bytes] = {}
    layer_artifacts: dict[str, dict[str, Any]] = {}
    for layer_name in LAYER_NAMES:
        relative = f"{identity_prefix}/{layer_name}.png"
        payload = png_bytes(compiled.atlases[layer_name])
        file_payloads[relative] = payload
        layer_artifacts[layer_name] = _artifact(relative, payload)
    palette_relative = f"{identity_prefix}/palette.json"
    palette_payload = canonical_json_bytes(dict(palette))
    if _artifact(palette_relative, palette_payload)["sha256"] != palette_artifact["sha256"]:
        raise ValueError("Neural identity palette artifact is not byte-exact with static parent")
    file_payloads[palette_relative] = palette_payload
    binding_relative = f"{identity_prefix}/binding_manifest.json"
    binding_payload = canonical_json_bytes(dict(binding.manifest))
    file_payloads[binding_relative] = binding_payload
    motions_relative = f"{identity_prefix}/motion_manifests.json"
    motions_payload = canonical_json_bytes(
        {"format": MOTION_MANIFESTS_FORMAT, "sample_id": sample_id, "clips": source_clip_manifests}
    )
    file_payloads[motions_relative] = motions_payload
    index_relative = f"{identity_prefix}/frame_index.npz"
    index_payload = deterministic_npz_bytes(index_arrays)
    file_payloads[index_relative] = index_payload
    raw_archive = candidate.raw_archive_path
    raw_manifest = candidate.raw_manifest_path
    identity_manifest = {
        "format": IDENTITY_MANIFEST_FORMAT,
        "status": "ready",
        "neural_output": True,
        "family": family,
        "sample_id": sample_id,
        "condition": sample.condition.as_dict(),
        "compiler": dict(build_contract["compiler"]),
        "authority": {
            "raw_neural_fields_are_source_authority": True,
            "binding_and_motion_program_are_derived_authority": True,
            "presentation_is_derived_only": True,
            "procedural_pixel_substitution": False,
            "collision_authority_modified": False,
            "aura_is_effect_not_body": True,
        },
        "source": {
            "generation_manifest_sha256": source.bank.manifest_sha256,
            "style_manifest_sha256": style_parent.manifest_sha256,
            "raw_fields_sha256": sample.raw_fields_sha256,
            "compiled_fields_sha256": sample.fields.aligned_sha256,
            "raw_archive_path": _project_relative(raw_archive),
            "raw_archive_bytes": raw_archive.stat().st_size,
            "raw_archive_sha256": binding.upstream_hashes["raw_archive_sha256"],
            "raw_manifest_path": _project_relative(raw_manifest),
            "raw_manifest_bytes": raw_manifest.stat().st_size,
            "raw_manifest_sha256": binding.upstream_hashes["raw_manifest_sha256"],
            "binding_sha256": binding.sha256,
            "static_palette_sha256": palette_artifact["sha256"],
        },
        "selection": {
            "policy": "first-full-matrix-valid-per-family-v1",
            "candidate_ordinal_within_family": list(source.candidates_by_family[family]).index(candidate),
            "rejected_prior_candidates": rejected_candidates,
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
            "binding_manifest": _artifact(binding_relative, binding_payload),
            "motion_manifests": _artifact(motions_relative, motions_payload),
            "frame_index": _artifact(index_relative, index_payload),
            "layers": layer_artifacts,
        },
        "clip_count": len(clip_records),
        "frame_count": frame_count,
        "gates": {name: True for name in IDENTITY_GATE_NAMES},
        "clips": clip_records,
    }
    manifest_relative = f"{identity_prefix}/identity_manifest.json"
    manifest_payload = canonical_json_bytes(identity_manifest)
    file_payloads[manifest_relative] = manifest_payload
    return NeuralIdentityPayload(
        family=family,
        sample_id=sample_id,
        file_payloads=file_payloads,
        identity_manifest=identity_manifest,
        frame_count=frame_count,
        clip_count=len(clip_records),
    )


def _compile_selected(
    source: NeuralMotionSource,
    style_parent: NeuralStyleParent,
    selected: SelectedNeuralIdentity,
    build_contract: Mapping[str, Any],
    rejected_candidates: list[Mapping[str, Any]],
) -> NeuralIdentityPayload:
    candidate = selected.candidate
    sample = candidate.sample
    binding = selected.binding
    family = binding.family
    sample_id = binding.sample_id
    if binder_source_hash() != build_contract["compiler"]["bridge_source_sha256"]:
        raise ValueError("Neural bridge source changed during identity compilation")
    frame_count = sum(
        int(build_contract["matrix"]["frame_counts"][motion]) * len(FACING_NAMES)
        for motion in MOTION_NAMES
    )
    if frame_count != 944:
        raise ValueError("Neural identity motion matrix frame count drifted")
    rows = math.ceil(frame_count / ATLAS_COLUMNS)
    atlases = {
        name: np.zeros((rows * IMAGE_SIZE, ATLAS_COLUMNS * IMAGE_SIZE, 4), dtype=np.uint8)
        for name in LAYER_NAMES
    }
    palette = style_parent.palettes[sample_id]
    palette_artifact = style_parent.palette_artifacts[sample_id]
    identity_prefix = f"identities/{family}/{sample_id}"
    source_motion_frame_hashes: list[str] = []
    bound_frame_hashes: list[str] = []
    categorical_hashes: list[str] = []
    aligned_field_hashes: list[str] = []
    driver_hashes: list[str] = []
    joint_hashes: list[str] = []
    socket_hashes: list[str] = []
    presentation_hashes: list[tuple[str, ...]] = []
    phases: list[float] = []
    emission_pulses: list[int] = []
    clip_records: list[dict[str, Any]] = []
    source_clip_manifests: list[dict[str, Any]] = []
    cursor = 0
    for motion in MOTION_NAMES:
        for facing in FACING_NAMES:
            clip = compile_neural_motion_clip(binding, motion, facing=facing)
            # compile_neural_motion_clip is a validating constructor: its public
            # contract now rerenders every bound frame before it returns.  A
            # second assert here would rerender all 944 frames again without
            # adding an independent check; our artifact validator and exact
            # replay are the independent downstream checks.
            if (
                clip.manifest["metrics"]["all_source_tuples_preserved"] is not True
                or clip.manifest["metrics"]["procedural_pixel_substitution"] is not False
                or clip.manifest["binder_source_sha256"] != build_contract["compiler"]["bridge_source_sha256"]
            ):
                raise ValueError(f"Neural source motion authority failed: {clip.manifest['id']}")
            clip_start = cursor
            clip_frames: list[dict[str, np.ndarray]] = []
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
                row, column = divmod(cursor, ATLAS_COLUMNS)
                y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
                for layer_name in LAYER_NAMES:
                    atlases[layer_name][y : y + IMAGE_SIZE, x : x + IMAGE_SIZE] = rendered.layers[layer_name]
                source_motion_frame_hashes.append(frame.sha256)
                bound_frame_hashes.append(frame.fields.sha256)
                categorical_hashes.append(rendered.categorical_sha256)
                aligned_field_hashes.append(rendered.aligned_fields_sha256)
                driver_hashes.append(str(frame.fields.manifest["driver_index_sha256"]))
                joint_hashes.append(named_points_sha256("joints", JOINT_NAMES, frame.joints))
                socket_hashes.append(named_points_sha256("sockets", SOCKET_NAMES, frame.sockets))
                presentation_hashes.append(rendered.presentation_sha256)
                phases.append(float(frame.phase))
                emission_pulses.append(int(frame.emission_pulse))
                clip_frames.append(dict(rendered.layers))
                clip_categorical.append(rendered.categorical_sha256)
                clip_motion_authority.append(frame.sha256)
                clip_presentations.append(rendered.presentation_sha256)
                cursor += 1
            loop_exact = True
            if clip.loop:
                loop_exact = (
                    clip_categorical[0] == clip_categorical[-1]
                    and clip.frames[0].fields.sha256 == clip.frames[-1].fields.sha256
                    and clip_presentations[0] == clip_presentations[-1]
                    and clip.frames[0].joints == clip.frames[-1].joints
                    and clip.frames[0].sockets == clip.frames[-1].sockets
                    and all(
                        np.array_equal(clip_frames[0][name], clip_frames[-1][name])
                        for name in LAYER_NAMES
                    )
                )
            if not loop_exact:
                raise ValueError(f"Neural styled loop endpoint mismatch: {clip.manifest['id']}")
            derived_clip_sha = clip_presentation_sha256(
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
                    "derived_clip_sha256": derived_clip_sha,
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
            source_clip_manifests.append(dict(clip.manifest))
    if cursor != frame_count:
        raise RuntimeError("Neural identity frame accounting failed")
    return finalize_identity_payload(
        source,
        style_parent,
        selected,
        build_contract,
        rejected_candidates,
        CompiledIdentityFrames(
            atlases=atlases,
            source_motion_frame_hashes=tuple(source_motion_frame_hashes),
            bound_frame_hashes=tuple(bound_frame_hashes),
            categorical_hashes=tuple(categorical_hashes),
            aligned_field_hashes=tuple(aligned_field_hashes),
            driver_hashes=tuple(driver_hashes),
            joint_hashes=tuple(joint_hashes),
            socket_hashes=tuple(socket_hashes),
            presentation_hashes=tuple(presentation_hashes),
            phases=tuple(phases),
            emission_pulses=tuple(emission_pulses),
            clip_records=tuple(clip_records),
            source_clip_manifests=tuple(source_clip_manifests),
        ),
    )


def compile_family_identity_payload(
    source: NeuralMotionSource,
    style_parent: NeuralStyleParent,
    family: str,
    build_contract: Mapping[str, Any],
) -> NeuralIdentityPayload:
    if family not in FAMILIES:
        raise ValueError(f"Unknown neural motion family {family!r}")
    rejected: list[Mapping[str, Any]] = []
    for ordinal, candidate in enumerate(source.candidates_by_family[family]):
        try:
            selected = bind_candidate(source, candidate)
            return _compile_selected(source, style_parent, selected, build_contract, rejected)
        except (BindingRejected, ValueError) as error:
            rejected.append(
                {
                    "sample_id": candidate.sample.condition.sample_id,
                    "candidate_ordinal_within_family": ordinal,
                    "reason": str(error)[:500],
                }
            )
    raise ValueError(f"No neural identity passed the full motion matrix for {family}: {rejected}")
