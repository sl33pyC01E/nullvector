from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from forge.morphology import (
    FAMILIES,
    FACING_NAMES,
    LOOPING_MOTIONS,
    MOTION_NAMES,
    allowed_training_field_tuples,
    blend_motion_poses,
    frame_training_tuples,
    generate_motion_clip,
    genome_from_seed,
    render_specimen,
    validate_motion_clip,
)
from forge.morphology.motion import STABLE_STANCE_MOTIONS
from forge.morphology.motion_preview import (
    SHOWCASE_MOTIONS,
    build_motion_bank,
    build_motion_contact_sheet,
    build_showcase_frames,
    build_vertical_sprite_sheet,
    preview_specimens,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def motion_bank():
    return build_motion_bank()


def test_all_families_and_motions_pass_strict_contract(motion_bank) -> None:
    specimens, clips = motion_bank
    assert len(specimens) == len(FAMILIES)
    assert len(clips) == len(FAMILIES) * len(MOTION_NAMES)
    assert {specimen.genome.family_name for specimen in specimens} == set(FAMILIES)
    assert {clip.motion for clip in clips} == set(MOTION_NAMES)
    for clip in clips:
        assert validate_motion_clip(clip) == []
        metrics = clip.manifest["metrics"]
        assert metrics["unique_semantic_frames"] >= 2
        assert metrics["max_changed_pixel_fraction"] > 0.0
        assert metrics["max_structural_components"] == 1
        assert metrics["margin_clear"] is True
        assert metrics["field_tuples_valid"] is True
        assert clip.loop == (clip.motion in LOOPING_MOTIONS)
        assert clip.manifest["events"]
        assert all(0 <= event["frame"] < len(clip.frames) for event in clip.manifest["events"])
        assert all(event["socket"] in clip.manifest["socket_names"] for event in clip.manifest["events"])


def test_loop_closure_and_stable_idle_stance(motion_bank) -> None:
    _, clips = motion_bank
    for clip in clips:
        if clip.loop:
            first, last = clip.frames[0], clip.frames[-1]
            assert np.array_equal(first.layers, last.layers)
            assert np.array_equal(first.tokens, last.tokens)
            assert np.array_equal(first.rgba, last.rgba)
            assert first.joints == last.joints
            assert first.sockets == last.sockets
        if clip.motion in STABLE_STANCE_MOTIONS:
            assert clip.manifest["metrics"]["left_foot_span"] == [0, 0]
            assert clip.manifest["metrics"]["right_foot_span"] == [0, 0]
            assert clip.manifest["metrics"]["root_span"] == [0, 0]


def test_training_field_tuples_remain_in_cross_field_vocabulary(motion_bank) -> None:
    _, clips = motion_bank
    allowed = allowed_training_field_tuples()
    assert (0, 0, 0) in allowed
    assert (16, 9, 3) in allowed
    for clip in clips:
        for frame in clip.frames:
            assert frame_training_tuples(frame, clip.specimen) <= allowed


def test_motion_manifest_matches_strict_json_schema(motion_bank) -> None:
    schema_path = (
        PROJECT_ROOT
        / "shared"
        / "schema"
        / "morphology_motion_manifest.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    _, clips = motion_bank
    for clip in clips:
        errors = list(validator.iter_errors(clip.manifest))
        assert errors == [], (clip.manifest["id"], errors)


def test_generation_is_bit_deterministic() -> None:
    hashes: list[str] = []
    for family_index, family in enumerate(FAMILIES):
        specimen = render_specimen(
            genome_from_seed(0xAC710000 + family_index * 7919, family)
        )
        first = generate_motion_clip(specimen, "locomote", facing="southeast")
        second = generate_motion_clip(specimen, "locomote", facing="southeast")
        assert first.sha256 == second.sha256
        assert first.manifest == second.manifest
        for first_frame, second_frame in zip(first.frames, second.frames, strict=True):
            assert np.array_equal(first_frame.layers, second_frame.layers)
            assert np.array_equal(first_frame.rgba, second_frame.rgba)
            assert first_frame.sha256 == second_frame.sha256
        hashes.append(first.sha256)
    assert len(set(hashes)) == len(FAMILIES)


def test_eight_way_facing_is_safe_and_directionally_distinct() -> None:
    for family_index, family in enumerate(FAMILIES):
        specimen = render_specimen(
            genome_from_seed(0xFACE0000 + family_index * 1543, family)
        )
        clips = [
            generate_motion_clip(specimen, "locomote", facing=facing)
            for facing in FACING_NAMES
        ]
        assert all(validate_motion_clip(clip) == [] for clip in clips)
        signatures = {
            clip.frames[len(clip.frames) // 4].layers.tobytes() for clip in clips
        }
        assert len(signatures) >= 6


def test_every_combat_role_survives_motion_and_remains_observable() -> None:
    for family_index, family in enumerate(FAMILIES):
        base_genome = genome_from_seed(0x701E0000 + family_index * 4099, family)
        signatures = set()
        for role_id in range(8):
            specimen = render_specimen(replace(base_genome, role_id=role_id))
            clip = generate_motion_clip(specimen, "cast", facing="northwest")
            assert validate_motion_clip(clip) == []
            signatures.add(clip.frames[len(clip.frames) // 2].layers.tobytes())
        assert len(signatures) == 8


def test_emotes_and_actions_are_visibly_distinct(motion_bank) -> None:
    _, clips = motion_bank
    for family in FAMILIES:
        family_clips = [clip for clip in clips if clip.specimen.genome.family_name == family]
        peak_signatures = set()
        for clip in family_clips:
            if clip.motion == "death":
                frame = clip.frames[-1]
            else:
                frame = clip.frames[len(clip.frames) // 2]
            peak_signatures.add(frame.layers.tobytes())
        assert len(peak_signatures) >= 11


def test_validator_detects_hash_corruption(motion_bank) -> None:
    _, clips = motion_bank
    clip = clips[0]
    damaged_frame = replace(clip.frames[0], sha256="0" * 64)
    damaged_clip = replace(clip, frames=(damaged_frame, *clip.frames[1:]))
    errors = validate_motion_clip(damaged_clip)
    assert any("frame hash is incorrect" in error for error in errors)
    assert any("manifest" in error for error in errors)


def test_preview_compositors_have_versioned_deterministic_layout(motion_bank) -> None:
    specimens, clips = motion_bank
    contact = build_motion_contact_sheet(specimens, clips, scale=1)
    assert contact.mode == "RGBA"
    assert contact.size == (74 + len(MOTION_NAMES) * 48, 42 + len(FAMILIES) * 48)
    frames = build_showcase_frames(
        specimens,
        clips,
        motions=SHOWCASE_MOTIONS[:3],
        frame_count=4,
        scale=1,
    )
    assert len(frames) == 4
    assert len({frame.tobytes() for frame in frames}) == 4
    strip = build_vertical_sprite_sheet(frames)
    assert strip.size == (frames[0].width, frames[0].height * len(frames))


def test_preview_sources_are_stable_and_invalid_inputs_fail() -> None:
    first = preview_specimens()
    second = preview_specimens()
    assert [specimen.manifest for specimen in first] == [
        specimen.manifest for specimen in second
    ]
    specimen = first[0]
    with pytest.raises(ValueError, match="Unsupported motion"):
        generate_motion_clip(specimen, "teleport")
    with pytest.raises(ValueError, match="Unsupported facing"):
        generate_motion_clip(specimen, "locomote", facing="up-ish")
    with pytest.raises(ValueError, match="frame_count"):
        generate_motion_clip(specimen, "locomote", frame_count=2)
    with pytest.raises(ValueError, match="frame_count"):
        generate_motion_clip(specimen, "locomote", frame_count=257)
    with pytest.raises(ValueError, match="fps"):
        generate_motion_clip(specimen, "locomote", fps=0)


def test_pose_blending_has_exact_endpoints_and_deterministic_midpoint() -> None:
    first = blend_motion_poses(
        "idle_breathe", "attack", weight=0.0, phase=0.4, family=0
    )
    second = blend_motion_poses(
        "idle_breathe", "attack", weight=1.0, phase=0.4, family=0
    )
    midpoint_a = blend_motion_poses(
        "idle_breathe", "attack", weight=0.5, phase=0.4, family=0
    )
    midpoint_b = blend_motion_poses(
        "idle_breathe", "attack", weight=0.5, phase=0.4, family=0
    )
    assert first.pose != second.pose
    assert midpoint_a == midpoint_b
    assert midpoint_a.pose.root_dy == pytest.approx(
        (first.pose.root_dy + second.pose.root_dy) * 0.5
    )
    with pytest.raises(ValueError, match="weight"):
        blend_motion_poses("joy", "fear", weight=1.1, phase=0.5, family=3)
