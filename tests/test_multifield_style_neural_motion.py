from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pytest

from forge.morphology import FAMILIES
from forge.multifield_style_neural_motion import (
    bind_candidate,
    compute_binding_census,
    load_neural_motion_source,
    load_neural_style_parent,
    render_neural_motion_frame,
)
from forge.multifield_style_neural_motion.compiler import build_contract
from forge.multifield_style_neural_motion.replay import replay_neural_motion_style_bank
from forge.multifield_style_neural_motion.sharding import (
    MOTION_SHARDS,
    compile_motion_shard_payload,
    load_motion_shard,
)
from forge.multifield_style_motion.hashing import canonical_json_bytes
from forge.neural_rig_bridge import (
    BindingRejected,
    assert_exact_neural_motion_replay,
    assert_valid_neural_motion_clip,
    compile_neural_motion_clip,
    replay_neural_motion_clip,
)
from forge.neural_rig_bridge.hashing import aligned_fields_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "production_handoff_v2"
    / "final_best_stratified80_bank_attempt1"
    / "generation_manifest.json"
)
STYLE_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "multifield_style"
    / "final_best_stratified80_v3"
    / "style_manifest.json"
)


@pytest.fixture(scope="module")
def source():
    return load_neural_motion_source(GENERATION_MANIFEST)


@pytest.fixture(scope="module")
def style_parent(source):
    return load_neural_style_parent(STYLE_MANIFEST, source)


@pytest.fixture(scope="module")
def humanoid_motion_shard(tmp_path_factory, source, style_parent):
    root = tmp_path_factory.mktemp("neural_motion_shard")
    contract = build_contract(source, style_parent)
    payload = compile_motion_shard_payload(source, style_parent, "humanoid", 3, contract)
    for relative, data in payload.file_payloads.items():
        path = root / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    loaded = load_motion_shard(root, source, style_parent, "humanoid", 3, contract)
    return root, loaded, contract


def test_final_neural_source_and_train_tuple_provenance_are_exact(source) -> None:
    assert source.bank.manifest_sha256 == "02f8aacc10688c27bf3e2123b758e637ed7ab0aabf1c92e7728e1aa6b7b976c9"
    assert source.corpus_sha256 == "77dc7313ca6411295bad883f483a6edf4be75016ebfd7c107d0f286d2cb1cd7b"
    assert source.split_fingerprint == "5e400872460dc527c01a2a301f006e761abd1621773c5f67b45568d68886007b"
    assert source.legal_tuple_fingerprint == "0b15074b76ca69ea9a93e0b73db7e5df0b242dc0ecc46c5e842342fb0378948d"
    assert source.legal_tuples.shape == (69, 3)
    assert source.legal_tuples.dtype == np.uint8
    assert source.legal_tuples.flags.writeable is False
    assert list(source.candidates_by_family) == list(FAMILIES)
    assert all(len(source.candidates_by_family[family]) == 16 for family in FAMILIES)


def test_first_production_identity_per_family_binds_to_exact_neural_fields(source) -> None:
    binding_hashes: set[str] = set()
    for family in FAMILIES:
        selected = bind_candidate(source, source.candidates_by_family[family][0])
        binding = selected.binding
        sample = selected.candidate.sample
        assert binding.family == family
        assert binding.sample_id == sample.condition.sample_id
        assert binding.raw_fields_sha256 == sample.raw_fields_sha256
        assert aligned_fields_hash(
            binding.part_owner,
            binding.material,
            binding.emission_level,
        ) == sample.fields.aligned_sha256
        assert np.array_equal(binding.part_owner, sample.fields.part)
        assert np.array_equal(binding.material, sample.fields.material)
        assert np.array_equal(binding.emission_level, sample.fields.emission)
        binding_hashes.add(binding.sha256)
    assert len(binding_hashes) == len(FAMILIES)


def test_all_80_binding_census_is_explicit_and_does_not_claim_80_animations(source) -> None:
    census = compute_binding_census(source)
    assert census["sample_count"] == 80
    assert census["bindable_count"] == 70
    assert census["rejected_count"] == 10
    assert [item["count"] for item in census["rejection_categories"]] == [3, 1, 3, 3]
    assert census["animation_bank_scope"]["selected_identity_count"] == 5
    assert census["animation_bank_scope"]["all_80_animated"] is False


def test_evaluator_tuple_table_is_required_for_raw_provenance(source) -> None:
    candidate = source.candidates_by_family["humanoid"][0]
    from forge.neural_rig_bridge import bind_raw_sample_archive

    with pytest.raises(BindingRejected, match="legal tuple fingerprint"):
        bind_raw_sample_archive(
            candidate.raw_archive_path,
            raw_manifest_path=candidate.raw_manifest_path,
        )


def test_static_neural_style_parent_is_fully_hash_bound(source, style_parent) -> None:
    assert style_parent.manifest_sha256 == "d0b09a06a407992e2cbdd9479fac8b9d39f7d6cb2ef44a0746c0135d5e2bf86a"
    assert style_parent.manifest["parent"]["manifest_sha256"] == source.bank.manifest_sha256
    assert style_parent.manifest["compiler"]["source_sha256"] == "af90198ae33642c345627a0fe0211de4eba189eabf8b55f469a4aec1ebb68c2c"
    assert len(style_parent.palettes) == 80
    assert all(
        palette["format"] == "nullvector-perceptual-palette-v1"
        for palette in style_parent.palettes.values()
    )


def test_public_motion_and_presentation_are_exact_and_authority_preserving(source, style_parent) -> None:
    selected = bind_candidate(source, source.candidates_by_family["humanoid"][0])
    clip = compile_neural_motion_clip(selected.binding, "locomote", facing="southeast")
    assert_valid_neural_motion_clip(clip)
    assert_exact_neural_motion_replay(replay_neural_motion_clip(clip))
    sample = selected.candidate.sample
    palette = style_parent.palettes[sample.condition.sample_id]
    palette_sha256 = style_parent.palette_artifacts[sample.condition.sample_id]["sha256"]
    rendered = [
        render_neural_motion_frame(
            frame,
            sample.condition,
            sample.fields.aligned_sha256,
            palette,
            palette_sha256,
        )
        for frame in clip.frames
    ]
    assert all(all(frame.gates.values()) for frame in rendered)
    assert len({frame.palette_sha256 for frame in rendered}) == 1
    assert rendered[0].categorical_sha256 == rendered[-1].categorical_sha256
    assert rendered[0].presentation_sha256 == rendered[-1].presentation_sha256
    assert all(
        np.array_equal(rendered[0].layers[name], rendered[-1].layers[name])
        for name in rendered[0].layers
    )


def test_plantlike_half_pixel_boundary_clip_is_canonical_and_exact(source) -> None:
    selected = bind_candidate(source, source.candidates_by_family["plantlike"][0])
    clip = compile_neural_motion_clip(selected.binding, "joy", facing="southeast")
    assert_valid_neural_motion_clip(clip)
    assert_exact_neural_motion_replay(replay_neural_motion_clip(clip))
    for frame in clip.frames:
        assert all(
            not (value == 0.0 and math.copysign(1.0, value) < 0.0)
            for matrix in frame.fields.manifest["source_to_destination"].values()
            for row in matrix
            for value in row
        )


def test_production_humanoid_cast_has_canonical_signed_zero_transforms(source) -> None:
    selected = bind_candidate(source, source.candidates_by_family["humanoid"][0])
    clip = compile_neural_motion_clip(selected.binding, "cast", facing="north")
    assert_valid_neural_motion_clip(clip)
    assert_exact_neural_motion_replay(replay_neural_motion_clip(clip))
    for frame in clip.frames:
        for matrix in frame.fields.manifest["source_to_destination"].values():
            assert all(
                not (value == 0.0 and str(value).startswith("-"))
                for row in matrix
                for value in row
            )


def test_production_plantlike_joy_southeast_replays_raster_boundary(source) -> None:
    selected = bind_candidate(source, source.candidates_by_family["plantlike"][0])
    clip = compile_neural_motion_clip(selected.binding, "joy", facing="southeast")
    assert_valid_neural_motion_clip(clip)
    assert_exact_neural_motion_replay(replay_neural_motion_clip(clip))


def test_bounded_motion_shard_has_exact_arrays_and_authority(humanoid_motion_shard) -> None:
    _, loaded, _ = humanoid_motion_shard
    assert loaded.manifest["motions"] == list(MOTION_SHARDS[3])
    assert loaded.manifest["clip_count"] == len(MOTION_SHARDS[3]) * 8
    assert loaded.manifest["frame_count"] == 200
    assert all(loaded.manifest["gates"].values())
    assert loaded.arrays["presentation_sha256"].shape == (200, 7)


def test_shard_runtime_rejects_canonical_manifest_tamper(
    source, style_parent, humanoid_motion_shard
) -> None:
    root, loaded, contract = humanoid_motion_shard
    manifest_path = next(root.glob("_build_shards/humanoid/*/shard_03/shard_manifest.json"))
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["clips"][0], tampered["clips"][1] = tampered["clips"][1], tampered["clips"][0]
    manifest_path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="clip order"):
        load_motion_shard(root, source, style_parent, "humanoid", 3, contract)
    manifest_path.write_bytes(canonical_json_bytes(dict(loaded.manifest)))


def test_replay_source_enforces_ffmpeg_and_encoding_provenance() -> None:
    source_text = Path(replay_neural_motion_style_bank.__code__.co_filename).read_text(encoding="utf-8")
    assert 'dict(showcase.ffmpeg) != manifest["showcase"]["ffmpeg"]' in source_text
    assert 'dict(showcase.encoding) != manifest["showcase"]["encoding"]' in source_text
