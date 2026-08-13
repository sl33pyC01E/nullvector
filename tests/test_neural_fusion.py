from __future__ import annotations

import numpy as np
import pytest

from forge.neural_fusion import FUSION_MODES, LATENT_MODES, MUTATION_MODES, build_fusion_binding, fuse_specimen, latent_fuse
from forge.neural_rig_repair.motion import compile_motion_clip_audit
from forge.neural_rig_repair_style import load_repair_style_authority


@pytest.fixture(scope="module")
def source():
    return load_repair_style_authority().repair_source


def test_cross_family_fusion_is_deterministic_legal_connected_and_riggable(source):
    first = fuse_specimen(
        source.samples[0],
        source.samples[32],
        seed=0xF0510A,
        fusion_mode="chimera",
        mutation_mode="phase_bloom",
        mutation_strength=2,
    )
    second = fuse_specimen(
        source.samples[0],
        source.samples[32],
        seed=0xF0510A,
        fusion_mode="chimera",
        mutation_mode="phase_bloom",
        mutation_strength=2,
    )
    assert first.fields_sha256 == second.fields_sha256
    assert first.provenance_sha256 == second.provenance_sha256
    assert np.array_equal(first.part_owner, second.part_owner)
    assert first.metrics["component_count"] == 1
    assert first.metrics["parent_a_pixels"] >= 20
    assert first.metrics["parent_b_pixels"] >= 20
    binding = build_fusion_binding(first)
    audit = compile_motion_clip_audit(binding, "idle_wiggle", "northeast")
    assert audit["frame_count"] == 9
    assert all(audit["gates"].values())


@pytest.mark.parametrize("fusion_mode", FUSION_MODES)
@pytest.mark.parametrize("mutation_mode", MUTATION_MODES)
def test_operator_matrix_preserves_tuple_vocabulary(source, fusion_mode, mutation_mode):
    specimen = fuse_specimen(
        source.samples[16],
        source.samples[64],
        seed=0xC0DEC0DE + FUSION_MODES.index(fusion_mode) * 97 + MUTATION_MODES.index(mutation_mode),
        fusion_mode=fusion_mode,
        mutation_mode=mutation_mode,
        mutation_strength=1,
        dominant_parent="b" if mutation_mode in {"scar", "bilateral_break"} else "a",
    )
    legal = {tuple(map(int, row)) for row in specimen.legal_tuples}
    observed = {
        tuple(map(int, row))
        for row in np.stack((specimen.part_owner, specimen.material, specimen.emission_level), axis=-1).reshape(-1, 3)
    }
    assert observed <= legal
    assert specimen.metrics["component_count"] == 1


@pytest.mark.parametrize("mode", LATENT_MODES)
def test_experimental_learned_latent_fusion_is_legal_connected_and_riggable(source, mode):
    specimen = latent_fuse(
        source.samples[0],
        source.samples[48],
        seed=0x1A7E0000 + LATENT_MODES.index(mode),
        alpha=0.45,
        mode=mode,
    )
    assert specimen.metrics["quality_tier"] == "experimental-smoke-codec-not-production"
    assert specimen.metrics["component_count"] == 1
    binding = build_fusion_binding(specimen)
    audit = compile_motion_clip_audit(binding, "idle_breathe", "north")
    assert all(audit["gates"].values())
