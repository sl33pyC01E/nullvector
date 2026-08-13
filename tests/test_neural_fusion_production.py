from __future__ import annotations

import numpy as np
import pytest

from forge.neural_fusion.genetics import _components
from forge.neural_fusion.rig import build_fusion_binding
from forge.neural_fusion_production import FUSION_MODES, MUTATION_MODES, production_fusion_source_hash, production_latent_fuse
from forge.neural_fusion_production.codec import load_production_codec
from forge.neural_rig_repair_style import load_repair_style_authority


@pytest.fixture(scope="module")
def sources():
    return load_repair_style_authority().repair_source.samples


def test_production_codec_authority_is_accepted_and_frozen() -> None:
    authority = load_production_codec()
    assert authority.manifest["status"] == "ready"
    assert authority.manifest["gates"]["full_quality_accepted"] is True
    assert authority.manifest["best"]["epoch"] == 24
    assert authority.ema_state_sha256 == "0f07e2946f313e18036944fa50658ab04d625a2e323521f23416f02171723e6b"
    assert len(production_fusion_source_hash()) == 64


def test_production_latent_fusion_is_deterministic_legal_and_riggable(sources) -> None:
    keywords = dict(seed=0x50524F44, alpha=0.5, fusion_mode="spatial_weave", mutation_mode="spatial_burst", mutation_strength=2)
    first = production_latent_fuse(sources[0], sources[16], **keywords)
    second = production_latent_fuse(sources[0], sources[16], **keywords)
    assert first.fields_sha256 == second.fields_sha256
    assert first.provenance_sha256 == second.provenance_sha256
    assert np.array_equal(first.part_owner, second.part_owner)
    assert len(_components(first.part_owner != 0)) == 1
    assert first.metrics["quality_tier"] == "production-learned-latent-authority-v1"
    assert min(first.metrics["parent_a_attributed_pixels"], first.metrics["parent_b_attributed_pixels"]) >= 8
    assert first.metrics["unique_codes"] > 1
    binding = build_fusion_binding(first)
    assert binding.raw_fields_sha256 == first.fields_sha256


@pytest.mark.parametrize("index", range(len(FUSION_MODES)))
def test_operator_matrix_preserves_topology_and_tuple_authority(sources, index: int) -> None:
    mutation = MUTATION_MODES[index]
    specimen = production_latent_fuse(
        sources[(index * 7) % 16],
        sources[16 + (index * 9) % 64],
        seed=0x600D0000 + index * 101,
        alpha=(0.3, 0.5, 0.7)[index % 3],
        fusion_mode=FUSION_MODES[index],
        mutation_mode=mutation,
        mutation_strength=0 if mutation == "none" else 1 + index % 3,
    )
    legal = {tuple(map(int, row)) for row in specimen.legal_tuples}
    observed = {tuple(map(int, row)) for row in np.stack((specimen.part_owner, specimen.material, specimen.emission_level), axis=-1).reshape(-1, 3)}
    assert observed <= legal
    assert len(_components(specimen.part_owner != 0)) == 1
    assert 0.02 <= float((specimen.part_owner != 0).mean()) <= 0.60
    assert specimen.metrics["production_ema_sha256"] == "0f07e2946f313e18036944fa50658ab04d625a2e323521f23416f02171723e6b"


def test_production_latent_fusion_rejects_invalid_operator(sources) -> None:
    with pytest.raises(ValueError, match="operator contract"):
        production_latent_fuse(sources[0], sources[16], seed=1, fusion_mode="bogus")
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        production_latent_fuse(sources[0], sources[16], seed=2**40)
