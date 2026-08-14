from __future__ import annotations

from pathlib import Path

import numpy as np

from forge.map_decorator_production.teacher import semantic_teacher_targets
from forge.maps.io import load_map_pack
from forge.neural_decorated_maps.compiler import feature_seed, select_source_maps
from forge.neural_decorated_maps.contract import (
    NEURAL_DECORATED_MAP_CONTRACT_SHA256,
    contract_manifest,
)
from forge.neural_decorated_maps.renderer import composite_frame, render_selected_map
from forge.map_decorator.hashing import json_sha256
from forge.maps.model import THEMES


MAPS = Path("outputs/maps_v2_forge_lab")


def test_neural_decorated_map_contract_preserves_topology_and_runtime_boundary() -> None:
    manifest = contract_manifest()
    assert NEURAL_DECORATED_MAP_CONTRACT_SHA256 == json_sha256(manifest)
    assert manifest["authority"]["topology_v2_arrays_immutable"] is True
    assert manifest["authority"]["selection_checkpoint_not_shipped"] is True
    assert manifest["authority"]["runtime_assets"] == [".json", ".png"]
    assert manifest["authority"]["neural_heads_authorized"] == ["decal", "prop"]
    assert manifest["authority"]["semantic_heads_authorized"] == ["variant", "emission"]
    assert manifest["authority"]["unsupported_neural_heads_cross_runtime_boundary"] is False


def test_source_registry_selects_exactly_one_current_v2_map_per_theme() -> None:
    sources = select_source_maps(MAPS)
    assert tuple(sources) == THEMES
    assert all(load_map_pack(path).theme == theme for theme, path in sources.items())


def test_selected_renderer_is_exact_and_does_not_change_topology() -> None:
    data = load_map_pack(select_source_maps(MAPS)["garden"])
    original = {name: value.copy() for name, value in data.arrays().items()}
    fields, _, report = semantic_teacher_targets(data)
    assert report["passed"]
    first = render_selected_map(data, fields)
    second = render_selected_map(data, fields)
    assert np.array_equal(first.base_color, second.base_color)
    assert np.array_equal(first.emissive, second.emissive)
    assert np.array_equal(first.hazard_color_frames, second.hazard_color_frames)
    assert np.array_equal(composite_frame(first, 3), composite_frame(second, 3))
    assert all(np.array_equal(data.arrays()[name], value) for name, value in original.items())
    assert not bool((first.collision != (data.walkability == 0)).any())


def test_feature_seed_is_deterministic_unsigned_and_identity_sensitive() -> None:
    values = [feature_seed(seed) for seed in (0, 1, 2, (1 << 64) - 1)]
    assert len(set(values)) == 4
    assert all(0 <= value < (1 << 64) for value in values)
