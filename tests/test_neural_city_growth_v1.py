import numpy as np
import torch

from forge.neural_city_growth_v1 import ACTIONS, GrowthCondition, GrowthModelConfig, NeuralCityGrowth, apply_teacher_growth, build_growth_corpus
from forge.neural_city_growth_v1.contract import GROWTH_CONDITION_NAMES, PATCH_SIZE, PURPOSE_COSTS
from forge.neural_city_growth_v1.projection import compile_growth_state, growth_authority_mask, project_neural_growth
from forge.neural_city_growth_v1.teacher import extract_local_patch, paste_local_patch
from forge.neural_city_layout_v1.contract import GRID_SIZE
from forge.neural_city_layout_v1.teacher import _condition


def test_growth_corpus_is_deterministic_and_contains_build_and_noop() -> None:
    left = build_growth_corpus(256, seed=87); right = build_growth_corpus(256, seed=87)
    assert [item.identity for item in left] == [item.identity for item in right]
    counts = [int(item.changed.sum()) for item in left]
    assert any(value == 0 for value in counts) and any(value > 50 for value in counts)
    assert all(item.current.shape == item.target.shape == (GRID_SIZE, GRID_SIZE) for item in left)
    assert len({item.condition.city.building_target for item in left}) >= 8
    for stage in range(4):
        assert len({item.condition.city.building_target for item in left if item.condition.stage == stage}) >= 5


def test_growth_requires_resources_and_preserves_current_on_failure() -> None:
    city = _condition(91); current = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8); action = ACTIONS[0]
    failed = GrowthCondition(city, action, (0, 0, 0, 0), (.5, .5), 0); unchanged, report = apply_teacher_growth(current, failed)
    assert report["affordable"] is False and np.array_equal(unchanged, current)
    passed = GrowthCondition(city, action, tuple(min(1, value + .1) for value in PURPOSE_COSTS[action]), (.5, .5), 0); changed, report = apply_teacher_growth(current, passed)
    assert report["affordable"] is True and int((changed != current).sum()) > 50


def test_growth_model_contract_and_gradients() -> None:
    model = NeuralCityGrowth(GrowthModelConfig(width=16, levels=2, blocks_per_level=1)); current = torch.zeros((2, GRID_SIZE, GRID_SIZE), dtype=torch.long); conditions = torch.zeros((2, len(GROWTH_CONDITION_NAMES))); logits = model(current, conditions); assert logits.shape == (2, 8, GRID_SIZE, GRID_SIZE); logits.mean().backward(); assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_projection_rejects_remote_edits_and_forces_resource_noop() -> None:
    city = _condition(77); action = ACTIONS[0]; current = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8); proposal = np.ones_like(current)
    condition = GrowthCondition(city, action, (1, 1, 1, 1), (.25, .75), 0); projected, report = project_neural_growth(current, proposal, condition); authority = growth_authority_mask(condition)
    assert np.array_equal(projected != 0, authority)
    assert report["rejected_changes"] > report["accepted_changes"] > 0
    blocked = GrowthCondition(city, action, (0, 0, 0, 0), (.25, .75), 0); result, report = project_neural_growth(current, proposal, blocked)
    assert np.array_equal(result, current) and report["accepted_changes"] == 0


def test_projection_never_repaints_committed_material() -> None:
    city = _condition(78); action = ACTIONS[1]
    current = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8); current[24:40, 24:40] = 3
    proposal = np.full_like(current, 6)
    condition = GrowthCondition(city, action, (1, 1, 1, 1), (.5, .5), 2)
    projected, report = project_neural_growth(current, proposal, condition)
    assert np.array_equal(projected[current != 0], current[current != 0])
    assert report["preserved_existing_cells"] >= int((current != 0).sum())


def test_growth_compiler_joins_all_toroidal_islands_without_repainting() -> None:
    raw = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)
    raw[2:8, 2:8] = 3; raw[28:34, 28:34] = 3; raw[55:61, 55:61] = 3
    compiled, report = compile_growth_state(raw)
    assert report["toroidal_components"] == 1
    assert np.array_equal(compiled[raw != 0], raw[raw != 0])
    assert report["edited_cells"] > 0


def test_local_patch_roundtrip_is_exact_inside_toroidal_window() -> None:
    field = np.arange(GRID_SIZE * GRID_SIZE, dtype=np.uint16).reshape(GRID_SIZE, GRID_SIZE).astype(np.uint8); site = (.98, .02); patch = extract_local_patch(field, site); assert patch.shape == (PATCH_SIZE, PATCH_SIZE); replacement = np.full_like(patch, 7); result = paste_local_patch(field, replacement, site); assert np.array_equal(extract_local_patch(result, site), replacement)
