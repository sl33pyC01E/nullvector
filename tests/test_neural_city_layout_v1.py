import numpy as np
import torch

from forge.neural_city_layout_v1 import CLASSES, GRID_SIZE, ModelConfig, NeuralCityLayout, build_corpus, compile_city_layout, render_teacher_city, validate_compiled_city
from forge.neural_city_layout_v1.contract import CONDITION_NAMES, MASK_TOKEN
from forge.neural_city_layout_v1.evaluation import PALETTE


def test_teacher_corpus_is_deterministic_diverse_and_bounded() -> None:
    left = build_corpus(64, seed=731)
    right = build_corpus(64, seed=731)
    assert [item.identity for item in left] == [item.identity for item in right]
    assert len(set(item.identity for item in left)) == 64
    assert all(item.target.shape == (GRID_SIZE, GRID_SIZE) and item.target.dtype == np.uint8 for item in left)
    assert all(int(item.target.max()) < len(CLASSES) for item in left)
    assert len(set(item.target.tobytes() for item in left)) == 64


def test_city_model_contract_and_gradient() -> None:
    model = NeuralCityLayout(ModelConfig(width=16, levels=2, blocks_per_level=1))
    tokens = torch.full((2, GRID_SIZE, GRID_SIZE), MASK_TOKEN, dtype=torch.long)
    conditions = torch.zeros((2, len(CONDITION_NAMES)))
    logits = model(tokens, conditions)
    assert logits.shape == (2, len(CLASSES), GRID_SIZE, GRID_SIZE)
    logits.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_compiler_removes_isolated_structural_cells() -> None:
    raw = render_teacher_city(91)
    raw[0, 0] = CLASSES.index("wall")
    compiled, diagnostics = compile_city_layout(raw)
    validation = validate_compiled_city(compiled)
    assert diagnostics["edited_cells"] >= 1
    assert validation["passed"] is True
    assert PALETTE.shape == (len(CLASSES), 3)
