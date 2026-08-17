import pytest
import torch

from forge.mobile_cell_nca_v1 import MobileCellNCA, MobileCellNCAConfig
from forge.mobile_cell_nca_v1.contract import MobileCellNCAPlan, tensor_state_sha256
from forge.mobile_cell_nca_v1.training import _load, _split


def test_mobile_cell_nca_shape_bounds_and_budget() -> None:
    model = MobileCellNCA(MobileCellNCAConfig(width=32, depth=4))
    static = torch.zeros(2, 85, 48, 48); static[:, 0, 8:40, 8:40] = 1
    state = torch.rand(2, 12, 48, 48); bonds = torch.ones(2, 8, 48, 48)
    output = model(static, state, bonds)
    assert output.shape == state.shape and output.min() >= 0 and output.max() <= 1
    assert not output[:, :9, :8].any() and model.parameter_count < 1_000_000


def test_production_default_is_mobile_bounded_and_input_contracts_fail_closed() -> None:
    model = MobileCellNCA()
    assert model.parameter_count == 492_492 and model.parameter_count < 750_000
    with pytest.raises(ValueError, match="static tensor"):
        model(torch.zeros(1, 84, 48, 48), torch.zeros(1, 12, 48, 48), torch.zeros(1, 8, 48, 48))
    with pytest.raises(ValueError):
        MobileCellNCAConfig(width=70)
    with pytest.raises(ValueError):
        MobileCellNCAPlan(rollout_steps=3)


def test_authoritative_corpus_and_teacher_are_frozen_and_family_heldout_is_exact() -> None:
    arrays, teacher = _load()
    train, heldout = _split(arrays["family_id"])
    assert len(train) == 40 and len(heldout) == 5
    assert arrays["family_id"][heldout].tolist() == [0, 1, 2, 3, 4]
    assert sum(value.numel() for value in teacher.parameters()) == 9_991_180


def test_model_state_fingerprint_is_order_independent_and_sensitive() -> None:
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert tensor_state_sha256(first) == tensor_state_sha256(second)
    second["b"][0] = 3
    assert tensor_state_sha256(first) != tensor_state_sha256(second)
