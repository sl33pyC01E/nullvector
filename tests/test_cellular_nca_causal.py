from __future__ import annotations

import torch

from forge.cellular_nca.contract import CellularNCAConfig
from forge.cellular_nca.corpus import load_corpus
from forge.cellular_nca.model import OrganismCellularAutomaton
from forge.cellular_nca.teacher import teacher_step
from forge.cellular_nca_causal.contract import PARENT_OUTPUT, causal_source_sha256
from forge.cellular_nca_causal.curriculum import SYSTEMS, apply_system_ablation, causal_contrast_loss, make_targeted_pairs, system_mask


def _batch(count: int = 4):
    arrays = load_corpus(PARENT_OUTPUT)["arrays"]
    return torch.from_numpy(arrays["static"][:count]), torch.from_numpy(arrays["initial_state"][:count]), torch.from_numpy(arrays["live_bonds"][:count])


def test_targeted_ablation_covers_all_systems_without_chassis_escape() -> None:
    static, initial, _ = _batch(); systems = torch.arange(4, dtype=torch.long); mask = system_mask(static, systems); damaged = apply_system_ablation(initial, static, systems)
    assert mask.shape == (4, 1, 48, 48) and bool((mask.sum((1, 2, 3)) > 0).all())
    assert bool((damaged[:, 0] < initial[:, 0]).any()) and float((damaged[:, :9] * (1 - static[:, :1])).abs().max()) == 0


def test_pre_roll_is_deterministic_and_develops_counterfactuals() -> None:
    static, initial, bonds = _batch(); systems = torch.arange(4, dtype=torch.long); rolls = torch.tensor((0, 4, 8, 16), dtype=torch.long)
    control_a, damaged_a = make_targeted_pairs(static, initial, bonds, systems, rolls); control_b, damaged_b = make_targeted_pairs(static, initial, bonds, systems, rolls)
    assert torch.equal(control_a, control_b) and torch.equal(damaged_a, damaged_b)
    assert bool(((control_a - damaged_a).abs().sum((1, 2, 3)) > 0).all())


def test_contrast_loss_penalizes_identity_and_has_gradient() -> None:
    static, initial, bonds = _batch(); systems = torch.arange(4, dtype=torch.long); rolls = torch.tensor((0, 4, 8, 16), dtype=torch.long); control, damaged = make_targeted_pairs(static, initial, bonds, systems, rolls); target_control = teacher_step(static, control, bonds); target_damaged = teacher_step(static, damaged, bonds)
    predicted_control = control.clone().requires_grad_(True); predicted_damaged = damaged.clone().requires_grad_(True); loss, pieces = causal_contrast_loss(predicted_control, predicted_damaged, target_control, target_damaged, static, systems); loss.backward()
    assert float(loss) > 0 and float(pieces["magnitude"]) > 0 and predicted_control.grad is not None and float(predicted_control.grad.abs().sum()) > 0


def test_small_model_can_backpropagate_causal_pair() -> None:
    static, initial, bonds = _batch(); systems = torch.arange(4, dtype=torch.long); rolls = torch.tensor((0, 4, 8, 16), dtype=torch.long); control, damaged = make_targeted_pairs(static, initial, bonds, systems, rolls); model = OrganismCellularAutomaton(CellularNCAConfig(width=64, depth=4)); pair_static = torch.cat((static, static)); pair_bonds = torch.cat((bonds, bonds)); predicted = model(pair_static, torch.cat((control, damaged)), pair_bonds); target_control = teacher_step(static, control, bonds); target_damaged = teacher_step(static, damaged, bonds); loss, _ = causal_contrast_loss(predicted[:4], predicted[4:], target_control, target_damaged, static, systems); loss.backward()
    assert torch.isfinite(loss) and any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for parameter in model.parameters())


def test_contract_binds_parent_and_four_named_systems() -> None:
    assert [row[0] for row in SYSTEMS] == ["circulation", "respiration", "digestion", "neural"]
    assert len(causal_source_sha256()) == 64
