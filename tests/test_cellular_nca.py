from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.cellular_nca.contract import CellularNCAConfig, DYNAMIC_CHANNELS, STATIC_CHANNELS
from forge.cellular_nca.corpus import build_corpus, load_corpus
from forge.cellular_nca.model import OrganismCellularAutomaton, parameter_count
from forge.cellular_nca.teacher import cellular_loss, make_scenarios, teacher_step


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("cellular-nca-corpus")
    report = build_corpus(output)
    assert report["passed"] and report["identity_count"] == 45 and report["cell_count"] == 25668
    return load_corpus(output)


def test_authoritative_corpus_is_complete_and_bounded(corpus) -> None:
    arrays = corpus["arrays"]
    assert arrays["static"].shape == (45, STATIC_CHANNELS, 48, 48)
    assert arrays["initial_state"].shape == (45, DYNAMIC_CHANNELS, 48, 48)
    assert arrays["live_bonds"].shape == (45, 8, 48, 48)
    assert set(arrays["family_id"].tolist()) == set(range(5))
    assert np.array_equal(arrays["initial_state"][:, 11], arrays["static"][:, 0])


def test_scenario_generation_is_exact_in_seed_and_severs_bonds(corpus) -> None:
    arrays = corpus["arrays"]; static = torch.from_numpy(arrays["static"][:5]); state = torch.from_numpy(arrays["initial_state"][:5]); bonds = torch.from_numpy(arrays["live_bonds"][:5])
    a_state, a_bonds = make_scenarios(static, state, bonds, torch.Generator().manual_seed(71)); b_state, b_bonds = make_scenarios(static, state, bonds, torch.Generator().manual_seed(71))
    assert torch.equal(a_state, b_state) and torch.equal(a_bonds, b_bonds)
    assert bool((a_bonds < bonds).any()) and bool((a_state[:, 7] > 0).any()) and bool((a_state[:, 9] > 0).any())


def test_teacher_preserves_chassis_and_diffuses_surface_fluid(corpus) -> None:
    arrays = corpus["arrays"]; static = torch.from_numpy(arrays["static"][:5]); state = torch.from_numpy(arrays["initial_state"][:5]); bonds = torch.from_numpy(arrays["live_bonds"][:5]); state, bonds = make_scenarios(static, state, bonds, torch.Generator().manual_seed(19)); result = teacher_step(static, state, bonds)
    outside = 1 - static[:, :1]
    assert float((result[:, :9] * outside).abs().max()) == 0
    assert float((result[:, 9:10] * outside).sum()) > 0
    assert torch.isfinite(result).all() and float(result.min()) >= 0 and float(result.max()) <= 1


def test_model_has_real_capacity_and_bounded_updates(corpus) -> None:
    arrays = corpus["arrays"]; static = torch.from_numpy(arrays["static"][:2]); state = torch.from_numpy(arrays["initial_state"][:2]); bonds = torch.from_numpy(arrays["live_bonds"][:2]); config = CellularNCAConfig(width=64, depth=4); model = OrganismCellularAutomaton(config)
    assert parameter_count(OrganismCellularAutomaton()) > 9_000_000
    predicted = model(static, state, bonds)
    assert predicted.shape == state.shape and torch.isfinite(predicted).all()
    assert float((predicted[:, :9] * (1 - static[:, :1])).abs().max()) == 0
    assert float((predicted - state).abs().max()) <= config.max_delta + 1e-6


def test_velocity_loss_penalizes_identity_solution(corpus) -> None:
    arrays = corpus["arrays"]; static = torch.from_numpy(arrays["static"][:3]); state = torch.from_numpy(arrays["initial_state"][:3]); bonds = torch.from_numpy(arrays["live_bonds"][:3]); state, bonds = make_scenarios(static, state, bonds, torch.Generator().manual_seed(7)); target = teacher_step(static, state, bonds)
    loss, pieces = cellular_loss(state, target, static, state)
    assert float(loss) > float(pieces["reconstruction"])
    assert float(pieces["velocity"]) > 0

