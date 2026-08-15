from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forge.creature_stage_developmental_motion.compiler import validate_candidate_corpus
from forge.creature_stage_developmental_motion.contract import (
    DEFAULT_CORPUS,
    MAX_APPENDAGES,
    MAX_CELLS,
    MAX_MUSCLES,
    MAX_NODES,
    DevelopmentalActuatorConfig,
    DevelopmentalTrainingConfig,
    corpus_source_sha256,
    source_sha256,
)
from forge.creature_stage_developmental_motion.dataset import (
    DevelopmentalMotionTeacher,
    DevelopmentalSequenceSampler,
)
from forge.creature_stage_developmental_motion.model import (
    DevelopmentalCellularMotionTransformer,
    developmental_actuator_loss,
)
from forge.creature_stage_developmental_motion.smoke import validate_parent_adapter_smoke


SMOKE = Path("outputs/creature_stage_developmental_motion/smoke_cpu_v8")


def test_sealed_corpus_replays_exactly() -> None:
    result = validate_candidate_corpus(DEFAULT_CORPUS, replay=True)
    assert result["passed"]
    assert result["training_permitted"]
    assert result["specimens"] == 10


def test_structural_authority_is_complete_and_normalized() -> None:
    teacher = DevelopmentalMotionTeacher(DEFAULT_CORPUS)
    arrays = teacher.arrays
    assert arrays["static"].shape == (10, MAX_CELLS, 61)
    assert arrays["node_features"].shape == (10, MAX_NODES, 8)
    assert arrays["muscle_features"].shape == (10, MAX_MUSCLES, 10)
    assert arrays["planted_contacts"].shape == (10, 72, MAX_APPENDAGES)
    assert int(arrays["node_count"].max()) == 43
    assert int(arrays["muscle_count"].max()) == 60
    active = arrays["mask"]
    assert np.allclose(arrays["cell_node_weights"].sum(axis=2)[active], 1.0, atol=1e-6)
    assert np.count_nonzero(arrays["muscle_incidence"]) == int(arrays["muscle_count"].sum()) * 2


def test_cyclic_sampler_is_family_balanced_and_deterministic() -> None:
    teacher = DevelopmentalMotionTeacher(DEFAULT_CORPUS)
    sampler = DevelopmentalSequenceSampler(teacher, batch_size=10, sequence_frames=12)
    assert sampler.coordinates(7) == sampler.coordinates(7)
    coordinates = sampler.coordinates(0)
    assert [item.specimen // 2 for item in coordinates] == [0, 1, 2, 3, 4] * 2
    frames, _ = sampler.sequence(0)
    assert len(frames) == 12
    assert set(frames[0]["family"].tolist()) == set(range(5))
    assert frames[0]["static"].shape == (10, MAX_CELLS, 61)


def test_small_actuator_has_physical_output_contract_and_gradients() -> None:
    teacher = DevelopmentalMotionTeacher(DEFAULT_CORPUS)
    sampler = DevelopmentalSequenceSampler(teacher, batch_size=5, sequence_frames=6)
    frame = sampler.sequence(1)[0][0]
    config = DevelopmentalActuatorConfig(
        width=96, depth=2, heads=4, feedforward_multiplier=2,
        condition_width=96, cell_width=64, cell_graph_blocks=2, dropout=0.0,
    )
    model = DevelopmentalCellularMotionTransformer(config)
    parent_prior = torch.zeros_like(frame["state"])
    output = model(
        frame["static"], frame["state"], frame["mask"], frame["adjacency"],
        frame["node_features"], frame["node_state"], frame["node_mask"], frame["node_adjacency"],
        frame["muscle_features"], frame["muscle_state"], frame["muscle_mask"],
        frame["muscle_incidence"], frame["cell_node_weights"], parent_prior, frame["family"],
        frame["morphotype"], frame["phase"], frame["traits"],
    )
    assert output["cell_state"].shape == (5, MAX_CELLS, 4)
    assert output["node_state"].shape == (5, MAX_NODES, 4)
    assert output["muscle_activation"].shape == (5, MAX_MUSCLES)
    assert torch.count_nonzero(output["cell_state"][~frame["mask"]]) == 0
    assert torch.count_nonzero(output["node_state"][~frame["node_mask"]]) == 0
    assert torch.count_nonzero(output["muscle_activation"][~frame["muscle_mask"]]) == 0
    loss, pieces = developmental_actuator_loss(
        output, frame, frame["state"].float(), frame["node_state"].float(),
        DevelopmentalTrainingConfig(sequence_frames=6),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(pieces["outside"]) == 0.0
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in model.parameters() if parameter.requires_grad)


def test_invalid_padding_fails_closed() -> None:
    teacher = DevelopmentalMotionTeacher(DEFAULT_CORPUS)
    sampler = DevelopmentalSequenceSampler(teacher, batch_size=5, sequence_frames=6)
    frame = sampler.sequence(2)[0][0]
    config = DevelopmentalActuatorConfig(
        width=96, depth=2, heads=4, feedforward_multiplier=2,
        condition_width=96, cell_width=64, cell_graph_blocks=2, dropout=0.0,
    )
    model = DevelopmentalCellularMotionTransformer(config)
    with pytest.raises(ValueError, match="input contract"):
        model(
            frame["static"][:, :-1], frame["state"][:, :-1], frame["mask"][:, :-1], frame["adjacency"][:, :-1, :-1],
            frame["node_features"], frame["node_state"], frame["node_mask"], frame["node_adjacency"],
            frame["muscle_features"], frame["muscle_state"], frame["muscle_mask"],
            frame["muscle_incidence"], frame["cell_node_weights"][:, :-1], torch.zeros_like(frame["state"][:, :-1]), frame["family"],
            frame["morphotype"], frame["phase"], frame["traits"],
        )


def test_cpu_smoke_replays_exactly() -> None:
    result = validate_parent_adapter_smoke(SMOKE, replay=True)
    assert result["passed"]
    assert result["diagnostics"]["families"] == 5
    assert result["diagnostics"]["outside_max_abs"] == 0.0


def test_source_contracts_are_distinct_and_stable() -> None:
    assert len(corpus_source_sha256()) == 64
    assert len(source_sha256()) == 64
    assert corpus_source_sha256() != source_sha256()
