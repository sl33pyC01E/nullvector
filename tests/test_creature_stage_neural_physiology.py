from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.creature_stage_neural_physiology.contract import (
    DEFAULT_TEACHER,
    FLUID_SLOTS,
    MAX_CELLS,
    CellularPhysiologyTransformerConfig,
    source_sha256,
)
from forge.creature_stage_neural_physiology.dataset import NativeInterventionTeacher, PhysiologyBatchSampler
from forge.creature_stage_neural_physiology.model import CellularPhysiologyTransformer, cellular_physiology_loss
from forge.creature_stage_neural_physiology.training import (
    assert_training_window,
    prepare_production,
    run_cpu_smoke,
    validate_cpu_smoke,
)


@pytest.fixture(scope="module")
def teacher() -> NativeInterventionTeacher:
    return NativeInterventionTeacher(DEFAULT_TEACHER)


def _small_model() -> CellularPhysiologyTransformer:
    return CellularPhysiologyTransformer(
        CellularPhysiologyTransformerConfig(
            width=64, depth=2, heads=4, feedforward_multiplier=3,
            condition_width=128, fluid_width=64, fluid_depth=2, dropout=0.0,
        )
    )


def _forward(model: CellularPhysiologyTransformer, batch: dict[str, torch.Tensor]):
    return model(
        batch["static"], batch["cell_state"], batch["summary_state"], batch["fluid_state"],
        batch["mask"], batch["adjacency"], batch["family"], batch["morphotype"],
        batch["intervention"], batch["phase"], batch["events"],
    )


def test_teacher_contract_covers_cells_organs_fluids_and_splits(teacher: NativeInterventionTeacher) -> None:
    assert [len(teacher.split_chassis("train", family)) for family in range(5)] == [2] * 5
    assert [len(teacher.split_chassis("validation", family)) for family in range(5)] == [1] * 5
    assert [len(teacher.split_chassis("test", family)) for family in range(5)] == [1] * 5
    sample = teacher.sample(0, 4, 16)
    assert sample["static"].shape == (MAX_CELLS, 61)
    assert sample["cell_state"].shape == sample["cell_target"].shape == (MAX_CELLS, 4)
    assert sample["summary_state"].shape == sample["summary_target"].shape == (10,)
    assert sample["fluid_state"].shape == sample["fluid_target"].shape == (FLUID_SLOTS, 7)
    assert sample["events"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert sample["mask"].sum() == sample["cell_count"]
    assert not sample["static"].flags.writeable
    active = sample["fluid_target"][:, 6] > 0.5
    assert int(active.sum()) == sample["fluid_count"]
    assert float(abs(sample["fluid_target"][~active]).max(initial=0.0)) == 0.0


def test_sampler_is_deterministic_and_family_balanced(teacher: NativeInterventionTeacher) -> None:
    first = PhysiologyBatchSampler(teacher, batch_size=10, seed=71)
    second = PhysiologyBatchSampler(teacher, batch_size=10, seed=71)
    coordinates = [first.coordinates(step) for step in range(12)]
    assert coordinates == [second.coordinates(step) for step in range(12)]
    for row in coordinates:
        assert [teacher.chassis[chassis]["family_id"] for chassis, _, _ in row] == [0, 1, 2, 3, 4] * 2
    assert len({item for row in coordinates for item in row}) > 80


def test_transformer_is_substantial_and_preserves_cell_support(teacher: NativeInterventionTeacher) -> None:
    assert CellularPhysiologyTransformer().parameter_count == 16_711_701
    batch = PhysiologyBatchSampler(teacher, batch_size=5, seed=7).batch(2)
    model = _small_model().eval()
    with torch.inference_mode():
        cell, summary, fluid = _forward(model, batch)
    assert cell.shape == (5, MAX_CELLS, 4)
    assert summary.shape == (5, 10)
    assert fluid.shape == (5, FLUID_SLOTS, 7)
    assert torch.isfinite(cell).all() and torch.isfinite(summary).all() and torch.isfinite(fluid).all()
    assert float(cell[~batch["mask"]].abs().max()) == 0.0
    assert float(summary.min()) >= 0.0 and float(summary.max()) <= 1.0


def test_intervention_condition_changes_future_without_changing_support(teacher: NativeInterventionTeacher) -> None:
    batch = PhysiologyBatchSampler(teacher, batch_size=5, seed=8).batch(3)
    model = _small_model().eval()
    changed_intervention = (batch["intervention"] + 1) % 9
    with torch.inference_mode():
        original = _forward(model, batch)[0]
        changed = model(
            batch["static"], batch["cell_state"], batch["summary_state"], batch["fluid_state"],
            batch["mask"], batch["adjacency"], batch["family"], batch["morphotype"],
            changed_intervention, batch["phase"], batch["events"],
        )[0]
    assert float((original - changed).abs()[batch["mask"]].mean()) > 1e-4
    assert float(changed[~batch["mask"]].abs().max()) == 0.0


def test_joint_cell_organ_and_fluid_loss_backpropagates(teacher: NativeInterventionTeacher) -> None:
    batch = PhysiologyBatchSampler(teacher, batch_size=5, seed=9).batch(4)
    model = _small_model()
    predicted = _forward(model, batch)
    loss, pieces = cellular_physiology_loss(
        predicted, batch["cell_target"], batch["summary_target"], batch["fluid_target"],
        batch["mask"], batch["adjacency"],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in pieces.values())
    assert float(model.blocks[0].attention.in_proj_weight.grad.abs().sum()) > 0.0
    assert float(model.blocks[0].graph[0].weight.grad.abs().sum()) > 0.0
    assert float(model.fluid_out[-1].weight.grad.abs().sum()) > 0.0
    assert float(model.summary_out[-2].weight.grad.abs().sum()) > 0.0


def test_cpu_smoke_and_exact_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "smoke"
    report = run_cpu_smoke(output, steps=3)
    assert report["passed"] and report["production_parameters"] >= 15_000_000
    assert validate_cpu_smoke(output) == report
    manifest_path = output / "smoke_manifest.json"
    original = manifest_path.read_bytes()
    payload = json.loads(original)
    payload["gates"]["fluid_predictions_finite"] = False
    manifest_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="ascii")
    try:
        with pytest.raises(ValueError, match="authority drifted"):
            validate_cpu_smoke(output)
    finally:
        manifest_path.write_bytes(original)


def test_production_contract_refuses_busy_or_absent_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "production"
    contract = prepare_production(output, total_steps=4000, segment_steps=400, batch_size=5)
    assert contract["model"] == CellularPhysiologyTransformerConfig().to_dict()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="16 GiB free VRAM"):
        assert_training_window(output)


def test_source_identity_is_stable() -> None:
    assert len(source_sha256()) == 64
    assert source_sha256() == source_sha256()
