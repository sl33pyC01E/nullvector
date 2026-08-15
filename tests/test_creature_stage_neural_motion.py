from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.creature_stage_neural_motion.contract import (
    DEFAULT_TEACHER,
    MAX_CELLS,
    CellularMotionTransformerConfig,
    source_sha256,
)
from forge.creature_stage_neural_motion.dataset import MotionBatchSampler, NativeMotionTeacher
from forge.creature_stage_neural_motion.model import CellularMotionTransformer, cellular_motion_loss
from forge.creature_stage_neural_motion.training import (
    prepare_production,
    run_cpu_smoke,
    train_segment,
    validate_cpu_smoke,
)


@pytest.fixture(scope="module")
def teacher() -> NativeMotionTeacher:
    return NativeMotionTeacher(DEFAULT_TEACHER)


def _small_model() -> CellularMotionTransformer:
    return CellularMotionTransformer(
        CellularMotionTransformerConfig(
            width=64,
            depth=2,
            heads=4,
            feedforward_multiplier=3,
            condition_width=128,
            dropout=0.0,
        )
    )


def test_teacher_split_and_cell_tensor_contract(teacher: NativeMotionTeacher) -> None:
    assert [len(teacher.split_chassis("train", family)) for family in range(5)] == [2] * 5
    assert [len(teacher.split_chassis("validation", family)) for family in range(5)] == [1] * 5
    assert [len(teacher.split_chassis("test", family)) for family in range(5)] == [1] * 5
    sample = teacher.sample(0, 9, 20)
    assert sample["static"].shape == (MAX_CELLS, 61)
    assert sample["state"].shape == sample["target"].shape == (MAX_CELLS, 4)
    assert sample["adjacency"].shape == (MAX_CELLS, MAX_CELLS)
    assert sample["mask"].sum() == sample["cell_count"]
    assert not sample["static"].flags.writeable
    assert not sample["target"].flags.writeable
    assert sample["adjacency"][: sample["cell_count"], : sample["cell_count"]].diagonal().all()


def test_sampler_is_deterministic_and_family_balanced(teacher: NativeMotionTeacher) -> None:
    first = MotionBatchSampler(teacher, batch_size=10, seed=91)
    second = MotionBatchSampler(teacher, batch_size=10, seed=91)
    coordinates = [first.coordinates(step) for step in range(20)]
    assert coordinates == [second.coordinates(step) for step in range(20)]
    for row in coordinates:
        families = [teacher.chassis[chassis]["family_id"] for chassis, _, _ in row]
        assert families == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert len({coordinate for row in coordinates for coordinate in row}) > 120


def test_transformer_is_substantial_and_masks_padding(teacher: NativeMotionTeacher) -> None:
    assert CellularMotionTransformer().parameter_count == 27_409_156
    batch = MotionBatchSampler(teacher, batch_size=5, seed=7).batch(2)
    model = _small_model().eval()
    with torch.no_grad():
        output = model(
            batch["static"], batch["state"], batch["mask"], batch["adjacency"],
            batch["family"], batch["morphotype"], batch["motion"], batch["phase"],
            batch["controls"],
        )
    assert output.shape == (5, MAX_CELLS, 4)
    assert torch.isfinite(output).all()
    assert float(output[~batch["mask"]].abs().max()) == 0.0


def test_conditions_change_pose_without_changing_support(teacher: NativeMotionTeacher) -> None:
    batch = MotionBatchSampler(teacher, batch_size=5, seed=8).batch(3)
    model = _small_model().eval()
    changed_motion = (batch["motion"] + 1) % 13
    with torch.no_grad():
        original = model(
            batch["static"], batch["state"], batch["mask"], batch["adjacency"],
            batch["family"], batch["morphotype"], batch["motion"], batch["phase"],
            batch["controls"],
        )
        changed = model(
            batch["static"], batch["state"], batch["mask"], batch["adjacency"],
            batch["family"], batch["morphotype"], changed_motion, batch["phase"],
            batch["controls"],
        )
    assert float((original - changed).abs()[batch["mask"]].mean()) > 1e-4
    assert float(changed[~batch["mask"]].abs().max()) == 0.0


def test_loss_backpropagates_through_graph_and_attention(teacher: NativeMotionTeacher) -> None:
    batch = MotionBatchSampler(teacher, batch_size=5, seed=9).batch(4)
    model = _small_model()
    predicted = model(
        batch["static"], batch["state"], batch["mask"], batch["adjacency"],
        batch["family"], batch["morphotype"], batch["motion"], batch["phase"],
        batch["controls"],
    )
    loss, pieces = cellular_motion_loss(
        predicted, batch["target"], batch["state"], batch["mask"], batch["adjacency"]
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in pieces.values())
    assert float(model.blocks[0].attention.in_proj_weight.grad.abs().sum()) > 0.0
    assert float(model.blocks[0].graph[0].weight.grad.abs().sum()) > 0.0


def test_cpu_smoke_and_exact_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "smoke"
    report = run_cpu_smoke(output, steps=3)
    assert report["passed"] and report["production_parameters"] >= 20_000_000
    assert validate_cpu_smoke(output) == report
    manifest_path = output / "smoke_manifest.json"
    original = manifest_path.read_bytes()
    payload = json.loads(original)
    payload["gates"]["outside_cells_exact_zero"] = False
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
    contract = prepare_production(output, total_steps=1000, segment_steps=500, batch_size=5)
    assert contract["model"] == CellularMotionTransformerConfig().to_dict()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="16 GiB free VRAM"):
        train_segment(output, end_step=500)


def test_source_identity_is_stable_and_complete() -> None:
    assert len(source_sha256()) == 64
    assert source_sha256() == source_sha256()
