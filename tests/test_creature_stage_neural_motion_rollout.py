from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from forge.config import PROJECT_ROOT
from forge.creature_stage_neural_motion.contract import source_sha256 as parent_source_sha256
from forge.creature_stage_neural_motion.dataset import NativeMotionTeacher
from forge.creature_stage_neural_motion.training import _canonical
from forge.creature_stage_neural_motion_rollout.contract import RolloutTrainingConfig, source_sha256
from forge.creature_stage_neural_motion_rollout.evaluation import (
    REPORT_NAME,
    evaluate_checkpoint,
    evaluation_source_sha256,
    validate_evaluation,
)
from forge.creature_stage_neural_motion_rollout.training import (
    RolloutBatchSampler,
    rollout_frame_loss,
    run_cpu_smoke,
    validate_cpu_smoke,
)


TEACHER = PROJECT_ROOT / "outputs/creature_stage_motion_corpus_v1_final_a"
PILOT = PROJECT_ROOT / "outputs/creature_stage_neural_motion_rollout/production_v1/cell_motion_rollout_0000500.pt"


def test_rollout_sampler_is_deterministic_balanced_and_consecutive() -> None:
    teacher = NativeMotionTeacher(TEACHER)
    sampler = RolloutBatchSampler(teacher, batch_size=10, sequence_frames=4)
    assert sampler.coordinates(17) == sampler.coordinates(17)
    assert sampler.coordinates(17) != sampler.coordinates(18)
    coordinates = sampler.coordinates(17)
    assert [teacher.chassis[chassis]["family_id"] for chassis, _, _ in coordinates] == [0, 1, 2, 3, 4] * 2
    frames = sampler.sequence(17)
    assert len(frames) == 4
    assert all(frame["family"].tolist() == frames[0]["family"].tolist() for frame in frames)
    for offset, frame in enumerate(frames):
        expected = teacher.sample(*coordinates[0][:2], coordinates[0][2] + offset)
        assert torch.equal(frame["target"][0], torch.from_numpy(expected["target"].copy()))


def test_rollout_loss_penalizes_energy_collapse_and_moves_appendages() -> None:
    teacher = NativeMotionTeacher(TEACHER)
    frame = RolloutBatchSampler(teacher, sequence_frames=2).sequence(0)[0]
    target = frame["target"].clone()
    collapsed = torch.zeros_like(target, requires_grad=True)
    loss, pieces = rollout_frame_loss(
        collapsed,
        target,
        frame["state"],
        frame["state"],
        frame["static"],
        frame["mask"],
        frame["adjacency"],
        RolloutTrainingConfig(),
    )
    assert float(pieces["energy"]) > 0.0
    assert float(pieces["appendage"]) > 0.0
    loss.backward()
    assert collapsed.grad is not None
    assert float(collapsed.grad.abs().sum()) > 0.0
    assert bool(torch.isfinite(collapsed.grad).all())


def test_rollout_cpu_smoke_replays_and_rehashed_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "smoke"
    result = run_cpu_smoke(output, steps=8)
    assert result["passed"]
    assert result["diagnostics"]["families"] == 5
    assert result["diagnostics"]["prediction_fed_frames"] == 3
    assert result["diagnostics"]["outside_max_abs"] == 0.0
    assert validate_cpu_smoke(output, replay=True) == result

    path = output / "smoke_manifest.json"
    payload = json.loads(path.read_bytes())
    payload["diagnostics"]["energy_ratio"] += 0.5
    payload["semantic_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})
    ).hexdigest()
    path.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="deterministic replay"):
        validate_cpu_smoke(output, replay=True)


def test_rollout_successor_is_additive_to_frozen_parent_model() -> None:
    assert len(source_sha256()) == 64
    assert len(evaluation_source_sha256()) == 64
    assert parent_source_sha256() == "2300cacade824488a69d1f191519e5809222f1de14ecd8d92f64f3ea1f3b5ec5"


def test_rollout_pilot_evaluation_is_prediction_fed_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "evaluation"
    result = evaluate_checkpoint(PILOT, output, motion_ids=(0,), rollout_frames=2)
    assert result["passed"]
    assert result["update"] == 500
    assert result["clips"] == 5
    assert not result["promotion_eligible"]
    assert not result["gates"]["full_validation_matrix"]
    assert not result["gates"]["final_rollout_checkpoint"]
    assert result["gates"]["outside_cells_exact_zero"]
    assert validate_evaluation(output / REPORT_NAME, replay=True) == result

    path = output / REPORT_NAME
    payload = json.loads(path.read_bytes())
    payload["clips"][0]["metrics"]["position_mae_px"] += 0.25
    payload["semantic_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})
    ).hexdigest()
    path.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="aggregate drifted"):
        validate_evaluation(path)
