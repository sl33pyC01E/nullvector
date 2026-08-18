from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from forge.config import PROJECT_ROOT
from forge.creature_stage_neural_motion.dataset import NativeMotionTeacher
from forge.creature_stage_neural_motion.training import _canonical
from forge.creature_stage_neural_motion_rollout.contract import source_sha256 as parent_rollout_source_sha256
from forge.creature_stage_neural_motion_loop.contract import LoopTrainingConfig, source_sha256
from forge.creature_stage_neural_motion_loop.sampler import LoopAwareRolloutBatchSampler
from forge.creature_stage_neural_motion_loop.smoke import run_cpu_smoke, validate_cpu_smoke
from forge.creature_stage_neural_motion_loop.production import (
    production_source_sha256,
    prepare_production,
    train_segment,
)
from forge.creature_stage_neural_motion_loop.evaluation import evaluation_source_sha256


TEACHER = PROJECT_ROOT / "outputs/creature_stage_motion_corpus_v1_final_a"


def test_forced_loop_sequences_cross_exact_71_to_zero_seam_for_all_families() -> None:
    teacher = NativeMotionTeacher(TEACHER)
    sampler = LoopAwareRolloutBatchSampler(teacher)
    frames, coordinates = sampler.sequence(0, force_seam=True)
    assert len(frames) == 6
    assert [teacher.chassis[row.chassis]["family_id"] for row in coordinates] == list(range(5))
    assert all(row.loop and row.forced_seam for row in coordinates)
    for row in coordinates:
        indices = sampler.frame_indices(row)
        assert any(left == 71 and right == 0 for left, right in zip(indices, indices[1:]))
    assert all(frame["family"].tolist() == list(range(5)) for frame in frames)


def test_regular_sampler_is_deterministic_and_never_wraps_nonloops() -> None:
    teacher = NativeMotionTeacher(TEACHER)
    sampler = LoopAwareRolloutBatchSampler(teacher, batch_size=10)
    assert sampler.coordinates(31) == sampler.coordinates(31)
    assert sampler.coordinates(31) != sampler.coordinates(32)
    seam_count = 0
    nonloop_count = 0
    for update in range(64):
        for row in sampler.coordinates(update):
            indices = sampler.frame_indices(row)
            if row.forced_seam:
                seam_count += 1
                assert any(left == 71 and right == 0 for left, right in zip(indices, indices[1:]))
            if not row.loop:
                nonloop_count += 1
                assert tuple(sorted(indices)) == indices
                assert indices[-1] < 72
    assert seam_count > 0
    assert nonloop_count > 0


def test_loop_successor_rebalances_loss_after_energy_recovery() -> None:
    loop = LoopTrainingConfig()
    assert loop.sequence_frames == 6
    assert loop.energy_weight == 0.10
    assert loop.delta_weight == 1.00
    assert loop.energy_weight < 0.25
    assert loop.delta_weight > 0.25


def test_loop_cpu_smoke_exact_replay_and_rehashed_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    output = tmp_path / "smoke"
    result = run_cpu_smoke(output, steps=8)
    assert result["passed"]
    assert result["diagnostics"]["families"] == 5
    assert result["diagnostics"]["forced_seam_sequences"] == 5
    assert result["diagnostics"]["seam_transitions"] == 5
    assert result["diagnostics"]["prediction_fed_frames"] == 5
    assert result["diagnostics"]["outside_max_abs"] == 0.0
    assert validate_cpu_smoke(output, replay=True) == result

    manifest = output / "smoke_manifest.json"
    payload = json.loads(manifest.read_bytes())
    payload["diagnostics"]["seam_transitions"] = 4
    payload["semantic_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})
    ).hexdigest()
    manifest.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="deterministic replay"):
        validate_cpu_smoke(output, replay=True)


def test_loop_successor_preserves_parent_rollout_authority() -> None:
    assert len(source_sha256()) == 64
    assert len(production_source_sha256()) == 64
    assert len(evaluation_source_sha256()) == 64
    assert parent_rollout_source_sha256() == "045b88495134b8dfdcadbd76a6587f4138ea37a5645e60ddd5d33d3b4bb80856"


def test_loop_production_contract_is_parent_bound_and_cuda_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "production"
    contract = prepare_production(output, total_updates=100)
    assert contract["parent"]["update"] == 1000
    assert contract["total_updates"] == contract["segment_updates"] == 100
    assert contract["training"] == LoopTrainingConfig().to_dict()
    assert contract["source_sha256"] == production_source_sha256()
    assert prepare_production(output, total_updates=100) == contract
    with pytest.raises(ValueError, match="exactly one bounded segment"):
        train_segment(output, end_update=50)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="deterministic CUDA BF16"):
        train_segment(output, end_update=100)


def test_loop_production_rejects_rehashed_parent_tamper(tmp_path: Path) -> None:
    output = tmp_path / "production"
    prepare_production(output, total_updates=100)
    path = output / "production_contract.json"
    payload = json.loads(path.read_bytes())
    payload["parent"]["ema_state_sha256"] = "0" * 64
    payload["semantic_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})
    ).hexdigest()
    path.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="parent provenance"):
        train_segment(output, end_update=100)
