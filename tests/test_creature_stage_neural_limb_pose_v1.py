from __future__ import annotations

import numpy as np
import pytest
import torch

from forge.creature_stage_developmental.development import develop
from forge.creature_stage_developmental.genomes import review_genomes
from forge.creature_stage_manipulation_v1.articulation import ArticulatedBody, curved_muscle_pose
from forge.creature_stage_neural_limb_pose_v1.contract import MAX_NODES, ModelConfig, source_sha256
from forge.creature_stage_neural_limb_pose_v1.dataset import build_corpus
from forge.creature_stage_neural_limb_pose_v1.model import NeuralLimbPose


def test_contract_and_corpus_are_exact() -> None:
    assert len(source_sha256()) == 64
    first = build_corpus(split="validation", cases_per_appendage=64)
    second = build_corpus(split="validation", cases_per_appendage=64)
    assert first.semantic_sha256 == second.semantic_sha256
    assert first.nodes.shape[1:] == (MAX_NODES, 8)
    assert first.context.shape[1] == 21 and first.target.shape[1:] == (MAX_NODES, 2)
    assert torch.equal(first.nodes, second.nodes) and torch.equal(first.target, second.target)


def test_model_is_masked_differentiable_and_strict() -> None:
    corpus = build_corpus(split="validation", cases_per_appendage=64)
    model = NeuralLimbPose(ModelConfig(width=96, depth=3, heads=3, dropout=0))
    output = model(corpus.nodes[:8], corpus.context[:8], corpus.mask[:8])
    assert output.pose.shape == (8, MAX_NODES, 2) and output.confidence.shape == (8,)
    loss = output.pose.square().mean() + output.confidence.mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    with pytest.raises(ValueError, match="input drifted"):
        model(corpus.nodes[:8, :-1], corpus.context[:8], corpus.mask[:8, :-1])


def test_curved_teacher_pins_root_and_hand_and_respects_reach() -> None:
    root = np.asarray((2.0, -1.0), np.float32)
    lengths = np.asarray((4.0, 5.0), np.float32)
    target = np.asarray((30.0, 2.0), np.float32)
    pose = curved_muscle_pose(root, target, lengths, -1.0)
    assert np.array_equal(pose[0], root)
    assert np.isclose(np.linalg.norm(pose[-1] - root), lengths.sum(), atol=1e-5)
    assert np.isfinite(pose).all()


def test_pose_driver_runs_inside_full_skeleton_projector() -> None:
    class TeacherDriver:
        def predict_pose(self, organism, appendage, positions, velocities, root, target, lengths, **values):
            return curved_muscle_pose(root, target, lengths, values["bend_sign"])

    organism = develop(review_genomes()[0])
    baseline = ArticulatedBody.from_organism(organism)
    driven = ArticulatedBody.from_organism(organism); driven.pose_driver = TeacherDriver()
    target = baseline.root(0) + np.asarray((-7.0, -1.5))
    for _ in range(80):
        expected = baseline.solve(0, target, .65)
        actual = driven.solve(0, target, .65)
    assert np.allclose(actual, expected, atol=1e-5)
    assert np.allclose(driven.nodes, baseline.nodes, atol=1e-5)
    assert driven.max_length_error() < 1e-4
