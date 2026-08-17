from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from forge.creature_stage_developmental.development import develop
from forge.creature_stage_developmental.genomes import review_genomes
from forge.creature_stage_neural_grounded_feedback_v2.dataset import encode_live
from forge.creature_stage_neural_target_field_v1.contract import (
    TARGET_FEATURES, TARGET_PHASE_HARMONICS,
)
from forge.creature_stage_neural_target_field_v1.dataset import (
    build_target_augmentation, encode_target_context,
)
from forge.creature_stage_neural_target_field_v1.model import NeuralGroundedTargetField
from forge.creature_stage_neural_target_field_v1.bank import load_training_bank, validate_training_bank
from forge.creature_stage_neural_target_field_v1.audit import validate_audit


def _inputs():
    organism = develop(review_genomes()[0])
    nodes = organism.skeleton_nodes[:, :2].astype(np.float32)
    velocity = np.zeros_like(nodes)
    contacts = np.zeros(len(organism.genome.appendages), dtype=np.bool_)
    live = encode_live(organism, nodes, velocity, contacts, .25, 0.0)
    target = encode_target_context(organism, nodes, velocity, .25)
    tensors = tuple(torch.from_numpy(value[None]) for value in live)
    return organism, tensors, torch.from_numpy(target[None])


def test_target_context_is_periodic_and_carries_eight_local_harmonics() -> None:
    organism = develop(review_genomes()[0])
    nodes = organism.skeleton_nodes[:, :2].astype(np.float32)
    velocity = np.zeros_like(nodes)
    first = encode_target_context(organism, nodes, velocity, 0.0)
    closed = encode_target_context(organism, nodes + 99, velocity + 99, 1.0)
    assert first.shape[1] == TARGET_FEATURES == 4 + TARGET_PHASE_HARMONICS * 2
    np.testing.assert_allclose(first, closed, atol=2e-6)
    for owner, gene in enumerate(organism.genome.appendages):
        np.testing.assert_allclose(first[owner, :2] * 24, gene.endpoint, atol=1e-6)


def test_target_field_does_not_feed_rollout_error_back_into_periodic_target() -> None:
    _organism, tensors, target = _inputs()
    model = NeuralGroundedTargetField().eval()
    with torch.inference_mode():
        baseline = model(*tensors, target).terminal_target
        changed = list(tensors)
        changed[0] = changed[0].clone(); changed[0][:, :, 16:23] += 7
        changed[1] = changed[1].clone(); changed[1][:, 20:23] -= 5
        perturbed = model(*changed, target).terminal_target
    torch.testing.assert_close(baseline, perturbed, rtol=0, atol=0)


def test_target_augmentation_is_balanced_over_all_reviewed_chassis() -> None:
    corpus = build_target_augmentation(variants_per_chassis=1)
    assert corpus.samples == 10 * 72
    assert torch.bincount(corpus.family, minlength=5).tolist() == [144] * 5
    assert corpus.target_context.shape[1:] == (8, TARGET_FEATURES)
    assert torch.isfinite(corpus.terminal_target).all()


def test_published_training_bank_is_source_bound_and_loadable() -> None:
    root = Path("outputs/creature_stage_neural_target_field_v1/training_bank_v2_modular")
    if not root.is_dir():
        return
    manifest = validate_training_bank(root)
    train, augmentation = load_training_bank(root)
    assert train.samples == manifest["train_samples"]
    assert augmentation.samples == manifest["augmentation_samples"]


def test_published_exhaustive_audit_is_hash_closed() -> None:
    root = Path("outputs/creature_stage_neural_target_field_v1/production_6000_v11_exhaustive_audit_v2")
    if root.is_dir():
        assert validate_audit(root)["gates"]["all_passed"]
