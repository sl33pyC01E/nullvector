from __future__ import annotations

from forge.neural_ensemble_v1.release import _quality


def test_behavior_promotion_requires_rare_forage_support():
    report = {"validation": {"selected": "model", "model": {"intent_accuracy": .99, "direction_cosine": .8, "per_intent": {"forage": {"accuracy": .2}}}}}
    passed, gates = _quality("behavior", report)
    assert not passed
    assert not gates["forage_accuracy"]


def test_world_transition_requires_visual_and_latent_improvement():
    passed, gates = _quality("world_latent_dit", {"latent_improvement": .1, "rgb_improvement": -.01, "horizon": 4})
    assert not passed
    assert not gates["rgb_improvement"]
