from __future__ import annotations

import numpy as np

from forge.creature_stage_developmental.development import develop
from forge.creature_stage_morphology_v2.genomes import morphology_review_genomes
from forge.living_body_substrate import LivingBody


def _bodies() -> list[LivingBody]:
    genomes = morphology_review_genomes()
    return [LivingBody(develop(genomes[index]), seed=index) for index in (5, 11, 17, 23, 29)]


def test_all_families_spawn_healthy_without_terminal_cascade() -> None:
    bodies = _bodies()
    assert [body.family for body in bodies] == list(range(5))
    for body in bodies:
        initial = body.snapshot()
        assert not initial.incapacitated and not initial.dead
        assert initial.systems["integrity"] == 1
        for _ in range(200):
            final = body.tick(.1)
        assert not final.incapacitated and not final.dead
        assert final.systems["integrity"] == 1


def test_neural_damage_changes_behavior_before_death() -> None:
    body = _bodies()[0]
    brain = body.organism.genome.components[2]
    before = body.systems()["neural"]
    body.impact(brain.anchor, max(brain.radius) * 1.5, .78)
    snapshot = body.tick(.1)
    assert snapshot.systems["neural"] < before - .35
    assert snapshot.systems["integrity"] > snapshot.systems["neural"]


def test_appendage_severing_reduces_locomotion_and_leaks() -> None:
    body = _bodies()[1]
    appendage = body.organism.genome.appendages[0]
    start = np.asarray(appendage.root_offset, dtype=np.float32) + np.asarray((-2, 0), dtype=np.float32)
    end = np.asarray(appendage.root_offset, dtype=np.float32) + np.asarray((2, 0), dtype=np.float32)
    before = body.systems()["locomotion"]
    assert body.cut(tuple(start), tuple(end), width=1.1) > 0
    after = body.tick(.1)
    assert after.systems["locomotion"] < before
    assert after.leak_amount > 0


def test_detached_fates_are_family_specific_and_consolidated() -> None:
    animal = _bodies()[1]
    plant = _bodies()[2]
    for body in (animal, plant):
        # Isolate a broad lower portion from the core.
        body.cut((-30, 4), (30, 4), width=1.5)
        for _ in range(120):
            body.tick(.1)
    assert animal.snapshot().biomass_count >= 1
    assert animal.snapshot().polyp_count == 0
    assert plant.snapshot().polyp_count >= 1


def test_damage_replay_is_deterministic() -> None:
    left, right = _bodies()[4], _bodies()[4]
    for body in (left, right):
        body.impact((0, -2), 4.5, .63)
        body.cut((-8, 2), (8, 2), width=.8)
        body.heal((0, -2), 5.0, .22)
        for _ in range(20):
            snapshot = body.tick(.1)
    assert left.snapshot().semantic_sha256 == right.snapshot().semantic_sha256
    assert left.snapshot() == right.snapshot()
