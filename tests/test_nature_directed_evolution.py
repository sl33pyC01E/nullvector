from __future__ import annotations

import numpy as np

from forge.creature_stage_developmental import develop
from forge.nature_sim_v2 import NatureWorld, apply_offer, evolution_offers, founder_genomes, metamorphose


def test_offers_are_deterministic_diverse_and_structurally_valid() -> None:
    genome=founder_genomes(variants_per_family=1)[0];left=evolution_offers(genome,epoch=3);right=evolution_offers(genome,epoch=3)
    assert left==right and len({item.offer_id for item in left})==3
    evolved=apply_offer(genome,left[0],seed=811);organism=develop(evolved.developmental)
    assert organism.cell_count>0 and evolved.developmental.generation==genome.developmental.generation+1
    assert evolved.developmental.appendages==genome.developmental.appendages
    assert evolved.mutation_log[-1].startswith("directed:")


def test_metamorphosis_preserves_wounds_scars_and_cellular_body_state() -> None:
    world=NatureWorld(seed=91,size=32);entity_id=world.add_organism(founder_genomes(variants_per_family=1)[1],(12,12),energy=.8);entity=world.organisms[entity_id];entity.body.impact((0,0),6,.58);entity.body.heal((0,0),6,.18);before_integrity=entity.body.systems()["integrity"];before_genome=entity.genome.semantic_sha256();offer=evolution_offers(entity.genome,epoch=1)[0]
    metamorphose(entity,offer,seed=991)
    assert entity.genome.semantic_sha256()!=before_genome
    assert np.any(entity.body.health<.99) and np.any(entity.body.scar>0)
    assert abs(entity.body.systems()["integrity"]-before_integrity)<.18
    assert len(entity.neural_contacts)==len(entity.body.organism.genome.appendages)
    assert len(entity.neural_muscles)==len(entity.body.organism.muscles)
