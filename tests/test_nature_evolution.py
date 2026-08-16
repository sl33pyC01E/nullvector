from __future__ import annotations

from forge.nature_sim_v2 import EvolutionLedger, NatureWorld, founder_genomes, recombine


def test_ecotypes_cluster_microvariation_but_split_large_phenotypes() -> None:
    genomes=founder_genomes(variants_per_family=2);ledger=EvolutionLedger()
    assert ledger.clade_id(genomes[0]) != ledger.clade_id(genomes[2])
    child=recombine(genomes[0],genomes[1],seed=444)
    assert len(ledger.signature(child))==23


def test_selection_ledger_tracks_population_fitness_and_ancestry() -> None:
    world=NatureWorld(seed=33,size=48);world.seed_founders(variants_per_family=1);ledger=EvolutionLedger();ledger.observe(world)
    assert sum(record.population for record in ledger.clades.values())==5
    assert ledger.diversity>1 and ledger.dominant()
    parent=world.organisms[1]
    parent.stage="mature";parent.energy=1;parent.gestation_remaining=.01;parent.mate_id=1
    world.step(.02);ledger.observe(world)
    assert sum(record.births for record in ledger.clades.values())>=1
    assert sum(record.population for record in ledger.clades.values())==world.snapshot().population
