from __future__ import annotations

from forge.nature_sim_v2 import NatureWorld, founder_genomes
from forge.nature_sim_v2.directed_evolution import EvolutionOffer,apply_offer,metamorphose


def _offer(structural:str)->EvolutionOffer:
    return EvolutionOffer("test-"+structural,structural,structural,(),(),1.0,structural)


def test_structural_mutations_grow_symmetric_organs_and_appendages() -> None:
    genome=founder_genomes(variants_per_family=1)[1]
    for kind in ("armor_lobes","storage_lobes","neural_lobes"):
        evolved=apply_offer(genome,_offer(kind),seed=90)
        added=evolved.developmental.components[len(genome.developmental.components):]
        assert len(added)==2 and {item.side for item in added}=={-1,1}
        assert added[0].anchor[0]+added[1].anchor[0]==2*genome.developmental.components[0].anchor[0]
    for kind in ("hardpoint_pair","locomotor_pair"):
        evolved=apply_offer(genome,_offer(kind),seed=91)
        added=evolved.developmental.appendages[len(genome.developmental.appendages):]
        assert len(added)==2 and added[0].paired_with==added[1].appendage_id and added[1].paired_with==added[0].appendage_id


def test_metamorphosis_rebuilds_physical_cells_and_preserves_damage() -> None:
    world=NatureWorld(seed=93,size=40);entity_id=world.add_organism(founder_genomes(variants_per_family=1)[0],(20,20),energy=.8);entity=world.organisms[entity_id];entity.body.impact((0,0),4,.5);old_cells=entity.body.organism.cell_count
    metamorphose(entity,_offer("armor_lobes"),seed=94)
    assert entity.body.organism.cell_count>old_cells
    assert entity.genome.developmental.generation==1
    assert float(entity.body.health.min())<1
