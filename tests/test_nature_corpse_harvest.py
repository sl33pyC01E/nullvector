from __future__ import annotations

from forge.nature_sim_v2 import AdventureState,NatureWorld,founder_genomes


def test_corpse_butchery_depletes_real_cells_and_yields_family_matter() -> None:
    world=NatureWorld(seed=2201,size=40);genomes=founder_genomes(variants_per_family=1);player=world.organisms[world.add_organism(genomes[0],(20,20),energy=.8)];corpse=world.organisms[world.add_organism(genomes[4],(21,20),energy=.8)];corpse.alive=False;corpse.stage="dead";adventure=AdventureState(seed=2202,size=40);before_cells=int(corpse.body.alive_mask.sum());before_metal=adventure.inventory["metal"];message=adventure.interact(world,player)
    assert message.startswith("BUTCHERED")
    assert int(corpse.body.alive_mask.sum())<before_cells
    assert adventure.inventory["metal"]>before_metal
    assert world.events[-1]["type"]=="corpse_harvest"


def test_corpse_harvest_is_local_not_remote() -> None:
    world=NatureWorld(seed=2203,size=40);genomes=founder_genomes(variants_per_family=1);player=world.organisms[world.add_organism(genomes[0],(2,2),energy=.8)];corpse=world.organisms[world.add_organism(genomes[1],(20,20),energy=.8)];corpse.alive=False;adventure=AdventureState(seed=2204,size=40);before=int(corpse.body.alive_mask.sum());message=adventure.interact(world,player)
    assert not message.startswith("BUTCHERED")
    assert int(corpse.body.alive_mask.sum())==before
