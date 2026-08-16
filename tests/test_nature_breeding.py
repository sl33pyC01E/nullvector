from __future__ import annotations

from forge.nature_sim_v2 import BreedingSystem,NatureWorld,founder_genomes,load_world,save_world


def _mature(entity,energy=.9):entity.age=100;entity.energy=energy;entity.reserve=.6;entity.reproduction_cooldown=0;entity.update_stage()


def test_mate_selection_rejects_close_kin_and_prefers_fitter_symmetric_partner() -> None:
    world=NatureWorld(seed=601,size=32);genome=founder_genomes(variants_per_family=1)[1];chooser_id=world.add_organism(genome,(10,10),parents=(91,92));sibling_id=world.add_organism(genome,(10.2,10),parents=(91,92));weak_id=world.add_organism(genome,(10.3,10));fit_id=world.add_organism(genome,(10.4,10));chooser,sibling,weak,fit=[world.organisms[value] for value in (chooser_id,sibling_id,weak_id,fit_id)]
    for entity in (chooser,sibling,weak,fit):_mature(entity)
    weak.body.impact((0,0),8,.82);weak.energy=.65;choice=world.breeding.choose(chooser,[sibling,weak,fit]);assert choice is fit and world.breeding.related_rejections==1


def test_courtship_records_selection_and_produces_developed_offspring() -> None:
    world=NatureWorld(seed=602,size=32,max_population=40);genome=founder_genomes(variants_per_family=1)[1];left=world.add_organism(genome,(12,12),energy=1);right=world.add_organism(genome,(12.4,12),energy=1);_mature(world.organisms[left],1);_mature(world.organisms[right],1);world.organisms[left].intent="mate";world._interactions(world.organisms[left],.2);assert world.breeding.pairings==1 and world.organisms[left].mate_id==right
    world.organisms[left].gestation_remaining=.1;world.step(.1,publish=False);children=[item for item in world.organisms.values() if item.parent_ids];assert children and children[0].genome.developmental.generation==1


def test_breeding_ledger_survives_save(tmp_path) -> None:
    world=NatureWorld(seed=603,size=32);genome=founder_genomes(variants_per_family=1)[0];left=world.add_organism(genome,(8,8),energy=1);right=world.add_organism(genome,(8.3,8),energy=1);_mature(world.organisms[left]);_mature(world.organisms[right]);world.breeding.record(world.organisms[left],world.organisms[right],3);path=tmp_path/"breeding.nvz";save_world(world,path);restored=load_world(path);assert restored.breeding.payload()==world.breeding.payload()
