from __future__ import annotations

import numpy as np
from pathlib import Path

from forge.creature_stage_developmental import develop
from forge.nature_sim_v2 import AdventureState,NatureWorld,VisibleBodyPhysics,cohort_conservation,demote_to_cohort,founder_genomes,graft_appendage_pair,graft_organ,harvest_appendage_pair,recombine
from forge.nature_sim_v2.demo import OVERLAY_TOGGLES
from forge.creature_stage_grounded_locomotion.physics import primary_mode


def test_founders_cover_families_and_spawn_without_cascade() -> None:
    founders=founder_genomes(variants_per_family=2)
    assert len(founders)==10
    assert [sum(g.family==family for g in founders) for family in range(5)]==[2]*5
    for genome in founders:
        organism=develop(genome.developmental)
        assert organism.cell_count>30
        assert len(organism.genome.components)>=4


def test_structural_recombination_changes_body_and_preserves_pairs() -> None:
    parents=founder_genomes(variants_per_family=2)
    child=recombine(parents[2],parents[3],seed=0xBEEFBEEF)
    assert child.developmental.generation==1
    assert len(child.developmental.parent_ids)==2
    assert child.semantic_sha256() not in {parents[2].semantic_sha256(),parents[3].semantic_sha256()}
    lookup={a.appendage_id:a for a in child.developmental.appendages}
    for appendage in lookup.values():
        if appendage.paired_with is not None:
            assert lookup[appendage.paired_with].paired_with==appendage.appendage_id
    assert develop(child.developmental).cell_count>30


def test_material_consumption_is_spatial_and_family_specific() -> None:
    world=NatureWorld(seed=7,size=32)
    founders=founder_genomes(variants_per_family=1)
    ids=[world.add_organism(g,(8+family*3,8),energy=.25) for family,g in enumerate(founders)]
    before=world.fields.copy()
    for _ in range(8):world.step(.25)
    assert np.any(world.fields<before)
    assert world.organisms[ids[2]].consumed[1]>0  # plant light
    assert world.organisms[ids[3]].consumed[4]>0  # anomaly phase
    assert world.organisms[ids[4]].consumed[3]>0  # machine charge


def test_world_replay_and_colonies_are_deterministic() -> None:
    left=NatureWorld(seed=91,size=32);right=NatureWorld(seed=91,size=32)
    for world in (left,right):world.seed_founders(variants_per_family=2)
    for _ in range(180):
        a=left.step(.25);b=right.step(.25)
        assert a.semantic_sha256==b.semantic_sha256
    assert left.snapshot()==right.snapshot()
    assert left.snapshot().population>0
    assert left.snapshot().colony_count>0


def test_reproduction_is_delayed_and_creates_developed_offspring() -> None:
    world=NatureWorld(seed=123,size=32,max_population=80)
    founders=founder_genomes(variants_per_family=1)
    # Put a compatible animal pair together and advance them to maturity.
    first=world.add_organism(founders[1],(12,12),energy=1.0)
    second=world.add_organism(founders[1],(12.5,12),energy=1.0)
    world.organisms[first].age=70;world.organisms[second].age=70
    for _ in range(200):
        world.step(.25)
        if world.births:break
    assert world.births>0
    children=[o for o in world.organisms.values() if o.parent_ids]
    assert children and children[0].genome.developmental.generation==1
    assert children[0].body.snapshot().systems["integrity"]==1


def test_death_decomposes_gradually_without_explosion() -> None:
    world=NatureWorld(seed=3,size=32)
    entity_id=world.add_organism(founder_genomes(variants_per_family=1)[1],(10,10),energy=.5)
    entity=world.organisms[entity_id]
    entity.body.health[:]=0
    world.step(.25)
    assert not entity.alive and entity.stage=="dead"
    assert entity_id in world.organisms
    start=entity.decomposition
    for _ in range(20):world.step(.25)
    assert entity.decomposition>start and entity.decomposition<1


def test_animals_physically_harvest_plant_entities() -> None:
    world=NatureWorld(seed=44,size=32)
    founders=founder_genomes(variants_per_family=1)
    animal_id=world.add_organism(founders[1],(12,12),energy=.28)
    plant_id=world.add_organism(founders[2],(12.4,12),energy=.62)
    before=world.organisms[plant_id].body.snapshot().systems["integrity"]
    for _ in range(12):world.step(.25)
    after=world.organisms[plant_id].body.snapshot().systems["integrity"]
    assert world.predation_events>0 and after<before
    assert world.organisms[animal_id].intent=="hunt"


def test_lod_demotion_conserves_lineage_mass_and_ancestry() -> None:
    world=NatureWorld(seed=8,size=32)
    genome=founder_genomes(variants_per_family=1)[2]
    ids=[world.add_organism(genome,(10+i*.2,10),energy=.4+i*.05) for i in range(4)]
    organisms=[world.organisms[i] for i in ids]
    cohort=demote_to_cohort(organisms,region_id="garden-4-9")
    assert cohort.count==4 and cohort.family==2
    assert all(cohort_conservation(cohort,organisms).values())


def test_native_nature_demo_and_launcher_are_present() -> None:
    root=Path(__file__).resolve().parents[1]
    source=(root/"forge/nature_sim_v2/demo.py").read_text("utf-8")
    for capability in ("PlayableNeuralRuntime","_step_neural_physiology","_damage_at","show_cells","show_organs","WASD PLAY","VAE"):
        assert capability in source
    assert (root/"Launch Neural Nature Stage.bat").is_file()


def test_overlay_controls_only_change_presentation_and_information() -> None:
    attributes=[attribute for attribute,_label,_key in OVERLAY_TOGGLES]
    assert len(attributes)==len(set(attributes))
    assert {"show_vision_cone","show_senses","show_health_bars","show_cells","show_organs"}<=set(attributes)
    forbidden={"paused","neural_raster","show_dream","tool","action_latch"}
    assert forbidden.isdisjoint(attributes)


def test_body_leaks_death_and_weapons_enter_material_world() -> None:
    world=NatureWorld(seed=61,size=32);founders=founder_genomes(variants_per_family=1)
    attacker=world.add_organism(founders[4],(8,16),energy=.9);target=world.add_organism(founders[1],(15,16),energy=.8)
    before=world.organisms[target].body.snapshot().alive_cells
    result=world.fire_beam(attacker,(22,16),energy=8,width=.8)
    assert result["bodies_hit"]==1 and world.organisms[target].body.snapshot().alive_cells<before
    world.organisms[target].body.impact((0,0),4,.9);world.step(.2)
    assert float(world.materials.mass.sum())>0
    projectile=world.fire_projectile(attacker,(22,16),speed=20,energy=2)
    for _ in range(5):world.step(.1)
    assert all(p.projectile_id!=projectile for p in world.materials.projectiles)


def test_visible_body_uses_general_neural_controller_for_unseen_muscle_census() -> None:
    class FixedWidthPolicy:
        def predict(self,*_args):raise ValueError("grounded feedback muscle census drifted")
    world=NatureWorld(seed=613,size=32);world.seed_founders(variants_per_family=1)
    entity=next(iter(world.organisms.values()));entity.neural_muscles=np.zeros(len(entity.body.organism.muscles),np.float32);entity.neural_contacts=np.zeros(len(entity.body.organism.genome.appendages),np.bool_)
    points=VisibleBodyPhysics(FixedWidthPolicy()).step(world,entity,1/60)
    assert points.shape==entity.body.organism.cell_xy.shape and np.isfinite(points).all()


def test_organs_and_paired_locomotors_can_cross_lineages() -> None:
    founders=founder_genomes(variants_per_family=1);animal,machine=founders[1],founders[4]
    wheel=machine.developmental.appendages[0];harvest=harvest_appendage_pair(machine,wheel.appendage_id)
    assert len(harvest.source_ids)==2 and harvest.mineral>0
    wheeled=graft_appendage_pair(animal,machine,wheel.appendage_id,seed=901)
    organism=develop(wheeled.developmental)
    assert primary_mode(organism)=="wheel" and any("graft_appendage" in item for item in wheeled.mutation_log)
    battery=next(c for c in machine.developmental.components if c.organ=="battery")
    augmented=graft_organ(animal,machine,battery.component_id,seed=902)
    assert any(c.organ=="battery" for c in augmented.developmental.components)
    assert develop(augmented.developmental).cell_count>develop(animal.developmental).cell_count


def test_world_graft_preserves_old_wounds_and_injures_donor() -> None:
    world=NatureWorld(seed=772,size=32,max_population=32);world.seed_founders(variants_per_family=1)
    animal=next(o for o in world.organisms.values() if o.family==1);machine=next(o for o in world.organisms.values() if o.family==4)
    animal.body.impact((0,0),2.2,.44);prior_min=float(animal.body.health.min());donor_mean=float(machine.body.health.mean())
    event=world.graft_from(animal.entity_id,machine.entity_id,kind="locomotor")
    assert event["installed_cells"]>0
    assert any(item.startswith("graft_appendage") for item in animal.genome.mutation_log)
    assert float(animal.body.health.min())<=prior_min+.02
    assert float(machine.body.health.mean())<donor_mean
    assert animal.body.snapshot().systems["integrity"]<1


def test_visible_body_uses_neural_muscles_and_persistent_contacts() -> None:
    world=NatureWorld(seed=773,size=32,max_population=32);world.seed_founders(variants_per_family=1);entity=next(o for o in world.organisms.values() if o.family==1);solver=VisibleBodyPhysics();organism=entity.body.organism
    entity.neural_muscles=np.linspace(0,1,len(organism.muscles),dtype=np.float32);entity.neural_contacts=np.asarray([item.kind=="leg" and item.side<0 for item in organism.genome.appendages])
    first=solver.step(world,entity,1/60).copy();entity.position+=np.asarray((.35,.22));second=solver.step(world,entity,1/60).copy();appendage=organism.appendage_index>=0
    assert float(np.mean(np.linalg.norm(second[appendage]-first[appendage],axis=1)))>.04
    assert np.allclose(solver.states[entity.entity_id].nodes[0],organism.skeleton_nodes[0,:2],atol=.2)


def test_visible_body_accepts_periodic_neural_target_field() -> None:
    class Policy:
        calls=0
        def predict(self,organism,nodes,velocity,previous_contact,phase,body_velocity):
            self.calls+=1
            targets=np.asarray([appendage.endpoint for appendage in organism.genome.appendages],np.float32)
            targets[:,0]+=np.asarray([appendage.side for appendage in organism.genome.appendages],np.float32)*.35
            contact=np.asarray([appendage.kind=="leg" and appendage.side<0 for appendage in organism.genome.appendages])
            return np.full(len(organism.muscles),.35,np.float32),contact,targets
    world=NatureWorld(seed=1773,size=32,max_population=32);world.seed_founders(variants_per_family=1);entity=next(o for o in world.organisms.values() if o.family==1);entity.velocity[:]=(.8,0);policy=Policy();solver=VisibleBodyPhysics(policy)
    first=solver.step(world,entity,1/60).copy();world.time+=.1;second=solver.step(world,entity,1/60).copy();state=solver.states[entity.entity_id]
    assert policy.calls==2 and np.isfinite(second).all()
    assert bool(state.previous_contact.any()) and float(np.mean(np.linalg.norm(second-first,axis=1)))>.005


def test_adventure_sites_inventory_building_and_collision_are_physical() -> None:
    world=NatureWorld(seed=774,size=48,max_population=32);world.seed_founders(variants_per_family=1);entity=world.organisms[1];adventure=AdventureState(seed=44,size=48);site=adventure.sites[0];entity.position=site.position.copy()
    assert "SALVAGED" in adventure.interact(world,entity) and adventure.discoveries
    adventure.inventory.update(rock=3,metal=2,biomass=2);message=adventure.build(world,entity)
    assert "PHYSICAL WALL CELLS" in message and np.any(world.materials.structure_id>0)
    wall=np.argwhere(world.materials.structure_id>0)[0];entity.position=np.asarray((float(wall[1])-1,float(wall[0])));entity.velocity=np.asarray((5.0,0.0));world._move(entity,np.asarray((1.0,0.0)),.25)
    assert not (int(entity.position[0])==int(wall[1]) and int(entity.position[1])==int(wall[0]))


def test_polyp_offspring_and_bounded_plant_tessellation() -> None:
    world=NatureWorld(seed=775,size=48,max_population=32);plant=next(g for g in founder_genomes(variants_per_family=1) if g.family==2);parent_id=world.add_organism(plant,(24,24),energy=1);parent=world.organisms[parent_id];parent.age=100;parent.reserve=.8;parent.update_stage()
    parent.body.polyps.append({"viability":.72,"centroid":[4,1],"cell_count":9});before=len(world.organisms)
    assert world._spawn_polyps(parent)==1 and len(world.organisms)==before+1
    parent.reproduction_cooldown=0;parent.energy=1;parent.reserve=.8
    assert world._vegetative_spread(parent)
    children=[o for o in world.organisms.values() if o.parent_ids]
    assert len(children)==2 and all(o.genome.developmental.generation>=1 for o in children)
    assert np.linalg.norm(world._delta(parent.position,children[-1].position))>2
    assert parent.reproduction_cooldown>20 and parent.energy<1
