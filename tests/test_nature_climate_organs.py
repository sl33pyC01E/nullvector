from __future__ import annotations

from forge.nature_sim_v2 import NatureWorld,founder_genomes
from forge.nature_sim_v2.climate import ClimateState


def test_extreme_toxin_causes_local_respiratory_organ_damage() -> None:
    world=NatureWorld(seed=1801,size=40);entity_id=world.add_organism(founder_genomes(variants_per_family=1)[0],(20,20),energy=.8);entity=world.organisms[entity_id];world.fields[7,20,20]=1;before=entity.body.health.copy();state=ClimateState("sporefall",0,.5,.5,.5,.1,1,"spore_bloom");system,damage=world.climate.stress_body(world,entity,state,15)
    assert system in ("respiration","digestion") and damage>0
    damaged=(entity.body.health<before)
    assert damaged.any()
    owner=entity.body.component_owner[damaged]
    expected={"respiration":"lung","digestion":"gut"}[system]
    assert any(entity.genome.developmental.components[int(index)].organ==expected for index in owner)


def test_phase_storm_spares_anomaly_neural_tissue_but_stresses_humanoid() -> None:
    world=NatureWorld(seed=1802,size=40);genomes=founder_genomes(variants_per_family=1);human=world.organisms[world.add_organism(genomes[0],(10,10),energy=.8)];anomaly=world.organisms[world.add_organism(genomes[3],(30,30),energy=.8)];state=ClimateState("storm",0,.5,.5,.5,1,0,"phase_storm");human_result=world.climate.stress_body(world,human,state,15);anomaly_result=world.climate.stress_body(world,anomaly,state,15)
    assert human_result[0]=="neural" and human_result[1]>0
    assert anomaly_result==(None,0.0)
