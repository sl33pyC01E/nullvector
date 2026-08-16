from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import NatureWorld,founder_genomes,load_world,save_world


def test_pollination_and_seed_dispersal_are_material_ecology() -> None:
    world=NatureWorld(seed=801,size=32);genomes=founder_genomes(variants_per_family=1);animal=world.add_organism(genomes[1],(12,12),energy=.3);plant=world.add_organism(genomes[2],(12.7,12),energy=.8);before=world.fields[8].copy()
    for _ in range(60):world.step(.1,publish=False)
    assert world.ecosystem.pollinations>0 and world.organisms[animal].energy>.3
    assert world.organisms[plant].reserve>world.organisms[plant].energy*.0
    assert world.ecosystem.seed_dispersals>0 and np.any(world.fields[8]>before)


def test_plant_root_network_and_anomaly_machine_coupling() -> None:
    world=NatureWorld(seed=802,size=32);genomes=founder_genomes(variants_per_family=1);left=world.add_organism(genomes[2],(8,8),energy=.2);right=world.add_organism(genomes[2],(8.6,8),energy=.9);world.organisms[left].colony_id=1;world.organisms[right].colony_id=1;anomaly=world.add_organism(genomes[3],(18,18),energy=.7);machine=world.add_organism(genomes[4],(18.5,18),energy=.2);before_gap=abs(world.organisms[left].energy-world.organisms[right].energy);before_machine=world.organisms[machine].energy
    for _ in range(40):world.step(.1,publish=False)
    assert world.ecosystem.root_transfers>0 and abs(world.organisms[left].energy-world.organisms[right].energy)<before_gap
    assert world.ecosystem.phase_couplings>0 and world.organisms[machine].energy>before_machine


def test_ecosystem_counters_and_cooldowns_survive_exact_save(tmp_path) -> None:
    world=NatureWorld(seed=803,size=32);genomes=founder_genomes(variants_per_family=1);world.add_organism(genomes[1],(12,12),energy=.3);world.add_organism(genomes[2],(12.6,12),energy=.8)
    for _ in range(25):world.step(.1,publish=False)
    path=tmp_path/"network.nvz";save_world(world,path);restored=load_world(path)
    assert restored.ecosystem.payload()==world.ecosystem.payload()
    world.step(.1,publish=False);restored.step(.1,publish=False);assert restored.snapshot().semantic_sha256==world.snapshot().semantic_sha256
