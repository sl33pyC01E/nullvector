from __future__ import annotations

import numpy as np

from forge.living_body_substrate.contract import ORGAN_SYSTEM
from forge.nature_sim_v2 import NatureWorld,founder_genomes


def test_digestive_and_circulatory_organs_gate_real_food_assimilation() -> None:
    world=NatureWorld(seed=44,size=32);genome=founder_genomes(variants_per_family=1)[0];healthy_id=world.add_organism(genome,(12,12),energy=.25);injured_id=world.add_organism(genome,(12,12),energy=.25);healthy=world.organisms[healthy_id];injured=world.organisms[injured_id]
    digestive=np.asarray([ORGAN_SYSTEM.get(organ)=="digestion" for organ in injured.body.organ]);assert digestive.any();injured.body.health[digestive]=.01
    original=world.fields.copy();world._consume(healthy,.5);healthy_gain=healthy.energy-.25;world.fields[:]=original;world._consume(injured,.5);injured_gain=injured.energy-.25
    assert healthy_gain>injured_gain*2
    assert healthy.body.energy>injured.body.energy


def test_feeding_replenishes_the_cellular_repair_energy_pool() -> None:
    world=NatureWorld(seed=45,size=32);entity_id=world.add_organism(founder_genomes(variants_per_family=1)[1],(10,10),energy=.3);entity=world.organisms[entity_id];entity.body.energy=.1;before=entity.body.energy;world._consume(entity,.5);assert entity.body.energy>before
