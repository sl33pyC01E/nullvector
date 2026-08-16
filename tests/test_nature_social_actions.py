from __future__ import annotations

from forge.nature_sim_v2 import NatureWorld,founder_genomes
from forge.nature_sim_v2.social_actions import bond_nearby


def test_player_can_found_and_expand_a_compatible_kin_colony() -> None:
    world=NatureWorld(seed=2301,size=40);genome=founder_genomes(variants_per_family=1)[0];left=world.organisms[world.add_organism(genome,(20,20),energy=.8)];right=world.organisms[world.add_organism(genome,(21,20),energy=.8)];third=world.organisms[world.add_organism(genome,(22,20),energy=.8)];message=bond_nearby(world,left)
    assert "FOUNDED" in message and left.colony_id==right.colony_id
    message=bond_nearby(world,third)
    assert "JOINED" in message and third.colony_id==left.colony_id
    assert len(world.colonies[left.colony_id].member_ids)==3


def test_kin_bond_rejects_distant_or_incompatible_entities() -> None:
    world=NatureWorld(seed=2302,size=40);genomes=founder_genomes(variants_per_family=1);player=world.organisms[world.add_organism(genomes[0],(2,2),energy=.8)];world.add_organism(genomes[1],(2.5,2),energy=.8);world.add_organism(genomes[0],(20,20),energy=.8)
    assert "NO COMPATIBLE" in bond_nearby(world,player)
    assert not world.colonies
