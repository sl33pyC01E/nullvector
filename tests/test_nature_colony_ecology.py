from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import ColonyEcology, ColonyState, NatureWorld, founder_genomes


def _colony_world()->NatureWorld:
    world=NatureWorld(seed=61,size=48);genome=founder_genomes(variants_per_family=1)[0];ids=[world.add_organism(genome,(20+i*.25,20),energy=.9) for i in range(8)];world.colonies[1]=ColonyState(1,0,genome.lineage_id,set(ids),np.asarray((20.9,20.0)));world.next_colony_id=2
    for entity_id in ids:world.organisms[entity_id].colony_id=1
    return world


def test_colony_roles_are_reproducible_and_differentiated() -> None:
    left=_colony_world();right=_colony_world();left.colony_ecology.step(left,.25);right.colony_ecology.step(right,.25)
    assert left.colony_ecology.states[1].assignments==right.colony_ecology.states[1].assignments
    assert len(set(left.colony_ecology.states[1].assignments.values()))>=2


def test_colony_commons_transfers_only_real_surplus() -> None:
    world=_colony_world();members=[world.organisms[i] for i in sorted(world.colonies[1].member_ids)];members[0].energy=.12
    before=sum(member.energy for member in members)
    for _ in range(80):world.colony_ecology.step(world,.25)
    state=world.colony_ecology.states[1];after=sum(member.energy for member in members)+state.energy_store
    assert members[0].energy>.12 and state.transfers>0
    assert after<=before+1e-9


def test_colony_medicine_repairs_real_body_cells() -> None:
    world=_colony_world();world.colony_ecology.step(world,.25);state=world.colony_ecology.states[1]
    medic_id=next((entity_id for entity_id,role in state.assignments.items() if role=="medic"),None)
    if medic_id is None:
        # The role scorer is anatomy-dependent; turn its deterministic winner
        # into a medic to exercise the physical treatment path.
        medic_id=min(state.assignments);state.assignments[medic_id]="medic"
        world.colony_ecology.choose_role=lambda entity:"medic" if entity.entity_id==medic_id else "gatherer"
    target=world.organisms[max(state.assignments)];target.body.impact((0,0),6,.8);before=target.body.systems()["integrity"]
    for _ in range(20):world.colony_ecology.step(world,.25)
    assert target.body.systems()["integrity"]>before and world.colony_ecology.states[1].repairs>0
