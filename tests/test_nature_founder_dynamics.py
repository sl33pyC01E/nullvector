from __future__ import annotations

from forge.nature_sim_v2 import NatureWorld


def test_founders_start_as_a_heterogeneous_living_ecology() -> None:
    left=NatureWorld(seed=72,size=48);right=NatureWorld(seed=72,size=48);left.seed_founders(variants_per_family=3);right.seed_founders(variants_per_family=3);a=list(left.organisms.values());b=list(right.organisms.values());assert [(item.age,item.energy,item.reserve,item.stage,item.reproduction_cooldown) for item in a]==[(item.age,item.energy,item.reserve,item.stage,item.reproduction_cooldown) for item in b];assert len({item.stage for item in a})>=2;assert len({round(item.energy,3) for item in a})>=8;assert len({round(item.reproduction_cooldown,3) for item in a})>=10;assert all(item.alive and not item.body.incapacitated for item in a)
