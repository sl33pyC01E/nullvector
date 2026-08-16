from __future__ import annotations
from dataclasses import dataclass

FORMAT="nullvector-topdown-powder-world/1.0.0"
MATERIALS=("empty","soil","rock","water","blood","sap","oil","metal","biomass","acid","fire","smoke","crystal")
STATE=("empty","solid","solid","liquid","liquid","liquid","liquid","solid","solid","liquid","energy","gas","solid")
DENSITY=(0,.8,1,.52,.58,.62,.42,1,.72,.56,.01,.04,.96)
VISCOSITY=(0,1,1,.12,.28,.38,.22,1,1,.16,0,.03,1)
FLAMMABILITY=(0,.08,0,0,.14,.36,.92,0,.30,0,0,0,0)
CORROSION=(0,0,0,0,0,0,0,0,0,1,0,0,0)

@dataclass(slots=True)
class Projectile:
    projectile_id:int
    position:tuple[float,float]
    velocity:tuple[float,float]
    radius:float
    energy:float
    material:int
    owner_id:int|None=None
    alive:bool=True

