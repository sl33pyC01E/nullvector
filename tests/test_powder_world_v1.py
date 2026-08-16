from __future__ import annotations
import numpy as np
from forge.powder_world_v1 import MATERIALS,MaterialGrid

def test_topdown_fluid_spreads_radially_without_falling() -> None:
    world=MaterialGrid(48,48,seed=1);world.deposit("blood",(24,24),10,2)
    before=float(world.mass.sum())
    for _ in range(80):world.step(.1)
    assert abs(float(world.mass.sum())-before)<1e-3
    yy,xx=np.mgrid[:48,:48];mass=world.mass;cx=float((mass*xx).sum()/mass.sum());cy=float((mass*yy).sum()/mass.sum())
    assert abs(cx-24)<.35 and abs(cy-24)<.35

def test_beam_and_projectile_destroy_real_material_cells() -> None:
    world=MaterialGrid(48,48);mask=np.zeros((48,48),np.bool_);mask[20:28,20:28]=True;world.add_structure(mask,structure_id=1,material="rock")
    before=int(np.count_nonzero(world.material));
    for _ in range(20):world.beam((10,24),(38,24),energy=5,width=.6)
    assert np.count_nonzero(world.material)<before
    world.fire_projectile((5,22),(80,0),energy=3)
    for _ in range(8):world.step(.1)
    assert not world.projectiles

def test_structures_reject_single_pixel_shear_hazards() -> None:
    world=MaterialGrid(32,32);bad=np.zeros((32,32),np.bool_);bad[10,10]=True
    try:world.add_structure(bad,structure_id=2)
    except ValueError:pass
    else:raise AssertionError("singleton structure was accepted")

def test_material_replay_is_exact() -> None:
    worlds=[MaterialGrid(32,32,seed=9),MaterialGrid(32,32,seed=9)]
    for world in worlds:
        world.deposit("sap",(12,12),4,2);world.deposit("acid",(18,18),3,2)
        for _ in range(20):world.step(.1)
    assert worlds[0].semantic_sha256()==worlds[1].semantic_sha256()

