from __future__ import annotations

import hashlib
import math
import numpy as np

from .contract import BuildingPlan


PURPOSES=("habitat","workshop","clinic","granary","observatory","graft_house","battery_hall","shrine","market")


def generate_building(*,seed:int,origin:tuple[int,int],purpose:str|None=None)->BuildingPlan:
    rng=np.random.default_rng(seed);purpose=purpose or PURPOSES[int(rng.integers(len(PURPOSES)))]
    width=int(rng.integers(9,20));height=int(rng.integers(9,18));width+=width%2;height+=height%2
    ox,oy=origin;cells=[];door_x=width//2
    # Two-cell-thick conglomerate walls prevent a one-pixel structural shard
    # from shearing a fast cellular creature.
    for y in range(height):
        for x in range(width):
            boundary=x<2 or y<2 or x>=width-2 or y>=height-2
            material="wall" if boundary else "floor"
            if y>=height-2 and door_x-1<=x<=door_x:material="door"
            cells.append((ox+x,oy+y,material))
    interior=[i for i,c in enumerate(cells) if c[2]=="floor"]
    feature={"clinic":"utility","granary":"storage","observatory":"utility","graft_house":"utility","battery_hall":"utility","market":"storage"}.get(purpose,"garden" if purpose=="habitat" else "utility")
    for index in rng.choice(interior,size=min(len(interior),max(2,width*height//24)),replace=False):
        x,y,_=cells[int(index)];cells[int(index)]=(x,y,feature)
    building_id=f"b-{hashlib.sha256(f'{seed}:{origin}:{purpose}'.encode()).hexdigest()[:14]}"
    return BuildingPlan(building_id,purpose,origin,width,height,tuple(cells),((ox+door_x-1,oy+height-1),(ox+door_x,oy+height-1)))


def expand_settlement(*,seed:int,center:tuple[float,float],count:int)->tuple[list[BuildingPlan],set[tuple[int,int]]]:
    if not 1<=count<=128:raise ValueError("settlement building count drifted")
    buildings=[];roads=set();cx,cy=int(center[0]),int(center[1]);golden=math.pi*(3-math.sqrt(5))
    for index in range(count):
        radius=9+math.sqrt(index)*18;angle=index*golden+(seed%997)*.001
        origin=(int(round(cx+math.cos(angle)*radius)),int(round(cy+math.sin(angle)*radius)))
        building=generate_building(seed=seed+index*7919,origin=origin,purpose=PURPOSES[index%len(PURPOSES)]);buildings.append(building)
        door=building.entrances[0]
        x,y=cx,cy
        while x!=door[0]:roads.add((x,y));x+=1 if door[0]>x else -1
        while y!=door[1]:roads.add((x,y));y+=1 if door[1]>y else -1
        roads.add(door)
    return buildings,roads


def validate_building(plan:BuildingPlan)->dict[str,bool]:
    lookup={(x,y):material for x,y,material in plan.cells};ox,oy=plan.origin
    wall_thick=all(lookup.get((ox+x,oy))=="wall" and lookup.get((ox+x,oy+1))=="wall" for x in range(plan.width) if not (plan.width//2-1<=x<=plan.width//2))
    floors={(x,y) for x,y,m in plan.cells if m!="wall"};stack=[plan.entrances[0]];seen=set(stack)
    while stack:
        x,y=stack.pop()
        for point in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if point in floors and point not in seen:seen.add(point);stack.append(point)
    return {"two_cell_wall":wall_thick,"two_cell_door":len(plan.entrances)==2,"interior_connected":floors<=seen,"no_single_cell_wall":sum(m=="wall" for m in lookup.values())>=plan.width*4}

