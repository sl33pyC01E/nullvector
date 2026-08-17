from __future__ import annotations

import math
import numpy as np

from ..action_teacher_v2 import extract_actor_features
from ..creature_stage_developmental.contract import FAMILIES
from ..living_body_substrate.contract import ORGAN_SYSTEM
from ..action_teacher_v2.contract import ACTOR_FIELD_SHAPE
from ..powder_world_v1.contract import MATERIALS
from .contract import ORGANISM_FEATURES, ORGANISM_LIMIT, ORGANISM_SHAPE, SPATIAL_SHAPE


def _view_geometry(world, camera, span: float):
    axis = (np.arange(32, dtype=np.float64) + .5) / 32 * span - span * .5
    xx, yy = np.meshgrid(axis, axis)
    wx = (float(camera[0]) + xx) % world.size
    wy = (float(camera[1]) + yy) % world.size
    ix = np.floor(wx).astype(np.int32) % world.size
    iy = np.floor(wy).astype(np.int32) % world.size
    return ix, iy


def _pixel(world, camera, span: float, position) -> tuple[int, int] | None:
    delta = (np.asarray(position, np.float64) - np.asarray(camera, np.float64) + world.size * .5) % world.size - world.size * .5
    x = int(math.floor((delta[0] / span + .5) * 32))
    y = int(math.floor((delta[1] / span + .5) * 32))
    return (x, y) if 0 <= x < 32 and 0 <= y < 32 else None


def extract_spatial_state(world, society, feeding, *, camera, span: float, selected: int, topology=None) -> np.ndarray:
    if not 8 <= span <= world.size * 1.5:
        raise ValueError("whole-viewport span drifted")
    ix, iy = _view_geometry(world, camera, span)
    field = np.zeros(SPATIAL_SHAPE, np.float32)
    field[:10] = np.asarray(world.fields[:, iy, ix], np.float32)
    material = world.materials.material[iy, ix]
    for index in range(len(MATERIALS)):
        field[10 + index] = material == index
    field[23] = np.clip(world.materials.mass[iy, ix], 0, 1)
    field[24] = np.clip(world.materials.temperature[iy, ix], 0, 1)
    field[25] = np.clip(world.materials.damage[iy, ix], 0, 1)
    field[26] = world.materials.structure_id[iy, ix] > 0
    for entity in sorted(world.organisms.values(), key=lambda item: item.entity_id):
        point = _pixel(world, camera, span, entity.position)
        if point is None: continue
        x, y = point; body = entity.body; snapshot = body.snapshot(); count = max(1, body.organism.cell_count)
        field[27 + entity.family, y, x] = 1
        field[32, y, x] = max(field[32, y, x], snapshot.alive_cells / count)
        field[33, y, x] = max(field[33, y, x], float(body.fluid.sum() / max(float(body.fluid_capacity.sum()), 1e-6)))
        field[34, y, x] = max(field[34, y, x], float(body.scar.mean()))
        systems = snapshot.systems
        field[35, y, x] = max(field[35, y, x], systems.get("neural", 0.0))
        field[36, y, x] = max(field[36, y, x], min(systems.get("circulation", 0.0), systems.get("respiration", 0.0), systems.get("digestion", 0.0)))
        field[37, y, x] = max(field[37, y, x], float(not entity.alive))
        field[38, y, x] = max(field[38, y, x], float(entity.entity_id == selected))
        field[39, y, x] = np.clip(entity.velocity[0] / 12, -1, 1)
        field[40, y, x] = np.clip(entity.velocity[1] / 12, -1, 1)
    for projectile in world.materials.projectiles:
        point = _pixel(world, camera, span, projectile.position)
        if point is None: continue
        x, y = point; field[41, y, x] = max(field[41, y, x], min(1, projectile.energy))
        field[42, y, x] = np.clip(projectile.velocity[0] / 20, -1, 1); field[43, y, x] = np.clip(projectile.velocity[1] / 20, -1, 1)
    for clump in feeding.clumps.values():
        point = _pixel(world, camera, span, clump.food.position)
        if point is None: continue
        x, y = point; field[44, y, x] = max(field[44, y, x], min(1, clump.food.mass)); field[45, y, x] = max(field[45, y, x], min(1, clump.height / 8))
    for settlement in society.settlements.values():
        for road in settlement.roads:
            point = _pixel(world, camera, span, road)
            if point is not None: field[46, point[1], point[0]] = 1
        point = _pixel(world, camera, span, settlement.center)
        if point is not None: field[47, point[1], point[0]] = min(1, settlement.population / 32)
    if topology is not None:
        if topology.shape != (world.size, world.size):
            raise ValueError("whole-viewport topology/world size drifted")
        for index in range(9):
            field[48 + index] = topology.terrain[iy, ix] == index
        for index in range(5):
            field[57 + index] = topology.hazard[iy, ix] == index
        field[62] = np.clip(topology.elevation[iy, ix] / 8, -1, 1)
        field[63] = np.clip(topology.nav_cost[iy, ix] / 8, 0, 1)
        zones = topology.zone[iy, ix]
        field[64] = np.where(zones >= 0, (zones % 16) / 15, 0)
        field[65] = topology.protected_backbone[iy, ix]
        field[66] = topology.required_clearance[iy, ix]
        field[67] = topology.decoration_forbidden[iy, ix]
    if field.shape != SPATIAL_SHAPE or not np.isfinite(field).all():
        raise ValueError("whole-viewport spatial state drifted")
    return field.astype(np.float16)


def _pose_features(entity, points) -> np.ndarray:
    points=np.asarray(points,np.float32);result=np.zeros(32,np.float32);appendage=entity.body.organism.appendage_index;bottom=float(points[:,1].max())
    for index in range(8):
        chosen=appendage==index
        if not chosen.any():continue
        local=points[chosen];health=entity.body.health[chosen];offset=index*4;result[offset:offset+4]=(float(local[:,0].mean()/24),float(local[:,1].mean()/24),float(health.mean()),float(np.any(local[:,1]>=bottom-1)))
    return result

def extract_posed_actor_field(entity,points) -> np.ndarray:
    field=np.zeros(ACTOR_FIELD_SHAPE,np.float32);body=entity.body;organism=body.organism;points=np.asarray(points,np.float32);xy=np.rint(points*.6+15.5).astype(np.int32);valid=(xy[:,0]>=0)&(xy[:,0]<32)&(xy[:,1]>=0)&(xy[:,1]<32);connected=body._connected_to_core();capacity=np.maximum(body.fluid_capacity,1e-6);vital=np.asarray([ORGAN_SYSTEM.get(organ,"") in ("neural","circulation","respiration","digestion") for organ in body.organ],np.float32);neural=np.asarray([ORGAN_SYSTEM.get(organ,"")=="neural" for organ in body.organ],np.float32);values=np.stack((np.ones(organism.cell_count),body.health,body.fluid/capacity,body.scar,connected.astype(np.float32),neural,vital,body._appendage_mask.astype(np.float32))).astype(np.float32);counts=np.zeros((32,32),np.float32)
    for index in np.flatnonzero(valid):
        x,y=xy[index];field[0,y,x]=1;field[1:5,y,x]+=values[1:5,index];field[5:,y,x]=np.maximum(field[5:,y,x],values[5:,index]);counts[y,x]+=1
    occupied=counts>0;field[1:5,occupied]/=counts[occupied];return field.astype(np.float16)

def extract_organism_tokens(world, *, camera, span: float, selected: int, posed_points=None) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for entity in world.organisms.values():
        delta = (entity.position - np.asarray(camera) + world.size * .5) % world.size - world.size * .5
        if max(abs(delta[0]), abs(delta[1])) > span * .62: continue
        prefix = np.asarray((delta[0] / span, delta[1] / span, float(entity.entity_id == selected), float(entity.alive)), np.float32)
        points=entity.body.organism.cell_xy if posed_points is None or entity.entity_id not in posed_points else posed_points[entity.entity_id]
        rows.append((float(np.dot(delta, delta)), entity.entity_id, np.concatenate((prefix, extract_actor_features(world, entity.entity_id),_pose_features(entity,points)))))
    rows.sort(key=lambda item: (item[0], item[1]))
    tokens = np.zeros(ORGANISM_SHAPE, np.float16); mask = np.zeros((ORGANISM_LIMIT,), np.bool_)
    for index, (_, _, row) in enumerate(rows[:ORGANISM_LIMIT]):
        if row.shape != (ORGANISM_FEATURES,): raise ValueError("whole-viewport organism token drifted")
        tokens[index] = row.astype(np.float16); mask[index] = True
    return tokens, mask
