from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

import numpy as np
import torch

from ..cellular_nca.contract import BOND_CHANNELS, DIRECTION_XY, DYNAMIC_CHANNELS, STATIC_CHANNELS
from ..cellular_nca_causal.evaluation import validate_output
from ..cellular_nca_causal.training import load_final_checkpoint
from ..cellular_nca_selection_v1 import load_runtime as load_selected_runtime
from ..cellular_nca_selection_v1 import validate as validate_selected_runtime
from ..creature_stage_developmental.contract import TISSUES, TRAITS
from ..living_body_substrate.contract import ORGAN_SYSTEM
from .contract import CANVAS, DEFAULT_AUTHORITY


SYSTEM_NAMES = ("circulation", "respiration", "digestion", "neural", "senses", "locomotion", "repair", "immune")


@dataclass(slots=True)
class BodyRaster:
    static: np.ndarray
    state: np.ndarray
    live_bonds: np.ndarray
    canvas_xy: np.ndarray
    organism_sha256: str


def _canvas_coordinates(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.int16)
    minimum = points.min(0)
    maximum = points.max(0)
    extent = maximum - minimum + 1
    if np.any(extent > CANVAS - 2):
        raise ValueError("living body exceeds causal NCA canvas")
    offset = ((CANVAS - extent) // 2 + 1) - minimum
    result = points + offset
    if np.any(result < 1) or np.any(result >= CANVAS - 1) or len({tuple(row) for row in result.tolist()}) != len(result):
        raise ValueError("living body raster coordinates drifted")
    return result.astype(np.int16)


def _boundary(points: np.ndarray) -> np.ndarray:
    occupied = np.zeros((CANVAS, CANVAS), np.bool_)
    y, x = points[:, 1].astype(np.intp), points[:, 0].astype(np.intp)
    occupied[y, x] = True
    interior = occupied[y - 1, x] & occupied[y + 1, x] & occupied[y, x - 1] & occupied[y, x + 1]
    return (~interior).astype(np.float32)


def _system_weights(body) -> np.ndarray:
    count = body.organism.cell_count
    weights = np.zeros((len(SYSTEM_NAMES), count), np.float32)
    authority = body.organism.component_weights.max(1).astype(np.float32)
    for index, organ in enumerate(body.organ):
        system = ORGAN_SYSTEM.get(organ)
        if system in SYSTEM_NAMES:
            weights[SYSTEM_NAMES.index(system), index] = authority[index]
    weights[5] = (body.organism.appendage_index >= 0).astype(np.float32)
    regeneration = float(body.organism.genome.traits[TRAITS.index("regeneration")])
    weights[6] = regeneration * (.35 + .65 * (body.organism.tissue != TISSUES.index("bone")))
    weights[7] = .25 + .55 * np.isin(body.organism.tissue, (TISSUES.index("skin"), TISSUES.index("vascular")))
    return weights


def _dynamic_raster(body, static: np.ndarray, state: np.ndarray, bonds: np.ndarray, canvas_xy: np.ndarray) -> BodyRaster:
    y, x = canvas_xy[:, 1].astype(np.intp), canvas_xy[:, 0].astype(np.intp)
    state[0, y, x] = body.health
    state[1, y, x] = body.fluid / np.maximum(body.fluid_capacity, 1e-6)
    state[3, y, x] = body.energy
    state[6, y, x] = body.scar
    state[7, y, x] = np.clip(1 - body.health, 0, 1)
    state[11, y, x] = body.alive_mask
    live_grid = np.zeros((CANVAS, CANVAS), np.float32)
    live_grid[y, x] = body.alive_mask
    for channel, (dx, dy) in enumerate(DIRECTION_XY):
        bonds[channel] *= live_grid * np.roll(live_grid, shift=(-dy, -dx), axis=(0, 1))
    if not (np.isfinite(static).all() and np.isfinite(state).all() and np.isfinite(bonds).all()):
        raise FloatingPointError("living body NCA raster became non-finite")
    return BodyRaster(static, np.clip(state, 0, 1), bonds, canvas_xy, body.organism.identity_sha256)


def rasterize_body(body, previous_state: np.ndarray | BodyRaster | None = None) -> BodyRaster:
    organism = body.organism
    count = organism.cell_count
    if isinstance(previous_state, BodyRaster) and previous_state.organism_sha256 == organism.identity_sha256:
        static = previous_state.static
        state = previous_state.state.copy()
        canvas_xy = previous_state.canvas_xy
        bonds = (static[77 : 77 + BOND_CHANNELS] > 0).astype(np.float32)
        return _dynamic_raster(body, static, state, bonds, canvas_xy)
    canvas_xy = _canvas_coordinates(organism.cell_xy)
    y, x = canvas_xy[:, 1].astype(np.intp), canvas_xy[:, 0].astype(np.intp)
    static = np.zeros((STATIC_CHANNELS, CANVAS, CANVAS), np.float32)
    state = np.zeros((DYNAMIC_CHANNELS, CANVAS, CANVAS), np.float32) if previous_state is None else np.asarray(previous_state, np.float32).copy()
    bonds = np.zeros((BOND_CHANNELS, CANVAS, CANVAS), np.float32)
    static[0, y, x] = 1
    tissue = organism.tissue.astype(np.intp)
    for tissue_id in range(len(TISSUES)):
        if tissue_id: static[tissue_id, y, x] = tissue == tissue_id
    boundary = _boundary(canvas_xy)
    appendage = organism.appendage_index >= 0
    static[15, y, x] = boundary
    static[16, y[body.main_seed_cell], x[body.main_seed_cell]] = 1
    static[17, y, x] = appendage
    static[18, y, x] = organism.side > 0
    static[19, y, x] = organism.side < 0
    static[20, y, x] = tissue == TISSUES.index("sensor")
    static[21, y, x] = tissue == TISSUES.index("weapon")
    static[22, y, x] = np.isin(tissue, (TISSUES.index("phase"), TISSUES.index("machine"), TISSUES.index("root")))
    static[23 + body.family, y, x] = 1
    system_weights = _system_weights(body)
    for system in range(len(SYSTEM_NAMES)):
        active = system_weights[system] > 0
        static[28 + system * 3, y, x] = active
        static[52 + system, y, x] = system_weights[system]
    heal_class = np.select(
        (tissue == TISSUES.index("bone"), tissue == TISSUES.index("muscle"), tissue == TISSUES.index("vascular"), tissue == TISSUES.index("neural"), tissue == TISSUES.index("machine")),
        (2, 3, 4, 5, 6), default=1,
    )
    for value in range(1, 7): static[59 + value, y, x] = heal_class == value
    static[66, y, x] = .25 + .65 * (tissue == TISSUES.index("vascular")); static[67, y, x] = .18 + .55 * (tissue == TISSUES.index("skin")); static[68, y, x] = system_weights[6]
    static[69, y, x] = 1; static[70, y, x] = np.clip(body.fluid_capacity / 1.07, 0, 1); static[71, y, x] = .45
    static[72, y, x] = .65; static[73, y, x] = np.clip(.35 + organism.trait_fields[:, TRAITS.index("size")] * .65, 0, 1); static[74, y, x] = organism.trait_fields[:, TRAITS.index("stiffness")]
    direction_index = {direction: index for index, direction in enumerate(DIRECTION_XY)}
    degree = np.zeros(count, np.float32)
    for left_raw, right_raw in body.adjacency:
        left, right = int(left_raw), int(right_raw)
        delta = tuple((canvas_xy[right] - canvas_xy[left]).tolist())
        if delta not in direction_index or (-delta[0], -delta[1]) not in direction_index:
            continue
        forward, reverse = direction_index[delta], direction_index[(-delta[0], -delta[1])]
        bonds[forward, y[left], x[left]] = 1; bonds[reverse, y[right], x[right]] = 1; static[77 + forward, y[left], x[left]] = .75; static[77 + reverse, y[right], x[right]] = .75; degree[left] += 1; degree[right] += 1
    static[75, y, x] = degree / 8; static[76, y, x] = np.where(degree > 0, .75, 0)
    if previous_state is None:
        state[2, y, x] = .35 + .45 * system_weights[2]; state[4, y, x] = .88; state[8, y, x] = np.clip(.08 + .92 * system_weights[3], 0, 1)
    return _dynamic_raster(body, static, state, bonds, canvas_xy)


class LivingBodyNCARuntime:
    def __init__(self, model, device: torch.device, *, blend: float = .55) -> None:
        if not 0 < blend <= 1: raise ValueError("living body NCA blend drifted")
        self.model=model.eval();self.device=device;self.blend=float(blend);self.states:dict[Hashable,BodyRaster]={}

    @classmethod
    def from_output(cls, output: Path = DEFAULT_AUTHORITY, *, device: str = "cuda", blend: float = .55):
        output=Path(output).resolve()
        if (output / "selection_manifest.json").is_file():
            validation=validate_selected_runtime(output)
            if not validation["passed"]: raise ValueError("selected causal NCA authority is not ready")
            model,_,_=load_selected_runtime(output)
        else:
            validation=validate_output(output,rerun_evaluation=False,device_name="cpu")
            if not validation["passed"] or validation["status"]!="ready": raise ValueError("causal NCA authority is not ready")
            model,_,_=load_final_checkpoint(output)
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");return cls(model.to(target),target,blend=blend)

    def forget(self,key:Hashable)->None:self.states.pop(key,None)

    @torch.inference_mode()
    def step_many(self, items: Iterable[tuple[Hashable, object]]) -> dict[Hashable, object]:
        rows=[]
        for key,body in items:
            previous=self.states.get(key);row=rasterize_body(body,previous);rows.append((key,body,row))
        if not rows:return {}
        static=torch.from_numpy(np.stack([row.static for _,_,row in rows])).to(self.device);state=torch.from_numpy(np.stack([row.state for _,_,row in rows])).to(self.device);bonds=torch.from_numpy(np.stack([row.live_bonds for _,_,row in rows])).to(self.device)
        predicted=self.model(static,state,bonds).float().cpu().numpy();result={}
        for index,(key,body,row) in enumerate(rows):
            row.state=np.ascontiguousarray(predicted[index]);self.states[key]=row;result[key]=self._apply(body,row)
        return result

    def step(self,key:Hashable,body):return self.step_many(((key,body),))[key]

    def _apply(self,body,row:BodyRaster):
        y=row.canvas_xy[:,1].astype(np.intp);x=row.canvas_xy[:,0].astype(np.intp);state=row.state;blend=self.blend
        health=state[0,y,x];fluid=state[1,y,x]*body.fluid_capacity;scar=state[6,y,x]
        body.health=np.clip(body.health*(1-blend)+health*blend,0,1).astype(np.float32);body.fluid=np.clip(body.fluid*(1-blend)+fluid*blend,0,body.fluid_capacity).astype(np.float32);body.scar=np.clip(body.scar*(1-blend)+scar*blend,0,1).astype(np.float32)
        energy=float(np.mean(state[3,y,x]));body.energy=float(np.clip(body.energy*(1-blend)+energy*blend,0,1.2));body._connectivity_alive=None;body._systems_health=None;body._systems_fluid=None;body._systems_cache=None
        systems=body.systems();body.incapacitated=systems["neural"]<.16 or systems["locomotion"]<.06;body.dead=systems["neural"]<.035 or systems["integrity"]<.10 or (systems["circulation"]<.04 and systems["respiration"]<.04)
        return body.snapshot()
