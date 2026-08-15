from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..creature_stage_developmental.contract import FAMILIES, TISSUES, TRAITS
from ..creature_stage_developmental.development import DevelopedOrganism, develop
from ..creature_stage_developmental.motion import pose
from ..creature_stage_morphology_v2.genomes import morphology_review_genomes
from .contract import INPUT_CHANNELS


PALETTES=np.asarray([
    (55,213,238),(246,91,159),(126,226,76),(183,82,247),(239,114,65),
],dtype=np.float32)/255
TISSUE_RGB=np.asarray([
    (68,191,216),(232,226,191),(235,70,98),(239,147,68),(127,150,177),
    (240,68,184),(205,47,67),(61,220,204),(223,178,45),(251,235,119),
    (169,111,229),(96,210,71),(171,72,235),(229,91,64),(252,66,76),
],dtype=np.float32)/255
PRIORITY=np.asarray((1,5,4,3,2,10,8,8,8,9,7,2,3,6,9),dtype=np.int16)


def _rasterize(organism: DevelopedOrganism, phase_index: int) -> dict[str,np.ndarray]:
    phase=phase_index/16; motion=pose(organism,phase); points=motion.cells.astype(np.float32)
    low=organism.cell_xy.min(0).astype(np.float32); high=organism.cell_xy.max(0).astype(np.float32); midpoint=(low+high)*.5
    local=points-midpoint[None]+np.asarray((23.5,23.5),dtype=np.float32)
    tissue=organism.tissue.astype(np.int64); family=int(np.argmax(organism.genome.family_mix))
    occupancy=np.zeros((48,48),dtype=np.float32); tissue_grid=np.zeros((48,48),dtype=np.int64); score=np.full((48,48),-1,dtype=np.int16)
    for index,(xf,yf) in enumerate(local):
        x=int(np.clip(round(float(xf)),0,47)); y=int(np.clip(round(float(yf)),0,47)); p=int(PRIORITY[tissue[index]])
        occupancy[y,x]=1
        if p>=score[y,x]: tissue_grid[y,x]=tissue[index]; score[y,x]=p
    onehot=np.moveaxis(np.eye(len(TISSUES),dtype=np.float32)[tissue_grid],-1,0)*occupancy[None]
    traits=np.zeros((len(TRAITS),48,48),dtype=np.float32)
    for index,(xf,yf) in enumerate(local):
        x=int(np.clip(round(float(xf)),0,47)); y=int(np.clip(round(float(yf)),0,47)); traits[:,y,x]=organism.trait_fields[index]
    yy,xx=np.mgrid[:48,:48]; coords=np.stack(((xx-23.5)/24,(yy-23.5)/24)).astype(np.float32)*occupancy[None]
    family_map=np.zeros((5,48,48),dtype=np.float32); family_map[family]=occupancy
    load_path=(tissue_grid==TISSUES.index("bone")).astype(np.float32)*occupancy
    boundary=np.zeros((48,48),dtype=np.float32)
    boundary[1:-1,1:-1]=occupancy[1:-1,1:-1]*(1-(occupancy[:-2,1:-1]*occupancy[2:,1:-1]*occupancy[1:-1,:-2]*occupancy[1:-1,2:]))
    phase_maps=np.stack((np.full((48,48),math.sin(math.tau*phase),dtype=np.float32),np.full((48,48),math.cos(math.tau*phase),dtype=np.float32)))
    living=np.concatenate((occupancy[None],onehot,traits,coords,family_map,boundary[None],load_path[None],phase_maps),axis=0)
    if living.shape!=(INPUT_CHANNELS,48,48): raise RuntimeError(living.shape)
    # Two neural output pixels per logical cell.  Bilinear splats retain the
    # cellular lattice while avoiding the old giant-square presentation.
    alpha=np.zeros((96,96),dtype=np.float32); rgb_sum=np.zeros((3,96,96),dtype=np.float32); weight=np.zeros((96,96),dtype=np.float32)
    colors=np.clip(TISSUE_RGB[tissue]*.72+PALETTES[family][None]*.28,0,1)
    for index,(xf,yf) in enumerate(local*2+0.5):
        x0=int(math.floor(float(xf))); y0=int(math.floor(float(yf))); dx=float(xf-x0); dy=float(yf-y0)
        for oy,wy in ((0,1-dy),(1,dy)):
            for ox,wx in ((0,1-dx),(1,dx)):
                x=x0+ox; y=y0+oy
                if 0<=x<96 and 0<=y<96:
                    w=max(.12,wx*wy); alpha[y,x]=max(alpha[y,x],min(1.0,.32+w)); rgb_sum[:,y,x]+=colors[index]*w; weight[y,x]+=w
        # Connected cells get a small opaque core, leaving subpixel shoulders.
        xc=int(np.clip(round(float(xf)),0,95)); yc=int(np.clip(round(float(yf)),0,95)); alpha[yc,xc]=1
    rgb=rgb_sum/np.maximum(weight[None],1e-6)
    # Chromatic edge light makes individual cells legible without black boxes.
    edge_hi=np.maximum(alpha-np.roll(alpha,1,axis=0),0); rgb=np.clip(rgb+edge_hi[None]*.12,0,1)
    rgba=np.concatenate((rgb,alpha[None]),axis=0).astype(np.float32)
    return {"living":living,"rgba":rgba,"occupancy":occupancy,"tissue":tissue_grid,"family":np.asarray(family,dtype=np.int64),"traits":np.asarray(organism.genome.traits,dtype=np.float32),"phase":np.asarray((math.sin(math.tau*phase),math.cos(math.tau*phase)),dtype=np.float32)}


class MorphologyMotionCorpus(Dataset[dict[str,Tensor]]):
    def __init__(self) -> None:
        self.organisms=tuple(develop(genome) for genome in morphology_review_genomes())
        self.rows=tuple((identity,phase) for identity in range(len(self.organisms)) for phase in range(16))
        digest=hashlib.sha256(b"nullvector-morphology-motion-raster-v3\0")
        for organism in self.organisms: digest.update(organism.identity_sha256.encode("ascii")+b"\0")
        self.semantic_sha256=digest.hexdigest()

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self,index: int) -> dict[str,Tensor]:
        identity,phase=self.rows[index]; row=_rasterize(self.organisms[identity],phase)
        return {name:torch.from_numpy(value) for name,value in row.items()}|{"identity":torch.tensor(identity),"phase_index":torch.tensor(phase)}
