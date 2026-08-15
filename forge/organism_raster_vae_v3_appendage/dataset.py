from __future__ import annotations

import math

import numpy as np
import torch

from ..creature_stage_developmental.contract import APPENDAGE_KINDS
from ..creature_stage_developmental.motion import pose
from ..organism_raster_vae_v3.dataset import MorphologyMotionCorpus


APPENDAGE_CLASSES=("body",)+APPENDAGE_KINDS
INPUT_CHANNELS=42+len(APPENDAGE_CLASSES)


class AppendageMotionCorpus(MorphologyMotionCorpus):
    def __getitem__(self,index: int) -> dict[str,torch.Tensor]:
        result=super().__getitem__(index); identity,phase_index=self.rows[index]; organism=self.organisms[identity]; points=pose(organism,phase_index/16).cells.astype(np.float32); low=organism.cell_xy.min(0).astype(np.float32); high=organism.cell_xy.max(0).astype(np.float32); midpoint=(low+high)*.5; local=points-midpoint[None]+np.asarray((23.5,23.5),dtype=np.float32)
        classes=np.zeros(len(points),dtype=np.int64)
        for cell,appendage_index in enumerate(organism.appendage_index):
            if appendage_index>=0: classes[cell]=APPENDAGE_KINDS.index(organism.genome.appendages[int(appendage_index)].kind)+1
        grid=np.zeros((48,48),dtype=np.int64); priority=np.full((48,48),-1,dtype=np.int16); highres=np.zeros((96,96),dtype=np.float32)
        for cell,(xf,yf) in enumerate(local):
            x=int(np.clip(round(float(xf)),0,47)); y=int(np.clip(round(float(yf)),0,47)); score=2 if classes[cell]>0 else 1
            if score>=priority[y,x]: grid[y,x]=classes[cell]; priority[y,x]=score
            if classes[cell]>0:
                hx=float(xf*2+.5); hy=float(yf*2+.5); x0=int(math.floor(hx)); y0=int(math.floor(hy)); dx=hx-x0; dy=hy-y0
                for oy,wy in ((0,1-dy),(1,dy)):
                    for ox,wx in ((0,1-dx),(1,dx)):
                        xx=x0+ox; yy=y0+oy
                        if 0<=xx<96 and 0<=yy<96: highres[yy,xx]=max(highres[yy,xx],min(1.0,.32+max(.12,wx*wy)))
                highres[int(np.clip(round(hy),0,95)),int(np.clip(round(hx),0,95))]=1
        onehot=np.moveaxis(np.eye(len(APPENDAGE_CLASSES),dtype=np.float32)[grid],-1,0)*result["occupancy"].numpy()[None]
        result["living"]=torch.cat((result["living"],torch.from_numpy(onehot)),0); result["appendage"]=torch.from_numpy(grid); result["appendage_alpha"]=torch.from_numpy(highres[None]); return result
