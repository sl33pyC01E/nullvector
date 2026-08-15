from __future__ import annotations

import math

import numpy as np
import torch

from ..creature_stage_developmental.contract import APPENDAGE_KINDS
from ..creature_stage_developmental.motion import pose
from ..organism_raster_vae_v3_appendage.dataset import AppendageMotionCorpus


MAX_TOKENS=8;TOKEN_FEATURES=26


class GraphTokenCorpus(AppendageMotionCorpus):
    def __getitem__(self,index: int) -> dict[str,torch.Tensor]:
        result=super().__getitem__(index);identity,phase_index=self.rows[index];organism=self.organisms[identity];phase=phase_index/16;points=pose(organism,phase).cells.astype(np.float32);low=organism.cell_xy.min(0).astype(np.float32);high=organism.cell_xy.max(0).astype(np.float32);midpoint=(low+high)*.5;local=points-midpoint[None]+np.asarray((23.5,23.5),dtype=np.float32)
        tokens=np.zeros((MAX_TOKENS,TOKEN_FEATURES),dtype=np.float32);mask=np.zeros(MAX_TOKENS,dtype=np.bool_)
        if len(organism.genome.appendages)>MAX_TOKENS:raise ValueError("graph token appendage census exceeded")
        for token,gene in enumerate(organism.genome.appendages):
            tokens[token,APPENDAGE_KINDS.index(gene.kind)]=1;tokens[token,8]=gene.side;tokens[token,9]=gene.segments/5;tokens[token,10]=math.sin(math.tau*gene.phase);tokens[token,11]=math.cos(math.tau*gene.phase);tokens[token,12:14]=np.asarray(gene.root_offset)/24;tokens[token,14:16]=np.asarray(gene.endpoint)/24;delta=np.asarray(gene.endpoint)-np.asarray(gene.root_offset);tokens[token,16:18]=delta/24;tokens[token,18]=np.linalg.norm(delta)/32;tokens[token,19]=gene.bend;tokens[token,20]=float(gene.paired_with is not None)
            mode={"leg":0,"root":1,"wheel":2}.get(gene.kind,3);tokens[token,21+mode]=1;local_phase=(phase+gene.phase)%1;stance={"leg":.58,"root":.76,"wheel":.56}.get(gene.kind,0);tokens[token,25]=float(stance>0 and local_phase<stance);mask[token]=True
        owner=np.full((48,48),-1,dtype=np.int64);priority=np.full((48,48),-1,dtype=np.int16)
        for cell,(xf,yf) in enumerate(local):
            token=int(organism.appendage_index[cell]);x=int(np.clip(round(float(xf)),0,47));y=int(np.clip(round(float(yf)),0,47));score=2 if token>=0 else 1
            if score>=priority[y,x]:owner[y,x]=token;priority[y,x]=score
        result["living"]=result["living"][:42];result["tokens"]=torch.from_numpy(tokens);result["token_mask"]=torch.from_numpy(mask);result["token_owner"]=torch.from_numpy(owner);return result
