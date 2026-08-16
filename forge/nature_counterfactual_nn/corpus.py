from __future__ import annotations
import hashlib
import numpy as np
from .contract import ACTIONS,FEATURES,FORMAT
from ..nature_timeline_nn.corpus import build_corpus as build_timeline_corpus

def build_corpus(*,groups:int=6554,seed:int=0x434F554E544552):
    base=build_timeline_corpus(samples=groups,seed=seed^0x54494D45);sequence=np.repeat(base["sequence"],len(ACTIONS),axis=0);unmodified=np.repeat(base["target"],len(ACTIONS),axis=0);target=unmodified.copy();action=np.tile(np.arange(len(ACTIONS),dtype=np.int64),groups);score=np.zeros(len(action),np.float32);risk=np.zeros(len(action),np.float32)
    for index in range(len(ACTIONS)):
        selected=action==index;rows=target[selected];original=unmodified[selected]
        if index==0:
            need=1-original[:,[12,20,21]].mean(1);gain=.05+.18*need;rows[:,12]+=gain;rows[:,13]+=gain*.45;rows[:,20]+=gain*1.2;rows[:,21]+=gain*.8;rows[:,0]+=gain*.18;rows[:,8]+=gain*.28;score[selected]=np.clip(.22+need*.68,0,1);risk[selected]=np.clip(.12+original[:,19]*.22,0,1)
        elif index==1:
            need=(original[:,18]+original[:,19]+1-original[:,12])/3;gain=.04+.2*need;rows[:,12]+=gain;rows[:,17]+=gain*.8;rows[:,18]-=gain*.75;rows[:,19]-=gain;rows[:,35]+=gain*.35;score[selected]=np.clip(.18+need*.76,0,1);risk[selected]=np.clip(.16+original[:,15]*.12,0,1)
        elif index==2:
            need=(original[:,16]+original[:,19])/2;gain=.04+.24*need;rows[:,16]-=gain;rows[:,19]-=gain*.62;rows[:,14]+=gain*.42;rows[:,39]+=gain*.24;score[selected]=np.clip(.16+need*.82,0,1);risk[selected]=np.clip(.24+(1-original[:,39])*.18,0,1)
        elif index==3:
            need=np.clip(original[:,10]*1.7+original[:,43]*.7,0,1);gain=.03+.19*need;rows[:,10]-=gain*.55;rows[:,43]-=gain*.9;rows[:,44]+=gain;rows[:,9]-=gain*.16;score[selected]=np.clip(.14+need*.83,0,1);risk[selected]=np.clip(.20+original[:,9]*.16,0,1)
        else:
            need=np.clip(original[:,0]*(1-original[:,7])+.25*(1-original[:,20]),0,1);gain=.04+.18*need;rows[:,7]+=gain*.8;rows[:,20]+=gain*.5;rows[:,21]+=gain*.35;rows[:,0]+=gain*.12;score[selected]=np.clip(.17+need*.76,0,1);risk[selected]=np.clip(.27+(1-original[:,14])*.14,0,1)
        target[selected]=np.clip(rows,0,1)
    digest=hashlib.sha256();digest.update(FORMAT.encode())
    for array in (sequence,action,target,score,risk):digest.update(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
    return {"format":FORMAT,"sequence":sequence.astype(np.float32),"action":action,"target":target.astype(np.float32),"score":score,"risk":risk,"groups":groups,"upstream_sha256":base["semantic_sha256"],"semantic_sha256":digest.hexdigest()}
