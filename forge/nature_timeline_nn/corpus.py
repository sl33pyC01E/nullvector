from __future__ import annotations
import hashlib
import numpy as np
from .contract import FEATURES,SEQUENCE

def build_corpus(*,samples:int=32768,seed:int=0x54494D45434F5250):
    rng=np.random.default_rng(seed);sequence=np.zeros((samples,SEQUENCE,FEATURES),np.float32);target=np.zeros((samples,FEATURES),np.float32);events=np.zeros(samples,np.int64)
    for sample in range(samples):
        pop=float(rng.uniform(.05,.9));families=rng.dirichlet(np.ones(5));resources=rng.beta(2,2,10);lineages=float(rng.uniform(.05,.8));colonies=float(rng.uniform(0,.6));cumulative=np.zeros(4);season=int(rng.integers(6));systems=rng.beta(7,1.8,7);intents=rng.dirichlet(np.ones(12))
        for tick in range(SEQUENCE+1):
            climate=np.asarray((.5+.35*np.sin(tick*.11),.5+.35*np.cos(tick*.09),.4+.3*np.sin(tick*.07),.1+.25*max(0,np.sin(tick*.13)),.04+.08*max(0,np.cos(tick*.17)),tick/SEQUENCE),np.float64);birth=max(0,pop*(resources[[0,8,9]].mean()-.28)*.018);death=max(0,pop*((1-systems.mean())+.2-resources[[0,8,9]].mean())*.012);pred=max(0,pop*families[1]*families[2]*.009);mutation=birth*(.08+.12*resources[4]);delta=np.asarray((birth,death,pred,mutation));cumulative=np.clip(cumulative+delta,0,1);pop=np.clip(pop+birth-death,0.01,1);resources=np.clip(resources+(.5-resources)*.004-rng.uniform(0,.002,10)*pop,0,1);systems=np.clip(systems+(resources.mean()-.45)*.002-rng.uniform(0,.001,7),0,1);colonies=np.clip(colonies+birth*.08-death*.02,0,1);lineages=np.clip(lineages+mutation*.05-death*.01,0,1);row=np.concatenate(([pop],families,[lineages,colonies],cumulative,resources,climate,np.eye(6)[season],systems,intents));row=np.pad(row,(0,FEATURES-len(row))).astype(np.float32)
            if tick<SEQUENCE:sequence[sample,tick]=row
            else:target[sample]=row
        scores=np.asarray((.15,birth*120,death*150,pred*220,mutation*260,colonies*.08,climate[3]*.16,resources[2]*.08,lineages*.07,max(0,.25-resources.mean())*.18));events[sample]=int(np.argmax(scores+rng.uniform(0,.025,len(scores))))
    digest=hashlib.sha256(sequence.astype("<f4").tobytes()+target.astype("<f4").tobytes()+events.astype("<i8").tobytes()).hexdigest();return {"sequence":sequence,"target":target,"event":events,"semantic_sha256":digest}
