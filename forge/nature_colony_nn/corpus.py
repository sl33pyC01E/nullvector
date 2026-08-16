from __future__ import annotations

import hashlib
import numpy as np

from ..nature_sim_v2 import founder_genomes,phenotype_vector
from .contract import FEATURES,MAX_MEMBERS,ROLES


def build_corpus(*,colonies:int=12000,seed:int=0x434F52505553)->dict[str,np.ndarray|str]:
    rng=np.random.default_rng(seed);founders=founder_genomes(variants_per_family=6);base=np.stack([phenotype_vector(genome) for genome in founders]);families=np.asarray([genome.family for genome in founders]);features=np.zeros((colonies,MAX_MEMBERS,FEATURES),np.float32);roles=np.full((colonies,MAX_MEMBERS),-100,np.int64);actions=np.zeros((colonies,MAX_MEMBERS,3),np.float32);mask=np.zeros((colonies,MAX_MEMBERS),np.bool_);world_ids=np.arange(colonies,dtype=np.int64)
    for colony in range(colonies):
        count=int(rng.integers(4,MAX_MEMBERS+1));family=int(rng.integers(5));choices=np.flatnonzero(families==family);cohesion=float(rng.uniform(.2,1));store=float(rng.beta(1.5,5));size=count/MAX_MEMBERS
        for member in range(count):
            genome_index=int(rng.choice(choices));vector=np.clip(base[genome_index]+rng.normal(0,.035,44),0,1);systems=np.clip(rng.beta(7,1.8,7),0,1);energy=float(rng.beta(3.2,2.2));reserve=float(rng.beta(2.4,3));stage=float(rng.integers(4)/3);role_phase=(colony+member+genome_index)%len(ROLES);phase=role_phase/len(ROLES);row=np.concatenate((vector,np.eye(5,dtype=np.float32)[family],systems,np.asarray((energy,reserve,stage,size,store,cohesion,np.sin(phase*np.pi*2),np.cos(phase*np.pi*2)),np.float32)))
            features[colony,member]=row;mask[colony,member]=True
            eco=vector[15:31];dev=vector[:15];score=np.asarray((.15+.4*(vector[39]+vector[40])/2,.15+.4*(dev[13]+eco[11])/2,.15+.4*(dev[3]+dev[5]+eco[9])/3,.15+.4*(dev[11]+eco[8]+dev[9])/3,.15+.4*(eco[3]+eco[12])/2,.15+.4*(dev[3]+dev[6]+eco[15])/3),np.float32);score[role_phase]+=.58;roles[colony,member]=int(np.argmax(score));actions[colony,member]=(max(0,energy-.88)*(.4+eco[10]),(1-systems[0])*(.25+dev[11]+eco[8]),dev[13]*(1-cohesion))
    digest=hashlib.sha256(features.astype("<f4").tobytes()+roles.astype("<i8").tobytes()+actions.astype("<f4").tobytes()+mask.tobytes()).hexdigest();return {"features":features,"roles":roles,"actions":actions,"mask":mask,"world_ids":world_ids,"semantic_sha256":digest}
