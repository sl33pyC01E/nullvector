from __future__ import annotations

import hashlib
import numpy as np

from .contract import ACTIVITIES,FEATURES,PROJECTS


def teacher(features:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    culture=features[:,0:8];tech=features[:,8:18];family=features[:,18:23];state=features[:,23:39];biome=features[:,39:47];climate=features[:,47:55];relation=features[:,55:59];buildings=features[:,59:64]
    population,wealth,food,power,integrity,danger,scarcity,knowledge=state[:,:8].T
    score=np.full((len(features),len(ACTIVITIES)),.08,np.float32)
    def add(name,value):score[:,ACTIVITIES.index(name)]+=value
    add("forage",scarcity*1.8+(1-food)*.8);add("hunt",culture[:,2]*.8+danger*.25);add("heal",(1-integrity)*2+tech[:,8]*.8);add("breed",food*.55+culture[:,0]*.25);add("graft",tech[:,1]*.7+family[:,0]*.2);add("craft",culture[:,5]*.8+tech[:,2]*.5);add("build",culture[:,5]+population*.45);add("trade",wealth*.45+(1-culture[:,2])*.3);add("explore",culture[:,1]*.85);add("map",culture[:,1]*.65+knowledge*.25);add("study_anomaly",culture[:,7]+biome[:,4]*.6);add("defend",danger*1.3+culture[:,2]*.5);add("negotiate",culture[:,3]*.9+relation[:,2]*.4);add("raid",culture[:,2]*1.1+(1-food)*.25);add("found_colony",population*.55+culture[:,0]*.45);add("recover_relic",culture[:,1]*.55+danger*.2)
    directive=state[:,8:16].argmax(1)*2+(culture[np.arange(len(features)),state[:,8:16].argmax(1)]>.5);score[np.arange(len(features)),directive]+=1.35
    activity=score.argmax(1).astype(np.int64)
    labor=np.stack((.15+scarcity+np.maximum(0,.45-food),.1+culture[:,5]+population*.3,.08+(1-integrity)+tech[:,8]*.4,.08+danger+culture[:,2]*.4,.08+culture[:,1]+knowledge*.25,.07+wealth+relation[:,2]*.3),1).astype(np.float32);labor/=labor.sum(1,keepdims=True)
    hostility=culture[:,2]*.9+danger*.25-relation[:,2]*.7;cooperation=culture[:,3]*.65+culture[:,0]*.35+relation[:,2]*.75;diplomacy=np.where(hostility>cooperation+.2,2,np.where(cooperation>hostility+.15,0,1)).astype(np.int64)
    project_score=np.stack((population+.5*(1-food),culture[:,5]+tech[:,2],(1-integrity)+tech[:,8],(1-food)*1.5+scarcity,culture[:,1]+biome[:,4]*.25,tech[:,1]+(1-integrity)*.3,(1-power)+tech[:,9]*.3,culture[:,4]+culture[:,7],wealth+relation[:,2]),1);project=project_score.argmax(1).astype(np.int64)
    return activity,labor,diplomacy,project


def build_corpus(*,samples:int=240000,seed:int=0x534F43434F5250)->dict:
    rng=np.random.default_rng(seed);features=np.zeros((samples,FEATURES),np.float32);features[:,0:8]=rng.beta(2,2,(samples,8));features[:,8:18]=(rng.random((samples,10))<rng.uniform(.15,.9,(samples,1))).astype(np.float32);families=rng.integers(0,5,samples);features[np.arange(samples),18+families]=1;features[:,23:39]=rng.beta(2,2,(samples,16));biomes=rng.integers(0,8,samples);features[np.arange(samples),39+biomes]=1;climates=rng.integers(0,8,samples);features[np.arange(samples),47+climates]=1;features[:,55:59]=rng.uniform(-1,1,(samples,4));features[:,59:64]=rng.random((samples,5));activity,labor,diplomacy,project=teacher(features);digest=hashlib.sha256(features.astype("<f4").tobytes()+activity.astype("<i8").tobytes()+labor.astype("<f4").tobytes()+diplomacy.astype("<i8").tobytes()+project.astype("<i8").tobytes()).hexdigest();return {"features":features,"activity":activity,"labor":labor,"diplomacy":diplomacy,"project":project,"semantic_sha256":digest}
