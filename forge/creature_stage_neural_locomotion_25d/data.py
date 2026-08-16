from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from ..creature_stage_developmental import TRAITS,develop
from ..creature_stage_morphology_v2 import morphology_review_genomes
from ..creature_stage_neural_grounded_controller.dataset import owner_metadata
from ..creature_stage_grounded_locomotion_25d import Grounded25DConfig,simulate_25d
from .contract import MAX_APPENDAGES,MAX_MUSCLES


VALIDATION_IDENTITIES=(5,11,17,23,29)


def _muscle_metadata(organism):
    values=np.zeros((MAX_MUSCLES,8),np.float32);owners=np.zeros(MAX_MUSCLES,np.int64);mask=np.zeros(MAX_MUSCLES,np.bool_)
    if len(organism.muscles)>MAX_MUSCLES:raise ValueError("2.5D muscle census exceeded")
    for index,muscle in enumerate(organism.muscles):
        appendage=int(muscle[2]);joint=float(muscle[6]);gene=organism.genome.appendages[appendage]
        owners[index]=appendage;values[index]=(float(muscle[3]),float(muscle[4]),float(muscle[5]),joint/5,np.sin(np.pi*2*gene.phase),np.cos(np.pi*2*gene.phase),np.sin(np.pi*2*joint/5),np.cos(np.pi*2*joint/5));mask[index]=True
    return values,owners,mask


def control_program(program:int,frame:int,total:int)->np.ndarray:
    u=frame/max(total-1,1)
    if program==0:
        angle=np.pi*2*u;speed=.45+.55*np.sin(np.pi*u)**2
        return np.asarray((np.cos(angle),np.sin(angle)),np.float32)*speed
    if program==1:
        directions=((1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1))
        return np.asarray(directions[min(7,int(u*8))],np.float32)
    if program==2:
        return np.asarray((np.sin(np.pi*4*u),np.sin(np.pi*2*u)),np.float32)
    if program==3:
        angle=np.pi*2*(u*u*1.5+.125);return np.asarray((np.cos(angle),np.sin(angle)),np.float32)*(1 if int(frame/15)%3 else .25)
    raise ValueError("2.5D control program drifted")


@dataclass(slots=True)
class LocomotionCorpus:
    global_static:np.ndarray
    appendage_meta:np.ndarray
    appendage_mask:np.ndarray
    muscle_meta:np.ndarray
    muscle_owner:np.ndarray
    muscle_mask:np.ndarray
    dynamic:np.ndarray
    contact:np.ndarray
    muscle:np.ndarray
    velocity:np.ndarray
    identity:np.ndarray
    program:np.ndarray
    semantic_sha256:str

    @property
    def sequences(self)->int:return int(self.dynamic.shape[0])


def build_corpus(output:Path|None=None,*,programs:int=4,frames:int=120)->LocomotionCorpus:
    if programs!=4:raise ValueError("2.5D corpus requires canonical four programs")
    config=Grounded25DConfig(frames=frames)
    static=[];appendage=[];appendage_mask=[];muscle_meta_values=[];muscle_owner_values=[];muscle_mask_values=[]
    dynamic=[];contacts=[];muscles=[];velocities=[];identities=[];program_ids=[]
    for identity,genome in enumerate(morphology_review_genomes()):
        organism=develop(genome);om,omask=owner_metadata(organism);mm,mo,mmask=_muscle_metadata(organism)
        for program in range(programs):
            controls=np.stack([control_program(program,f,frames) for f in range(frames)])
            rollout=simulate_25d(organism,config,control=lambda f,t,p=program:control_program(p,f,t))
            contact=np.stack([f.contact_active for f in rollout.frames])
            contact=np.pad(contact,((0,0),(0,MAX_APPENDAGES-contact.shape[1])))
            previous=np.concatenate((np.zeros((1,MAX_APPENDAGES),np.float32),contact[:-1].astype(np.float32)),axis=0)
            phase=np.stack([(np.sin(np.pi*2*f.phase),np.cos(np.pi*2*f.phase)) for f in rollout.frames]).astype(np.float32)
            velocity=np.stack([f.ground_velocity for f in rollout.frames]).astype(np.float32)
            inputs=np.concatenate((phase,controls,velocity,previous),axis=1).astype(np.float32)
            muscle=np.stack([f.muscle_activation for f in rollout.frames])
            muscle=np.pad(muscle,((0,0),(0,MAX_MUSCLES-muscle.shape[1])))
            family=np.asarray(genome.family_mix,np.float32);traits=np.asarray(genome.traits,np.float32)
            static.append(np.concatenate((family,traits)));appendage.append(om);appendage_mask.append(omask)
            muscle_meta_values.append(mm);muscle_owner_values.append(mo);muscle_mask_values.append(mmask)
            dynamic.append(inputs);contacts.append(contact);muscles.append(muscle);velocities.append(velocity);identities.append(identity);program_ids.append(program)
    arrays={"global_static":np.stack(static).astype(np.float32),"appendage_meta":np.stack(appendage).astype(np.float32),"appendage_mask":np.stack(appendage_mask),"muscle_meta":np.stack(muscle_meta_values).astype(np.float32),"muscle_owner":np.stack(muscle_owner_values).astype(np.int64),"muscle_mask":np.stack(muscle_mask_values),"dynamic":np.stack(dynamic).astype(np.float32),"contact":np.stack(contacts).astype(np.float32),"muscle":np.stack(muscles).astype(np.float32),"velocity":np.stack(velocities).astype(np.float32),"identity":np.asarray(identities,np.int16),"program":np.asarray(program_ids,np.uint8)}
    digest=hashlib.sha256(b"nullvector-neural-locomotion-25d-corpus-v1\0")
    for name in sorted(arrays):digest.update(name.encode()+b"\0"+np.ascontiguousarray(arrays[name]).tobytes())
    corpus=LocomotionCorpus(**arrays,semantic_sha256=digest.hexdigest())
    if output is not None:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output,**arrays,semantic_sha256=np.asarray(corpus.semantic_sha256))
    return corpus


def load_corpus(path:Path)->LocomotionCorpus:
    with np.load(path,allow_pickle=False) as archive:
        values={name:archive[name] for name in ("global_static","appendage_meta","appendage_mask","muscle_meta","muscle_owner","muscle_mask","dynamic","contact","muscle","velocity","identity","program")};recorded=str(archive["semantic_sha256"].item())
    digest=hashlib.sha256(b"nullvector-neural-locomotion-25d-corpus-v1\0")
    for name in sorted(values):digest.update(name.encode()+b"\0"+np.ascontiguousarray(values[name]).tobytes())
    if digest.hexdigest()!=recorded:raise ValueError("2.5D locomotion corpus hash drifted")
    return LocomotionCorpus(**values,semantic_sha256=recorded)
