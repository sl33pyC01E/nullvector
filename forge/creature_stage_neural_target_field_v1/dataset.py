from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np
import torch

from ..creature_stage_developmental.motion import pose
from ..creature_stage_developmental.development import develop
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_grounded_locomotion.physics import _terminal_nodes
from ..creature_stage_neural_grounded_cyclic.curriculum import _scaled_genome
from ..creature_stage_neural_grounded_feedback_v2.dataset import FeedbackCorpus, build_corpus, encode_live
from ..creature_stage_neural_grounded_feedback_v2.contract import MAX_APPENDAGES
from .contract import TARGET_FEATURES, TARGET_PHASE_HARMONICS

def encode_target_context(organism,nodes_local,node_velocity,phase):
    values=np.zeros((MAX_APPENDAGES,TARGET_FEATURES),np.float32);terminals=_terminal_nodes(organism)
    for appendage,terminal in enumerate(terminals):
        gene=organism.genome.appendages[appendage]
        # Target authority is a learned periodic field over immutable anatomy.
        # Local Fourier phase removes the need for an MLP to discover trig
        # addition between gait phase and inherited appendage phase.  It also
        # prevents accumulated live-state error from changing the forcing
        # function on the next cycle.
        local_phase=(float(phase)+float(gene.phase))%1.0
        values[appendage,:4]=(
            float(gene.endpoint[0])/24,float(gene.endpoint[1])/24,
            float(gene.root_offset[0])/24,float(gene.root_offset[1])/24,
        )
        for harmonic in range(1,TARGET_PHASE_HARMONICS+1):
            angle=math.tau*local_phase*harmonic
            offset=4+(harmonic-1)*2
            values[appendage,offset:offset+2]=(math.sin(angle),math.cos(angle))
    return values

@dataclass(slots=True)
class TargetFieldCorpus:
    feedback:FeedbackCorpus; target_context:torch.Tensor; terminal_target:torch.Tensor; semantic_sha256:str
    @property
    def samples(self): return self.feedback.samples
    def batch(self,indices,device):
        result=self.feedback.batch(indices,device); result["target_context"]=self.target_context[indices].to(device);result["terminal_target"]=self.terminal_target[indices].to(device); return result

@dataclass(slots=True)
class TargetAugmentationCorpus:
    owner_state:torch.Tensor; global_state:torch.Tensor; owner_mask:torch.Tensor
    muscle_meta:torch.Tensor; muscle_owner:torch.Tensor; muscle_mask:torch.Tensor
    target_context:torch.Tensor; terminal_target:torch.Tensor; family:torch.Tensor
    semantic_sha256:str
    @property
    def samples(self): return int(self.family.numel())
    def batch(self,indices,device):
        names=("owner_state","global_state","owner_mask","muscle_meta","muscle_owner","muscle_mask","target_context","terminal_target")
        return {name:getattr(self,name)[indices].to(device) for name in names}

def build_target_corpus(*,split:str,variants_per_family:int=2)->TargetFieldCorpus:
    base=build_corpus(split=split,variants_per_family=variants_per_family)
    targets=np.zeros((base.samples,MAX_APPENDAGES,2),np.float32)
    contexts=np.zeros((base.samples,MAX_APPENDAGES,TARGET_FEATURES),np.float32)
    for row,(identity,frame) in enumerate(zip(base.identity.tolist(),base.frame.tolist(),strict=True)):
        organism=base.organisms[identity]; authored=pose(organism,frame/72); terminals=_terminal_nodes(organism);previous=base.cycles[identity].frames[(frame-1)%72]
        contexts[row]=encode_target_context(organism,previous.nodes_local,previous.node_velocity,frame/72)
        for appendage,terminal in enumerate(terminals): targets[row,appendage]=authored.nodes[int(terminal),:2]/24
    d=hashlib.sha256(b"nullvector-target-field-corpus-v1\0"+base.semantic_sha256.encode());d.update(memoryview(contexts));d.update(memoryview(targets))
    return TargetFieldCorpus(base,torch.from_numpy(contexts),torch.from_numpy(targets),d.hexdigest())

def build_target_augmentation(*,variants_per_chassis:int=3)->TargetAugmentationCorpus:
    if not 1<=variants_per_chassis<=8: raise ValueError("target augmentation quota drifted")
    organisms=[]
    for base_index,base in enumerate(review_genomes()):
        family=int(np.argmax(np.asarray(base.family_mix,np.float32)));accepted=0
        for ordinal in range(24+base_index*3,96):
            candidate=_scaled_genome(base,family,ordinal)
            if len(candidate.appendages)>MAX_APPENDAGES: continue
            organisms.append(develop(candidate));accepted+=1
            if accepted==variants_per_chassis: break
        if accepted!=variants_per_chassis: raise RuntimeError("target augmentation chassis quota failed")
    rows=[]
    for organism in organisms:
        family=int(np.argmax(np.asarray(organism.genome.family_mix,np.float32)))
        for frame in range(72):
            phase=frame/72;previous=pose(organism,(frame-1)%72/72);before=pose(organism,(frame-2)%72/72)
            velocity=previous.nodes[:,:2]-before.nodes[:,:2]
            live=encode_live(organism,previous.nodes[:,:2],velocity,previous.planted_contacts,phase,0.0)
            target=np.zeros((MAX_APPENDAGES,2),np.float32);authored=pose(organism,phase);terminals=_terminal_nodes(organism)
            for owner,terminal in enumerate(terminals): target[owner]=authored.nodes[int(terminal),:2]/24
            rows.append((*live,encode_target_context(organism,previous.nodes[:,:2],velocity,phase),target,np.int64(family)))
    arrays=[np.ascontiguousarray(np.stack([row[index] for row in rows])) for index in range(9)]
    arrays[2]=arrays[2].astype(np.bool_);arrays[4]=arrays[4].astype(np.int64);arrays[5]=arrays[5].astype(np.bool_);arrays[8]=arrays[8].astype(np.int64)
    digest=hashlib.sha256(b"nullvector-target-augmentation-v1\0")
    for value in arrays:
        digest.update(value.dtype.str.encode()+np.asarray(value.shape,dtype="<i8").tobytes()+memoryview(value))
    return TargetAugmentationCorpus(*(torch.from_numpy(value) for value in arrays),digest.hexdigest())
