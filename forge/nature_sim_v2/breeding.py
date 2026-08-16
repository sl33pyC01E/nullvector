from __future__ import annotations

import hashlib
import numpy as np


class BreedingSystem:
    def __init__(self)->None:self.pairings=0;self.hybrid_pairings=0;self.related_rejections=0;self.selection_log=[]

    @staticmethod
    def related(left,right)->bool:
        if left.entity_id==right.entity_id:return True
        if left.entity_id in right.parent_ids or right.entity_id in left.parent_ids:return True
        return bool(set(left.parent_ids)&set(right.parent_ids))

    @staticmethod
    def _hybrid_gate(left,right)->bool:
        affinity=min(left.genome.trait("graft_affinity"),right.genome.trait("graft_affinity"))
        if affinity<.78:return False
        digest=hashlib.sha256(f"{left.genome.semantic_sha256()}:{right.genome.semantic_sha256()}:hybrid".encode()).digest();value=int.from_bytes(digest[:8],"little")/2**64;return value<.012+.055*affinity

    def compatible(self,left,right)->bool:
        if not right.alive or right.stage!="mature" or right.reproduction_cooldown>0 or right.energy<=.62:return False
        if self.related(left,right):return False
        return left.family==right.family or self._hybrid_gate(left,right)

    def score(self,left,right)->float:
        systems=right.body.systems();symmetry=right.genome.developmental.traits[1];complement=float(np.mean(np.abs(np.asarray(left.genome.eco_traits)-np.asarray(right.genome.eco_traits))));generation=right.genome.developmental.generation
        return systems["integrity"]*.32+right.energy*.22+right.reserve*.12+symmetry*.18+min(.16,complement*.35)+min(.05,generation*.005)

    def choose(self,left,candidates):
        eligible=[]
        for right in candidates:
            if self.related(left,right):self.related_rejections+=1;continue
            if self.compatible(left,right):eligible.append(right)
        return max(eligible,key=lambda right:(self.score(left,right),-right.entity_id),default=None)

    def record(self,left,right,tick:int)->None:
        self.pairings+=1;hybrid=left.family!=right.family;self.hybrid_pairings+=int(hybrid);self.selection_log.append((int(tick),left.entity_id,right.entity_id,round(self.score(left,right),6),hybrid))
        if len(self.selection_log)>512:self.selection_log=self.selection_log[-512:]

    def payload(self)->dict:return {"pairings":self.pairings,"hybrid_pairings":self.hybrid_pairings,"related_rejections":self.related_rejections,"selection_log":self.selection_log}

    def restore(self,payload:dict)->None:
        self.pairings=int(payload.get("pairings",0));self.hybrid_pairings=int(payload.get("hybrid_pairings",0));self.related_rejections=int(payload.get("related_rejections",0));self.selection_log=[tuple(item) for item in payload.get("selection_log",[])]
