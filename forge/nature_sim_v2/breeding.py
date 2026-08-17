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

    @staticmethod
    def offspring_potential(left,right)->float:
        """Estimate viable descendant potential from heritable, costly traits.

        This does not alter either genome.  It only gives mate/reproduction
        behavior a noisy ecological preference; actual fitness still comes
        from survival, nourishment, reproduction, and descendant survival.
        """
        left_systems=left.body.systems();right_systems=right.body.systems()
        physiology=min(
            left_systems["integrity"],right_systems["integrity"],
            left_systems["circulation"],right_systems["circulation"],
            left_systems["digestion"],right_systems["digestion"],
        )
        fertility=(left.genome.trait("fertility")+right.genome.trait("fertility"))*.5
        investment=(left.genome.trait("offspring_investment")+right.genome.trait("offspring_investment"))*.5
        efficiency=1-(left.genome.trait("move_cost")+right.genome.trait("move_cost"))*.5
        perception=(left.genome.trait("perception")+right.genome.trait("perception"))*.5
        provisioning=(left.energy+right.energy+left.reserve+right.reserve)*.25
        return float(np.clip(
            physiology*.30+fertility*.18+investment*.14+efficiency*.13+perception*.10+provisioning*.15,
            0,1,
        ))

    def reproduction_drive(self,entity,candidates,*,local_capacity:float=1.0)->float:
        mates=[item for item in candidates if self.compatible(entity,item)]
        if not mates:return 0.0
        potential=max(self.offspring_potential(entity,item) for item in mates)
        fertility=entity.genome.trait("fertility")
        investment=entity.genome.trait("offspring_investment")
        readiness=np.clip((entity.energy-.58)/.42,0,1)*np.clip((entity.reserve-.08)/.72,0,1)
        return float(np.clip(readiness*(.28*fertility+.22*investment+.50*potential)*local_capacity,0,1))

    def score(self,left,right)->float:
        systems=right.body.systems();symmetry=right.genome.developmental.traits[1];complement=float(np.mean(np.abs(np.asarray(left.genome.eco_traits)-np.asarray(right.genome.eco_traits))));generation=right.genome.developmental.generation
        return systems["integrity"]*.24+right.energy*.16+right.reserve*.10+symmetry*.14+min(.12,complement*.26)+min(.04,generation*.004)+self.offspring_potential(left,right)*.20

    def choose(self,left,candidates):
        eligible=[]
        for right in candidates:
            if self.related(left,right):self.related_rejections+=1;continue
            if self.compatible(left,right):eligible.append(right)
        return max(eligible,key=lambda right:(self.score(left,right),-right.entity_id),default=None)

    def record(self,left,right,tick:int)->None:
        self.pairings+=1;hybrid=left.family!=right.family;self.hybrid_pairings+=int(hybrid);self.selection_log.append((int(tick),left.entity_id,right.entity_id,round(self.score(left,right),6),hybrid,round(self.offspring_potential(left,right),6)))
        if len(self.selection_log)>512:self.selection_log=self.selection_log[-512:]

    def payload(self)->dict:return {"pairings":self.pairings,"hybrid_pairings":self.hybrid_pairings,"related_rejections":self.related_rejections,"selection_log":self.selection_log}

    def restore(self,payload:dict)->None:
        self.pairings=int(payload.get("pairings",0));self.hybrid_pairings=int(payload.get("hybrid_pairings",0));self.related_rejections=int(payload.get("related_rejections",0));self.selection_log=[tuple(item) for item in payload.get("selection_log",[])]
