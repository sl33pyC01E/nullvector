from __future__ import annotations

import numpy as np
import torch

from ..nature_sim_v2.contract import INTENTS
from .features import extract_observation
from .training import load_model


class NeuralBehaviorRuntime:
    """Batched neural intent and steering authority for a living world."""
    def __init__(self,model,*,device="cuda",decision_interval:int=3):self.model=model;self.device=torch.device(device);self.decision_interval=decision_interval;self.cache={};self.last_tick=-1

    @classmethod
    def from_checkpoint(cls,path,*,device="cuda",decision_interval=3):model,_=load_model(path,device=device);return cls(model,device=device,decision_interval=decision_interval)

    @torch.inference_mode()
    def prepare(self,world)->None:
        if self.last_tick>=0 and world.tick_index%self.decision_interval:return
        entities=[item for item in sorted(world.organisms.values(),key=lambda item:item.entity_id) if item.alive and not item.body.incapacitated]
        if not entities:return
        system_cache={item.entity_id:item.body.systems() for item in world.organisms.values() if item.alive};observations=[extract_observation(world,item,system_cache) for item in entities];tensor=lambda values:torch.from_numpy(np.stack(values)).to(self.device)
        result=self.model(tensor([v[0] for v in observations]),tensor([v[1] for v in observations]),tensor([v[2] for v in observations]),tensor([v[3] for v in observations]));intent=result.intent_logits.argmax(-1).cpu().numpy();direction=result.direction.float().cpu().numpy();urgency=torch.sigmoid(result.urgency).float().cpu().numpy()
        for index,entity in enumerate(entities):self.cache[entity.entity_id]=(INTENTS[int(intent[index])],direction[index]*urgency[index])
        self.last_tick=world.tick_index

    def choose(self,entity):return self.cache.get(entity.entity_id)

    def forget(self,entity_id:int)->None:self.cache.pop(entity_id,None)
