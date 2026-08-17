from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_grounded_feedback_v2.dataset import encode_live
from .dataset import encode_target_context
from .contract import CHECKPOINT_FORMAT,ModelConfig,source_sha256
from .model import NeuralGroundedTargetField

class NeuralTargetFieldRuntime:
    def __init__(self,model,device): self.model=model.eval();self.device=device
    @classmethod
    def from_checkpoint(cls,path:Path,*,device="cuda"):
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu")
        p=torch.load(path,map_location="cpu",weights_only=True)
        if p.get("format")!=CHECKPOINT_FORMAT or p.get("source_sha256")!=source_sha256(): raise ValueError("target field provenance drifted")
        if _state_sha256(p["model_state"])!=p.get("model_state_sha256") or not p["report"]["gates"]["all_passed"]: raise ValueError("target field state failed")
        m=NeuralGroundedTargetField(ModelConfig(**p["model_config"]));m.load_state_dict(p["model_state"]);return cls(m.to(target),target)
    @torch.inference_mode()
    def predict(self,organism,nodes_local,node_velocity,previous_contact,phase,body_velocity):
        values=encode_live(organism,nodes_local,node_velocity,previous_contact,phase,body_velocity)
        context=encode_target_context(organism,nodes_local,node_velocity,phase)
        out=self.model(*(torch.from_numpy(v[None]).to(self.device) for v in values),torch.from_numpy(context[None]).to(self.device));a=len(organism.genome.appendages);m=len(organism.muscles)
        return (out.feedback.muscle_activation[0,:m].float().cpu().numpy(),
            (torch.sigmoid(out.feedback.contact_logits[0,:a])>=.5).cpu().numpy(),
            out.terminal_target[0,:a].float().cpu().numpy()*24)
