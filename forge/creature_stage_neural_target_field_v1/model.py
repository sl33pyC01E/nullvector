from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn,Tensor
from ..creature_stage_neural_grounded_feedback_v2.contract import GLOBAL_FEATURES,MAX_APPENDAGES,OWNER_FEATURES,ModelConfig as FeedbackConfig
from ..creature_stage_neural_grounded_feedback_v2.model import NeuralGroundedFeedback,FeedbackOutput
from .contract import (
    ModelConfig,TARGET_FEATURES,TARGET_GLOBAL_FEATURES,TARGET_OWNER_FEATURES,
    TARGET_ROOT_BASIS,
)

@dataclass(slots=True)
class TargetFieldOutput:
    feedback:FeedbackOutput; terminal_target:Tensor

class NeuralGroundedTargetField(nn.Module):
    def __init__(self,config:ModelConfig=ModelConfig()):
        super().__init__();self.config=config
        self.feedback=NeuralGroundedFeedback(FeedbackConfig(config.width,6,config.dropout))
        self.register_buffer("root_basis_centers",torch.linspace(-.05,.25,TARGET_ROOT_BASIS))
        self.target_head=nn.Sequential(nn.Linear(TARGET_FEATURES+TARGET_OWNER_FEATURES+TARGET_GLOBAL_FEATURES+TARGET_ROOT_BASIS,config.width),nn.SiLU(),
            *sum(([nn.Linear(config.width,config.width),nn.SiLU(),nn.Dropout(config.dropout)] for _ in range(config.depth)),[]),
            nn.Linear(config.width,2),nn.Tanh())
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
    def forward(self,owner_state,global_state,owner_mask,muscle_meta,muscle_owner,muscle_mask,target_context):
        feedback=self.feedback(owner_state,global_state,owner_mask,muscle_meta,muscle_owner,muscle_mask)
        # Only immutable morphology, traits and a Fourier phase code drive the
        # target field. Live state still drives neural muscles and contacts,
        # but it cannot make the periodic target chase its own rollout error.
        static_owner=owner_state[:,:,:TARGET_OWNER_FEATURES].float()
        static_global=global_state[:,:TARGET_GLOBAL_FEATURES].float()
        context=static_global[:,None].expand(-1,MAX_APPENDAGES,-1)
        root_y=target_context[:,:,3:4].float()
        root_basis=torch.exp(-torch.square((root_y-self.root_basis_centers)*40.0))
        terminal=target_context[:,:,:2].float()+self.target_head(torch.cat((target_context.float(),static_owner,context,root_basis),-1))*.35
        terminal=terminal*owner_mask[:,:,None]
        return TargetFieldOutput(feedback,terminal)
