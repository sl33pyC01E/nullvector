from __future__ import annotations

from torch import nn

from ..actor_state_student_v1.model import ActorStateStudent
from ..recurrent_action_dit_v2.model import RecurrentActionDiT
from ..world_latent_dit.contract import ModelConfig


class RecurrentWorldStudent(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.action = RecurrentActionDiT(config)
        self.actor = ActorStateStudent()

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
