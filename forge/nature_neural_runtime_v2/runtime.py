from __future__ import annotations

from dataclasses import dataclass

import torch

from ..creature_stage_neural_locomotion_25d.runtime import NeuralLocomotionRuntime
from ..living_body_nca_v1 import LivingBodyNCARuntime
from ..nature_colony_nn import NeuralColonyRuntime
from ..nature_counterfactual_nn import NeuralCounterfactualRuntime
from ..nature_society_nn import NeuralSocietyRuntime
from ..nature_timeline_nn import NeuralTimelineRuntime
from ..organism_cell_vae_runtime_v1 import ContinuousCellVAERuntime
from ..playable_neural_runtime_v1.runtime import _component_table, _load_behavior


@dataclass(slots=True)
class NatureNeuralRuntime:
    """Only the promoted specialists used by the live nature simulation.

    World-frame prediction is deliberately absent here.  The demo owns one
    recurrent V6 + rollout-decoder V3 pipeline, avoiding the former duplicate
    Action-DiT, VAE, refiner, and actor-state allocation.
    """

    locomotion: NeuralLocomotionRuntime
    behavior: object
    colony: NeuralColonyRuntime
    society: NeuralSocietyRuntime
    timeline: NeuralTimelineRuntime
    counterfactual: NeuralCounterfactualRuntime
    organism: ContinuousCellVAERuntime
    physiology: LivingBodyNCARuntime
    component_count: int

    @classmethod
    def from_release(cls, *, device: str = "cuda") -> "NatureNeuralRuntime":
        target = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        rows = _component_table()
        artifact = lambda name: rows[name]["artifact"]["path"]
        from ..config import PROJECT_ROOT

        return cls(
            locomotion=NeuralLocomotionRuntime.from_checkpoint(PROJECT_ROOT / artifact("locomotion_25d"), device=target),
            behavior=_load_behavior(PROJECT_ROOT / artifact("behavior"), target),
            colony=NeuralColonyRuntime.from_checkpoint(PROJECT_ROOT / artifact("colony"), device=target),
            society=NeuralSocietyRuntime.from_checkpoint(PROJECT_ROOT / artifact("society"), device=target),
            timeline=NeuralTimelineRuntime.from_checkpoint(PROJECT_ROOT / artifact("timeline"), device=target),
            counterfactual=NeuralCounterfactualRuntime.from_checkpoint(PROJECT_ROOT / artifact("counterfactual"), device=target),
            organism=ContinuousCellVAERuntime.from_release(device=target),
            physiology=LivingBodyNCARuntime.from_output(device=target),
            component_count=8,
        )
