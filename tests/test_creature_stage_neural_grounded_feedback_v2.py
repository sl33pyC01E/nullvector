from __future__ import annotations

import numpy as np
import torch

from forge.creature_stage_developmental.development import develop
from forge.creature_stage_developmental.genomes import review_genomes
from forge.creature_stage_developmental.motion import pose
from forge.creature_stage_grounded_locomotion.contract import GroundedLocomotionConfig
from forge.creature_stage_grounded_locomotion.physics import _contact_schedule, locomotor_modes, simulate_grounded_cycle
from forge.creature_stage_neural_grounded_feedback_v2.contract import (
    GLOBAL_FEATURES, MAX_APPENDAGES, MAX_MUSCLES, MUSCLE_FEATURES, OWNER_FEATURES, ModelConfig, source_sha256,
)
from forge.creature_stage_neural_grounded_feedback_v2.dataset import encode_live
from forge.creature_stage_neural_grounded_feedback_v2.model import NeuralGroundedFeedback
from forge.creature_stage_neural_grounded_feedback_v2.physics import simulate_feedback_cycle


def test_live_encoding_and_model_are_causal_and_differentiable() -> None:
    organism=develop(review_genomes()[0]); cycle=simulate_grounded_cycle(organism); frame=cycle.frames[0]
    encoded=encode_live(organism,frame.nodes_local,frame.node_velocity,frame.contact_active,0,frame.body_velocity_x)
    assert [value.shape for value in encoded]==[(MAX_APPENDAGES,OWNER_FEATURES),(GLOBAL_FEATURES,),
        (MAX_APPENDAGES,),(MAX_MUSCLES,MUSCLE_FEATURES),(MAX_MUSCLES,),(MAX_MUSCLES,)]
    model=NeuralGroundedFeedback(ModelConfig(width=128,depth=3,dropout=0))
    tensors=[torch.from_numpy(value[None]) for value in encoded]; output=model(*tensors)
    loss=output.muscle_activation.sum()+output.contact_logits.sum()+output.body_velocity.sum(); loss.backward()
    assert model.parameter_count>400_000 and all(parameter.grad is not None for parameter in model.parameters())


class _TeacherPolicy:
    def predict(self, organism, nodes_local, node_velocity, previous_contact, phase, body_velocity):
        config=GroundedLocomotionConfig()
        return pose(organism,phase).muscle_activation,_contact_schedule(organism,locomotor_modes(organism),phase,config),.1


def test_feedback_executes_through_grounded_physics() -> None:
    organism=develop(review_genomes()[0]); cycle=simulate_feedback_cycle(organism,_TeacherPolicy())
    assert len(cycle.frames)==72 and cycle.distance_px>.5
    assert cycle.maximum_contact_slip_px<.05 and cycle.maximum_edge_strain<.12
    assert cycle.vertical_axis_max_degrees<5


def test_source_contract_is_stable() -> None:
    assert len(source_sha256())==64

