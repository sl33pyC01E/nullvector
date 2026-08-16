from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from forge.nature_sim_v2 import NatureWorld
from forge.nature_timeline_nn import EVENTS,FEATURES,SEQUENCE,ModelConfig,NeuralTimelineRuntime,TimelineTransformer,extract_world_features
from forge.nature_timeline_nn.corpus import build_corpus

def test_timeline_corpus_is_deterministic_causal_and_bounded():
    left=build_corpus(samples=32,seed=9);right=build_corpus(samples=32,seed=9);assert left["semantic_sha256"]==right["semantic_sha256"];assert left["sequence"].shape==(32,SEQUENCE,FEATURES);assert left["target"].shape==(32,FEATURES);assert (left["sequence"]>=0).all() and (left["sequence"]<=1).all()

def test_timeline_transformer_predicts_state_event_and_confidence():
    model=TimelineTransformer(ModelConfig(width=128,layers=2,heads=4));x=torch.rand(3,SEQUENCE,FEATURES);state,event,confidence=model(x);assert state.shape==(3,FEATURES) and event.shape==(3,10) and confidence.shape==(3,);(state.mean()+event.mean()+confidence.mean()).backward();assert any(parameter.grad is not None for parameter in model.parameters())

def test_real_world_feature_projection_matches_training_contract():
    world=NatureWorld(seed=7,size=32);world.seed_founders(variants_per_family=1);features=extract_world_features(world);assert features.shape==(FEATURES,) and features.dtype.name=="float32" and (features>=0).all() and (features<=1).all()

def test_compact_production_timeline_forecasts_a_live_world():
    checkpoint=Path(__file__).resolve().parents[1]/"game/generated/models/nature_timeline/timeline_v1.pt";runtime=NeuralTimelineRuntime.from_checkpoint(checkpoint,device="cpu");world=NatureWorld(seed=19,size=32);world.seed_founders(variants_per_family=1);forecast=runtime.observe(world);assert forecast.event in EVENTS;assert np.isfinite(forecast.confidence) and 0<=forecast.confidence<=1;assert len(forecast.state)==FEATURES
