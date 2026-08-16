from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from forge.nature_counterfactual_nn import ACTIONS,FEATURES,SEQUENCE,CounterfactualTransformer,ModelConfig,NeuralCounterfactualRuntime
from forge.nature_counterfactual_nn.corpus import build_corpus

def test_counterfactual_corpus_is_grouped_deterministic_and_action_complete():
    left=build_corpus(groups=12,seed=7);right=build_corpus(groups=12,seed=7);assert left["semantic_sha256"]==right["semantic_sha256"];assert left["sequence"].shape==(60,SEQUENCE,FEATURES);assert (left["action"].reshape(-1,5)==range(5)).all();assert (left["target"]>=0).all() and (left["target"]<=1).all();assert (left["score"]>=0).all() and (left["score"]<=1).all()

def test_counterfactual_transformer_is_differentiable_and_action_conditioned():
    model=CounterfactualTransformer(ModelConfig(width=128,layers=2,heads=4));sequence=torch.rand(5,SEQUENCE,FEATURES);action=torch.arange(len(ACTIONS));state,benefit,risk=model(sequence,action);assert state.shape==(5,FEATURES) and benefit.shape==(5,) and risk.shape==(5,);(state.mean()+benefit.mean()+risk.mean()).backward();assert model.action.weight.grad is not None

def test_compact_production_counterfactual_ranks_all_interventions():
    checkpoint=Path(__file__).resolve().parents[1]/"game/generated/models/nature_counterfactual/counterfactual_v1.pt";runtime=NeuralCounterfactualRuntime.from_checkpoint(checkpoint,device="cpu");sequence=build_corpus(groups=1,seed=13)["sequence"][0];result=runtime.evaluate([row for row in sequence]);assert tuple(item.action for item in result)==ACTIONS;assert all(np.isfinite(item.benefit) and 0<=item.benefit<=1 and np.isfinite(item.risk) and 0<=item.risk<=1 for item in result);assert runtime.report["heldout_top_action_accuracy"]>.9
