from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from forge.nature_sim_v2 import ColonyState,NatureWorld,founder_genomes
from forge.nature_society_nn.contract import FEATURES,LABOR_SECTORS,ModelConfig
from forge.nature_society_nn.corpus import build_corpus,teacher
from forge.nature_society_nn.model import SocietyStrategist
from forge.nature_society_nn.runtime import NeuralSocietyRuntime,extract_features
from forge.qud_society_v1 import SocietyLayer


def test_society_corpus_is_deterministic_and_all_heads_are_diverse() -> None:
    left=build_corpus(samples=4096,seed=7);right=build_corpus(samples=4096,seed=7);assert left["semantic_sha256"]==right["semantic_sha256"] and np.array_equal(left["features"],right["features"])
    assert len(np.unique(left["activity"]))>=10 and len(np.unique(left["diplomacy"]))==3 and len(np.unique(left["project"]))>=7
    assert np.allclose(left["labor"].sum(1),1)


def test_society_model_shapes_and_gradient_flow() -> None:
    model=SocietyStrategist(ModelConfig(width=96,depth=2,dropout=0));features=torch.randn(11,FEATURES);outputs=model(features);assert outputs[0].shape==(11,16) and outputs[1].shape==(11,len(LABOR_SECTORS)) and outputs[2].shape==(11,3) and outputs[3].shape==(11,9);sum(value.square().mean() for value in outputs).backward();assert all(parameter.grad is not None for parameter in model.parameters())


def test_live_feature_projection_is_finite_and_teacher_legal() -> None:
    world=NatureWorld(seed=44,size=40);genome=founder_genomes(variants_per_family=1)[0];ids=[world.add_organism(genome,(12+i*.2,12),energy=.8) for i in range(4)];world.colonies[1]=ColonyState(1,0,genome.lineage_id,set(ids),np.asarray((12.,12.)));[setattr(world.organisms[i],"colony_id",1) for i in ids];society=SocietyLayer(world,seed=55);faction_id=society.found_from_colony(1);faction=society.factions[faction_id];settlement=society.settlements[next(iter(faction.settlement_ids))];features=extract_features(faction,settlement,world)
    assert features.shape==(FEATURES,) and np.isfinite(features).all();activity,labor,diplomacy,project=teacher(features[None]);assert 0<=activity[0]<16 and np.isclose(labor[0].sum(),1) and 0<=diplomacy[0]<3 and 0<=project[0]<9


def test_quality_gated_production_checkpoint_drives_a_live_settlement() -> None:
    checkpoint=Path(__file__).parents[1]/"game/generated/models/nature_society/strategist_v1.pt";runtime=NeuralSocietyRuntime.from_checkpoint(checkpoint,device="cpu");world=NatureWorld(seed=144,size=40);world.biome="fungal_garden";genome=founder_genomes(variants_per_family=1)[0];ids=[world.add_organism(genome,(14+i*.2,14),energy=.8) for i in range(4)];world.colonies[1]=ColonyState(1,0,genome.lineage_id,set(ids),np.asarray((14.,14.)));[setattr(world.organisms[i],"colony_id",1) for i in ids];society=SocietyLayer(world,seed=155,policy=runtime);society.found_from_colony(1);society.step_history(1);decision=next(iter(society.strategies.values()))
    assert decision.activity in tuple(__import__("forge.qud_society_v1.contract",fromlist=["ACTIVITIES"]).ACTIVITIES)
    assert np.isclose(sum(decision.labor),1,atol=1e-5) and decision.project
