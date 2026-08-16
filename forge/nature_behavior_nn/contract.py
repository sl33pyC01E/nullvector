from __future__ import annotations

from dataclasses import dataclass,asdict
import ast
import hashlib
from pathlib import Path

from ..config import PROJECT_ROOT

FORMAT="nullvector-nature-behavior-corpus/1.0.0"
CHECKPOINT_FORMAT="nullvector-nature-behavior-controller/1.0.0"
SELF_FEATURES=94
RESOURCE_FEATURES=4
NEIGHBOR_FEATURES=14
MAX_NEIGHBORS=12
SOURCE_FILES=(
    "forge/nature_behavior_nn/contract.py","forge/nature_behavior_nn/features.py",
    "forge/nature_behavior_nn/model.py","forge/nature_behavior_nn/corpus.py",
    "forge/nature_behavior_nn/training.py","forge/nature_behavior_nn/runtime.py",
    "forge/nature_sim_v2/contract.py","forge/nature_sim_v2/state.py",
    "forge/nature_sim_v2/genetics.py",
    "forge/living_body_substrate/state.py",
    "forge/creature_stage_developmental/contract.py",
    "forge/creature_stage_developmental/development.py",
)
CORPUS_SOURCE_FILES=(
    "forge/nature_behavior_nn/contract.py","forge/nature_behavior_nn/features.py","forge/nature_behavior_nn/corpus.py",
    "forge/nature_sim_v2/contract.py","forge/nature_sim_v2/state.py","forge/nature_sim_v2/genetics.py",
    "forge/living_body_substrate/state.py","forge/creature_stage_developmental/contract.py","forge/creature_stage_developmental/development.py",
)


@dataclass(frozen=True,slots=True)
class ModelConfig:
    width:int=256
    layers:int=4
    heads:int=8
    dropout:float=.08


@dataclass(frozen=True,slots=True)
class TrainingConfig:
    updates:int=1200
    batch_size:int=384
    learning_rate:float=2e-4
    weight_decay:float=1e-3
    ema_decay:float=.999
    seed:int=0x4E455552414C


def _source_hash(files:tuple[str,...],domain:bytes)->str:
    digest=hashlib.sha256(domain+b"\0")
    for relative in files:
        path=PROJECT_ROOT/relative
        digest.update(relative.encode()+b"\0"+path.read_bytes()+b"\0")
    # Bind only the world methods that define observations and teacher actions.
    # Material physics, rendering, projectiles, and buildings can evolve without
    # falsely invalidating an unchanged behavior model.
    tree=ast.parse((PROJECT_ROOT/"forge/nature_sim_v2/world.py").read_text("utf-8"));selected={"_cell","_delta","_neighbors","_local_gradient","_choose_intent","_can_harvest"}
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in selected:digest.update(node.name.encode()+b"\0"+ast.dump(node,include_attributes=False).encode()+b"\0")
    return digest.hexdigest()


def source_sha256()->str:return _source_hash(SOURCE_FILES,b"nullvector-nature-behavior-source-v1")


def corpus_source_sha256()->str:return _source_hash(CORPUS_SOURCE_FILES,b"nullvector-nature-behavior-corpus-source-v1")
