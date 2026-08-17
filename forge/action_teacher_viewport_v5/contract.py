from __future__ import annotations

import hashlib
import json

from ..action_teacher_v1.contract import ACTIONS, COUNTERFACTUAL_SHAPE, FRAME_SIZE, STATE_FEATURES
from ..action_teacher_v2.contract import ACTOR_FEATURES, ACTOR_FIELD_SHAPE
from ..config import PROJECT_ROOT

FORMAT = "nullvector-whole-viewport-teacher/5.0.0"
DEFAULT_ROOT = PROJECT_ROOT / "outputs/action_teacher_viewport_v5/pilot_v1"
SPATIAL_NAMES = (
    "water", "light", "mineral", "charge", "phase", "oxygen", "heat", "toxin", "flora", "biomass",
    "mat_empty", "mat_soil", "mat_rock", "mat_water", "mat_blood", "mat_sap", "mat_oil", "mat_metal",
    "mat_biomass", "mat_acid", "mat_fire", "mat_smoke", "mat_crystal", "mass", "temperature", "damage",
    "structure", "family_humanoid", "family_animalian", "family_plantlike", "family_anomaly", "family_machine",
    "organism_health", "organism_fluid", "organism_scar", "organism_neural", "organism_vital", "corpse",
    "selected", "velocity_x", "velocity_y", "projectile", "projectile_velocity_x", "projectile_velocity_y",
    "food_clump", "food_height", "settlement_road", "settlement_center",
    "terrain_void", "terrain_floor", "terrain_wall", "terrain_water", "terrain_bridge", "terrain_growth",
    "terrain_crystal", "terrain_chasm", "terrain_sand", "hazard_none", "hazard_laser", "hazard_lava",
    "hazard_spores", "hazard_arc", "elevation", "nav_cost", "zone", "protected_backbone",
    "required_clearance", "decoration_forbidden",
)
SPATIAL_SHAPE = (len(SPATIAL_NAMES), 32, 32)
ORGANISM_LIMIT = 64
ORGANISM_FEATURES = 164
ORGANISM_SHAPE = (ORGANISM_LIMIT, ORGANISM_FEATURES)
ARRAY_NAMES = (
    "frame", "spatial", "organisms", "organism_mask", "state", "actor_state", "actor_field",
    "visibility", "memory", "control", "action", "selected", "timeline_event", "timeline",
    "counterfactual", "tick", "episode_step",
)
SOURCE_FILES = (
    "forge/action_teacher_viewport_v5/__init__.py",
    "forge/action_teacher_viewport_v5/__main__.py",
    "forge/action_teacher_viewport_v5/contract.py",
    "forge/action_teacher_viewport_v5/state.py",
    "forge/action_teacher_viewport_v5/recorder.py",
    "forge/action_teacher_viewport_v5/curriculum.py",
    "forge/action_teacher_viewport_v5/aesthetic.py",
    "forge/action_teacher_viewport_v5/build_corpus.py",
    "forge/nature_sim_v2/demo.py",
    "forge/nature_sim_v2/world.py",
    "forge/nature_neural_runtime_v2/coordinator.py",
    "forge/nature_neural_feeding_v1/system.py",
    "forge/living_body_nca_v1/adapter.py",
    "forge/maps/model.py",
    "forge/maps/generator.py",
    "forge/maps/validate.py",
    "forge/creature_stage_neural_grasper_v1/feeding.py",
)

def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-whole-viewport-teacher-v5\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
