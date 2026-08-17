from __future__ import annotations

import hashlib,json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT="nullvector-neural-foundation-build-1/1.0.0";DEFAULT_OUTPUT=PROJECT_ROOT/"outputs/neural_foundation_v1/build_001";MANIFEST_NAME="neural_foundation_manifest.json"
SOURCE_FILES=("forge/neural_foundation_v1/__init__.py","forge/neural_foundation_v1/__main__.py","forge/neural_foundation_v1/contract.py","forge/neural_foundation_v1/registry.py")
COMPONENTS=(
    ("morphology_diffusion","morphology","validated_candidate","outputs/multifield_production_v2/training_summary.json","checkpoints/multifield_production_v2/best.pt"),
    ("latent_fusion_evolution","evolution","ready","outputs/neural_fusion_production_evolution_v1_run2/production_evolution_manifest.json",None),
    ("continuous_cell_vae","rasterization","ready","outputs/organism_cell_vae_v1/production_v3_calibrated/evaluation_manifest.json","outputs/organism_cell_vae_v1/production_v3_calibrated/cell_vae_0001200.pt"),
    ("causal_cellular_physiology","physiology","ready","outputs/cellular_nca/nca_causal_v3_selected/selection_manifest.json","outputs/cellular_nca/nca_causal_v3_selected/runtime.pt"),
    ("living_body_promotion","physiology","ready","outputs/living_body_nca_v1/promotion_audit_v3_selected.json",None),
    ("grounded_target_field","locomotion","ready","outputs/creature_stage_neural_target_field_v1/production_6000_v11_tail/report.json","outputs/creature_stage_neural_target_field_v1/production_6000_v11_tail/runtime.pt"),
    ("limb_pose_controller","locomotion","ready","outputs/creature_stage_neural_limb_pose_v1/production_2400_catalog/report.json","outputs/creature_stage_neural_limb_pose_v1/production_2400_catalog/runtime.pt"),
    ("grounded_feedback_controller","locomotion","ready","outputs/creature_stage_neural_grounded_feedback_v2/production_3000_v2/report.json","outputs/creature_stage_neural_grounded_feedback_v2/production_3000_v2/runtime.pt"),
    ("locomotion_25d","locomotion","ready","outputs/creature_stage_neural_locomotion_25d/evaluation_1200_runtime.json","outputs/creature_stage_neural_locomotion_25d/controller_1200_runtime.pt"),
    ("physical_grasper_feeder","manipulation","ready","outputs/creature_stage_neural_grasper_v1/production_v3_physical_feeder/report.json","outputs/creature_stage_neural_grasper_v1/production_v3_physical_feeder/runtime.pt"),
    ("behavior_controller","behavior","validated_candidate","outputs/nature_behavior_nn/controller_v4_fastbody.json","outputs/nature_behavior_nn/controller_v4_fastbody.pt"),
    ("macro_patch_dynamics","ecology","ready","outputs/nature_macro_nn/production_v2_calibrated/report.json","outputs/nature_macro_nn/production_v2_calibrated/runtime.pt"),
    ("colony_controller","colony","validated_candidate","outputs/nature_colony_nn/production_v4_material_economy/report.json","outputs/nature_colony_nn/production_v4_material_economy/checkpoint.pt"),
    ("society_controller","society","validated_candidate","outputs/nature_society_nn/production_v5_final/report.json","outputs/nature_society_nn/production_v5_final/runtime.pt"),
    ("world_timeline","ecology","validated_candidate","outputs/nature_timeline_nn/production_v1_b96/report.json","outputs/nature_timeline_nn/production_v1_b96/checkpoint.pt"),
    ("ecology_counterfactual","ecology","validated_candidate","outputs/nature_counterfactual_nn/production_v2_25m/report.json","outputs/nature_counterfactual_nn/production_v2_25m/checkpoint.pt"),
    ("map_decorator","maps","ready","outputs/map_decorator_production_v4_selection/protected_selection_audit_v1/selection_audit.json",None),
    ("neural_map_topology","maps","experimental","outputs/map_topology_neural_smoke_v2/smoke_manifest.json","outputs/map_topology_neural_production/calibration_500step_v2_hardened/checkpoint_final.pt"),
    ("world_frame_vae","frame_codec","validated_candidate","outputs/world_frame_vae/production_v2_high_fidelity/report.json","outputs/world_frame_vae/production_v2_high_fidelity/checkpoint.pt"),
    ("world_pixel_refiner","frame_codec","validated_candidate","outputs/world_frame_vae_refiner/production_v1/report.json","outputs/world_frame_vae_refiner/production_v1/checkpoint.pt"),
    ("world_latent_action_dit","world_transition","validated_candidate","outputs/world_latent_dit/production_v2_residual/report.json","outputs/world_latent_dit/production_v2_residual/checkpoint.pt"),
    ("cellular_temporal_action","world_transition","experimental","outputs/world_action_cellular_v7/production_v4_dual_gate/report.json","outputs/world_action_cellular_v7/production_v4_dual_gate/evaluated.pt"),
)
REQUIRED_DOMAINS=("morphology","evolution","rasterization","physiology","locomotion","manipulation","behavior","ecology","colony","society","maps","frame_codec","world_transition")
def canonical(value:object)->bytes:return(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()
def source_sha256()->str:
    digest=hashlib.sha256(b"nullvector-neural-foundation-v1\0")
    for relative in SOURCE_FILES:digest.update(relative.encode()+b"\0"+(PROJECT_ROOT/relative).read_bytes()+b"\0")
    return digest.hexdigest()
