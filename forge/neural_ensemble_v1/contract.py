from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import PROJECT_ROOT


FORMAT = "nullvector-factorized-neural-ensemble-v1/1.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/neural_ensemble_v1/build_001"
SOURCE_FILES = (
    "forge/neural_ensemble_v1/__init__.py",
    "forge/neural_ensemble_v1/__main__.py",
    "forge/neural_ensemble_v1/contract.py",
    "forge/neural_ensemble_v1/release.py",
)

COMPONENTS = (
    ("cell_raster", "local_frame", 30.0, "outputs/organism_cell_vae_v1/production_v3_calibrated/evaluation_manifest.json", "outputs/organism_cell_vae_v1/production_v3_calibrated/cell_vae_0001200.pt"),
    ("cell_physiology", "local_state", 15.0, "outputs/cellular_nca/nca_causal_v3_selected/selection_manifest.json", "outputs/cellular_nca/nca_causal_v3_selected/runtime.pt"),
    ("locomotion_25d", "local_control", 30.0, "outputs/creature_stage_neural_locomotion_25d/evaluation_1200_runtime.json", "outputs/creature_stage_neural_locomotion_25d/controller_1200_runtime.pt"),
    ("grasper_feeder", "local_control", 30.0, "outputs/creature_stage_neural_grasper_v1/production_v3_physical_feeder/report.json", "outputs/creature_stage_neural_grasper_v1/production_v3_physical_feeder/runtime.pt"),
    ("behavior", "organism_intent", 10.0, "outputs/nature_behavior_nn/controller_v4_fastbody.json", "outputs/nature_behavior_nn/controller_v4_fastbody.pt"),
    ("macro_patch", "patch_ecology", 1.0, "outputs/nature_macro_nn/production_v2_calibrated/report.json", "outputs/nature_macro_nn/production_v2_calibrated/runtime.pt"),
    ("colony", "colony_policy", 0.25, "outputs/nature_colony_nn/production_v4_material_economy/report.json", "outputs/nature_colony_nn/production_v4_material_economy/checkpoint.pt"),
    ("society", "society_policy", 0.05, "outputs/nature_society_nn/production_v5_final/report.json", "outputs/nature_society_nn/production_v5_final/runtime.pt"),
    ("timeline", "world_forecast", 0.02, "outputs/nature_timeline_nn/production_v1_b96/report.json", "outputs/nature_timeline_nn/production_v1_b96/checkpoint.pt"),
    ("counterfactual", "world_planning", 0.02, "outputs/nature_counterfactual_nn/production_v2_25m/report.json", "outputs/nature_counterfactual_nn/production_v2_25m/checkpoint.pt"),
    ("world_frame_vae", "distillation_codec", 30.0, "outputs/world_frame_vae/production_v2_high_fidelity/report.json", "outputs/world_frame_vae/production_v2_high_fidelity/checkpoint.pt"),
    ("world_pixel_refiner", "distillation_codec", 30.0, "outputs/world_frame_vae_refiner/production_v1/report.json", "outputs/world_frame_vae_refiner/production_v1/checkpoint.pt"),
    ("world_latent_dit", "distillation_transition", 30.0, "outputs/world_latent_dit/production_v2_residual/report.json", "outputs/world_latent_dit/production_v2_residual/checkpoint.pt"),
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-factorized-neural-ensemble-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()
