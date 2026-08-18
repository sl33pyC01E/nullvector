from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from forge.android_foundation_v1 import build, validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_android_foundation_exports_live_batched_controller(tmp_path):
    output = tmp_path / "foundation"
    report = build(output)
    assert validate(output)["semantic_sha256"] == report["semantic_sha256"]
    assert report["scope"]["frame_loop"] is False
    anatomy = json.loads((output / "foundation_anatomy.json").read_bytes())
    assert [row["family"] for row in anatomy["organisms"]] == anatomy["families"]
    assert all(row["cells"] and row["skeleton"]["muscles"] and row["appendages"] for row in anatomy["organisms"])
    assert all(len(cell["neural_style"]) == 7 for row in anatomy["organisms"] for cell in row["cells"])
    assert all(len(cell["nca_xy"]) == 2 for row in anatomy["organisms"] for cell in row["cells"])
    static = np.fromfile(output / "foundation_cell_static.f32", dtype="<f4").reshape(5, 85, 48, 48)
    state = np.fromfile(output / "foundation_cell_state.f32", dtype="<f4").reshape(5, 12, 48, 48)
    bonds = np.fromfile(output / "foundation_cell_bonds.f32", dtype="<f4").reshape(5, 8, 48, 48)
    assert np.all(static[:, 0].sum(axis=(1, 2)) == [len(row["cells"]) for row in anatomy["organisms"]])
    assert np.all(state[:, 0][static[:, 0] > 0] == 1)
    assert np.all((bonds == 0) | (bonds == 1))
    session = ort.InferenceSession(str(output / "grounded_feedback_fp32.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[0] == "batch"
    assert [value.name for value in session.get_outputs()] == ["muscle_activation", "contact_logits", "body_velocity"]
    grasper = ort.InferenceSession(str(output / "neural_grasper_fp32.onnx"), providers=["CPUExecutionProvider"])
    assert grasper.get_inputs()[0].shape[0] == "batch"
    assert [value.name for value in grasper.get_outputs()] == [
        "appendage_logits", "engage_logit", "reach", "force", "type_logits", "brace", "release_logit", "throw_impulse",
    ]


def test_android_foundation_supplies_live_perception_and_clean_view():
    source = (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/java/world/nullvector/mobile/NeuralWorldView.java").read_text(encoding="utf-8")
    assert "updatePerception(visibility,memory)" in source
    assert "visibleOffset" in source and "exploredWorld" in source
    assert "!creature.selected&&!isWorldVisible" in source
    assert "hudVisible" in source and "sightOverlay" in source and "labelsVisible" in source and "barsVisible" in source


def test_android_foundation_has_setup_dual_sticks_and_physical_actions():
    view = (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/java/world/nullvector/mobile/NeuralWorldView.java").read_text(encoding="utf-8")
    world = (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/java/world/nullvector/mobile/FoundationWorld.java").read_text(encoding="utf-8")
    for token in ("ENTER LIVING WORLD", "SURVIVAL", "CREATIVE / OBSERVER", "LEFT STICK MOVE · RIGHT STICK LOOK / AIM"):
        assert token in view
    assert "movementPointer" in view and "aimPointer" in view
    assert "THROW LOCKED · GRASP SOMETHING FIRST" in view
    assert "ENTITY GRIP CLOSED" in view and "TISSUE GRIP CLOSED" in view
    assert "AIMED BALLISTIC RELEASE" in view
    assert "throwCreature" in world and "carried" in world
    assert "groundSy-creature.z" in view


def test_android_projectiles_are_trait_gated_and_distinct_from_throwing():
    view = (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/java/world/nullvector/mobile/NeuralWorldView.java").read_text(encoding="utf-8")
    world = (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/java/world/nullvector/mobile/FoundationWorld.java").read_text(encoding="utf-8")
    assert '"GRASP","FEED","STRIKE","SCRAPE","CUT","THROW","FIRE"' in view
    assert "selectedProjectileAbility" in world and "consumeSelectedProjectileCost" in world
    assert "c.family==4" in world and "c.family==3" in world and "c.traits[13]>.72f" in world
    assert "MACHINE KINETIC FIRED" in view and "ANOMALY PHASE BOLT FIRED" in view and "GRAFTED EMITTER FIRED" in view
    assert "heldMaterial!=null" in view


def test_android_world_runs_the_coupled_teacher_ensemble():
    assets = PROJECT_ROOT / "android/nullvector-mobile/app/src/fp32/assets"
    manifest = json.loads((assets / "coupled_ensemble_manifest.json").read_bytes())
    assert manifest["status"] == "android_coupled_ensemble_export_ready"
    assert set(manifest["models"]) == {"macro", "colony", "society", "timeline", "counterfactual"}
    assert all((assets / row["path"]).stat().st_size == row["bytes"] for row in manifest["models"].values())
    world = (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/java/world/nullvector/mobile/FoundationWorld.java").read_text(encoding="utf-8")
    view = (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/java/world/nullvector/mobile/NeuralWorldView.java").read_text(encoding="utf-8")
    for token in ("encodeMacro()", "encodeColony()", "encodeSociety()", "timelineFeatures()", "encodeWorldContext()", "encodeSelectedVae()"):
        assert token in world
    for stem in ("macro", "colony", "society", "timeline", "counterfactual"):
        assert f'ensembleModel("{stem}")' in view
    assert "organismVaeFrame=rgbaBitmap(rgba)" in view
    assert "applySelectedVae(rgba)" not in view
    assert "STATE_ALIGNED_VIEWPORT_VAE_READY = true" in view
    assert "LIVE STATE → ACTION MODEL → VAE" in view
    assert "encodeViewportAction" in world
    assert 'assetFile("viewport_encoder_fp32.onnx")' in view
    renderer = (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/java/world/nullvector/mobile/MobileViewportGpuRenderer.java").read_text(encoding="utf-8")
    assert 'MobileViewportGpuRenderer.create(getContext(), "viewport_action_v5_fp16.tflite", "frame_vae_mobile_v1_fp16.tflite")' in view
    assert "LITERT GPU FP16 WEIGHTS · ACTION V5 + MOBILE VAE" in renderer
    assert "action.runForMultipleInputsOutputs" in renderer
    assert "decoder.runForMultipleInputsOutputs" in renderer
    assert "viewportLatent=frame.latent" in view
    assert "viewportLatent==null&&viewportSourceFrame" in view
    for artifact in ("viewport_action_v5_fp16.tflite", "frame_vae_mobile_v1_fp16.tflite"):
        assert (PROJECT_ROOT / "android/nullvector-mobile/app/src/main/assets" / artifact).stat().st_size > 1_000_000
    assert "drawLocalizedLeak" in view and "(.62f-cell.health)/.62f" in view
    assert "COUPLED_ENSEMBLE_OK" in view
    compact = PROJECT_ROOT / "android/nullvector-mobile/app/src/int8/assets"
    compact_manifest = json.loads((compact / "coupled_ensemble_int8_manifest.json").read_bytes())
    assert compact_manifest["status"] == "android_int8_ensemble_ready"
    assert compact_manifest["total_mib"] < 100
    assert all((compact / row["path"]).stat().st_size == row["bytes"] for row in compact_manifest["models"].values())
