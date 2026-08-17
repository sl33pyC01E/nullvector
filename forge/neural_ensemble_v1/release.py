from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import COMPONENTS, DEFAULT_OUTPUT, FORMAT, canonical, sha256_file, source_sha256


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _quality(name: str, report: dict) -> tuple[bool, dict]:
    if name == "cell_raster":
        metrics = report["evaluation"]["metrics"]
        gates = {"alpha_iou": metrics["alpha_iou"] >= .88, "rgba_mae": metrics["rgba_mae"] <= .018}
    elif name == "cell_physiology":
        gates = dict(report["gates"])
    elif name in {"locomotion_25d", "grasper_feeder", "macro_patch"}:
        gates = dict(report["gates"])
    elif name == "behavior":
        selected = report["validation"][report["validation"]["selected"]]
        gates = {"intent_accuracy": selected["intent_accuracy"] >= .97, "direction_cosine": selected["direction_cosine"] >= .70, "forage_accuracy": selected["per_intent"]["forage"]["accuracy"] >= .90}
    elif name == "colony":
        gates = {"role_accuracy": report["heldout_role_accuracy"] >= .98, "action_mae": report["heldout_action_mae"] <= .05}
    elif name == "society":
        gates = {"activity": report["heldout_activity_accuracy"] >= .85, "diplomacy": report["heldout_diplomacy_accuracy"] >= .95, "project": report["heldout_project_accuracy"] >= .95, "labor": report["heldout_labor_mae"] <= .02}
    elif name == "timeline":
        gates = {"event": report["heldout_event_accuracy"] >= .85, "state": report["heldout_state_mae"] <= .10}
    elif name == "counterfactual":
        gates = {"action": report["heldout_top_action_accuracy"] >= .90, "state": report["heldout_state_mae"] <= .05, "benefit": report["heldout_benefit_mae"] <= .05, "risk": report["heldout_risk_mae"] <= .03}
    elif name == "world_frame_vae":
        gates = {"psnr": report["heldout_psnr_db"] >= 30, "mae": report["heldout_mae"] <= .02}
    elif name == "world_pixel_refiner":
        gates = {"psnr": report["heldout_refined_psnr_db"] >= 31, "mae_improvement": report["mae_improvement"] >= .25, "edge_improvement": report["edge_improvement"] > 0}
    elif name == "world_latent_dit":
        gates = {"latent_improvement": report["latent_improvement"] > 0, "rgb_improvement": report["rgb_improvement"] > 0, "horizon": int(report["horizon"]) >= 4}
    else:
        raise KeyError(name)
    return all(value is True for value in gates.values()), gates


def _load_probe(name: str, artifact: Path):
    if name == "cell_raster":
        from ..organism_cell_vae_runtime_v1 import ContinuousCellVAERuntime
        return ContinuousCellVAERuntime.from_release(device="cpu")
    if name == "cell_physiology":
        from ..living_body_nca_v1 import LivingBodyNCARuntime
        return LivingBodyNCARuntime.from_output(device="cpu")
    if name == "locomotion_25d":
        from ..creature_stage_neural_locomotion_25d.runtime import NeuralLocomotionRuntime
        return NeuralLocomotionRuntime.from_checkpoint(artifact, device="cpu")
    if name == "grasper_feeder":
        from ..creature_stage_neural_grasper_v1.runtime import NeuralGrasperRuntime
        return NeuralGrasperRuntime.from_checkpoint(artifact, device="cpu")
    if name == "behavior":
        import torch
        from ..nature_behavior_nn.contract import CHECKPOINT_FORMAT, ModelConfig
        from ..nature_behavior_nn.model import NeuralNatureBehavior
        from ..nature_behavior_nn.runtime import NeuralBehaviorRuntime
        from ..nature_behavior_nn.training import _state_hash
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        report = json.loads(artifact.with_suffix(".json").read_text("utf-8"))
        selected = payload.get("selected")
        state = payload.get(selected, {})
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != report.get("source_sha256") or report.get("checkpoint_sha256") != sha256_file(artifact):
            raise ValueError("pinned behavior release provenance drifted")
        if _state_hash(state) != payload.get(f"{selected}_state_sha256"):
            raise ValueError("pinned behavior release state drifted")
        model = NeuralNatureBehavior(ModelConfig(**payload["model_config"])); model.load_state_dict(state, strict=True)
        return NeuralBehaviorRuntime(model.eval(), device="cpu", decision_interval=3)
    if name == "macro_patch":
        from ..nature_macro_nn.runtime import NeuralMacroPatchRuntime
        return NeuralMacroPatchRuntime.from_checkpoint(artifact, device="cpu")
    if name == "colony":
        from ..nature_colony_nn.runtime import NeuralColonyRuntime
        return NeuralColonyRuntime.from_checkpoint(artifact, device="cpu")
    if name == "society":
        from ..nature_society_nn.runtime import NeuralSocietyRuntime
        return NeuralSocietyRuntime.from_checkpoint(artifact, device="cpu")
    if name == "timeline":
        from ..nature_timeline_nn.runtime import NeuralTimelineRuntime
        return NeuralTimelineRuntime.from_checkpoint(artifact, device="cpu")
    if name == "counterfactual":
        from ..nature_counterfactual_nn.runtime import NeuralCounterfactualRuntime
        return NeuralCounterfactualRuntime.from_checkpoint(artifact, device="cpu")
    if name == "world_frame_vae":
        from ..world_frame_vae.runtime import WorldFrameVAERuntime
        return WorldFrameVAERuntime.from_checkpoint(artifact, device="cpu")
    return None


def build(output: Path = DEFAULT_OUTPUT, *, probe_loaders: bool = True) -> dict:
    output = Path(output).resolve()
    destination = output / "ensemble_manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=8 * 1024**2)
    rows = []
    for name, scale, rate, report_relative, artifact_relative in COMPONENTS:
        report_path, artifact_path = PROJECT_ROOT / report_relative, PROJECT_ROOT / artifact_relative
        report = json.loads(report_path.read_text("utf-8"))
        passed, gates = _quality(name, report)
        if not passed:
            raise ValueError(f"ensemble component failed promotion: {name}")
        loader_passed = None
        if probe_loaders and name not in {"world_pixel_refiner", "world_latent_dit"}:
            runtime = _load_probe(name, artifact_path)
            loader_passed = runtime is not None
            del runtime
            gc.collect()
        rows.append({
            "name": name, "scale": scale, "rate_hz": rate, "quality_gates": gates,
            "report": {"path": report_relative, "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
            "artifact": {"path": artifact_relative, "bytes": artifact_path.stat().st_size, "sha256": sha256_file(artifact_path)},
            "loader_probe_passed": loader_passed,
        })
    payload = {
        "format": FORMAT, "status": "teacher_ensemble_ready", "source_sha256": source_sha256(),
        "components": rows,
        "scheduler": {"local_frame_hz": 30, "local_control_hz": 30, "local_state_hz": 15, "intent_hz": 10, "patch_hz": 1, "colony_hz": .25, "society_hz": .05, "planet_forecast_hz": .02},
        "authority": {"neural_component_count": len(rows), "all_quality_gates_passed": True, "factorized_transition": True, "failed_cellular_action_v7_excluded": True, "deterministic_safety_projection_retained": True},
        "distillation": {"teacher_ready": True, "student_target": "recurrent action-DiT plus continuous VAE", "world_codec_ready": True, "monolithic_student_ready": False},
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    _atomic(destination, canonical(payload))
    return validate(output)


def validate(output: Path = DEFAULT_OUTPUT) -> dict:
    output = Path(output).resolve(); path = output / "ensemble_manifest.json"; raw = path.read_bytes(); payload = json.loads(raw)
    if raw != canonical(payload) or payload.get("format") != FORMAT or payload.get("source_sha256") != source_sha256():
        raise ValueError("neural ensemble manifest drifted")
    expected = hashlib.sha256(canonical({key: value for key, value in payload.items() if key != "manifest_sha256"})).hexdigest()
    if payload.get("manifest_sha256") != expected:
        raise ValueError("neural ensemble manifest hash drifted")
    for row in payload["components"]:
        for key in ("report", "artifact"):
            record = row[key]; source = PROJECT_ROOT / record["path"]
            if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
                raise ValueError(f"neural ensemble {key} drifted: {row['name']}")
    return {"passed": payload["status"] == "teacher_ensemble_ready", "components": len(payload["components"]), "manifest_sha256": payload["manifest_sha256"], "artifact_bytes": sum(row["artifact"]["bytes"] for row in payload["components"])}
