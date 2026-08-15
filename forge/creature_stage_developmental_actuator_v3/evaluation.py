from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_developmental_motion.dataset import DevelopmentalMotionTeacher, project_path
from ..creature_stage_developmental_motion.evaluation import _aggregate, _render_contact, _rollout_specimen, _specimen_metrics
from ..creature_stage_developmental_motion.training import atomic_bytes, canonical, read_json, semantic, sha256_file
from ..creature_stage_developmental_actuator_v2.contract import CausalActuatorConfig
from .contract import DEFAULT_OUTPUT, EVALUATION_FORMAT, EVALUATION_SCHEMA, BoneProjectionConfig, source_sha256
from .training import _validate_contract, checkpoint_name, load_checkpoint, load_model_state, make_model


def evaluate_checkpoint(output: Path = DEFAULT_OUTPUT, *, checkpoint: Path | None = None, destination: Path | None = None, device: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve(); contract = read_json(output / "production_contract.json"); _validate_contract(contract)
    update = int(contract["total_updates"]) if checkpoint is None else int(Path(checkpoint).stem.rsplit("_", 1)[-1])
    checkpoint_path = output / checkpoint_name(update) if checkpoint is None else Path(checkpoint).resolve()
    checked = load_checkpoint(checkpoint_path, contract, update)
    destination = output / f"evaluation_{update:07d}" if destination is None else Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100, planned_bytes=1024**3)
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("length-projected evaluation requested unavailable CUDA")
    model = make_model(CausalActuatorConfig(**contract["model"]), BoneProjectionConfig(**contract["projection"])).to(runtime_device)
    load_model_state(model, checked["ema_state"]); model.eval()
    teacher = DevelopmentalMotionTeacher(PROJECT_ROOT / contract["teacher"]["path"], prior=PROJECT_ROOT / contract["prior"]["path"], replay=False)
    predictions = [_rollout_specimen(model, teacher, specimen, runtime_device) for specimen in range(10)]
    records = [_specimen_metrics(teacher, specimen, predictions[specimen]) for specimen in range(10)]
    metrics = _aggregate(records)
    gates = {
        "all_five_families": len(metrics["families"]) == 5,
        "beats_sealed_parent_prior_by_10pct": metrics["cell_rmse_px"] < metrics["parent_prior_rmse_px"] * .90,
        "motion_energy_preserved": .75 <= metrics["energy_ratio"] <= 1.25,
        "appendage_energy_preserved": .70 <= metrics["appendage_energy_ratio"] <= 1.30,
        "copy_collapse_below_10pct": metrics["copy_collapse_fraction"] < .10,
        "mean_cell_rmse_below_0_40px": metrics["cell_rmse_px"] < .40,
        "worst_family_cell_rmse_below_0_55px": metrics["worst_family_cell_rmse_px"] < .55,
        "mean_node_rmse_below_0_60px": metrics["node_rmse_px"] < .60,
        "muscle_mae_below_0_18": metrics["muscle_mae"] < .18,
        "p99_bone_strain_below_0_18": metrics["p99_bone_strain"] < .18,
        "maximum_bone_strain_below_0_30": metrics["maximum_bone_strain"] < .30,
        "seam_rmse_below_0_30px": metrics["seam_rmse_px"] < .30,
    }
    destination.mkdir(parents=True); contact = destination / "evaluation_contact_sheet.png"; _render_contact(teacher, predictions).save(contact, format="PNG", compress_level=9)
    report: dict[str, Any] = {"format": EVALUATION_FORMAT, "status": "passed" if all(gates.values()) else "failed-quality", "source_sha256": source_sha256(), "production": {"path": project_path(output), "semantic_sha256": contract["semantic_sha256"]}, "checkpoint": {"path": project_path(checkpoint_path), "update": update, "sha256": sha256_file(checkpoint_path), "ema_state_sha256": checked["ema_state_sha256"]}, "teacher_semantic_sha256": teacher.semantic_sha256, "device": str(runtime_device), "specimens": records, "metrics": metrics, "gates": gates, "contact_sheet": {"path": contact.name, "bytes": contact.stat().st_size, "sha256": sha256_file(contact), "visually_inspected": False}}
    report["semantic_sha256"] = semantic(report); atomic_bytes(destination / "evaluation_manifest.json", canonical(report)); return validate_evaluation(destination)


def validate_evaluation(destination: Path) -> dict[str, Any]:
    destination = Path(destination).resolve(); report = read_json(destination / "evaluation_manifest.json")
    errors = sorted(Draft202012Validator(json.loads(EVALUATION_SCHEMA.read_text(encoding="utf-8"))).iter_errors(report), key=lambda error: list(error.path))
    if errors: raise ValueError(f"length-projected evaluation schema drifted: {errors[0].message}")
    if report["format"] != EVALUATION_FORMAT or report["source_sha256"] != source_sha256() or report["semantic_sha256"] != semantic({key: value for key, value in report.items() if key != "semantic_sha256"}) or (report["status"] == "passed") != all(report["gates"].values()) or len(report["specimens"]) != 10: raise ValueError("length-projected evaluation authority drifted")
    contact = destination / report["contact_sheet"]["path"]
    if contact.stat().st_size != report["contact_sheet"]["bytes"] or sha256_file(contact) != report["contact_sheet"]["sha256"]: raise ValueError("length-projected contact sheet drifted")
    checkpoint = PROJECT_ROOT / report["checkpoint"]["path"]
    if sha256_file(checkpoint) != report["checkpoint"]["sha256"]: raise ValueError("length-projected checkpoint drifted")
    return {"passed": report["status"] == "passed", "status": report["status"], "update": report["checkpoint"]["update"], "metrics": report["metrics"], "gates": report["gates"], "semantic_sha256": report["semantic_sha256"]}
