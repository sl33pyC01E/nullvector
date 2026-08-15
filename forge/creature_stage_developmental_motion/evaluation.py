from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ..config import PROJECT_ROOT
from ..creature_stage_developmental.contract import FAMILIES
from ..creature_stage_neural_motion.contract import TISSUES as MOTION_TISSUES
from ..safety import require_disk_floor
from .contract import (
    DEFAULT_OUTPUT,
    EVALUATION_FORMAT,
    EVALUATION_SCHEMA,
    FRAME_COUNT,
    MAX_DISPLACEMENT,
    DevelopmentalActuatorConfig,
    DevelopmentalTrainingConfig,
    source_sha256,
)
from .dataset import DevelopmentalMotionTeacher, DevelopmentalSequenceSampler, project_path
from .training import (
    _parent_authority,
    _validate_contract,
    atomic_bytes,
    canonical,
    checkpoint_name,
    load_checkpoint,
    load_successor_state,
    make_model,
    read_json,
    semantic,
    sha256_file,
)


MOTION_TISSUE_COLORS = {
    "skin": (80, 207, 236), "structure": (235, 229, 194), "armor": (139, 158, 178),
    "neural": (242, 78, 188), "circulatory": (208, 57, 74),
    "respiratory": (83, 225, 215), "digestive": (222, 186, 64),
    "sensor": (250, 239, 133), "locomotor": (239, 79, 102),
    "storage": (177, 123, 232), "phase": (184, 88, 249),
    "root": (119, 220, 89), "weapon": (255, 91, 91),
}


def _font(size: int):
    for path in (Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _frame_batch(teacher: DevelopmentalMotionTeacher, specimen: int, frame: int, device) -> dict[str, torch.Tensor]:
    return DevelopmentalSequenceSampler._stack([teacher.sample(specimen, frame)], device)


def _rollout_specimen(model, teacher, specimen: int, device) -> dict[str, np.ndarray]:
    first = _frame_batch(teacher, specimen, 0, device)
    cell_state = first["state"].float()
    node_state = first["node_state"].float()
    muscle_state = first["muscle_state"].float()
    predicted_cells: list[np.ndarray] = []
    predicted_nodes: list[np.ndarray] = []
    predicted_muscles: list[np.ndarray] = []
    parent_priors: list[np.ndarray] = []
    with torch.inference_mode():
        for frame_index in range(FRAME_COUNT):
            frame = _frame_batch(teacher, specimen, frame_index, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    frame["static"], cell_state, frame["mask"], frame["adjacency"],
                    frame["node_features"], node_state, frame["node_mask"], frame["node_adjacency"],
                    frame["muscle_features"], muscle_state, frame["muscle_mask"],
                    frame["muscle_incidence"], frame["cell_node_weights"], frame["parent_prior"], frame["family"],
                    frame["morphotype"], frame["phase"], frame["traits"],
                )
            cell_state = output["cell_state"].float()
            node_state = output["node_state"].float()
            muscle_state = output["muscle_activation"].float()
            predicted_cells.append(cell_state[0].cpu().numpy().copy())
            predicted_nodes.append(node_state[0].cpu().numpy().copy())
            predicted_muscles.append(muscle_state[0].cpu().numpy().copy())
            parent_priors.append(output["parent_prior"][0].float().cpu().numpy().copy())
    return {
        "cell": np.stack(predicted_cells), "node": np.stack(predicted_nodes),
        "muscle": np.stack(predicted_muscles), "parent": np.stack(parent_priors),
    }


def _specimen_metrics(teacher: DevelopmentalMotionTeacher, specimen: int, predicted: dict[str, np.ndarray]) -> dict[str, Any]:
    arrays = teacher.arrays
    cell_count = int(arrays["cell_count"][specimen])
    node_count = int(arrays["node_count"][specimen])
    muscle_count = int(arrays["muscle_count"][specimen])
    target_cell = arrays["trajectory"][specimen, :, :cell_count]
    target_node = arrays["node_trajectory"][specimen, :, :node_count]
    target_muscle = arrays["muscle_activation"][specimen, :, :muscle_count]
    prediction_cell = predicted["cell"][:, :cell_count, :2] * MAX_DISPLACEMENT
    prediction_node = predicted["node"][:, :node_count, :2] * MAX_DISPLACEMENT
    prediction_muscle = predicted["muscle"][:, :muscle_count]
    parent = predicted["parent"][:, :cell_count, :2] * MAX_DISPLACEMENT
    previous_cell = np.roll(target_cell, 1, axis=0)
    target_step = np.linalg.norm(target_cell - previous_cell, axis=2)
    predicted_step = np.linalg.norm(prediction_cell - np.roll(prediction_cell, 1, axis=0), axis=2)
    active_target = target_step > .08
    copy_fraction = float(np.mean((predicted_step < target_step * .25) & active_target)) if np.any(active_target) else 0.0
    target_energy = float(np.sqrt(np.mean(np.square(target_cell))))
    predicted_energy = float(np.sqrt(np.mean(np.square(prediction_cell))))

    rest = arrays["node_rest"][specimen, :node_count]
    adjacency = arrays["node_adjacency"][specimen, :node_count, :node_count].copy()
    np.fill_diagonal(adjacency, False)
    left, right = np.nonzero(np.triu(adjacency, 1))
    rest_length = np.linalg.norm(rest[right] - rest[left], axis=1)
    absolute = rest[None] + prediction_node
    current_length = np.linalg.norm(absolute[:, right] - absolute[:, left], axis=2)
    strain = np.abs(current_length - rest_length[None]) / np.maximum(rest_length[None], 1e-6)

    appendage = arrays["static"][specimen, :cell_count, 50] > .5
    appendage_target_energy = float(np.sqrt(np.mean(np.square(target_cell[:, appendage])))) if np.any(appendage) else 0.0
    appendage_predicted_energy = float(np.sqrt(np.mean(np.square(prediction_cell[:, appendage])))) if np.any(appendage) else 0.0
    return {
        "specimen": specimen,
        "genome_id": teacher.manifest["specimens"][specimen]["genome_id"],
        "family": FAMILIES[int(arrays["family"][specimen])],
        "role": teacher.manifest["specimens"][specimen]["role"],
        "cell_rmse_px": round(float(np.sqrt(np.mean(np.square(prediction_cell - target_cell)))), 9),
        "node_rmse_px": round(float(np.sqrt(np.mean(np.square(prediction_node - target_node)))), 9),
        "muscle_mae": round(float(np.mean(np.abs(prediction_muscle - target_muscle))), 9),
        "copy_previous_rmse_px": round(float(np.sqrt(np.mean(np.square(previous_cell - target_cell)))), 9),
        "parent_prior_rmse_px": round(float(np.sqrt(np.mean(np.square(parent - target_cell)))), 9),
        "energy_ratio": round(predicted_energy / max(target_energy, 1e-8), 9),
        "appendage_energy_ratio": round(appendage_predicted_energy / max(appendage_target_energy, 1e-8), 9),
        "copy_collapse_fraction": round(copy_fraction, 9),
        "maximum_bone_strain": round(float(np.max(strain)), 9),
        "p99_bone_strain": round(float(np.quantile(strain, .99)), 9),
        "seam_rmse_px": round(float(np.sqrt(np.mean(np.square(prediction_cell[0] - target_cell[0])))), 9),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "cell_rmse_px", "node_rmse_px", "muscle_mae", "copy_previous_rmse_px",
        "parent_prior_rmse_px", "energy_ratio", "appendage_energy_ratio",
        "copy_collapse_fraction", "maximum_bone_strain", "p99_bone_strain", "seam_rmse_px",
    )
    result: dict[str, Any] = {
        name: round(float(np.mean([float(record[name]) for record in records])), 9)
        for name in numeric
    }
    result["worst_family_cell_rmse_px"] = round(max(
        float(np.mean([record["cell_rmse_px"] for record in records if record["family"] == family]))
        for family in FAMILIES
    ), 9)
    result["families"] = {
        family: {
            "cell_rmse_px": round(float(np.mean([record["cell_rmse_px"] for record in records if record["family"] == family])), 9),
            "node_rmse_px": round(float(np.mean([record["node_rmse_px"] for record in records if record["family"] == family])), 9),
            "energy_ratio": round(float(np.mean([record["energy_ratio"] for record in records if record["family"] == family])), 9),
            "copy_collapse_fraction": round(float(np.mean([record["copy_collapse_fraction"] for record in records if record["family"] == family])), 9),
        }
        for family in FAMILIES
    }
    return result


def _render_contact(teacher: DevelopmentalMotionTeacher, predictions: list[dict[str, np.ndarray]]) -> Image.Image:
    width, height = 1800, 1060
    image = Image.new("RGBA", (width, height), (3, 8, 14, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width, 62), fill=(6, 15, 24, 255))
    draw.text((24, 16), "NEURAL ACTUATOR ROLLOUT // TARGET GHOST + PREDICTED CELLS", font=_font(25), fill=(219, 242, 249, 255))
    phase_indices = (0, 12, 24, 36, 48, 60)
    panel_w = width / 5
    panel_h = (height - 62) / 2
    for specimen in range(10):
        column, row = specimen // 2, specimen % 2
        x0, y0 = column * panel_w, 62 + row * panel_h
        family = FAMILIES[int(teacher.arrays["family"][specimen])]
        role = teacher.manifest["specimens"][specimen]["role"]
        draw.text((x0 + 14, y0 + 12), f"{family.upper()} // {role.upper()}", font=_font(15), fill=(215, 236, 243, 255))
        count = int(teacher.arrays["cell_count"][specimen])
        rest = teacher.arrays["rest_xy"][specimen, :count]
        tissue = teacher.arrays["static"][specimen, :count, 4:17].argmax(axis=1)
        target = teacher.arrays["trajectory"][specimen, :, :count]
        predicted = predictions[specimen]["cell"][:, :count, :2] * MAX_DISPLACEMENT
        for phase_slot, phase in enumerate(phase_indices):
            cx = x0 + 42 + phase_slot * 54
            cy = y0 + 270
            target_points = rest + target[phase]
            predicted_points = rest + predicted[phase]
            for x, y in target_points:
                draw.point((round(cx + x * 1.35), round(cy + y * 1.35)), fill=(80, 110, 125, 90))
            for cell_index, (x, y) in enumerate(predicted_points):
                color = MOTION_TISSUE_COLORS[MOTION_TISSUES[int(tissue[cell_index])]]
                px, py = cx + x * 1.35, cy + y * 1.35
                draw.ellipse((px - 1.15, py - 1.15, px + 1.15, py + 1.15), fill=(*color, 235))
            draw.text((cx - 10, y0 + panel_h - 32), f"{phase:02d}", font=_font(10), fill=(91, 214, 233, 255))
        draw.line((x0, y0, x0 + panel_w, y0), fill=(31, 68, 80, 255), width=1)
        draw.line((x0, y0, x0, y0 + panel_h), fill=(31, 68, 80, 255), width=1)
    return image


def evaluate_checkpoint(
    output: Path = DEFAULT_OUTPUT,
    *,
    checkpoint: Path | None = None,
    destination: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    output = Path(output).resolve()
    contract = read_json(output / "production_contract.json")
    _validate_contract(contract)
    update = int(contract["total_updates"]) if checkpoint is None else int(Path(checkpoint).stem.rsplit("_", 1)[-1])
    checkpoint_path = output / checkpoint_name(update) if checkpoint is None else Path(checkpoint).resolve()
    checked = load_checkpoint(checkpoint_path, contract, update)
    destination = (output / f"evaluation_{update:07d}") if destination is None else Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100, planned_bytes=1024**3)
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("developmental actuator evaluation requested unavailable CUDA")
    parent_contract, parent = _parent_authority(PROJECT_ROOT / contract["parent"]["path"])
    model = make_model(parent_contract, parent, DevelopmentalActuatorConfig(**contract["model"])).to(runtime_device)
    load_successor_state(model, checked["ema_state"])
    model.eval()
    teacher = DevelopmentalMotionTeacher(
        PROJECT_ROOT / contract["teacher"]["path"],
        prior=PROJECT_ROOT / contract["prior"]["path"], replay=False,
    )
    predictions = [_rollout_specimen(model, teacher, specimen, runtime_device) for specimen in range(10)]
    records = [_specimen_metrics(teacher, specimen, predictions[specimen]) for specimen in range(10)]
    metrics = _aggregate(records)
    gates = {
        "all_five_families": set(metrics["families"]) == set(FAMILIES),
        "beats_copy_previous_baseline": metrics["cell_rmse_px"] < metrics["copy_previous_rmse_px"] * .90,
        "beats_sealed_parent_prior": metrics["cell_rmse_px"] < metrics["parent_prior_rmse_px"] * .90,
        "motion_energy_preserved": .65 <= metrics["energy_ratio"] <= 1.35,
        "appendage_energy_preserved": .55 <= metrics["appendage_energy_ratio"] <= 1.50,
        "copy_collapse_below_15pct": metrics["copy_collapse_fraction"] < .15,
        "mean_cell_rmse_below_2px": metrics["cell_rmse_px"] < 2.0,
        "worst_family_cell_rmse_below_3px": metrics["worst_family_cell_rmse_px"] < 3.0,
        "mean_node_rmse_below_1_5px": metrics["node_rmse_px"] < 1.5,
        "muscle_mae_below_0_18": metrics["muscle_mae"] < .18,
        "p99_bone_strain_below_0_20": metrics["p99_bone_strain"] < .20,
        "seam_rmse_below_1_5px": metrics["seam_rmse_px"] < 1.5,
    }
    destination.mkdir(parents=True)
    contact_path = destination / "evaluation_contact_sheet.png"
    _render_contact(teacher, predictions).save(contact_path, format="PNG", compress_level=9)
    report: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "status": "passed" if all(gates.values()) else "failed-quality",
        "source_sha256": source_sha256(),
        "production": {"path": project_path(output), "semantic_sha256": contract["semantic_sha256"]},
        "checkpoint": {
            "path": project_path(checkpoint_path), "update": update,
            "sha256": sha256_file(checkpoint_path), "ema_state_sha256": checked["ema_state_sha256"],
        },
        "teacher_semantic_sha256": teacher.semantic_sha256,
        "device": str(runtime_device),
        "specimens": records,
        "metrics": metrics,
        "gates": gates,
        "contact_sheet": {
            "path": contact_path.name, "bytes": contact_path.stat().st_size,
            "sha256": sha256_file(contact_path), "visually_inspected": False,
        },
    }
    report["semantic_sha256"] = semantic(report)
    atomic_bytes(destination / "evaluation_manifest.json", canonical(report))
    return validate_evaluation(destination)


def validate_evaluation(destination: Path) -> dict[str, Any]:
    destination = Path(destination).resolve()
    report = read_json(destination / "evaluation_manifest.json")
    errors = sorted(
        Draft202012Validator(json.loads(EVALUATION_SCHEMA.read_text(encoding="utf-8"))).iter_errors(report),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"developmental actuator evaluation schema drifted: {errors[0].message}")
    if (
        report["format"] != EVALUATION_FORMAT or report["source_sha256"] != source_sha256()
        or report["semantic_sha256"]
        != semantic({key: value for key, value in report.items() if key != "semantic_sha256"})
        or (report["status"] == "passed") != all(report["gates"].values())
        or len(report["specimens"]) != 10
    ):
        raise ValueError("developmental actuator evaluation authority drifted")
    contact = destination / report["contact_sheet"]["path"]
    if contact.stat().st_size != report["contact_sheet"]["bytes"] or sha256_file(contact) != report["contact_sheet"]["sha256"]:
        raise ValueError("developmental actuator evaluation contact sheet drifted")
    checkpoint = PROJECT_ROOT / report["checkpoint"]["path"]
    if sha256_file(checkpoint) != report["checkpoint"]["sha256"]:
        raise ValueError("developmental actuator evaluation checkpoint drifted")
    return {
        "passed": report["status"] == "passed", "status": report["status"],
        "update": report["checkpoint"]["update"], "metrics": report["metrics"],
        "gates": report["gates"], "semantic_sha256": report["semantic_sha256"],
    }
