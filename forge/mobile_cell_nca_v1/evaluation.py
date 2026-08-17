from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor

from ..cellular_nca.contract import DYNAMIC_NAMES
from ..cellular_nca.evaluation import _damage_system, _mae, _render
from ..cellular_nca.teacher import make_scenarios


SYSTEMS = (
    ("circulation", 28, 1),
    ("respiration", 31, 4),
    ("digestion", 34, 3),
    ("neural", 37, 8),
)


def _support(static: Tensor) -> Tensor:
    body = static[:, :1].float()
    return torch.cat((body.expand(-1, 9, -1, -1), torch.ones_like(body).expand(-1, 2, -1, -1), body), 1)


def _response_metrics(predicted: Tensor, target: Tensor, initial: Tensor, static: Tensor) -> dict[str, float]:
    support = _support(static)
    predicted_delta = (predicted.float() - initial.float()) * support
    target_delta = (target.float() - initial.float()) * support
    numerator = (predicted_delta - target_delta).abs().sum((1, 2, 3))
    denominator = target_delta.abs().sum((1, 2, 3)).clamp_min(1e-6)
    dot = (predicted_delta * target_delta).sum((1, 2, 3))
    norm = predicted_delta.square().sum((1, 2, 3)).sqrt() * target_delta.square().sum((1, 2, 3)).sqrt()
    active = target_delta.abs() >= 2e-4
    sign_equal = ((predicted_delta.sign() == target_delta.sign()) & active).sum((1, 2, 3)) / active.sum((1, 2, 3)).clamp_min(1)
    return {
        "normalized_l1": float((numerator / denominator).mean()),
        "cosine": float((dot / norm.clamp_min(1e-8)).mean()),
        "active_sign_agreement": float(sign_equal.float().mean()),
        "teacher_change_mean": float((target_delta.abs().sum((1, 2, 3)) / support.sum((1, 2, 3)).clamp_min(1)).mean()),
    }


def _relative(state: Tensor, control: Tensor, body: Tensor, readout: int) -> float:
    damaged_value = (state[:, readout : readout + 1] * body).sum() / body.sum().clamp_min(1)
    control_value = (control[:, readout : readout + 1] * body).sum() / body.sum().clamp_min(1)
    return float((damaged_value - control_value) / control_value.clamp_min(1e-6))


def _contact_sheet(static: Tensor, injured: Tensor, target: Tensor, predicted: Tensor, family: Tensor) -> bytes:
    tile = 192
    header = 62
    label = 24
    rows = len(static)
    canvas = Image.new("RGB", (tile * 3, header + rows * (tile + label)), (3, 8, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), "MOBILE NEURAL CELL RULE // 32 RECURRENT PHYSIOLOGY STEPS", fill=(80, 236, 255))
    draw.text((12, 29), "injury, organ failure, fluids, clotting, scarring, neural activity, death and biomass", fill=(148, 174, 190))
    for column, title in enumerate(("INJURED T0", "10M TEACHER T32", "MOBILE STUDENT T32")):
        draw.text((column * tile + 8, 47), title, fill=(187, 255, 78))
    family_names = ("HUMANOID", "ANIMALIAN", "PLANTLIKE", "ANOMALY", "MACHINE")
    static_np = static.detach().cpu().numpy()
    family_np = family.detach().cpu().numpy()
    states = (injured.detach().cpu().numpy(), target.detach().cpu().numpy(), predicted.detach().cpu().numpy())
    for row in range(rows):
        y = header + row * (tile + label)
        for column, values in enumerate(states):
            canvas.paste(_render(static_np[row], values[row], int(family_np[row])), (column * tile, y))
        draw.text((8, y + tile + 4), family_names[int(family_np[row])], fill=(187, 255, 78))
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", compress_level=9, optimize=False)
    return stream.getvalue()


@torch.inference_mode()
def evaluate_candidate(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    static: Tensor,
    initial: Tensor,
    bonds: Tensor,
    family: Tensor,
    *,
    seed: int,
) -> tuple[dict[str, Any], bytes]:
    generator = torch.Generator(device=static.device).manual_seed(seed)
    injured, live = make_scenarios(static, initial, bonds, generator)
    target = injured.clone()
    predicted = injured.clone()
    metrics_by_step: dict[str, dict[str, float]] = {}
    response_by_step: dict[str, dict[str, float]] = {}
    for step in range(1, 33):
        target = teacher(static, target, live)
        predicted = model(static, predicted, live)
        if step in (1, 2, 4, 8, 16, 32):
            metrics_by_step[str(step)] = _mae(predicted, target, static)
            response_by_step[str(step)] = _response_metrics(predicted, target, injured, static)

    family_errors: dict[str, float] = {}
    support = _support(static)
    for family_id, family_name in enumerate(("humanoid", "animalian", "plantlike", "anomaly", "machine")):
        selected = family == family_id
        error = ((predicted[selected] - target[selected]).abs() * support[selected]).sum()
        family_errors[family_name] = float(error / support[selected].sum().clamp_min(1))

    clean_target = initial.clone()
    clean_predicted = initial.clone()
    for _ in range(16):
        clean_target = teacher(static, clean_target, bonds)
        clean_predicted = model(static, clean_predicted, bonds)
    clean_mae = _mae(clean_predicted, clean_target, static)

    body = static[:, :1]
    organ_rows: list[dict[str, Any]] = []
    for name, start, readout in SYSTEMS:
        teacher_control = initial.clone()
        teacher_damaged = _damage_system(initial, static, start)
        student_control = initial.clone()
        student_damaged = teacher_damaged.clone()
        for _ in range(16):
            teacher_control = teacher(static, teacher_control, bonds)
            teacher_damaged = teacher(static, teacher_damaged, bonds)
            student_control = model(static, student_control, bonds)
            student_damaged = model(static, student_damaged, bonds)
        teacher_change = _relative(teacher_damaged, teacher_control, body, readout)
        student_change = _relative(student_damaged, student_control, body, readout)
        organ_rows.append({
            "system": name,
            "readout": DYNAMIC_NAMES[readout],
            "teacher_relative_change": teacher_change,
            "student_relative_change": student_change,
            "absolute_error": abs(student_change - teacher_change),
        })

    final = metrics_by_step["32"]
    response = response_by_step["32"]
    causal_error = float(np.mean([row["absolute_error"] for row in organ_rows]))
    internal_escape = float((predicted[:, :9] * (1 - body)).abs().max())
    surface_outside = float((predicted[:, 9:10] * (1 - body)).sum() / (1 - body).sum().clamp_min(1))
    values = [value for row in metrics_by_step.values() for value in row.values()]
    values += [value for row in response_by_step.values() for value in row.values()]
    values += [value for row in organ_rows for key, value in row.items() if key.endswith("change") or key == "absolute_error"]
    gates = {
        "all_values_finite": bool(np.isfinite(values).all()),
        "internal_physiology_stays_in_chassis": internal_escape == 0,
        "surface_fluid_can_leave_chassis": surface_outside > 0,
        "health_32_mae_below_0_015": final["health"] < .015,
        "fluid_32_mae_below_0_03": final["fluid"] < .03,
        "neural_32_mae_below_0_04": final["neural_activity"] < .04,
        "worst_family_32_mae_below_0_035": max(family_errors.values()) < .035,
        "response_normalized_l1_below_0_35": response["normalized_l1"] < .35,
        "response_cosine_above_0_90": response["cosine"] > .90,
        "response_sign_agreement_above_0_85": response["active_sign_agreement"] > .85,
        "organ_counterfactual_mae_below_0_025": causal_error < .025,
        "all_organ_effect_directions_match": all(row["teacher_relative_change"] * row["student_relative_change"] > 0 for row in organ_rows),
    }
    metrics: dict[str, Any] = {
        "mae_by_step": metrics_by_step,
        "response_by_step": response_by_step,
        "family_32_mae": family_errors,
        "clean_16_mae": clean_mae,
        "organ_counterfactuals": organ_rows,
        "organ_counterfactual_mae": causal_error,
        "internal_escape_max": internal_escape,
        "surface_fluid_outside_mean": surface_outside,
        "gates": gates,
    }
    return metrics, _contact_sheet(static, injured, target, predicted, family)
