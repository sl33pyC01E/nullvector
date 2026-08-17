from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from ..cellular_nca.teacher import teacher_step


# name, first static organ-role channel, dynamic readout channel
SYSTEMS: tuple[tuple[str, int, int], ...] = (
    ("circulation", 28, 1),
    ("respiration", 31, 4),
    ("digestion", 34, 3),
    ("neural", 37, 8),
)
PRE_ROLL_CHOICES: tuple[int, ...] = (0, 4, 8, 16)


def system_mask(static: Tensor, system_ids: Tensor) -> Tensor:
    if static.ndim != 4 or static.shape[1:] != (85, 48, 48) or system_ids.shape != (len(static),) or system_ids.dtype != torch.long:
        raise ValueError("Causal system-mask contract drifted.")
    result = torch.zeros((len(static), 1, 48, 48), dtype=static.dtype, device=static.device)
    for system_id, (_, start, _) in enumerate(SYSTEMS):
        selected = system_ids == system_id
        if bool(selected.any()):
            result[selected] = static[selected, start : start + 3].sum(1, keepdim=True).clamp(0, 1)
    return result


def apply_system_ablation(initial: Tensor, static: Tensor, system_ids: Tensor) -> Tensor:
    if initial.ndim != 4 or initial.shape[1:] != (12, 48, 48) or len(initial) != len(static):
        raise ValueError("Causal ablation state contract drifted.")
    result = initial.clone(); mask = system_mask(static, system_ids)
    result[:, 0:1] *= 1 - .78 * mask
    result[:, 1:2] *= 1 - .52 * mask
    result[:, 7:8] = torch.maximum(result[:, 7:8], .9 * mask)
    result[:, 11:12] = (result[:, 0:1] > .025).to(result.dtype) * static[:, :1]
    return result


@torch.no_grad()
def make_targeted_pairs(static: Tensor, initial: Tensor, bonds: Tensor, system_ids: Tensor, pre_rolls: Tensor, *, dt: float = .1) -> tuple[Tensor, Tensor]:
    """Return matched healthy/damaged states after bounded teacher pre-roll.

    Pre-roll exposes the learner to both the tiny onset and the developed
    downstream consequences of organ failure without backpropagating through a
    long recurrent history. It is deterministic for identical tensors.
    """
    if pre_rolls.shape != (len(static),) or pre_rolls.dtype != torch.long or bool((pre_rolls < 0).any()) or bool((pre_rolls > 16).any()):
        raise ValueError("Causal pre-roll contract drifted.")
    control = initial.clone(); damaged = apply_system_ablation(initial, static, system_ids)
    for step in range(int(pre_rolls.max().item()) if len(pre_rolls) else 0):
        active = (pre_rolls > step)[:, None, None, None]
        control = torch.where(active, teacher_step(static, control, bonds, dt), control)
        damaged = torch.where(active, teacher_step(static, damaged, bonds, dt), damaged)
    return control, damaged


def causal_contrast_loss(predicted_control: Tensor, predicted_damaged: Tensor, target_control: Tensor, target_damaged: Tensor, static: Tensor, system_ids: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Match counterfactual response magnitude and spatial response per organ.

    A plain reconstruction objective can minimize average error while ignoring
    small but vital oxygen/energy deltas. Each selected readout is normalized by
    its teacher counterfactual magnitude, so respiration and digestion cannot be
    drowned out by circulation or neural signals.
    """
    body = static[:, :1].float(); losses: list[Tensor] = []; direction_losses: list[Tensor] = []; magnitude_errors: list[Tensor] = []
    for index, (_, _, readout) in enumerate(SYSTEMS):
        selected = system_ids == index
        if not bool(selected.any()):
            continue
        support = body[selected]
        predicted_delta = (predicted_control[selected, readout : readout + 1].float() - predicted_damaged[selected, readout : readout + 1].float()) * support
        target_delta = (target_control[selected, readout : readout + 1].float() - target_damaged[selected, readout : readout + 1].float()) * support
        target_scale = target_delta.abs().sum((1, 2, 3)).clamp_min(1e-3)
        spatial_error = (predicted_delta - target_delta).abs().sum((1, 2, 3)) / target_scale
        predicted_mean = predicted_delta.sum((1, 2, 3)) / support.sum((1, 2, 3)).clamp_min(1)
        target_mean = target_delta.sum((1, 2, 3)) / support.sum((1, 2, 3)).clamp_min(1)
        normalized_error = (predicted_mean - target_mean).abs() / target_mean.abs().clamp_min(2e-4)
        direction = F.relu(.35 * target_mean.abs() - predicted_mean * target_mean.sign()) / target_mean.abs().clamp_min(2e-4)
        losses.append((.55 * spatial_error + .45 * normalized_error).mean()); direction_losses.append(direction.mean()); magnitude_errors.append(normalized_error.mean())
    if not losses:
        raise ValueError("Causal contrast batch contains no systems.")
    contrast = torch.stack(losses).mean(); direction = torch.stack(direction_losses).mean(); magnitude = torch.stack(magnitude_errors).mean()
    return contrast + .35 * direction, {"contrast": contrast.detach(), "direction": direction.detach(), "magnitude": magnitude.detach()}
