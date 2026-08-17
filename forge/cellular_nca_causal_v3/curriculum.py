from __future__ import annotations

import torch
from torch import Tensor

from ..cellular_nca.teacher import cellular_loss, teacher_step
from ..cellular_nca_causal.curriculum import causal_contrast_loss


ROLLOUT_STEPS = 6
CONTRAST_STEPS = (2, 4, 6)


def long_horizon_loss(
    model,
    static: Tensor,
    control: Tensor,
    damaged: Tensor,
    bonds: Tensor,
    system_ids: Tensor,
    *,
    dt: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Supervise the recurrent trajectory, not only its first two updates."""
    batch = len(static)
    pair_static = torch.cat((static, static))
    pair_bonds = torch.cat((bonds, bonds))
    predicted = torch.cat((control, damaged))
    target = predicted.detach().clone()
    base_losses: list[Tensor] = []
    contrast_losses: list[Tensor] = []
    directions: list[Tensor] = []
    magnitudes: list[Tensor] = []
    final_pieces: dict[str, Tensor] = {}
    for step in range(1, ROLLOUT_STEPS + 1):
        previous = predicted
        target = teacher_step(pair_static, target, pair_bonds, dt)
        predicted = model(pair_static, predicted, pair_bonds)
        base, pieces = cellular_loss(predicted, target, pair_static, previous)
        base_losses.append(base)
        final_pieces = pieces
        if step in CONTRAST_STEPS:
            contrast, values = causal_contrast_loss(
                predicted[:batch], predicted[batch:], target[:batch], target[batch:], static, system_ids
            )
            contrast_losses.append(contrast)
            directions.append(values["direction"])
            magnitudes.append(values["magnitude"])
    base = torch.stack(base_losses).mean()
    contrast = torch.stack(contrast_losses).mean()
    return base + .45 * contrast, {
        "base": base.detach(),
        "contrast": contrast.detach(),
        "direction": torch.stack(directions).mean().detach(),
        "magnitude": torch.stack(magnitudes).mean().detach(),
        "health": final_pieces["channel_0"].detach(),
        "oxygen": final_pieces["channel_4"].detach(),
        "energy": final_pieces["channel_3"].detach(),
        "neural": final_pieces["channel_8"].detach(),
    }
