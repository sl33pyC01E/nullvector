from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from ..map_decorator_ml.contract import HEAD_CLASS_COUNTS
from ..map_decorator_ml.legality import (
    ILLEGAL_LOGIT,
    TorchLegalMasks,
    assert_selected_legal,
    mask_head_logits,
)
from ..map_decorator_ml.training import class_balanced_weights
from .contract import FactoredLossConfig
from .decoding import select_factored_legal_argmax
from .model import FactoredDecoratorOutput, OBJECT_HEADS


def _categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
    *,
    classes: int,
    empty_class: int | None,
) -> torch.Tensor:
    if not bool(selected.any()):
        raise ValueError("Categorical loss received no selected cells.")
    weights = class_balanced_weights(
        target,
        selected,
        classes,
        empty_class=empty_class,
    )
    per_cell = F.cross_entropy(logits, target, weight=weights, reduction="none")
    return per_cell[selected].mean()


def _balanced_presence_loss(
    presence_logit: torch.Tensor,
    foreground: torch.Tensor,
    selected: torch.Tensor,
    *,
    hard_negative_ratio: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    positives = selected & foreground
    negatives = selected & ~foreground
    if not bool(positives.any()):
        raise ValueError("Factored presence loss requires selected foreground cells.")
    if not bool(negatives.any()):
        raise ValueError("Factored presence loss requires selected empty cells.")
    positive_loss = F.softplus(-presence_logit[positives])
    negative_loss = F.softplus(presence_logit[negatives])
    keep = min(int(negative_loss.numel()), max(1, int(positive_loss.numel()) * hard_negative_ratio))
    hard_negative = torch.topk(negative_loss, k=keep, largest=True, sorted=False).values
    loss = 0.5 * (positive_loss.mean() + hard_negative.mean())
    return loss, {
        "positive_cells": int(positive_loss.numel()),
        "available_negative_cells": int(negative_loss.numel()),
        "hard_negative_cells": keep,
    }


def _foreground_type_loss(
    type_logits: torch.Tensor,
    legal_types: torch.Tensor,
    target: torch.Tensor,
    selected_foreground: torch.Tensor,
) -> torch.Tensor:
    if not bool(selected_foreground.any()):
        raise ValueError("Within-type loss requires selected foreground cells.")
    if legal_types.shape != type_logits.shape or legal_types.dtype != torch.bool:
        raise TypeError("Foreground type legality must exactly match type logits.")
    bounded = type_logits.masked_fill(~legal_types, ILLEGAL_LOGIT)
    foreground_target = target - 1
    classes = type_logits.shape[1]
    weights = class_balanced_weights(
        foreground_target,
        selected_foreground,
        classes,
        empty_class=None,
    )
    selected_logits = bounded.permute(0, 2, 3, 1)[selected_foreground]
    selected_targets = foreground_target[selected_foreground]
    return F.cross_entropy(selected_logits, selected_targets, weight=weights, reduction="mean")


def _count_calibration_loss(
    presence_logit: torch.Tensor,
    foreground: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    selected_float = selected.to(presence_logit.dtype)
    predicted = (torch.sigmoid(presence_logit) * selected_float).sum(dim=(1, 2))
    target = (foreground & selected).sum(dim=(1, 2)).to(presence_logit.dtype)
    active = selected.sum(dim=(1, 2)) > 0
    if not bool(active.any()):
        raise ValueError("Count calibration requires selected eligible cells.")
    loss = F.smooth_l1_loss(torch.log1p(predicted[active]), torch.log1p(target[active]))
    return loss, {
        "predicted_count": float(predicted[active].detach().sum().item()),
        "target_count": float(target[active].detach().sum().item()),
    }


def factored_refinement_loss(
    output: FactoredDecoratorOutput,
    targets: Mapping[str, torch.Tensor],
    masked: Mapping[str, torch.Tensor],
    legal_masks: Mapping[str, torch.Tensor],
    valid_cells: torch.Tensor,
    hard_empty: torch.Tensor,
    *,
    config: FactoredLossConfig = FactoredLossConfig(),
) -> tuple[torch.Tensor, dict[str, float | int], dict[str, torch.Tensor]]:
    legal = TorchLegalMasks(hard_empty=hard_empty, **legal_masks)
    assert_selected_legal(targets, legal)
    bounded = mask_head_logits(output.as_head_logits(), legal)
    losses: dict[str, torch.Tensor] = {}
    details: dict[str, float | int] = {}

    for name in ("variant", "emission"):
        selected = masked[name] & valid_cells
        loss = _categorical_loss(
            getattr(bounded, name),
            targets[name],
            selected,
            classes=HEAD_CLASS_COUNTS[name],
            empty_class=None if name == "variant" else 0,
        )
        losses[name] = loss
        details[f"{name}_categorical"] = float(loss.detach().item())

    for name in OBJECT_HEADS:
        selected = masked[name] & valid_cells
        eligible = legal_masks[name][:, 1:].any(dim=1) & valid_cells
        selected_eligible = selected & eligible
        foreground = targets[name] != 0
        presence, presence_counts = _balanced_presence_loss(
            output.presence_logits[name],
            foreground,
            selected_eligible,
            hard_negative_ratio=config.hard_negative_ratio,
        )
        selected_foreground = selected_eligible & foreground
        within_type = _foreground_type_loss(
            output.type_logits[name],
            legal_masks[name][:, 1:],
            targets[name],
            selected_foreground,
        )
        categorical = _categorical_loss(
            getattr(bounded, name),
            targets[name],
            selected,
            classes=HEAD_CLASS_COUNTS[name],
            empty_class=0,
        )
        count, count_details = _count_calibration_loss(
            output.presence_logits[name],
            foreground,
            selected_eligible,
        )
        losses[f"{name}_presence"] = presence
        losses[f"{name}_type"] = within_type
        losses[f"{name}_categorical"] = categorical
        losses[f"{name}_count"] = count
        details[f"{name}_presence"] = float(presence.detach().item())
        details[f"{name}_type"] = float(within_type.detach().item())
        details[f"{name}_categorical"] = float(categorical.detach().item())
        details[f"{name}_count"] = float(count.detach().item())
        for key, value in presence_counts.items():
            details[f"{name}_{key}"] = value
        for key, value in count_details.items():
            details[f"{name}_{key}"] = value

    overlap_mask = valid_cells & (masked["decal"] | masked["prop"])
    if not bool(overlap_mask.any()):
        raise ValueError("Object exclusivity loss requires selected valid cells.")
    overlap_probability = torch.sigmoid(output.presence_logits["decal"]) * torch.sigmoid(
        output.presence_logits["prop"]
    )
    exclusivity = overlap_probability[overlap_mask].mean()
    losses["object_exclusivity"] = exclusivity
    details["object_exclusivity"] = float(exclusivity.detach().item())

    weighted = (
        config.variant_weight * losses["variant"]
        + config.emission_weight * losses["emission"]
        + config.object_exclusivity_weight * exclusivity
    )
    denominator = config.variant_weight + config.emission_weight + config.object_exclusivity_weight
    for name in OBJECT_HEADS:
        weighted = (
            weighted
            + config.object_presence_weight * losses[f"{name}_presence"]
            + config.object_type_weight * losses[f"{name}_type"]
            + config.object_categorical_weight * losses[f"{name}_categorical"]
            + config.object_count_weight * losses[f"{name}_count"]
        )
        denominator += (
            config.object_presence_weight
            + config.object_type_weight
            + config.object_categorical_weight
            + config.object_count_weight
        )
    total = weighted / denominator
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Factored decorator loss became non-finite.")
    predictions = select_factored_legal_argmax(output, legal)
    details["total"] = float(total.detach().item())
    return total, details, predictions
