from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from ..map_decorator_ml.contract import HEAD_CLASS_COUNTS
from ..map_decorator_ml.legality import ILLEGAL_LOGIT, TorchLegalMasks, assert_selected_legal, mask_head_logits
from ..map_decorator_ml.training import class_balanced_weights
from ..map_decorator_production_v2.model import OBJECT_HEADS
from .contract import LocatorLossConfig
from .decoding import select_sparse_locator_argmax
from .model import SparseLocatorOutput


def _categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
    *,
    classes: int,
    empty_class: int | None,
) -> torch.Tensor:
    if not bool(selected.any()):
        raise ValueError("Categorical loss requires selected cells.")
    weights = class_balanced_weights(target, selected, classes, empty_class=empty_class)
    per_cell = F.cross_entropy(logits, target, weight=weights, reduction="none")
    return per_cell[selected].mean()


def _localized_presence_loss(
    logits: torch.Tensor,
    foreground: torch.Tensor,
    selected: torch.Tensor,
    *,
    config: LocatorLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
    positives = selected & foreground
    if not bool(positives.any()):
        raise ValueError("Localized presence loss requires masked foreground cells.")
    radius = config.halo_radius
    dilated = F.max_pool2d(
        foreground[:, None].to(torch.float32),
        kernel_size=radius * 2 + 1,
        stride=1,
        padding=radius,
    ).squeeze(1) > 0
    halo = selected & dilated & ~foreground
    negatives = selected & ~dilated
    if not bool(negatives.any()):
        negatives = selected & ~foreground
    if not bool(negatives.any()):
        raise ValueError("Localized presence loss requires background cells.")

    positive_logits = logits[positives]
    negative_losses = F.softplus(logits[negatives])
    keep = min(
        int(negative_losses.numel()),
        max(1, int(positive_logits.numel()) * config.hard_negative_ratio),
    )
    hard_indices = torch.topk(negative_losses, k=keep, largest=True, sorted=False).indices
    hard_negative_logits = logits[negatives][hard_indices]
    positive_loss = F.softplus(-positive_logits).mean()
    negative_loss = F.softplus(hard_negative_logits).mean()
    if bool(halo.any()):
        halo_loss = F.binary_cross_entropy_with_logits(
            logits[halo], torch.full_like(logits[halo], config.halo_target)
        )
        presence = (positive_loss + negative_loss + 0.35 * halo_loss) / 2.35
    else:
        halo_loss = logits.sum() * 0.0
        presence = 0.5 * (positive_loss + negative_loss)
    ranking = F.softplus(
        config.ranking_margin
        + hard_negative_logits[None, :]
        - positive_logits[:, None]
    ).mean()
    return presence, ranking, {
        "positive_cells": int(positive_logits.numel()),
        "halo_cells": int(halo.sum().item()),
        "available_negative_cells": int(negatives.sum().item()),
        "hard_negative_cells": keep,
        "halo_loss": float(halo_loss.detach().item()),
    }


def _foreground_type_loss(
    type_logits: torch.Tensor,
    legal_types: torch.Tensor,
    target: torch.Tensor,
    selected_foreground: torch.Tensor,
) -> torch.Tensor:
    if not bool(selected_foreground.any()):
        raise ValueError("Foreground type loss requires masked foreground cells.")
    bounded = type_logits.masked_fill(~legal_types, ILLEGAL_LOGIT)
    foreground_target = target - 1
    weights = class_balanced_weights(
        foreground_target,
        selected_foreground,
        type_logits.shape[1],
        empty_class=None,
    )
    return F.cross_entropy(
        bounded.permute(0, 2, 3, 1)[selected_foreground],
        foreground_target[selected_foreground],
        weight=weights,
    )


def _independent_count_loss(
    predicted_log1p: torch.Tensor,
    foreground: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if predicted_log1p.ndim != 1 or predicted_log1p.shape[0] != foreground.shape[0]:
        raise ValueError("Count prediction must have shape [B].")
    target_count = (foreground & eligible).sum(dim=(1, 2)).to(predicted_log1p.dtype)
    target_log1p = torch.log1p(target_count)
    loss = F.smooth_l1_loss(predicted_log1p, target_log1p)
    predicted_count = torch.expm1(predicted_log1p)
    return loss, {
        "predicted_count": float(predicted_count.detach().sum().item()),
        "target_count": float(target_count.detach().sum().item()),
        "absolute_count_error": float((predicted_count.detach() - target_count).abs().sum().item()),
    }


def sparse_locator_refinement_loss(
    output: SparseLocatorOutput,
    targets: Mapping[str, torch.Tensor],
    masked: Mapping[str, torch.Tensor],
    legal_masks: Mapping[str, torch.Tensor],
    valid_cells: torch.Tensor,
    hard_empty: torch.Tensor,
    *,
    config: LocatorLossConfig = LocatorLossConfig(),
) -> tuple[torch.Tensor, dict[str, float | int], dict[str, torch.Tensor]]:
    legal = TorchLegalMasks(hard_empty=hard_empty, **legal_masks)
    assert_selected_legal(targets, legal)
    bounded = mask_head_logits(output.as_head_logits(), legal)
    components: dict[str, torch.Tensor] = {}
    details: dict[str, float | int] = {}

    for name in ("variant", "emission"):
        selected = masked[name] & valid_cells
        components[name] = _categorical_loss(
            getattr(bounded, name),
            targets[name],
            selected,
            classes=HEAD_CLASS_COUNTS[name],
            empty_class=None if name == "variant" else 0,
        )
        details[f"{name}_categorical"] = float(components[name].detach().item())

    for name in OBJECT_HEADS:
        selected = masked[name] & valid_cells
        eligible = legal_masks[name][:, 1:].any(dim=1) & valid_cells & ~hard_empty
        foreground = targets[name] != 0
        presence, ranking, localization = _localized_presence_loss(
            output.presence_logits[name], foreground, selected & eligible, config=config
        )
        type_loss = _foreground_type_loss(
            output.type_logits[name],
            legal_masks[name][:, 1:],
            targets[name],
            selected & eligible & foreground,
        )
        categorical = _categorical_loss(
            getattr(bounded, name),
            targets[name],
            selected,
            classes=HEAD_CLASS_COUNTS[name],
            empty_class=0,
        )
        count, count_details = _independent_count_loss(
            output.log1p_counts[name], foreground, eligible
        )
        components[f"{name}_presence"] = presence
        components[f"{name}_ranking"] = ranking
        components[f"{name}_type"] = type_loss
        components[f"{name}_categorical"] = categorical
        components[f"{name}_count"] = count
        for key in ("presence", "ranking", "type", "categorical", "count"):
            details[f"{name}_{key}"] = float(components[f"{name}_{key}"].detach().item())
        for key, value in localization.items():
            details[f"{name}_{key}"] = value
        for key, value in count_details.items():
            details[f"{name}_{key}"] = value

    overlap_selected = valid_cells & (masked["decal"] | masked["prop"])
    if not bool(overlap_selected.any()):
        raise ValueError("Object exclusivity loss requires selected cells.")
    overlap_probability = torch.sigmoid(output.presence_logits["decal"]) * torch.sigmoid(
        output.presence_logits["prop"]
    )
    exclusivity = overlap_probability[overlap_selected].mean()
    components["object_exclusivity"] = exclusivity
    details["object_exclusivity"] = float(exclusivity.detach().item())

    weighted = (
        config.variant_weight * components["variant"]
        + config.emission_weight * components["emission"]
        + config.object_exclusivity_weight * exclusivity
    )
    denominator = config.variant_weight + config.emission_weight + config.object_exclusivity_weight
    for name in OBJECT_HEADS:
        weighted = (
            weighted
            + config.object_presence_weight * components[f"{name}_presence"]
            + config.object_ranking_weight * components[f"{name}_ranking"]
            + config.object_type_weight * components[f"{name}_type"]
            + config.object_categorical_weight * components[f"{name}_categorical"]
            + config.object_count_weight * components[f"{name}_count"]
        )
        denominator += (
            config.object_presence_weight
            + config.object_ranking_weight
            + config.object_type_weight
            + config.object_categorical_weight
            + config.object_count_weight
        )
    total = weighted / denominator
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Sparse locator loss became non-finite.")
    predictions = select_sparse_locator_argmax(output, legal)
    details["total"] = float(total.detach().item())
    return total, details, predictions
