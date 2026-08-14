from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from ..map_decorator_ml.contract import HEAD_CLASS_COUNTS
from ..map_decorator_ml.training import class_balanced_weights
from ..map_decorator_production_v2.model import OBJECT_HEADS
from ..map_decorator_production_v4.model import ProposalLocatorOutputV4
from .contract import ResidualLossConfig


def _categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
    *,
    classes: int,
    empty_class: int | None,
) -> torch.Tensor:
    if not bool(selected.any()):
        raise ValueError("V4 categorical loss requires selected cells.")
    weights = class_balanced_weights(target, selected, classes, empty_class=empty_class)
    per_cell = F.cross_entropy(logits, target, weight=weights, reduction="none")
    return per_cell[selected].mean()


def proposal_residual_loss(
    output: ProposalLocatorOutputV4,
    targets: Mapping[str, torch.Tensor],
    masked: Mapping[str, torch.Tensor],
    valid: torch.Tensor,
    *,
    config: ResidualLossConfig = ResidualLossConfig(),
    candidate_logit_prior: float = 4.0,
    noncandidate_logit_prior: float = -8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    components: dict[str, torch.Tensor] = {}
    for name in ("variant", "emission"):
        selected = masked[name] & valid
        components[name] = _categorical_loss(
            getattr(output, name),
            targets[name],
            selected,
            classes=HEAD_CLASS_COUNTS[name],
            empty_class=None if name == "variant" else 0,
        )
    for name in OBJECT_HEADS:
        proposal = output.proposals[name].any(dim=1) & valid
        foreground = (targets[name] != 0) & valid
        if not bool(foreground.any()) or not bool(proposal.any()):
            raise ValueError("V4 object residual loss requires proposed foreground.")
        selected = proposal & masked[name]
        if not bool(selected.any()):
            selected = proposal
        binary_target = foreground[selected].to(output.presence_logits[name].dtype)
        weights = torch.where(
            binary_target > 0,
            torch.ones_like(binary_target),
            torch.full_like(binary_target, config.extra_proposal_weight),
        )
        components[f"{name}_presence"] = F.binary_cross_entropy_with_logits(
            output.presence_logits[name][selected], binary_target, weight=weights
        )
        type_selected = foreground & masked[name]
        if not bool(type_selected.any()):
            type_selected = foreground
        components[f"{name}_type"] = F.cross_entropy(
            output.type_logits[name].permute(0, 2, 3, 1)[type_selected],
            (targets[name] - 1)[type_selected],
        )
        target_count = foreground.sum(dim=(1, 2)).to(output.log1p_counts[name].dtype)
        components[f"{name}_count"] = F.smooth_l1_loss(
            output.log1p_counts[name], torch.log1p(target_count)
        )
        prior = torch.where(
            proposal,
            torch.full_like(output.presence_logits[name], candidate_logit_prior),
            torch.full_like(output.presence_logits[name], noncandidate_logit_prior),
        )
        components[f"{name}_residual"] = ((output.presence_logits[name] - prior)[valid] ** 2).mean()
    total = config.variant_weight * components["variant"] + config.emission_weight * components["emission"]
    denominator = config.variant_weight + config.emission_weight
    for name in OBJECT_HEADS:
        total = (
            total
            + config.proposal_presence_weight * components[f"{name}_presence"]
            + config.proposal_type_weight * components[f"{name}_type"]
            + config.proposal_count_weight * components[f"{name}_count"]
            + config.residual_regularization_weight * components[f"{name}_residual"]
        )
        denominator += (
            config.proposal_presence_weight
            + config.proposal_type_weight
            + config.proposal_count_weight
            + config.residual_regularization_weight
        )
    total = total / denominator
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("V4 proposal residual loss became non-finite.")
    details = {name: float(value.detach().item()) for name, value in components.items()}
    details["total"] = float(total.detach().item())
    return total, details
