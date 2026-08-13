from __future__ import annotations

import torch

from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.legality import (
    ILLEGAL_LOGIT,
    TorchLegalMasks,
    apply_legal_mask,
    assert_selected_legal,
)
from .model import FactoredDecoratorOutput, OBJECT_HEADS


def select_factored_legal_argmax(
    output: FactoredDecoratorOutput,
    legal: TorchLegalMasks,
) -> dict[str, torch.Tensor]:
    """Decode calibrated sparse counts, then legal types, without overlap.

    The loss calibrates the sum of per-cell presence probabilities to the
    expected object count. A fixed 0.5 threshold discards that signal on
    sparse maps, where every individual probability may correctly be small.
    We round the summed legal probability into a deterministic per-head quota,
    take the strongest cells, resolve cross-head collisions by confidence,
    backfill without overlap, and only then select each legal foreground type.
    """

    categorical = output.as_head_logits()
    selected: dict[str, torch.Tensor] = {
        name: torch.argmax(
            apply_legal_mask(getattr(categorical, name), getattr(legal, name), name),
            dim=1,
        )
        for name in ("variant", "emission")
    }

    presence_scores: dict[str, torch.Tensor] = {}
    eligible_masks: dict[str, torch.Tensor] = {}
    type_choices: dict[str, torch.Tensor] = {}
    for name in OBJECT_HEADS:
        type_legal = getattr(legal, name)[:, 1:]
        eligible = type_legal.any(dim=1)
        bounded_type = output.type_logits[name].masked_fill(~type_legal, ILLEGAL_LOGIT)
        type_choices[name] = torch.argmax(bounded_type, dim=1) + 1
        eligible_masks[name] = eligible
        presence_scores[name] = output.presence_logits[name].masked_fill(~eligible, ILLEGAL_LOGIT)

    batch, height, width = presence_scores[OBJECT_HEADS[0]].shape
    object_cells = {
        name: torch.zeros(
            (batch, height, width), dtype=torch.bool, device=presence_scores[name].device
        )
        for name in OBJECT_HEADS
    }
    quotas: dict[str, list[int]] = {name: [] for name in OBJECT_HEADS}
    for name in OBJECT_HEADS:
        probability = torch.sigmoid(output.presence_logits[name]) * eligible_masks[name]
        eligible_count = eligible_masks[name].flatten(1).sum(dim=1)
        expected_count = torch.round(probability.flatten(1).sum(dim=1)).to(torch.long)
        expected_count = torch.minimum(expected_count, eligible_count)
        quotas[name] = [int(value) for value in expected_count.detach().cpu()]
        flat_score = presence_scores[name].flatten(1)
        flat_eligible = eligible_masks[name].flatten(1)
        flat_selected = object_cells[name].flatten(1)
        for batch_index, quota in enumerate(quotas[name]):
            if not quota:
                continue
            candidates = torch.nonzero(flat_eligible[batch_index], as_tuple=False).squeeze(1)
            strongest = torch.topk(
                flat_score[batch_index, candidates], k=quota, largest=True, sorted=False
            ).indices
            flat_selected[batch_index, candidates[strongest]] = True

    decal, prop = OBJECT_HEADS
    collisions = object_cells[decal] & object_cells[prop]
    decal_wins = presence_scores[decal] >= presence_scores[prop]
    object_cells[decal] &= ~collisions | decal_wins
    object_cells[prop] &= ~collisions | ~decal_wins

    # Preserve both quotas where mutually exclusive legal cells exist.
    for name, other in ((decal, prop), (prop, decal)):
        flat_score = presence_scores[name].flatten(1)
        flat_eligible = eligible_masks[name].flatten(1)
        flat_selected = object_cells[name].flatten(1)
        flat_occupied = object_cells[other].flatten(1)
        for batch_index, quota in enumerate(quotas[name]):
            shortfall = quota - int(flat_selected[batch_index].sum().item())
            if shortfall <= 0:
                continue
            available = flat_eligible[batch_index] & ~flat_selected[batch_index] & ~flat_occupied[batch_index]
            candidates = torch.nonzero(available, as_tuple=False).squeeze(1)
            keep = min(shortfall, int(candidates.numel()))
            if keep:
                strongest = torch.topk(
                    flat_score[batch_index, candidates], k=keep, largest=True, sorted=False
                ).indices
                flat_selected[batch_index, candidates[strongest]] = True

    for name in OBJECT_HEADS:
        selected[name] = torch.where(
            object_cells[name], type_choices[name], torch.zeros_like(type_choices[name])
        )

    ordered = {name: selected[name] for name in HEAD_NAMES}
    assert_selected_legal(ordered, legal)
    return ordered
