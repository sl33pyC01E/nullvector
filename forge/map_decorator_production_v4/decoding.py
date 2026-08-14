from __future__ import annotations

import torch

from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.legality import ILLEGAL_LOGIT, TorchLegalMasks, apply_legal_mask, assert_selected_legal
from ..map_decorator_production_v2.model import OBJECT_HEADS
from .model import ProposalLocatorOutputV4


def _stable_strongest(score: torch.Tensor, candidates: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or candidates.numel() == 0:
        return candidates[:0]
    keep = min(count, int(candidates.numel()))
    order = torch.argsort(score[candidates], descending=True, stable=True)
    return candidates[order[:keep]]


def proposal_count_quotas(
    output: ProposalLocatorOutputV4,
    eligible: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in OBJECT_HEADS:
        log_count = output.log1p_counts[name]
        mask = eligible[name]
        if log_count.ndim != 1 or log_count.shape[0] != mask.shape[0]:
            raise ValueError("V4 count output does not match its proposal batch.")
        if not bool(torch.isfinite(log_count).all()) or bool((log_count < 0).any()):
            raise ValueError("V4 count output must be finite nonnegative log1p counts.")
        quota = torch.round(torch.expm1(log_count)).to(torch.long)
        quota = quota.clamp(min=0, max=output.maximum_objects_per_head)
        result[name] = torch.minimum(quota, mask.flatten(1).sum(dim=1))
    return result


def select_proposal_conditioned_argmax(
    output: ProposalLocatorOutputV4,
    legal: TorchLegalMasks,
) -> dict[str, torch.Tensor]:
    categorical = output.as_head_logits()
    selected: dict[str, torch.Tensor] = {
        name: torch.argmax(
            apply_legal_mask(getattr(categorical, name), getattr(legal, name), name), dim=1
        )
        for name in ("variant", "emission")
    }
    scores: dict[str, torch.Tensor] = {}
    eligible: dict[str, torch.Tensor] = {}
    type_choice: dict[str, torch.Tensor] = {}
    for name in OBJECT_HEADS:
        legal_type = getattr(legal, name)[:, 1:]
        proposal_type = output.proposals[name]
        typed_eligible = legal_type & proposal_type & ~legal.hard_empty[:, None]
        eligible[name] = typed_eligible.any(dim=1)
        bounded_type = output.type_logits[name].masked_fill(~typed_eligible, ILLEGAL_LOGIT)
        type_choice[name] = torch.argmax(bounded_type, dim=1) + 1
        scores[name] = output.presence_logits[name].masked_fill(~eligible[name], ILLEGAL_LOGIT)
    quotas = proposal_count_quotas(output, eligible)

    batch, height, width = scores[OBJECT_HEADS[0]].shape
    occupied = {
        name: torch.zeros((batch, height, width), dtype=torch.bool, device=scores[name].device)
        for name in OBJECT_HEADS
    }
    for name in OBJECT_HEADS:
        flat_score = scores[name].flatten(1)
        flat_eligible = eligible[name].flatten(1)
        flat_selected = occupied[name].flatten(1)
        for batch_index, quota in enumerate(quotas[name].detach().cpu().tolist()):
            candidates = torch.nonzero(flat_eligible[batch_index], as_tuple=False).squeeze(1)
            chosen = _stable_strongest(flat_score[batch_index], candidates, quota)
            flat_selected[batch_index, chosen] = True

    decal, prop = OBJECT_HEADS
    collisions = occupied[decal] & occupied[prop]
    decal_wins = scores[decal] >= scores[prop]
    occupied[decal] &= ~collisions | decal_wins
    occupied[prop] &= ~collisions | ~decal_wins
    for name, other in ((decal, prop), (prop, decal)):
        flat_score = scores[name].flatten(1)
        flat_eligible = eligible[name].flatten(1)
        flat_selected = occupied[name].flatten(1)
        flat_other = occupied[other].flatten(1)
        for batch_index, quota in enumerate(quotas[name].detach().cpu().tolist()):
            shortfall = quota - int(flat_selected[batch_index].sum().item())
            if shortfall <= 0:
                continue
            candidates = torch.nonzero(
                flat_eligible[batch_index] & ~flat_selected[batch_index] & ~flat_other[batch_index],
                as_tuple=False,
            ).squeeze(1)
            flat_selected[batch_index, _stable_strongest(flat_score[batch_index], candidates, shortfall)] = True

    for name in OBJECT_HEADS:
        selected[name] = torch.where(occupied[name], type_choice[name], torch.zeros_like(type_choice[name]))
        proposed = output.proposals[name].gather(1, (selected[name] - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        if bool(((selected[name] != 0) & ~proposed).any()):
            raise RuntimeError("V4 decoder selected an off-proposal object cell.")
    ordered = {name: selected[name] for name in HEAD_NAMES}
    assert_selected_legal(ordered, legal)
    return ordered
