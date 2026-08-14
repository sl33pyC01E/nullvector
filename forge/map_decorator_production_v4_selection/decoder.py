from __future__ import annotations

from typing import Any

import torch

from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.legality import TorchLegalMasks, assert_selected_legal
from ..map_decorator_production_v4.decoding import select_proposal_conditioned_argmax
from ..map_decorator_production_v4.model import ProposalLocatorOutputV4
from .contract import ProtectedSelectionConfig


def apply_protected_proposals(
    selected: dict[str, torch.Tensor],
    output: ProposalLocatorOutputV4,
    legal: TorchLegalMasks,
    *,
    config: ProtectedSelectionConfig = ProtectedSelectionConfig(),
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if set(selected) != set(HEAD_NAMES):
        raise ValueError("Protected selection requires the complete head registry.")
    result = {name: value.clone() for name, value in selected.items()}
    restored: dict[str, dict[str, int]] = {"decal": {}, "prop": {}}
    for head, other, classes in (
        ("decal", "prop", config.decal_classes),
        ("prop", "decal", config.prop_classes),
    ):
        for class_id in classes:
            proposal = output.proposals[head][:, class_id - 1]
            class_legal = getattr(legal, head)[:, class_id]
            eligible = proposal & class_legal & ~legal.hard_empty
            restore = eligible & (result[head] == 0) & (result[other] == 0)
            result[head] = torch.where(restore, torch.full_like(result[head], class_id), result[head])
            restored[head][str(class_id)] = int(restore.sum().item())
    if bool(((result["decal"] != 0) & (result["prop"] != 0)).any()):
        raise RuntimeError("Protected selection introduced a decal/prop collision.")
    for head in ("decal", "prop"):
        proposed = output.proposals[head].gather(
            1, (result[head] - 1).clamp(min=0).unsqueeze(1)
        ).squeeze(1)
        if bool(((result[head] != 0) & ~proposed).any()):
            raise RuntimeError("Protected selection introduced an off-proposal object.")
    ordered = {name: result[name] for name in HEAD_NAMES}
    assert_selected_legal(ordered, legal)
    return ordered, {
        "restored": restored,
        "total_restored": sum(sum(values.values()) for values in restored.values()),
    }


def select_protected_proposal_argmax(
    output: ProposalLocatorOutputV4,
    legal: TorchLegalMasks,
    *,
    config: ProtectedSelectionConfig = ProtectedSelectionConfig(),
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    selected = select_proposal_conditioned_argmax(output, legal)
    return apply_protected_proposals(selected, output, legal, config=config)
