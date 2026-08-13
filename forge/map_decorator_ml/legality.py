from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import torch

from ..map_decorator.catalog import LegalClassMasks
from .contract import HEAD_CLASS_COUNTS, HEAD_NAMES
from .model import HeadLogits


ILLEGAL_LOGIT = -1.0e9


@dataclass(slots=True)
class TorchLegalMasks:
    variant: torch.Tensor
    decal: torch.Tensor
    prop: torch.Tensor
    emission: torch.Tensor
    hard_empty: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in HEAD_NAMES}


def legal_masks_to_torch(
    masks: LegalClassMasks,
    *,
    device: torch.device | str = "cpu",
    add_batch: bool = True,
) -> TorchLegalMasks:
    converted: dict[str, torch.Tensor] = {}
    for name, classes in HEAD_CLASS_COUNTS.items():
        array = getattr(masks, name)
        if array.shape[0] != classes or array.dtype != np.bool_:
            raise TypeError(f"Legal mask {name!r} violates the categorical contract.")
        tensor = torch.from_numpy(np.asarray(array).copy()).to(device=device, dtype=torch.bool)
        converted[name] = tensor[None] if add_batch else tensor
    hard_empty = torch.from_numpy(np.asarray(masks.hard_empty).copy()).to(
        device=device, dtype=torch.bool
    )
    if add_batch:
        hard_empty = hard_empty[None]
    return TorchLegalMasks(hard_empty=hard_empty, **converted)


def _validate_pair(name: str, logits: torch.Tensor, legal: torch.Tensor) -> None:
    classes = HEAD_CLASS_COUNTS[name]
    if logits.ndim != 4 or logits.shape[1] != classes:
        raise ValueError(f"{name} logits must have shape [B,{classes},H,W].")
    if legal.shape != logits.shape or legal.dtype != torch.bool:
        raise TypeError(f"{name} legal mask must be bool and exactly match logits.")
    if not bool(legal.any(dim=1).all()):
        raise ValueError(f"{name} legal mask leaves at least one cell without any class.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} logits contain a non-finite value.")


def apply_legal_mask(logits: torch.Tensor, legal: torch.Tensor, name: str) -> torch.Tensor:
    """Mask before prediction/sampling; callers never receive an unbounded choice."""
    _validate_pair(name, logits, legal)
    return logits.masked_fill(~legal, ILLEGAL_LOGIT)


def mask_head_logits(raw: HeadLogits, legal: TorchLegalMasks) -> HeadLogits:
    return HeadLogits(
        **{
            name: apply_legal_mask(getattr(raw, name), getattr(legal, name), name)
            for name in HEAD_NAMES
        }
    )


def select_legal_argmax(raw: HeadLogits, legal: TorchLegalMasks) -> dict[str, torch.Tensor]:
    """Decode deterministic legal predictions, including one joint object choice."""
    bounded = mask_head_logits(raw, legal)
    variant = torch.argmax(bounded.variant, dim=1)
    emission = torch.argmax(bounded.emission, dim=1)
    empty = torch.logsumexp(
        torch.stack((bounded.decal[:, 0], bounded.prop[:, 0]), dim=1), dim=1
    ) - torch.log(torch.tensor(2.0, device=bounded.decal.device))
    object_logits = torch.cat((empty[:, None], bounded.decal[:, 1:], bounded.prop[:, 1:]), dim=1)
    object_legal = torch.cat(
        (
            legal.decal[:, :1] & legal.prop[:, :1],
            legal.decal[:, 1:],
            legal.prop[:, 1:],
        ),
        dim=1,
    )
    object_logits = object_logits.masked_fill(~object_legal, ILLEGAL_LOGIT)
    joint = torch.argmax(object_logits, dim=1)
    decal_nonempty = HEAD_CLASS_COUNTS["decal"] - 1
    decal = torch.where(
        (joint >= 1) & (joint <= decal_nonempty), joint, torch.zeros_like(joint)
    )
    prop = torch.where(
        joint > decal_nonempty, joint - decal_nonempty, torch.zeros_like(joint)
    )
    selected = {"variant": variant, "decal": decal, "prop": prop, "emission": emission}
    assert_selected_legal(selected, legal)
    return selected


def assert_selected_legal(
    selected: Mapping[str, torch.Tensor], legal: TorchLegalMasks
) -> None:
    for name in HEAD_NAMES:
        field = selected[name]
        mask = getattr(legal, name)
        if field.dtype != torch.long or field.shape != (mask.shape[0], *mask.shape[-2:]):
            raise TypeError(f"Selected {name!r} must be int64 [B,H,W].")
        chosen = mask.gather(1, field[:, None]).squeeze(1)
        if not bool(chosen.all()):
            raise ValueError(f"Selected {name!r} contains an illegal class.")
    if bool(((selected["decal"] != 0) & (selected["prop"] != 0)).any()):
        raise ValueError("A cell may contain at most one decal or prop class.")
