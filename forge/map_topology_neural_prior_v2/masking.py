from __future__ import annotations

import math
from typing import Any, Final

import torch
from torch import Tensor

from .contract import MASK_TOKEN, PriorV2Config


MASK_MODES_V2: Final[tuple[str, ...]] = (
    "full", "high_random", "rectangle", "half_plane", "corridor", "islands",
)


def mask_tokens_v2(
    targets: Tensor,
    valid_mask: Tensor,
    *,
    generator: torch.Generator,
    config: PriorV2Config,
    step: int,
) -> dict[str, Any]:
    if targets.ndim != 3 or targets.dtype != torch.long:
        raise ValueError("Prior-v2 targets must be int64 B,H,W.")
    if valid_mask.shape != (targets.shape[0], 1, *targets.shape[-2:]) or valid_mask.dtype != torch.bool:
        raise ValueError("Prior-v2 valid mask disagrees with targets.")
    if bool(((targets < 0) | (targets >= MASK_TOKEN)).any()):
        raise ValueError("Prior-v2 targets exceed the codebook vocabulary.")
    masks = torch.zeros_like(targets, dtype=torch.bool)
    modes: list[str] = []
    fractions: list[float] = []
    height, width = targets.shape[-2:]
    for index in range(targets.shape[0]):
        valid = valid_mask[index, 0]
        valid_indices = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
        if not valid_indices.numel():
            raise ValueError("Prior-v2 sample has no valid latent cells.")
        mode = MASK_MODES_V2[(step * targets.shape[0] + index) % len(MASK_MODES_V2)]
        ratio = config.minimum_mask_fraction + (
            config.maximum_mask_fraction - config.minimum_mask_fraction
        ) * float(torch.rand((), generator=generator))
        if mode == "full":
            mask = valid.clone()
        elif mode == "high_random":
            ratio = max(0.80, ratio)
            mask = (torch.rand((height, width), generator=generator) < ratio) & valid
        elif mode == "rectangle":
            rect_h = max(1, min(height, int(round(height * math.sqrt(ratio)))))
            rect_w = max(1, min(width, int(round(width * math.sqrt(ratio)))))
            top = int(torch.randint(0, height - rect_h + 1, (), generator=generator))
            left = int(torch.randint(0, width - rect_w + 1, (), generator=generator))
            mask = torch.zeros((height, width), dtype=torch.bool)
            mask[top:top + rect_h, left:left + rect_w] = True
            mask &= valid
        elif mode == "half_plane":
            mask = torch.zeros((height, width), dtype=torch.bool)
            if (step + index) % 2:
                span = max(1, min(width, int(round(width * ratio))))
                mask[:, :span] = True
            else:
                span = max(1, min(height, int(round(height * ratio))))
                mask[-span:, :] = True
            mask &= valid
        elif mode == "corridor":
            mask = torch.zeros((height, width), dtype=torch.bool)
            stripe = max(1, int(round(min(height, width) * max(0.18, ratio * 0.45))))
            if (step + index) % 2:
                center = int(torch.randint(0, width, (), generator=generator))
                mask[:, max(0, center - stripe):min(width, center + stripe + 1)] = True
            else:
                center = int(torch.randint(0, height, (), generator=generator))
                mask[max(0, center - stripe):min(height, center + stripe + 1), :] = True
            mask &= valid
        else:
            coarse_h = max(2, math.ceil(height / 4))
            coarse_w = max(2, math.ceil(width / 4))
            coarse = torch.rand((coarse_h, coarse_w), generator=generator) < ratio
            mask = torch.nn.functional.interpolate(
                coarse[None, None].float(), size=(height, width), mode="nearest"
            )[0, 0].bool() & valid
        flat = mask.flatten()
        if not bool(flat[valid_indices].any()):
            flat[valid_indices[0]] = True
        if mode != "full" and valid_indices.numel() > 1 and bool(flat[valid_indices].all()):
            flat[valid_indices[-1]] = False
        mask = flat.view_as(mask)
        masks[index] = mask
        modes.append(mode)
        fractions.append(float(mask.sum()) / float(valid.sum()))
    corrupted = targets.clone()
    corrupted[masks] = MASK_TOKEN
    corrupted[~valid_mask[:, 0]] = 0
    return {
        "tokens": corrupted,
        "mask": masks,
        "mask_fraction": torch.tensor(fractions, dtype=torch.float32).unsqueeze(1),
        "modes": modes,
    }
