from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor

from .contract import MASK_TOKEN, MaskedPriorConfig


MASK_MODES = ("random", "rectangle", "half_plane", "corridor")


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(str(list(value.shape)).encode("ascii") + b"\0")
    digest.update(memoryview(array))
    return digest.hexdigest()


def _ensure_nontrivial(mask: Tensor, valid: Tensor) -> Tensor:
    flat_valid = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    if not flat_valid.numel():
        raise ValueError("Masked-prior sample has no valid latent cells.")
    flat = mask.flatten()
    if not bool(flat[flat_valid].any()):
        flat[flat_valid[0]] = True
    if flat_valid.numel() > 1 and bool(flat[flat_valid].all()):
        flat[flat_valid[-1]] = False
    return flat.view_as(mask)


def mask_tokens(
    targets: Tensor,
    valid_mask: Tensor,
    *,
    generator: torch.Generator,
    config: MaskedPriorConfig,
    step: int,
) -> dict[str, Any]:
    if targets.ndim != 3 or targets.dtype != torch.long:
        raise ValueError("Masked-prior targets must be int64 B,H,W.")
    if valid_mask.shape != (targets.shape[0], 1, targets.shape[1], targets.shape[2]) or valid_mask.dtype != torch.bool:
        raise ValueError("Masked-prior valid mask disagrees with targets.")
    masks = torch.zeros_like(targets, dtype=torch.bool)
    fractions: list[float] = []
    modes: list[str] = []
    height, width = targets.shape[-2:]
    for batch_index in range(targets.shape[0]):
        valid = valid_mask[batch_index, 0]
        ratio = config.minimum_mask_fraction + (
            config.maximum_mask_fraction - config.minimum_mask_fraction
        ) * float(torch.rand((), generator=generator))
        mode = MASK_MODES[(step + batch_index) % len(MASK_MODES)]
        if mode == "random":
            mask = torch.rand((height, width), generator=generator) < ratio
        elif mode == "rectangle":
            rect_h = max(1, min(height, int(round(height * math.sqrt(ratio)))))
            rect_w = max(1, min(width, int(round(width * math.sqrt(ratio)))))
            top = int(torch.randint(0, height - rect_h + 1, (), generator=generator))
            left = int(torch.randint(0, width - rect_w + 1, (), generator=generator))
            mask = torch.zeros((height, width), dtype=torch.bool)
            mask[top : top + rect_h, left : left + rect_w] = True
        elif mode == "half_plane":
            mask = torch.zeros((height, width), dtype=torch.bool)
            if (step + batch_index) % 2:
                span = max(1, min(width, int(round(width * ratio))))
                mask[:, :span] = True
            else:
                span = max(1, min(height, int(round(height * ratio))))
                mask[-span:, :] = True
        else:
            mask = torch.zeros((height, width), dtype=torch.bool)
            stripe = max(1, int(round(min(height, width) * max(0.12, ratio * 0.35))))
            if (step + batch_index) % 2:
                center = int(torch.randint(0, width, (), generator=generator))
                mask[:, max(0, center - stripe) : min(width, center + stripe + 1)] = True
            else:
                center = int(torch.randint(0, height, (), generator=generator))
                mask[max(0, center - stripe) : min(height, center + stripe + 1), :] = True
        mask = _ensure_nontrivial(mask & valid, valid)
        masks[batch_index] = mask
        fractions.append(float(mask.sum()) / float(valid.sum()))
        modes.append(mode)
    corrupted = targets.clone()
    corrupted[masks] = MASK_TOKEN
    corrupted[~valid_mask[:, 0]] = 0
    return {
        "tokens": corrupted,
        "mask": masks,
        "mask_fraction": torch.tensor(fractions, dtype=torch.float32).unsqueeze(1),
        "modes": modes,
    }


def sample_parallel(
    model: torch.nn.Module,
    conditions: dict[str, Tensor],
    *,
    sampling_steps: int,
) -> dict[str, Any]:
    valid = conditions["valid_mask"][:, 0]
    tokens = torch.zeros(valid.shape, dtype=torch.long)
    tokens[valid] = MASK_TOKEN
    trace: list[str] = []
    final_uncertainty = torch.ones(valid.shape, dtype=torch.float32)
    with torch.inference_mode():
        for iteration in range(sampling_steps):
            remaining = tokens == MASK_TOKEN
            fractions = remaining.sum(dim=(1, 2)).to(torch.float32) / valid.sum(dim=(1, 2)).clamp_min(1)
            logits = model({**conditions, "tokens": tokens, "mask_fraction": fractions[:, None]})
            probabilities = torch.softmax(logits.float(), dim=1)
            confidence, proposed = probabilities.max(dim=1)
            for batch_index in range(tokens.shape[0]):
                candidates = torch.nonzero(remaining[batch_index].flatten(), as_tuple=False).flatten()
                if not candidates.numel():
                    continue
                total = int(valid[batch_index].sum())
                target_revealed = math.ceil(total * (iteration + 1) / sampling_steps)
                already_revealed = total - int(candidates.numel())
                reveal = max(1, min(int(candidates.numel()), target_revealed - already_revealed))
                scores = confidence[batch_index].flatten().index_select(0, candidates)
                order = torch.argsort(scores, descending=True, stable=True)[:reveal]
                chosen = candidates.index_select(0, order)
                flat_tokens = tokens[batch_index].flatten()
                flat_proposed = proposed[batch_index].flatten()
                flat_tokens[chosen] = flat_proposed[chosen]
                final_uncertainty[batch_index].flatten()[chosen] = 1.0 - confidence[batch_index].flatten()[chosen]
            trace.append(tensor_sha256(tokens))
    if bool((tokens[valid] == MASK_TOKEN).any()):
        raise RuntimeError("Masked-prior sampler failed to reveal every valid token.")
    return {"tokens": tokens, "uncertainty": final_uncertainty, "trace_sha256": trace}

