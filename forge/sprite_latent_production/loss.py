from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from ..sprite_latent.codec import (
    EMISSION_COUNT,
    MATERIAL_COUNT,
    PART_COUNT,
    CodecOutput,
    SpriteLatentConfig,
    legal_tuple_scores,
)


def _flattened_cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
) -> Tensor:
    """Cross entropy through the deterministic 2-D CUDA NLL kernel.

    PyTorch's N,C,H,W CUDA NLL implementation is explicitly nondeterministic.
    Flattening aligned pixels to N*H*W,C is mathematically equivalent and uses
    the deterministic matrix-classification implementation.
    """

    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("production logits and categorical target are not aligned")
    classes = int(logits.shape[1])
    flattened_logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, classes)
    return F.cross_entropy(flattened_logits, target.contiguous().view(-1), weight=weight)


def _target_legal_indices(part: Tensor, material: Tensor, emission: Tensor, legal_tuples: Tensor) -> Tensor:
    if part.shape != material.shape or part.shape != emission.shape or part.ndim != 3:
        raise ValueError("production target categorical fields are not aligned")
    values = legal_tuples.long().to(part.device)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("production legal tuple table must have shape K,3")
    codes = values[:, 0] * MATERIAL_COUNT * EMISSION_COUNT + values[:, 1] * EMISSION_COUNT + values[:, 2]
    if len(torch.unique(codes)) != len(values) or not bool(torch.all(codes[1:] > codes[:-1])):
        raise ValueError("production legal tuple table must be unique and canonically sorted")
    lookup = torch.full((PART_COUNT * MATERIAL_COUNT * EMISSION_COUNT,), -1, dtype=torch.long, device=part.device)
    lookup[codes] = torch.arange(len(values), device=part.device)
    target_codes = part * MATERIAL_COUNT * EMISSION_COUNT + material * EMISSION_COUNT + emission
    targets = lookup[target_codes]
    if bool(torch.any(targets < 0)):
        raise ValueError("production target fields contain tuples outside the legal table")
    return targets


def deterministic_sprite_codec_loss(
    output: CodecOutput,
    part: Tensor,
    material: Tensor,
    emission: Tensor,
    legal_tuples: Tensor,
    *,
    config: SpriteLatentConfig,
    class_weights: Mapping[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = class_weights or {}
    part_loss = _flattened_cross_entropy(output.part_logits, part, weight=weights.get("part"))
    material_loss = _flattened_cross_entropy(output.material_logits, material, weight=weights.get("material"))
    emission_loss = _flattened_cross_entropy(output.emission_logits, emission, weight=weights.get("emission"))
    tuple_scores = legal_tuple_scores(output, legal_tuples)
    tuple_targets = _target_legal_indices(part, material, emission, legal_tuples)
    tuple_loss = F.cross_entropy(tuple_scores.contiguous().view(-1, tuple_scores.shape[-1]), tuple_targets.contiguous().view(-1))
    predicted_foreground = 1.0 - output.part_logits.softmax(dim=1)[:, 0]
    target_foreground = (part != 0).float()
    intersection = (predicted_foreground * target_foreground).sum(dim=(1, 2))
    denominator = predicted_foreground.sum(dim=(1, 2)) + target_foreground.sum(dim=(1, 2))
    foreground_dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    usage_penalty = 1.0 - output.soft_marginal_entropy
    reconstruction = part_loss + material_loss + emission_loss
    total = (
        reconstruction
        + config.joint_tuple_weight * tuple_loss
        + config.foreground_dice_weight * foreground_dice
        + (config.latent_usage_weight * usage_penalty if output.quantized else 0.0)
    )
    return total, {
        "loss": total.detach(),
        "reconstruction": reconstruction.detach(),
        "part_ce": part_loss.detach(),
        "material_ce": material_loss.detach(),
        "emission_ce": emission_loss.detach(),
        "joint_tuple_ce": tuple_loss.detach(),
        "foreground_dice_loss": foreground_dice.detach(),
        "usage_penalty": usage_penalty.detach(),
        "perplexity": output.perplexity.detach(),
        "utilization": output.utilization.detach(),
        "marginal_entropy": output.marginal_entropy.detach(),
        "soft_marginal_entropy": output.soft_marginal_entropy.detach(),
    }
