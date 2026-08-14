from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch

from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.training import corrupt_targets
from ..map_decorator_production_v2.training import WarmStartEMA
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from .contract import ResidualLossConfig, ResidualTrainingConfig
from .loss import proposal_residual_loss


def make_optimizer(
    model: ProposalConditionedDecoratorV4,
    config: ResidualTrainingConfig = ResidualTrainingConfig(),
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("V4 residual optimizer requires at least one trainable parameter.")
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def train_batch(
    model: ProposalConditionedDecoratorV4,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    batch: Mapping[str, object],
    *,
    generator: torch.Generator,
    training_config: ResidualTrainingConfig = ResidualTrainingConfig(),
    loss_config: ResidualLossConfig = ResidualLossConfig(),
    device: torch.device | str = "cpu",
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("V4 residual training supports only CPU or explicit CUDA calibration.")
    if device.type == "cuda" and autocast_dtype is not torch.bfloat16:
        raise ValueError("V4 CUDA residual training is restricted to BF16 calibration.")
    if autocast_dtype is not None and device.type != "cuda":
        raise ValueError("V4 mixed precision is restricted to CUDA calibration.")
    features = batch["features"].to(device)  # type: ignore[union-attr]
    targets = {name: batch["targets"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    valid = batch["valid_cells"].to(device)  # type: ignore[union-attr]
    theme = batch["theme_index"].to(device)  # type: ignore[union-attr]
    conditions = batch["global_conditions"].to(device)  # type: ignore[union-attr]
    proposals = {name: batch["proposals"][name].to(device) for name in ("decal", "prop")}  # type: ignore[index,union-attr]
    masked, probability = corrupt_targets(
        targets,
        valid,
        generator=generator,
        minimum=training_config.corruption_min,
        maximum=training_config.corruption_max,
    )
    full_slots = list(range(0, features.shape[0], training_config.full_mask_stride))
    for index in full_slots:
        for name in HEAD_NAMES:
            masked[name][index] = valid[index]
        probability[index] = 1.0
    optimizer.zero_grad(set_to_none=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with autocast:
        output = model(
            features,
            targets,
            masked,
            theme,
            conditions,
            probability.to(torch.float32),
            proposals,
        )
        loss, details = proposal_residual_loss(
            output,
            targets,
            masked,
            valid,
            config=loss_config,
            candidate_logit_prior=model.locator_config.candidate_logit_prior,
            noncandidate_logit_prior=model.locator_config.noncandidate_logit_prior,
        )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("V4 residual optimizer produced non-finite parameters.")
    ema.update(model)
    return {
        "loss": details,
        "gradient_norm": float(gradient_norm.detach().item()),
        "full_mask_sample_count": len(full_slots),
        "corruption_fraction": [float(value) for value in probability.detach().cpu()],
    }
