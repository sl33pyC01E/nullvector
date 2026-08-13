from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch

from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_ml.training import EMA, corrupt_targets
from .contract import FactoredLossConfig, V2TrainingConfig
from .loss import factored_refinement_loss
from .model import FactoredDecoratorV2


def make_optimizer(
    model: FactoredDecoratorV2,
    config: V2TrainingConfig = V2TrainingConfig(),
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_batch_v2(
    model: FactoredDecoratorV2,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    batch: Mapping[str, object],
    *,
    generator: torch.Generator,
    training_config: V2TrainingConfig = V2TrainingConfig(),
    loss_config: FactoredLossConfig = FactoredLossConfig(),
    device: torch.device | str = "cpu",
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    device = torch.device(device)
    if autocast_dtype is not None and device.type != "cuda":
        raise ValueError("Mixed precision is restricted to the eventual CUDA calibration path.")
    features = batch["features"].to(device)  # type: ignore[union-attr]
    targets = {name: batch["targets"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    legal_masks = {
        name: batch["legal_masks"][name].to(device) for name in HEAD_NAMES  # type: ignore[index,union-attr]
    }
    valid = batch["valid_cells"].to(device)  # type: ignore[union-attr]
    hard_empty = batch["hard_empty"].to(device)  # type: ignore[union-attr]
    theme = batch["theme_index"].to(device)  # type: ignore[union-attr]
    conditions = batch["global_conditions"].to(device)  # type: ignore[union-attr]
    masked, probability = corrupt_targets(
        targets,
        valid,
        generator=generator,
        minimum=training_config.corruption_min,
        maximum=training_config.corruption_max,
    )
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
        )
        loss, losses, predictions = factored_refinement_loss(
            output,
            targets,
            masked,
            legal_masks,
            valid,
            hard_empty,
            config=loss_config,
        )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("V2 train batch produced a non-finite loss.")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0, error_if_nonfinite=True
    )
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"Optimizer produced a non-finite parameter {name!r}.")
    ema.update(model)
    metrics = decoration_metrics(predictions, targets, valid)
    return {
        "loss": losses,
        "gradient_norm": float(gradient_norm.detach().item()),
        "metrics": metrics,
        "corruption_fraction": [float(value) for value in probability.detach().cpu()],
    }
