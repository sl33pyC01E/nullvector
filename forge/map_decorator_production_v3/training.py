from __future__ import annotations

from collections.abc import Mapping

import torch

from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_ml.training import corrupt_targets
from ..map_decorator_production_v2.training import WarmStartEMA
from .contract import LocatorLossConfig, LocatorTrainingConfig
from .loss import sparse_locator_refinement_loss
from .model import SparseLocatorDecoratorV3


def make_optimizer_v3(
    model: SparseLocatorDecoratorV3,
    config: LocatorTrainingConfig = LocatorTrainingConfig(),
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def train_batch_v3(
    model: SparseLocatorDecoratorV3,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    batch: Mapping[str, object],
    *,
    generator: torch.Generator,
    training_config: LocatorTrainingConfig = LocatorTrainingConfig(),
    loss_config: LocatorLossConfig = LocatorLossConfig(),
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    device = torch.device(device)
    if device.type != "cpu":
        raise ValueError("The v3 localization foundation is CPU-only until a new calibration is authorized.")
    features = batch["features"].to(device)  # type: ignore[union-attr]
    targets = {name: batch["targets"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    legal_masks = {name: batch["legal_masks"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
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
    full_mask_slots = list(range(0, features.shape[0], training_config.full_mask_stride))
    for index in full_mask_slots:
        for name in HEAD_NAMES:
            masked[name][index] = valid[index]
        probability[index] = 1.0

    optimizer.zero_grad(set_to_none=True)
    output = model(features, targets, masked, theme, conditions, probability.to(torch.float32))
    loss, details, predictions = sparse_locator_refinement_loss(
        output,
        targets,
        masked,
        legal_masks,
        valid,
        hard_empty,
        config=loss_config,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("V3 CPU training produced a non-finite loss.")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0, error_if_nonfinite=True
    )
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"Optimizer produced a non-finite parameter {name!r}.")
    ema.update(model)
    return {
        "loss": details,
        "gradient_norm": float(gradient_norm.detach().item()),
        "metrics": decoration_metrics(predictions, targets, valid),
        "corruption_fraction": [float(value) for value in probability.detach().cpu()],
        "full_mask_sample_count": len(full_mask_slots),
    }
