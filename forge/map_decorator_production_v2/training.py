from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch

from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_ml.training import EMA, corrupt_targets
from .contract import FactoredLossConfig, V2TrainingConfig
from .decoding import select_factored_legal_argmax
from .loss import factored_refinement_loss
from .model import FactoredDecoratorV2


FULL_MASK_SLOT_STRIDE = 2


class WarmStartEMA:
    """Bias-safe EMA for short calibration and crash-isolated segments.

    A fixed 0.999 decay leaves more than ninety percent of the random initial
    shadow after 100 updates.  That made the calibration gate evaluate an
    untrained model even when the live weights had learned object presence.
    This bounded inverse-time warm start is the same deterministic policy on
    every run and stores its update count for exact checkpoint resume.
    """

    POLICY = "inverse-time-min-base-decay-v1"

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1).")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    def effective_decay(self) -> float:
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        expected = {name for name, value in state.items() if value.is_floating_point()}
        if set(self.shadow) != expected:
            raise ValueError("EMA parameter set drifted from the model.")
        self.updates += 1
        decay = self.effective_decay()
        for name, target in self.shadow.items():
            observed = state[name].detach().to(device="cpu", dtype=torch.float32)
            target.lerp_(observed, 1.0 - decay)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "warmup_policy": self.POLICY,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"decay", "updates", "warmup_policy", "shadow"}:
            raise ValueError("EMA state members violate the v2 contract.")
        if float(state["decay"]) != self.decay or state["warmup_policy"] != self.POLICY:
            raise ValueError("EMA policy does not match the resume contract.")
        updates = state["updates"]
        if type(updates) is not int or updates < 0:
            raise ValueError("EMA update count is invalid.")
        shadow = state["shadow"]
        if not isinstance(shadow, dict) or set(shadow) != set(self.shadow):
            raise ValueError("EMA checkpoint tensors do not match the model.")
        loaded: dict[str, torch.Tensor] = {}
        for name in sorted(self.shadow):
            value = shadow[name]
            if not isinstance(value, torch.Tensor) or value.shape != self.shadow[name].shape:
                raise TypeError(f"Invalid EMA tensor {name!r}.")
            loaded[name] = value.detach().to(device="cpu", dtype=torch.float32).clone()
        self.updates = updates
        self.shadow = loaded

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))


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
    ema: EMA | WarmStartEMA,
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
    # Generation starts from a fully masked field, while the stochastic
    # corruption curriculum tops out below one.  Reserve one deterministic
    # slot per four samples for the exact generation boundary so calibration
    # measures a condition the model has actually learned.  This is source-
    # bound rather than an evaluation relaxation.
    full_mask_slots = list(range(0, features.shape[0], FULL_MASK_SLOT_STRIDE))
    for index in full_mask_slots:
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
        "full_mask_sample_count": len(full_mask_slots),
    }
