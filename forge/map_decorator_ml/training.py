from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .contract import HEAD_CLASS_COUNTS, HEAD_NAMES
from .legality import (
    TorchLegalMasks,
    assert_selected_legal,
    mask_head_logits,
    select_legal_argmax,
)
from .metrics import decoration_metrics
from .model import CategoricalRefinementUNet


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    corruption_min: float = 0.20
    corruption_max: float = 0.95
    ema_decay: float = 0.999
    seed: int = 0xDEC0A7E

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            for value in (
                self.learning_rate,
                self.weight_decay,
                self.corruption_min,
                self.corruption_max,
                self.ema_decay,
                self.seed,
            )
        ):
            raise TypeError("TrainingConfig numeric fields cannot be booleans.")
        if not 0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate must be in (0, 0.1].")
        if not 0 <= self.weight_decay <= 1:
            raise ValueError("weight_decay must be in [0, 1].")
        if not 0 < self.corruption_min <= self.corruption_max <= 1:
            raise ValueError("Corruption bounds must satisfy 0 < min <= max <= 1.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1).")
        if not 0 <= self.seed < (1 << 63):
            raise ValueError("seed must be in [0, 2**63).")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        if set(self.shadow) != {name for name, value in state.items() if value.is_floating_point()}:
            raise ValueError("EMA parameter set drifted from the model.")
        for name, target in self.shadow.items():
            observed = state[name].detach().to(device="cpu", dtype=torch.float32)
            target.lerp_(observed, 1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if float(state.get("decay", -1)) != self.decay:
            raise ValueError("EMA decay does not match the resume contract.")
        shadow = state.get("shadow")
        if not isinstance(shadow, dict) or set(shadow) != set(self.shadow):
            raise ValueError("EMA checkpoint tensors do not match the model.")
        loaded: dict[str, torch.Tensor] = {}
        for name in sorted(self.shadow):
            value = shadow[name]
            if not isinstance(value, torch.Tensor) or value.shape != self.shadow[name].shape:
                raise TypeError(f"Invalid EMA tensor {name!r}.")
            loaded[name] = value.detach().to(device="cpu", dtype=torch.float32).clone()
        self.shadow = loaded

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))


def _force_nonempty(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    result = mask.clone()
    for index in range(result.shape[0]):
        if not bool(result[index].any()) and bool(valid[index].any()):
            first = torch.nonzero(valid[index], as_tuple=False)[0]
            result[index, first[0], first[1]] = True
    return result


def _force_foreground(
    mask: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    result = mask.clone()
    foreground = (target != 0) & valid
    for index in range(result.shape[0]):
        if bool(foreground[index].any()) and not bool((result[index] & foreground[index]).any()):
            first = torch.nonzero(foreground[index], as_tuple=False)[0]
            result[index, first[0], first[1]] = True
    return result


def corrupt_targets(
    targets: Mapping[str, torch.Tensor],
    valid_cells: torch.Tensor,
    *,
    generator: torch.Generator,
    minimum: float,
    maximum: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if valid_cells.dtype != torch.bool or valid_cells.ndim != 3:
        raise TypeError("valid_cells must be bool [B,H,W].")
    batch = valid_cells.shape[0]
    device = valid_cells.device
    probability = torch.rand((batch,), generator=generator, device=device)
    probability = minimum + (maximum - minimum) * probability

    def draw() -> torch.Tensor:
        mask = torch.rand(valid_cells.shape, generator=generator, device=device)
        return _force_nonempty((mask < probability[:, None, None]) & valid_cells, valid_cells)

    object_mask = draw()
    object_mask = _force_foreground(object_mask, targets["decal"], valid_cells)
    object_mask = _force_foreground(object_mask, targets["prop"], valid_cells)
    emission_mask = _force_foreground(draw(), targets["emission"], valid_cells)
    masked = {
        "variant": draw(),
        "decal": object_mask,
        "prop": object_mask,
        "emission": emission_mask,
    }
    return masked, probability


def class_balanced_weights(
    target: torch.Tensor,
    selected_cells: torch.Tensor,
    classes: int,
    *,
    empty_class: int | None,
) -> torch.Tensor:
    counts = torch.bincount(target[selected_cells], minlength=classes).to(torch.float32)
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = counts[present].rsqrt()
    if empty_class is not None and present[empty_class] and int(present.sum()) > 1:
        foreground_mean = weights[present & (torch.arange(classes, device=counts.device) != empty_class)].mean()
        weights[empty_class] = torch.minimum(weights[empty_class], foreground_mean * 0.25)
    if bool(present.any()):
        weights[present] /= weights[present].mean()
    return weights


def masked_refinement_loss(
    raw_logits: object,
    targets: Mapping[str, torch.Tensor],
    masked: Mapping[str, torch.Tensor],
    legal_masks: Mapping[str, torch.Tensor],
    valid_cells: torch.Tensor,
    hard_empty: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    legal = TorchLegalMasks(hard_empty=hard_empty, **legal_masks)
    assert_selected_legal(targets, legal)
    bounded = mask_head_logits(raw_logits, legal)  # type: ignore[arg-type]
    losses: dict[str, torch.Tensor] = {}
    for name in HEAD_NAMES:
        logits = getattr(bounded, name)
        selected = masked[name] & valid_cells
        if not bool(selected.any()):
            raise ValueError(f"No corrupted valid cells available for head {name!r}.")
        weights = class_balanced_weights(
            targets[name],
            selected,
            HEAD_CLASS_COUNTS[name],
            empty_class=None if name == "variant" else 0,
        )
        per_cell = F.cross_entropy(logits, targets[name], weight=weights, reduction="none")
        losses[name] = per_cell[selected].mean()
    predictions = select_legal_argmax(raw_logits, legal)  # type: ignore[arg-type]
    total = torch.stack(tuple(losses.values())).mean()
    details = {name: float(value.detach().item()) for name, value in losses.items()}
    details["total"] = float(total.detach().item())
    return total, details, predictions


def train_batch(
    model: CategoricalRefinementUNet,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    batch: Mapping[str, object],
    *,
    generator: torch.Generator,
    config: TrainingConfig,
    device: torch.device | str = "cpu",
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    device = torch.device(device)
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
        minimum=config.corruption_min,
        maximum=config.corruption_max,
    )
    level = probability.to(torch.float32)
    optimizer.zero_grad(set_to_none=True)
    if autocast_dtype is not None and device.type != "cuda":
        raise ValueError("Mixed-precision training is currently restricted to CUDA.")
    from contextlib import nullcontext

    autocast = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with autocast:
        raw = model(features, targets, masked, theme, conditions, level)
        loss, losses, predictions = masked_refinement_loss(
            raw, targets, masked, legal_masks, valid, hard_empty
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    ema.update(model)
    metrics = decoration_metrics(predictions, targets, valid)
    return {
        "loss": losses,
        "metrics": metrics,
        "corruption_fraction": [float(value) for value in probability.detach().cpu()],
    }
