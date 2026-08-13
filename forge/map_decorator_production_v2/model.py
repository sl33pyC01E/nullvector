from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.model import CategoricalRefinementUNet, HeadLogits
from .contract import FactoredModelConfig


OBJECT_HEADS = ("decal", "prop")


@dataclass(slots=True)
class FactoredDecoratorOutput:
    categorical: HeadLogits
    presence_logits: dict[str, torch.Tensor]
    type_logits: dict[str, torch.Tensor]

    @property
    def variant(self) -> torch.Tensor:
        return self.categorical.variant

    @property
    def decal(self) -> torch.Tensor:
        return self.categorical.decal

    @property
    def prop(self) -> torch.Tensor:
        return self.categorical.prop

    @property
    def emission(self) -> torch.Tensor:
        return self.categorical.emission

    def as_head_logits(self) -> HeadLogits:
        return self.categorical


def compose_object_logits(presence_logit: torch.Tensor, type_logits: torch.Tensor) -> torch.Tensor:
    if presence_logit.ndim != 3:
        raise ValueError("presence_logit must have shape [B,H,W].")
    if type_logits.ndim != 4 or type_logits.shape[1] < 1:
        raise ValueError("type_logits must have shape [B,K,H,W] with K>=1.")
    if type_logits.shape[0] != presence_logit.shape[0] or type_logits.shape[-2:] != presence_logit.shape[-2:]:
        raise ValueError("Presence and type tensors must share batch/spatial dimensions.")
    if not bool(torch.isfinite(presence_logit).all()) or not bool(torch.isfinite(type_logits).all()):
        raise ValueError("Factored object logits must be finite.")
    log_empty = F.logsigmoid(-presence_logit)
    log_present = F.logsigmoid(presence_logit)
    type_log_probability = F.log_softmax(type_logits, dim=1)
    return torch.cat((log_empty[:, None], log_present[:, None] + type_log_probability), dim=1)


class FactoredDecoratorV2(nn.Module):
    """V1 categorical core with explicit learned object-presence/type factorization."""

    def __init__(self, config: FactoredModelConfig = FactoredModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.core = CategoricalRefinementUNet(config.core_config())
        self.presence_heads = nn.ModuleDict(
            {name: nn.Conv2d(3, 1, kernel_size=1) for name in OBJECT_HEADS}
        )
        self._initialize_presence_heads()

    def _initialize_presence_heads(self) -> None:
        with torch.no_grad():
            for layer in self.presence_heads.values():
                layer.weight.zero_()
                layer.weight[:, 0].fill_(-1.0)
                layer.weight[:, 1:].fill_(0.5)
                layer.bias.fill_(self.config.presence_bias_init)

    def forward(
        self,
        features: torch.Tensor,
        labels: Mapping[str, torch.Tensor],
        masked: Mapping[str, torch.Tensor],
        theme_index: torch.Tensor,
        global_conditions: torch.Tensor,
        refinement_level: torch.Tensor,
    ) -> FactoredDecoratorOutput:
        raw = self.core(
            features,
            labels,
            masked,
            theme_index,
            global_conditions,
            refinement_level,
        )
        presence: dict[str, torch.Tensor] = {}
        types: dict[str, torch.Tensor] = {}
        composed: dict[str, torch.Tensor] = {
            "variant": raw.variant,
            "emission": raw.emission,
        }
        for name in OBJECT_HEADS:
            categorical = getattr(raw, name)
            presence[name] = self.presence_heads[name](categorical).squeeze(1)
            types[name] = categorical[:, 1:]
            composed[name] = compose_object_logits(presence[name], types[name])
        output = HeadLogits(**{name: composed[name] for name in HEAD_NAMES})
        return FactoredDecoratorOutput(
            categorical=output,
            presence_logits=presence,
            type_logits=types,
        )
