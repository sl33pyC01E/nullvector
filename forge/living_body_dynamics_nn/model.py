from __future__ import annotations

from dataclasses import asdict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import DynamicsConfig, FEATURES, SYSTEMS


class MessageBlock(nn.Module):
    def __init__(self, width: int, family_width: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width))
        self.update = nn.Sequential(
            nn.LayerNorm(width * 2 + family_width),
            nn.Linear(width * 2 + family_width, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, value: Tensor, edges: Tensor, family: Tensor, graph_index: Tensor) -> Tensor:
        source, target = edges
        message = self.message(torch.cat((value[source], value[target]), 1))
        aggregate = torch.zeros_like(value)
        aggregate.index_add_(0, target, message)
        degree = torch.zeros((len(value), 1), device=value.device, dtype=value.dtype)
        degree.index_add_(0, target, torch.ones((len(target), 1), device=value.device, dtype=value.dtype))
        aggregate = aggregate / degree.clamp_min(1)
        update = self.update(torch.cat((value, aggregate, family[graph_index]), 1))
        return value + torch.tanh(self.gate) * update


class LivingBodyDynamicsNet(nn.Module):
    def __init__(self, config: DynamicsConfig = DynamicsConfig()) -> None:
        super().__init__()
        self.config = config
        self.family = nn.Embedding(5, config.family_width)
        self.input = nn.Sequential(
            nn.Linear(FEATURES, config.width), nn.SiLU(), nn.Linear(config.width, config.width)
        )
        self.blocks = nn.ModuleList(
            MessageBlock(config.width, config.family_width, config.dropout) for _ in range(config.depth)
        )
        self.cell_head = nn.Sequential(
            nn.LayerNorm(config.width), nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, 3)
        )
        self.system_head = nn.Sequential(
            nn.LayerNorm(config.width + config.family_width),
            nn.Linear(config.width + config.family_width, config.width),
            nn.SiLU(), nn.Linear(config.width, SYSTEMS),
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        value = self.input(batch["features"].float())
        family = self.family(batch["family"].long())
        for block in self.blocks:
            value = block(value, batch["edges"].long(), family, batch["graph_index"].long())
        # Residual next-state prediction keeps healthy identity transitions easy
        # while still allowing intervention-conditioned changes.
        baseline = batch["features"][:, 30:33].float()
        delta = torch.tanh(self.cell_head(value)) * torch.tensor((.85, .45, .30), device=value.device)
        cell = (baseline + delta).clamp(0, 1)
        graph_count = len(batch["family"])
        pooled = torch.zeros((graph_count, value.shape[1]), device=value.device, dtype=value.dtype)
        pooled.index_add_(0, batch["graph_index"], value)
        counts = torch.bincount(batch["graph_index"], minlength=graph_count).to(value.dtype)[:, None]
        pooled = pooled / counts.clamp_min(1)
        systems = torch.sigmoid(self.system_head(torch.cat((pooled, family), 1)))
        return cell, systems

    def config_dict(self) -> dict[str, int | float]:
        return asdict(self.config)


def loss(model: LivingBodyDynamicsNet, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
    cell, systems = model(batch)
    target = batch["target"].float()
    action = batch["features"][:, 34:37].abs().sum(1)
    changed = (target[:, 0] - batch["features"][:, 30]).abs() > 1e-5
    weight = 1 + action * 3 + changed.float() * 4
    health = ((cell[:, 0] - target[:, 0]).abs() * weight).sum() / weight.sum()
    fluid = ((cell[:, 1] - target[:, 1]).abs() * (1 + changed.float() * 2)).mean()
    scar = F.smooth_l1_loss(cell[:, 2], target[:, 2])
    system = F.smooth_l1_loss(systems, batch["systems"].float())
    # Cells beyond two graph hops of an intervention should remain stable.
    untouched = action <= 1e-6
    locality = (cell[untouched, 0] - batch["features"][untouched, 30]).abs().mean() if bool(untouched.any()) else cell.sum() * 0
    total = 3.0 * health + 1.5 * fluid + scar + 2.0 * system + .35 * locality
    return total, {
        "loss": float(total.detach()), "health_mae": float(health.detach()),
        "fluid_mae": float(fluid.detach()), "scar_smooth_l1": float(scar.detach()),
        "system_smooth_l1": float(system.detach()), "untouched_drift": float(locality.detach()),
    }
