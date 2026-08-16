from __future__ import annotations

from dataclasses import asdict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import (
    ACTION_SLICE,
    ACTIVITY_INDEX,
    CELL_STATE_SLICE,
    CONNECTED_INDEX,
    CONTACT_INDEX,
    DELTA_INDEX,
    DIGESTIVE_INDEX,
    DynamicsConfig,
    ENERGY_INDEX,
    FAMILY_NUTRITION_INDEX,
    FEEDER_INDEX,
    FEATURES,
    FOOD_MASS_INDEX,
    FULLNESS_INDEX,
    NUTRIENT_DENSITY_INDEX,
    RESERVE_INDEX,
    SYSTEMS,
)


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
        # Route integrity depends on sparse feeder/digestive cell populations;
        # mean pooling alone erased those ablations in v2.  Eleven explicit,
        # permutation-invariant organ statistics preserve the causal signal.
        self.feeding_summary_width = 11
        self.feeding_head = nn.Sequential(
            nn.LayerNorm(config.width + config.family_width + self.feeding_summary_width),
            nn.Linear(config.width + config.family_width + self.feeding_summary_width, config.width * 2),
            nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(config.width * 2, 4),
        )

    @staticmethod
    def _graph_mean(value: Tensor, graph_index: Tensor, graph_count: int, weight: Tensor | None = None) -> Tensor:
        if weight is None:
            weight = torch.ones((len(value), 1), device=value.device, dtype=value.dtype)
        elif weight.ndim == 1:
            weight = weight[:, None]
        total = torch.zeros((graph_count, value.shape[1]), device=value.device, dtype=value.dtype)
        count = torch.zeros((graph_count, 1), device=value.device, dtype=value.dtype)
        total.index_add_(0, graph_index, value * weight)
        count.index_add_(0, graph_index, weight)
        return total / count.clamp_min(1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        value = self.input(batch["features"].float())
        family = self.family(batch["family"].long())
        for block in self.blocks:
            value = block(value, batch["edges"].long(), family, batch["graph_index"].long())
        # Residual next-state prediction keeps healthy identity transitions easy
        # while still allowing intervention-conditioned changes.
        baseline = batch["features"][:, CELL_STATE_SLICE].float()
        delta = torch.tanh(self.cell_head(value)) * torch.tensor((.85, .45, .30), device=value.device)
        cell = (baseline + delta).clamp(0, 1)
        graph_count = len(batch["family"])
        graph_index = batch["graph_index"].long()
        pooled = self._graph_mean(value, graph_index, graph_count)
        graph = torch.cat((pooled, family), 1)
        systems = torch.sigmoid(self.system_head(graph))
        features = batch["features"].float()
        health_fluid_connected = features[:, (CELL_STATE_SLICE.start, CELL_STATE_SLICE.start + 1, CONNECTED_INDEX)]
        feeder = features[:, FEEDER_INDEX]
        digestive = features[:, DIGESTIVE_INDEX]
        node_fraction = torch.ones((len(features), 1), device=value.device, dtype=value.dtype)
        global_summary = self._graph_mean(health_fluid_connected, graph_index, graph_count)
        feeder_summary = self._graph_mean(health_fluid_connected, graph_index, graph_count, feeder)
        digestive_summary = self._graph_mean(health_fluid_connected, graph_index, graph_count, digestive)
        feeder_fraction = self._graph_mean(feeder[:, None], graph_index, graph_count, node_fraction)
        digestive_fraction = self._graph_mean(digestive[:, None], graph_index, graph_count, node_fraction)
        summary = torch.cat((global_summary, feeder_fraction, feeder_summary, digestive_fraction, digestive_summary), 1)
        raw = self.feeding_head(torch.cat((graph, summary), 1))

        # Global physical context is repeated on every node, so graph means are
        # exact.  Learned factors can modulate a legal event but can never
        # create contact, edible matter, free capacity, or mass.
        context = self._graph_mean(features, graph_index, graph_count)
        feed_action = (context[:, ACTION_SLICE.start + 4] + context[:, ACTION_SLICE.start + 6]).clamp(0, 1)
        metabolize_action = (context[:, ACTION_SLICE.start + 5] + context[:, ACTION_SLICE.start + 6]).clamp(0, 1)
        contact = context[:, CONTACT_INDEX].clamp(0, 1)
        mass = context[:, FOOD_MASS_INDEX].clamp(0, 1) * 3.0
        density = context[:, NUTRIENT_DENSITY_INDEX].clamp(0, 1) * 6.0
        nutrition_factor = context[:, FAMILY_NUTRITION_INDEX].clamp(0, 1)
        reserve_before = context[:, RESERVE_INDEX].clamp(0, 1) * 4.0
        fullness_before = context[:, FULLNESS_INDEX].clamp(0, 1) * 240.0
        energy_before = context[:, ENERGY_INDEX].clamp(0, 1) * 4.0
        activity = context[:, ACTIVITY_INDEX].clamp(0, 1)
        delta_time = context[:, DELTA_INDEX].clamp(.001, 1.0)
        available = (4.0 - reserve_before).clamp_min(0)
        conversion = density * nutrition_factor
        capacity_mass = available / conversion.clamp_min(1e-8)
        upper_absorption = torch.minimum(torch.minimum(mass, .45 * delta_time), capacity_mass)
        legal_absorption = feed_action * contact * (nutrition_factor > 0).to(value.dtype) * (mass > 0).to(value.dtype) * (available > 0).to(value.dtype)
        route_logit = raw[:, 3]
        route_probability = torch.sigmoid(route_logit)
        absorbed = upper_absorption * torch.sigmoid(raw[:, 0]) * route_probability * legal_absorption
        nutrition = absorbed * conversion
        upper_release = torch.minimum(reserve_before + nutrition, delta_time * (.0015 + .0035 * activity))
        released = upper_release * torch.sigmoid(raw[:, 1]) * route_probability * metabolize_action
        reserve_after = (reserve_before + nutrition - released).clamp(0, 4.0)
        fullness_after = (fullness_before - delta_time * metabolize_action + nutrition * 42.0).clamp(0, 240.0)
        # The body tick consumes at most ~.001 energy here. Predict only that
        # bounded cost, rather than re-predicting the full recurrent state.
        tick_cost = torch.sigmoid(raw[:, 2]) * .0012
        energy_after = (energy_before + released - tick_cost).clamp(0, 4.0)
        remaining_fraction = torch.where(mass > 1e-8, ((mass - absorbed) / mass).clamp(0, 1), torch.zeros_like(mass))
        contact_logit = torch.where(contact >= .5, torch.full_like(contact, 12.0), torch.full_like(contact, -12.0))
        feeding = torch.stack((
            absorbed,
            nutrition / 4.0,
            reserve_after / 4.0,
            fullness_after / 240.0,
            energy_after / 4.0,
            released / .01,
            contact_logit,
            route_logit,
            remaining_fraction,
        ), 1)
        return cell, systems, feeding

    def config_dict(self) -> dict[str, int | float]:
        return asdict(self.config)


def loss(model: LivingBodyDynamicsNet, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
    cell, systems, feeding = model(batch)
    target = batch["target"].float()
    action = batch["features"][:, 43:46].abs().sum(1)
    changed = (target[:, 0] - batch["features"][:, 30]).abs() > 1e-5
    weight = 1 + action * 3 + changed.float() * 4
    health = ((cell[:, 0] - target[:, 0]).abs() * weight).sum() / weight.sum()
    fluid = ((cell[:, 1] - target[:, 1]).abs() * (1 + changed.float() * 2)).mean()
    scar = F.smooth_l1_loss(cell[:, 2], target[:, 2])
    system = F.smooth_l1_loss(systems, batch["systems"].float())
    feeding_target = batch["feeding_target"].float()
    continuous_indices = torch.tensor((0, 1, 2, 3, 4, 5, 8), device=feeding.device)
    feeding_continuous = F.smooth_l1_loss(
        feeding.index_select(1, continuous_indices),
        feeding_target.index_select(1, continuous_indices),
    )
    contacted = F.binary_cross_entropy_with_logits(feeding[:, 6], feeding_target[:, 6])
    route = F.binary_cross_entropy_with_logits(feeding[:, 7], feeding_target[:, 7])
    positive = feeding_target[:, 0] > 1e-7
    absorption = (feeding[:, 0] - feeding_target[:, 0]).abs()
    absorption_weight = 1 + positive.float() * 8
    absorption_mae = (absorption * absorption_weight).sum() / absorption_weight.sum()
    # Cells beyond two graph hops of an intervention should remain stable.
    untouched = action <= 1e-6
    locality = (cell[untouched, 0] - batch["features"][untouched, 30]).abs().mean() if bool(untouched.any()) else cell.sum() * 0
    total = (
        3.0 * health + 1.5 * fluid + scar + 2.0 * system + .35 * locality
        + 2.0 * feeding_continuous + 3.0 * absorption_mae + .05 * contacted + 3.0 * route
    )
    return total, {
        "loss": float(total.detach()), "health_mae": float(health.detach()),
        "fluid_mae": float(fluid.detach()), "scar_smooth_l1": float(scar.detach()),
        "system_smooth_l1": float(system.detach()), "untouched_drift": float(locality.detach()),
        "feeding_smooth_l1": float(feeding_continuous.detach()),
        "absorption_mae": float(absorption_mae.detach()),
        "contact_bce": float(contacted.detach()), "route_bce": float(route.detach()),
    }
