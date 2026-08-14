from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .contract import DIRECTION_XY, DYNAMIC_CHANNELS


def _neighbor(value: Tensor, dx: int, dy: int) -> Tensor:
    return torch.roll(value, shifts=(-dy, -dx), dims=(-2, -1))


def _capacity(static: Tensor, state: Tensor, system: int) -> Tensor:
    weight = static[:, 52 + system : 53 + system]
    viable = state[:, 0:1] * state[:, 4:5] * state[:, 11:12]
    return (weight * viable).sum((-2, -1), keepdim=True) / weight.sum((-2, -1), keepdim=True).clamp_min(1e-5)


def make_scenarios(static: Tensor, initial: Tensor, bonds: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Create deterministic-in-seed wounds, severing, starvation and feeding states."""
    batch = len(static); device = static.device; state = initial.clone(); live = bonds.clone(); body = static[:, :1]
    yy, xx = torch.meshgrid(torch.arange(48, device=device), torch.arange(48, device=device), indexing="ij")
    cx = torch.randint(5, 43, (batch, 1, 1), device=device, generator=generator); cy = torch.randint(5, 43, (batch, 1, 1), device=device, generator=generator)
    radius = 2.0 + 8.0 * torch.rand((batch, 1, 1), device=device, generator=generator); severity = .25 + .72 * torch.rand((batch, 1, 1), device=device, generator=generator)
    distance = torch.sqrt((xx[None] - cx).float().square() + (yy[None] - cy).float().square())
    injury = torch.clamp(1 - distance / radius, 0, 1)[:, None] * severity[:, None] * body
    # A second elongated cut provides true topology damage rather than only bruising.
    vertical = torch.rand((batch, 1, 1), device=device, generator=generator) < .5
    line_position = torch.randint(9, 39, (batch, 1, 1), device=device, generator=generator)
    line_delta = torch.where(vertical, (xx[None] - line_position).abs(), (yy[None] - line_position).abs())
    along_center = torch.randint(8, 40, (batch, 1, 1), device=device, generator=generator)
    along_delta = torch.where(vertical, (yy[None] - along_center).abs(), (xx[None] - along_center).abs())
    cut = ((line_delta <= 0) & (along_delta <= radius * 1.25))[:, None] & (body > 0)
    cut_strength = (.35 + .55 * torch.rand((batch, 1, 1, 1), device=device, generator=generator))
    injury = torch.maximum(injury, cut.float() * cut_strength)
    state[:, 0:1] *= 1 - injury
    lost_fluid = state[:, 1:2] * injury * .62
    state[:, 1:2] -= lost_fluid; state[:, 9:10] = torch.clamp(state[:, 9:10] + lost_fluid, 0, 1)
    state[:, 7:8] = torch.maximum(state[:, 7:8], injury)
    state[:, 5:6] = injury * (.02 + .18 * torch.rand((batch, 1, 1, 1), device=device, generator=generator))
    state[:, 6:7] = injury * (.02 + .25 * torch.rand((batch, 1, 1, 1), device=device, generator=generator))
    # Random physiological stress and resource availability create nontrivial behavior.
    state[:, 3:4] *= .2 + .8 * torch.rand((batch, 1, 1, 1), device=device, generator=generator)
    state[:, 4:5] *= .25 + .75 * torch.rand((batch, 1, 1, 1), device=device, generator=generator)
    digestive = static[:, 34:37].sum(1, keepdim=True).clamp(0, 1)
    feeding = torch.rand((batch, 1, 1, 1), device=device, generator=generator)
    state[:, 2:3] = torch.clamp(state[:, 2:3] + digestive * feeding * .8, 0, 1)
    state[:, 11:12] = (state[:, 0:1] > .025).float() * body
    for index, (dx, dy) in enumerate(DIRECTION_XY):
        endpoint_injury = torch.maximum(injury, _neighbor(injury, dx, dy))
        live[:, index : index + 1] *= (endpoint_injury < .68).float()
    return state, live


def teacher_step(static: Tensor, state: Tensor, live_bonds: Tensor, dt: float = .1) -> Tensor:
    if static.shape[1] != 85 or state.shape[1] != DYNAMIC_CHANNELS or live_bonds.shape[1] != 8 or not 0 < dt <= .25:
        raise ValueError("Cellular NCA teacher input contract drifted.")
    body = static[:, :1]; alive = state[:, 11:12]; next_state = state.clone()
    health, fluid, nutrient, energy, oxygen = [state[:, index : index + 1] for index in range(5)]
    clot, scar, wound, neural, surface, biomass = [state[:, index : index + 1] for index in range(5, 11)]
    circulation = _capacity(static, state, 0); respiration = _capacity(static, state, 1); digestion = _capacity(static, state, 2); neural_capacity = _capacity(static, state, 3); immune = _capacity(static, state, 7)

    def diffuse(value: Tensor, rate: float) -> Tensor:
        delta = torch.zeros_like(value); normalizer = torch.zeros_like(value)
        for index, (dx, dy) in enumerate(DIRECTION_XY):
            edge = live_bonds[:, index : index + 1] * static[:, 77 + index : 78 + index]
            delta += edge * (_neighbor(value, dx, dy) - value); normalizer += edge
        return value + dt * rate * delta / normalizer.clamp_min(1.0)

    fluid = diffuse(fluid, .62); nutrient = diffuse(nutrient, .16); oxygen = diffuse(oxygen, .85); neural = diffuse(neural, .48)
    respiratory_cells = static[:, 31:34].sum(1, keepdim=True).clamp(0, 1)
    oxygen = oxygen + dt * .42 * respiratory_cells * respiration * (1 - oxygen)
    oxygen = oxygen - dt * .045 * alive * (static[:, 57:58] + .22) * (1 - respiratory_cells * .5)
    digestive_cells = static[:, 34:37].sum(1, keepdim=True).clamp(0, 1)
    digested = torch.minimum(nutrient, dt * .20 * digestive_cells * digestion * (oxygen * .65 + .35))
    nutrient = nutrient - digested; energy = energy + digested * 1.65
    photosynthetic = static[:, 19:20]; emitter = static[:, 22:23]; plant = static[:, 25:26]; anomaly = static[:, 26:27]; machine = static[:, 27:28]
    energy = energy + dt * (.026 * photosynthetic * plant + .018 * emitter * anomaly + .014 * emitter * machine)
    metabolism = dt * (.010 + .018 * static[:, 57:58] + .012 * static[:, 55:56]) * alive
    energy = torch.clamp(energy - metabolism, 0, 1)
    clot_gain = dt * .38 * static[:, 66:67] * wound * circulation * (.2 + .8 * immune) * (1 - clot)
    clot = torch.clamp(clot + clot_gain, 0, 1)
    leakage = torch.minimum(fluid, dt * .46 * wound * (1 - clot) * (fluid + .08) * alive)
    fluid = fluid - leakage; surface = surface + leakage
    healing = dt * .075 * static[:, 68:69] * immune * circulation * energy * clot * wound
    healing = torch.minimum(1 - health, healing); health = health + healing; energy = torch.clamp(energy - healing * .48, 0, 1)
    scar = torch.clamp(scar + healing * static[:, 67:68] * .26, 0, 1)
    wound = torch.clamp(wound - dt * (.09 * clot + .13 * healing) + dt * (1 - health) * .006, 0, 1)
    hypoxia = torch.relu(.24 - oxygen) * alive; starvation = torch.relu(.035 - energy) * alive
    health = torch.clamp(health - dt * (.16 * hypoxia + .11 * starvation), 0, 1)
    neural_cells = static[:, 37:40].sum(1, keepdim=True).clamp(0, 1)
    neural_target = neural_cells * health * oxygen * (energy * .7 + .3) * neural_capacity
    neural = torch.clamp(neural + dt * 1.2 * (neural_target - neural), 0, 1)
    # Top-down surface physics: radial diffusion, no screen-down gravity.
    surface = torch.clamp(surface + dt * 1.35 * (F.avg_pool2d(surface, 3, 1, 1) - surface) - dt * .025 * surface, 0, 1)
    alive_next = torch.clamp(health * 5.0, 0, 1) * body
    dying = torch.clamp(alive - alive_next, 0, 1)
    biomass = torch.clamp(biomass + dt * .16 * dying + dt * .03 * body * (1 - alive_next), 0, 1)
    biomass = torch.clamp(biomass + dt * .22 * (F.avg_pool2d(biomass, 3, 1, 1) - biomass), 0, 1)
    values = (health, fluid, nutrient, energy, oxygen, clot, scar, wound, neural, surface, biomass, alive_next)
    next_state = torch.cat(values, dim=1).clamp(0, 1)
    next_state[:, :9] *= body; next_state[:, 11:12] *= body
    return next_state


STATE_WEIGHTS = torch.tensor((2.2, 2.0, 1.0, 1.3, 1.6, 1.0, .8, 1.2, 2.2, 1.2, .8, 1.5), dtype=torch.float32)


def cellular_loss(predicted: Tensor, target: Tensor, static: Tensor, previous: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    body = static[:, :1]; support = torch.cat((body.expand(-1, 9, -1, -1), torch.ones_like(body).expand(-1, 2, -1, -1), body), dim=1)
    absolute = (predicted.float() - target.float()).abs() * support
    denominator = support.sum((0, 2, 3)).clamp_min(1); per_channel = absolute.sum((0, 2, 3)) / denominator
    weights = STATE_WEIGHTS.to(predicted.device); reconstruction = (per_channel * weights).sum() / weights.sum()
    # Preserve sharp organ boundaries while letting external fluid stay diffuse.
    edge = (predicted[:, :9, 1:] - predicted[:, :9, :-1] - (target[:, :9, 1:] - target[:, :9, :-1])).abs().mean()
    edge += (predicted[:, :9, :, 1:] - predicted[:, :9, :, :-1] - (target[:, :9, :, 1:] - target[:, :9, :, :-1])).abs().mean()
    velocity = torch.zeros((), device=predicted.device)
    if previous is not None:
        predicted_velocity = (predicted.float() - previous.float()) / .08; target_velocity = (target.float() - previous.float()) / .08
        velocity_error = torch.nn.functional.smooth_l1_loss(predicted_velocity, target_velocity, reduction="none") * support
        velocity_per_channel = velocity_error.sum((0, 2, 3)) / denominator
        velocity = (velocity_per_channel * weights).sum() / weights.sum()
    total = reconstruction + .8 * velocity + .12 * edge
    return total, {"reconstruction": reconstruction.detach(), "velocity": velocity.detach(), "edge": edge.detach(), **{f"channel_{index}": value.detach() for index, value in enumerate(per_channel)}}
