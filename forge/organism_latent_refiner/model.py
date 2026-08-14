from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..organism_latent_flow.model import FlowBlock, timestep_embedding
from .contract import OrganismRefinerConfig


class HierarchicalLatentRefiner(nn.Module):
    def __init__(self, config: OrganismRefinerConfig = OrganismRefinerConfig()) -> None:
        super().__init__(); self.config = config; cw, fw, cd = config.coarse_width, config.fine_width, config.condition_dim
        self.noise = nn.Sequential(nn.Linear(config.time_dim, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.condition = nn.Sequential(nn.Linear(192, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.coarse_in = nn.Conv2d(48, cw, 3, padding=1); self.fine_in = nn.Conv2d(16, fw, 3, padding=1)
        self.coarse_blocks = nn.ModuleList(FlowBlock(cw, cd) for _ in range(config.depth)); self.fine_blocks = nn.ModuleList(FlowBlock(fw, cd) for _ in range(config.depth))
        self.fine_to_coarse = nn.ModuleList(nn.Conv2d(fw, cw, 1) for _ in range(config.depth)); self.coarse_to_fine = nn.ModuleList(nn.Conv2d(cw, fw, 1) for _ in range(config.depth))
        self.coarse_out = nn.Sequential(nn.GroupNorm(32, cw), nn.SiLU(), nn.Conv2d(cw, 32, 3, padding=1)); self.fine_out = nn.Sequential(nn.GroupNorm(32, fw), nn.SiLU(), nn.Conv2d(fw, 16, 3, padding=1))
        # Start extremely close to identity while allowing the first update to
        # reach the conditioner and both cross-scale trunks.
        nn.init.normal_(self.coarse_out[-1].weight, std=1e-5); nn.init.zeros_(self.coarse_out[-1].bias)
        nn.init.normal_(self.fine_out[-1].weight, std=1e-5); nn.init.zeros_(self.fine_out[-1].bias)

    def forward(self, coarse: Tensor, fine: Tensor, noise_level: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        if coarse.ndim != 4 or coarse.shape[1:] != (32, 12, 12) or fine.ndim != 4 or fine.shape[1:] != (16, 24, 24) or noise_level.shape != (len(coarse),) or condition.shape != (len(coarse), 192):
            raise ValueError("Organism refiner input contract drifted.")
        modulation = self.condition(condition.float()) + self.noise(timestep_embedding(noise_level.float(), self.config.time_dim))
        fine_value = self.fine_in(fine.float()); coarse_value = self.coarse_in(torch.cat((coarse.float(), F.avg_pool2d(fine.float(), 2)), dim=1))
        for coarse_block, fine_block, down, up in zip(self.coarse_blocks, self.fine_blocks, self.fine_to_coarse, self.coarse_to_fine, strict=True):
            coarse_value = coarse_block(coarse_value + down(F.avg_pool2d(fine_value, 2)), modulation)
            fine_value = fine_block(fine_value + F.interpolate(up(coarse_value), scale_factor=2, mode="nearest"), modulation)
        return self.coarse_out(coarse_value), self.fine_out(fine_value)


@torch.no_grad()
def refine_latents(model: HierarchicalLatentRefiner, coarse: Tensor, fine: Tensor, condition: Tensor, schedule: tuple[float, ...] = (.45, .25, .12)) -> tuple[Tensor, Tensor]:
    if not schedule or any(not 0 < value <= 1 for value in schedule): raise ValueError("Organism refiner schedule drifted.")
    model.eval(); refined_coarse, refined_fine = coarse.float().clone(), fine.float().clone()
    for index, sigma in enumerate(schedule):
        level = torch.full((len(condition),), sigma, device=condition.device); coarse_delta, fine_delta = model(refined_coarse, refined_fine, level, condition)
        strength = .72 if index < len(schedule) - 1 else .9
        refined_coarse = refined_coarse + strength * coarse_delta.float(); refined_fine = refined_fine + strength * fine_delta.float()
    return refined_coarse, refined_fine


def latent_refinement_loss(predicted_coarse_delta: Tensor, predicted_fine_delta: Tensor, corrupted_coarse: Tensor, corrupted_fine: Tensor, clean_coarse: Tensor, clean_fine: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
    predicted_coarse = corrupted_coarse + predicted_coarse_delta.float(); predicted_fine = corrupted_fine + predicted_fine_delta.float()
    coarse_mse = F.mse_loss(predicted_coarse, clean_coarse); fine_mse = F.mse_loss(predicted_fine, clean_fine)
    coarse_l1 = F.l1_loss(predicted_coarse, clean_coarse); fine_l1 = F.l1_loss(predicted_fine, clean_fine)
    total = coarse_mse + fine_mse + .12 * (coarse_l1 + fine_l1)
    return total, {"latent_loss": total.detach(), "coarse_mse": coarse_mse.detach(), "fine_mse": fine_mse.detach(), "coarse_l1": coarse_l1.detach(), "fine_l1": fine_l1.detach()}, predicted_coarse, predicted_fine
