from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import OrganismVAEConfig


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__(); groups = min(16, channels)
        while channels % groups: groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels); self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, channels * 2)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm1(value) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden)); hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


@dataclass(slots=True)
class OrganismVAEOutput:
    rgba: Tensor
    occupancy_logits: Tensor
    tissue_logits: Tensor
    material_logits: Tensor
    part_logits: Tensor
    emission: Tensor
    physiology: Tensor
    cell_state: Tensor
    mean: Tensor
    log_variance: Tensor
    latent: Tensor


class ContinuousOrganismRasterVAE(nn.Module):
    """Gaussian field VAE whose decoder is the organism's learned rasterizer."""

    def __init__(self, config: OrganismVAEConfig = OrganismVAEConfig()) -> None:
        super().__init__(); self.config = config; width = config.width
        self.family = nn.Embedding(5, 12); self.subtype = nn.Embedding(20, 12); self.role = nn.Embedding(8, 12)
        self.gene = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 28))
        self.condition = nn.Sequential(nn.Linear(64, config.condition_dim), nn.SiLU(), nn.Linear(config.condition_dim, config.condition_dim))
        self.stem = nn.Conv2d(config.input_channels, width, 3, padding=1)
        self.encoder0 = nn.ModuleList(FiLMBlock(width, config.condition_dim) for _ in range(config.residual_depth))
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.encoder1 = nn.ModuleList(FiLMBlock(width * 2, config.condition_dim) for _ in range(config.residual_depth))
        self.down2 = nn.Conv2d(width * 2, width * 2, 4, stride=2, padding=1)
        self.mean = nn.Conv2d(width * 2, config.latent_channels, 1); self.log_variance = nn.Conv2d(width * 2, config.latent_channels, 1)
        self.from_latent = nn.Conv2d(config.latent_channels, width * 2, 1)
        self.decoder1 = nn.ModuleList(FiLMBlock(width * 2, config.condition_dim) for _ in range(config.residual_depth))
        self.up1 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.decoder0 = nn.ModuleList(FiLMBlock(width, config.condition_dim) for _ in range(config.residual_depth))
        self.up2 = nn.Conv2d(width, width, 3, padding=1)
        self.shared = nn.Conv2d(width, width, 3, padding=1)
        self.rgba_head = nn.Conv2d(width, 4, 1); self.occupancy_head = nn.Conv2d(width, 1, 1)
        self.tissue_head = nn.Conv2d(width, 15, 1); self.material_head = nn.Conv2d(width, 10, 1); self.part_head = nn.Conv2d(width, 17, 1)
        self.emission_head = nn.Conv2d(width, 1, 1); self.physiology_head = nn.Conv2d(width, 8, 1); self.cell_state_head = nn.Conv2d(width, 10, 1)

    def condition_vector(self, family: Tensor, subtype: Tensor, role: Tensor, genes: Tensor) -> Tensor:
        if genes.ndim != 2 or genes.shape[1] != 16 or family.shape != subtype.shape or family.shape != role.shape or family.shape != genes.shape[:1]:
            raise ValueError("Organism VAE condition tensors are misaligned.")
        return self.condition(torch.cat((self.family(family), self.subtype(subtype), self.role(role), self.gene(genes.float())), dim=1))

    def encode(self, living_field: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        if living_field.ndim != 4 or living_field.shape[1:] != (74, 48, 48): raise ValueError("Organism VAE living field must be B,74,48,48.")
        hidden = self.stem(living_field.float())
        for block in self.encoder0: hidden = block(hidden, condition)
        hidden = F.silu(self.down1(hidden))
        for block in self.encoder1: hidden = block(hidden, condition)
        hidden = F.silu(self.down2(hidden))
        return self.mean(hidden), self.log_variance(hidden).clamp(-12, 8)

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor, *, generator: torch.Generator | None = None, sample: bool = True) -> Tensor:
        if not sample: return mean
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return mean + torch.exp(.5 * log_variance) * noise

    def decode(self, latent: Tensor, condition: Tensor) -> OrganismVAEOutput:
        if latent.ndim != 4 or latent.shape[1:] != (self.config.latent_channels, 12, 12): raise ValueError("Organism VAE latent must be B,C,12,12.")
        hidden = self.from_latent(latent)
        for block in self.decoder1: hidden = block(hidden, condition)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest"); hidden = F.silu(self.up1(hidden))
        for block in self.decoder0: hidden = block(hidden, condition)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest"); hidden = F.silu(self.up2(hidden)); hidden = F.silu(self.shared(hidden))
        empty = torch.empty(0, device=latent.device)
        return OrganismVAEOutput(torch.sigmoid(self.rgba_head(hidden)), self.occupancy_head(hidden), self.tissue_head(hidden), self.material_head(hidden), self.part_head(hidden), torch.sigmoid(self.emission_head(hidden)), torch.sigmoid(self.physiology_head(hidden)), torch.sigmoid(self.cell_state_head(hidden)), empty, empty, latent)

    def forward(self, living_field: Tensor, family: Tensor, subtype: Tensor, role: Tensor, genes: Tensor, *, generator: torch.Generator | None = None, sample: bool = True) -> OrganismVAEOutput:
        condition = self.condition_vector(family, subtype, role, genes); mean, log_variance = self.encode(living_field, condition)
        result = self.decode(self.reparameterize(mean, log_variance, generator=generator, sample=sample), condition)
        result.mean = mean; result.log_variance = log_variance
        return result


def organism_vae_loss(output: OrganismVAEOutput, batch: dict[str, Tensor], config: OrganismVAEConfig, *, beta_scale: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
    occupancy = batch["occupancy"].float(); visible = occupancy[:, None]
    occupancy_loss = F.binary_cross_entropy_with_logits(output.occupancy_logits[:, 0], occupancy)
    pixel_weight = .2 + 1.8 * visible
    rgba_loss = ((output.rgba - batch["rgba"].float()).abs() * pixel_weight).sum() / (pixel_weight.sum() * 4)
    # CUDA's fused NLL backward is not deterministic.  The explicit gather is
    # mathematically identical for unweighted mean CE and supports exact
    # segmented/replay work on the target 4090.
    categorical = sum(((-logits.float().log_softmax(dim=1).gather(1, batch[name].long()[:, None])) * pixel_weight).sum() / pixel_weight.sum() for logits, name in ((output.tissue_logits, "tissue"), (output.material_logits, "material"), (output.part_logits, "part")))
    denominator = visible.sum().clamp_min(1)
    emission_loss = ((output.emission[:, 0] - batch["emission"].float()).abs() * occupancy).sum() / denominator
    physiology_loss = ((output.physiology - batch["physiology"].float()).abs() * visible).sum() / (denominator * 8)
    state_loss = ((output.cell_state - batch["cell_state"].float()).abs() * visible).sum() / (denominator * 10)
    kl_map = -.5 * (1 + output.log_variance - output.mean.square() - output.log_variance.exp())
    kl = torch.maximum(kl_map.mean(dim=(2, 3)), torch.full_like(kl_map.mean(dim=(2, 3)), config.free_bits)).mean()
    occupancy_probability = output.occupancy_logits.sigmoid(); symmetry_prior = torch.tensor((.88, .72, .62, .22, .84), device=occupancy.device)[batch["family"].long()]
    symmetry = ((occupancy_probability - occupancy_probability.flip(-1)).abs().mean(dim=(1, 2, 3)) * symmetry_prior).mean()
    alpha_consistency = (output.rgba[:, 3] - occupancy_probability).abs().mean()
    target_edges = (occupancy[:, :, 1:] - occupancy[:, :, :-1]).abs().mean() + (occupancy[:, 1:, :] - occupancy[:, :-1, :]).abs().mean()
    predicted_edges = (occupancy_probability[:, :, :, 1:] - occupancy_probability[:, :, :, :-1]).abs().mean() + (occupancy_probability[:, :, 1:, :] - occupancy_probability[:, :, :-1, :]).abs().mean()
    edge_loss = (predicted_edges - target_edges).abs()
    reconstruction = config.occupancy_weight * occupancy_loss + config.rgba_weight * rgba_loss + config.categorical_weight * categorical + config.physiology_weight * physiology_loss + config.cell_state_weight * state_loss + .25 * emission_loss + config.symmetry_weight * symmetry + config.alpha_consistency_weight * alpha_consistency + .12 * edge_loss
    total = reconstruction + config.beta * float(beta_scale) * kl
    return total, {"loss": total.detach(), "reconstruction": reconstruction.detach(), "occupancy_bce": occupancy_loss.detach(), "rgba_l1": rgba_loss.detach(), "categorical_ce": categorical.detach(), "emission_l1": emission_loss.detach(), "physiology_l1": physiology_loss.detach(), "cell_state_l1": state_loss.detach(), "kl": kl.detach(), "symmetry_l1": symmetry.detach(), "alpha_consistency_l1": alpha_consistency.detach(), "edge_l1": edge_loss.detach()}


@torch.no_grad()
def reconstruction_metrics(output: OrganismVAEOutput, batch: dict[str, Tensor]) -> dict[str, float]:
    predicted = output.occupancy_logits[:, 0] >= 0; target = batch["occupancy"].bool(); intersection = (predicted & target).flatten(1).sum(1).float(); union = (predicted | target).flatten(1).sum(1).float().clamp_min(1)
    visible = target[:, None]; denominator = visible.sum().clamp_min(1)
    return {
        "silhouette_iou": float((intersection / union).mean()), "rgba_mae": float((output.rgba - batch["rgba"]).abs().mean()),
        "tissue_accuracy": float((output.tissue_logits.argmax(1) == batch["tissue"]).float().mean()), "material_accuracy": float((output.material_logits.argmax(1) == batch["material"]).float().mean()),
        "part_accuracy": float((output.part_logits.argmax(1) == batch["part"]).float().mean()), "physiology_mae_visible": float(((output.physiology - batch["physiology"]).abs() * visible).sum() / (denominator * 8)),
        "cell_state_mae_visible": float(((output.cell_state - batch["cell_state"]).abs() * visible).sum() / (denominator * 10)), "latent_mean_abs": float(output.mean.abs().mean()), "latent_std_mean": float(torch.exp(.5 * output.log_variance).mean()),
    }
