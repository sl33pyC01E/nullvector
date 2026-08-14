from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import OrganismVAEV2Config


class ModulatedBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__(); groups = min(32, channels)
        while channels % groups: groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels); self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, channels * 2)
        squeezed = max(16, channels // 8); self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, squeezed, 1), nn.SiLU(), nn.Conv2d(squeezed, channels, 1), nn.Sigmoid())

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1); hidden = self.norm1(value) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden)); hidden = self.conv2(F.silu(self.norm2(hidden))); return value + hidden * self.attention(hidden)


class LearnedUpsample(nn.Module):
    def __init__(self, source: int, target: int) -> None:
        super().__init__(); self.project = nn.Conv2d(source, target * 4, 3, padding=1); self.shuffle = nn.PixelShuffle(2); self.refine = nn.Conv2d(target, target, 3, padding=1)
    def forward(self, value: Tensor) -> Tensor: return self.refine(F.silu(self.shuffle(self.project(value))))


@dataclass(slots=True)
class HierarchicalOutput:
    rgb: Tensor; occupancy_logits: Tensor; coarse_occupancy_logits: Tensor
    tissue_logits: Tensor; material_logits: Tensor; part_logits: Tensor
    emission: Tensor; physiology: Tensor; system_role_logits: Tensor; cell_state: Tensor
    coarse_mean: Tensor; coarse_log_variance: Tensor; coarse_latent: Tensor
    fine_mean: Tensor; fine_log_variance: Tensor; fine_latent: Tensor

    @property
    def rgba(self) -> Tensor: return torch.cat((self.rgb, self.occupancy_logits.sigmoid()), dim=1)


class HierarchicalOrganismRasterVAE(nn.Module):
    def __init__(self, config: OrganismVAEV2Config = OrganismVAEV2Config()) -> None:
        super().__init__(); self.config = config; w = config.width; c = config.coarse_width; mid = w * 2
        self.family = nn.Embedding(5, 16); self.subtype = nn.Embedding(20, 16); self.role = nn.Embedding(8, 16)
        self.gene = nn.Sequential(nn.Linear(16, 48), nn.SiLU(), nn.Linear(48, 32)); self.style = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 32))
        self.condition = nn.Sequential(nn.Linear(112, config.condition_dim), nn.SiLU(), nn.Linear(config.condition_dim, config.condition_dim))
        self.stem = nn.Conv2d(74, w, 3, padding=1); self.enc0 = nn.ModuleList(ModulatedBlock(w, config.condition_dim) for _ in range(config.residual_depth))
        self.down1 = nn.Conv2d(w, mid, 4, stride=2, padding=1); self.enc1 = nn.ModuleList(ModulatedBlock(mid, config.condition_dim) for _ in range(config.residual_depth))
        self.fine_mean = nn.Conv2d(mid, config.fine_latent_channels, 1); self.fine_logvar = nn.Conv2d(mid, config.fine_latent_channels, 1)
        self.down2 = nn.Conv2d(mid, c, 4, stride=2, padding=1); self.enc2 = nn.ModuleList(ModulatedBlock(c, config.condition_dim) for _ in range(config.residual_depth))
        self.coarse_mean = nn.Conv2d(c, config.coarse_latent_channels, 1); self.coarse_logvar = nn.Conv2d(c, config.coarse_latent_channels, 1)
        self.coarse_in = nn.Conv2d(config.coarse_latent_channels, c, 1); self.dec2 = nn.ModuleList(ModulatedBlock(c, config.condition_dim) for _ in range(config.residual_depth))
        self.up1 = LearnedUpsample(c, mid); self.fine_fuse = nn.Conv2d(mid + config.fine_latent_channels, mid, 3, padding=1); self.dec1 = nn.ModuleList(ModulatedBlock(mid, config.condition_dim) for _ in range(config.residual_depth))
        self.coarse_occupancy = nn.Conv2d(mid, 1, 1); self.up2 = LearnedUpsample(mid, w); self.dec0 = nn.ModuleList(ModulatedBlock(w, config.condition_dim) for _ in range(config.residual_depth)); self.shared = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.SiLU(), nn.Conv2d(w, w, 3, padding=1), nn.SiLU())
        self.rgb = nn.Conv2d(w, 3, 1); self.occupancy = nn.Conv2d(w, 1, 1); self.tissue = nn.Conv2d(w, 15, 1); self.material = nn.Conv2d(w, 10, 1); self.part = nn.Conv2d(w, 17, 1); self.emission = nn.Conv2d(w, 1, 1); self.physiology = nn.Conv2d(w, 8, 1); self.system_role = nn.Conv2d(w, 8 * 4, 1); self.cell_state = nn.Conv2d(w, 10, 1)

    def condition_vector(self, family: Tensor, subtype: Tensor, role: Tensor, genes: Tensor, style: Tensor) -> Tensor:
        if genes.shape != (len(family), 16) or style.shape != (len(family), 8) or family.shape != subtype.shape or family.shape != role.shape: raise ValueError("Organism VAE v2 conditions are misaligned.")
        return self.condition(torch.cat((self.family(family), self.subtype(subtype), self.role(role), self.gene(genes.float()), self.style(style.float())), dim=1))

    def encode(self, living: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if living.ndim != 4 or living.shape[1:] != (74, 48, 48): raise ValueError("Organism VAE v2 living field must be B,74,48,48.")
        value = self.stem(living.float())
        for block in self.enc0: value = block(value, condition)
        value = F.silu(self.down1(value))
        for block in self.enc1: value = block(value, condition)
        fine_mean, fine_logvar = self.fine_mean(value), self.fine_logvar(value).clamp(-10, 6); value = F.silu(self.down2(value))
        for block in self.enc2: value = block(value, condition)
        return self.coarse_mean(value), self.coarse_logvar(value).clamp(-10, 6), fine_mean, fine_logvar

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor, generator: torch.Generator | None, sample: bool) -> Tensor:
        if not sample: return mean
        return mean + torch.exp(.5 * log_variance) * torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)

    def decode(self, coarse: Tensor, fine: Tensor, condition: Tensor) -> HierarchicalOutput:
        if coarse.shape[1:] != (self.config.coarse_latent_channels, 12, 12) or fine.shape[1:] != (self.config.fine_latent_channels, 24, 24): raise ValueError("Organism VAE v2 latent pyramid drifted.")
        value = self.coarse_in(coarse)
        for block in self.dec2: value = block(value, condition)
        value = self.fine_fuse(torch.cat((self.up1(value), fine), dim=1))
        for block in self.dec1: value = block(value, condition)
        coarse_occupancy = self.coarse_occupancy(value); value = self.up2(value)
        for block in self.dec0: value = block(value, condition)
        value = self.shared(value); empty = torch.empty(0, device=value.device)
        roles = self.system_role(value).reshape(len(value), 8, 4, 48, 48)
        return HierarchicalOutput(torch.sigmoid(self.rgb(value)), self.occupancy(value), coarse_occupancy, self.tissue(value), self.material(value), self.part(value), torch.sigmoid(self.emission(value)), torch.sigmoid(self.physiology(value)), roles, torch.sigmoid(self.cell_state(value)), empty, empty, coarse, empty, empty, fine)

    def forward(self, living: Tensor, family: Tensor, subtype: Tensor, role: Tensor, genes: Tensor, style: Tensor, *, generator: torch.Generator | None = None, sample: bool = True) -> HierarchicalOutput:
        condition = self.condition_vector(family, subtype, role, genes, style); cm, clv, fm, flv = self.encode(living, condition); coarse = self.reparameterize(cm, clv, generator, sample); fine = self.reparameterize(fm, flv, generator, sample); result = self.decode(coarse, fine, condition); result.coarse_mean, result.coarse_log_variance, result.fine_mean, result.fine_log_variance = cm, clv, fm, flv; return result


def _stable_bce(logits: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    value = F.softplus(logits.float()) - target * logits.float(); return (value * weight).sum() / weight.sum()


def _categorical(logits: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    value = -logits.float().log_softmax(1).gather(1, target.long()[:, None]); return (value * weight).sum() / weight.sum()


def _kl(mean: Tensor, log_variance: Tensor, free_bits: float) -> Tensor:
    value = -.5 * (1 + log_variance.float() - mean.float().square() - log_variance.float().exp()).mean(dim=(2, 3)); return value.clamp_min(free_bits).mean()


def hierarchical_loss(output: HierarchicalOutput, batch: dict[str, Tensor], config: OrganismVAEV2Config, beta_scale: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
    occupancy = batch["occupancy"].float()[:, None]; weight = .15 + 2.35 * occupancy; probability = output.occupancy_logits.sigmoid()
    occupancy_bce = _stable_bce(output.occupancy_logits, occupancy, weight); intersection = (probability * occupancy).sum((1, 2, 3)); dice = 1 - ((2 * intersection + 1) / (probability.sum((1, 2, 3)) + occupancy.sum((1, 2, 3)) + 1)).mean()
    rgb_l1 = ((output.rgb - batch["rgba"][:, :3].float()).abs() * weight).sum() / (weight.sum() * 3)
    categorical = sum(_categorical(logits, batch[name], weight) for logits, name in ((output.tissue_logits, "tissue"), (output.material_logits, "material"), (output.part_logits, "part")))
    visible_count = occupancy.sum().clamp_min(1); role_target = batch["system_role"].long(); member = (role_target > 0).float(); physiology_weight = occupancy[:, None] * (.2 + 2.8 * member); physiology = ((output.physiology - batch["physiology"]).abs() * physiology_weight).sum() / physiology_weight.sum().clamp_min(1)
    role_weight = .02 + .18 * occupancy[:, None] + 3.8 * member; role_nll = -output.system_role_logits.float().log_softmax(2).gather(2, role_target[:, :, None]).squeeze(2); system_role = (role_nll * role_weight).sum() / role_weight.sum()
    state = ((output.cell_state - batch["cell_state"]).abs() * occupancy).sum() / (visible_count * 10); emission = ((output.emission - batch["emission"][:, None]).abs() * occupancy).sum() / visible_count
    target_coarse = F.avg_pool2d(occupancy, 2); coarse_bce = _stable_bce(output.coarse_occupancy_logits, target_coarse, .2 + 1.8 * target_coarse)
    target_rgb = batch["rgba"][:, :3].float(); per_sample_count = occupancy.sum((2, 3)).clamp_min(1); target_mean = (target_rgb * occupancy).sum((2, 3)) / per_sample_count; output_mean = (output.rgb * occupancy).sum((2, 3)) / per_sample_count
    target_std = (((target_rgb - target_mean[:, :, None, None]).square() * occupancy).sum((2, 3)) / per_sample_count).sqrt(); output_std = (((output.rgb - output_mean[:, :, None, None]).square() * occupancy).sum((2, 3)) / per_sample_count).sqrt(); palette = (target_mean - output_mean).abs().mean() + (target_std - output_std).abs().mean()
    edge = sum((a - b).abs().mean() for a, b in (((probability[:, :, :, 1:] - probability[:, :, :, :-1]), (occupancy[:, :, :, 1:] - occupancy[:, :, :, :-1])), ((probability[:, :, 1:, :] - probability[:, :, :-1, :]), (occupancy[:, :, 1:, :] - occupancy[:, :, :-1, :]))))
    family_prior = torch.tensor((.88, .72, .62, .22, .84), device=occupancy.device)[batch["family"].long()]; symmetry = ((probability - probability.flip(-1)).abs().mean((1, 2, 3)) * family_prior).mean()
    kl_coarse = _kl(output.coarse_mean, output.coarse_log_variance, config.free_bits); kl_fine = _kl(output.fine_mean, output.fine_log_variance, config.free_bits)
    reconstruction = occupancy_bce + .55 * dice + 2.8 * rgb_l1 + .75 * categorical + 1.05 * physiology + .65 * system_role + .55 * state + .25 * emission + .35 * coarse_bce + .45 * palette + .2 * edge + .08 * symmetry
    total = reconstruction + float(beta_scale) * (config.beta_coarse * kl_coarse + config.beta_fine * kl_fine)
    return total, {"loss": total.detach(), "reconstruction": reconstruction.detach(), "occupancy_bce": occupancy_bce.detach(), "silhouette_dice_loss": dice.detach(), "rgb_l1": rgb_l1.detach(), "categorical_ce": categorical.detach(), "physiology_l1": physiology.detach(), "system_role_ce": system_role.detach(), "cell_state_l1": state.detach(), "emission_l1": emission.detach(), "coarse_occupancy_bce": coarse_bce.detach(), "palette_l1": palette.detach(), "edge_l1": edge.detach(), "symmetry_l1": symmetry.detach(), "kl_coarse": kl_coarse.detach(), "kl_fine": kl_fine.detach()}


@torch.no_grad()
def metrics(output: HierarchicalOutput, batch: dict[str, Tensor]) -> dict[str, float]:
    predicted = output.occupancy_logits[:, 0] >= 0; target = batch["occupancy"].bool(); intersection = (predicted & target).flatten(1).sum(1).float(); union = (predicted | target).flatten(1).sum(1).float().clamp_min(1); visible = target[:, None]; count = visible.sum().clamp_min(1)
    predicted_roles = output.system_role_logits.argmax(2); target_roles = batch["system_role"]; members = target_roles > 0; cores = target_roles == 1
    member_accuracy = float((predicted_roles[members] == target_roles[members]).float().mean()) if bool(members.any()) else 1.0; core_recall = float((predicted_roles[cores] == 1).float().mean()) if bool(cores.any()) else 1.0
    return {"silhouette_iou": float((intersection / union).mean()), "rgba_mae": float((output.rgba - batch["rgba"]).abs().mean()), "foreground_rgb_mae": float(((output.rgb - batch["rgba"][:, :3]).abs() * visible).sum() / (count * 3)), "tissue_accuracy": float((output.tissue_logits.argmax(1) == batch["tissue"]).float().mean()), "material_accuracy": float((output.material_logits.argmax(1) == batch["material"]).float().mean()), "part_accuracy": float((output.part_logits.argmax(1) == batch["part"]).float().mean()), "physiology_mae_visible": float(((output.physiology - batch["physiology"]).abs() * visible).sum() / (count * 8)), "system_role_member_accuracy": member_accuracy, "system_core_recall": core_recall, "cell_state_mae_visible": float(((output.cell_state - batch["cell_state"]).abs() * visible).sum() / (count * 10)), "coarse_latent_mean_abs": float(output.coarse_mean.abs().mean()), "coarse_latent_std_mean": float(torch.exp(.5 * output.coarse_log_variance).mean()), "fine_latent_mean_abs": float(output.fine_mean.abs().mean()), "fine_latent_std_mean": float(torch.exp(.5 * output.fine_log_variance).mean())}
