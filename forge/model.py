from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.block(inputs)


class UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.residual = ResidualBlock(output_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        return self.residual(self.project(inputs))


@dataclass(slots=True)
class VAEOutput:
    logits: Tensor
    mu: Tensor
    logvar: Tensor
    latent: Tensor


class SemanticBetaVAE(nn.Module):
    """A conditional VAE that generates riggable semantic pixel layers."""

    def __init__(
        self,
        layer_count: int = 8,
        condition_count: int = 4,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.condition_count = condition_count
        self.latent_dim = latent_dim

        encoder_input = layer_count + condition_count
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_input, 48, 4, stride=2, padding=1),
            ResidualBlock(48),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),
            ResidualBlock(96),
            nn.Conv2d(96, 160, 4, stride=2, padding=1),
            ResidualBlock(160),
        )
        self.to_mu = nn.Linear(160 * 4 * 4, latent_dim)
        self.to_logvar = nn.Linear(160 * 4 * 4, latent_dim)

        self.condition_embedding = nn.Embedding(condition_count, 24)
        self.from_latent = nn.Linear(latent_dim + 24, 192 * 4 * 4)
        self.decoder = nn.Sequential(
            ResidualBlock(192),
            UpsampleBlock(192, 160),
            UpsampleBlock(160, 96),
            UpsampleBlock(96, 64),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, layer_count, 3, padding=1),
        )

    def condition_planes(self, labels: Tensor, height: int, width: int) -> Tensor:
        one_hot = F.one_hot(labels, self.condition_count).to(dtype=torch.float32)
        return one_hot[:, :, None, None].expand(-1, -1, height, width)

    def encode(self, layers: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        condition = self.condition_planes(labels, layers.shape[-2], layers.shape[-1])
        encoded = self.encoder(torch.cat((layers, condition), dim=1))
        flattened = encoded.flatten(1)
        return self.to_mu(flattened), self.to_logvar(flattened)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        noise = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * noise

    def decode(self, latent: Tensor, labels: Tensor) -> Tensor:
        condition = self.condition_embedding(labels)
        decoded = self.from_latent(torch.cat((latent, condition), dim=1))
        return self.decoder(decoded.view(-1, 192, 4, 4))

    def forward(self, layers: Tensor, labels: Tensor) -> VAEOutput:
        mu, logvar = self.encode(layers, labels)
        latent = self.reparameterize(mu, logvar)
        return VAEOutput(
            logits=self.decode(latent, labels),
            mu=mu,
            logvar=logvar,
            latent=latent,
        )

    @torch.no_grad()
    def sample(self, labels: Tensor, temperature: float = 1.0) -> Tensor:
        latent = torch.randn(
            labels.shape[0],
            self.latent_dim,
            device=labels.device,
            dtype=self.condition_embedding.weight.dtype,
        )
        return self.decode(latent * temperature, labels)


def silhouette_edges(mask: Tensor) -> Tensor:
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def vae_loss(
    output: VAEOutput,
    target: Tensor,
    beta: float,
    dice_weight: float,
    pos_weight: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    reconstruction = F.binary_cross_entropy_with_logits(
        output.logits,
        target,
        pos_weight=pos_weight.view(1, -1, 1, 1),
    )
    probability = output.logits.sigmoid()
    intersection = (probability * target).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    target_silhouette = target[:, :6].amax(dim=1, keepdim=True)
    predicted_silhouette = probability[:, :6].amax(dim=1, keepdim=True)
    boundary = F.l1_loss(
        silhouette_edges(predicted_silhouette),
        silhouette_edges(target_silhouette),
    )
    kl_per_dimension = -0.5 * (
        1.0 + output.logvar - output.mu.square() - output.logvar.exp()
    )
    kl = torch.clamp(kl_per_dimension, min=0.05).mean()
    total = reconstruction + dice_weight * dice + 0.12 * boundary + beta * kl
    return total, {
        "loss": total.detach(),
        "reconstruction": reconstruction.detach(),
        "dice": dice.detach(),
        "boundary": boundary.detach(),
        "kl": kl.detach(),
    }


@torch.no_grad()
def binary_metrics(logits: Tensor, target: Tensor) -> dict[str, float]:
    prediction = logits.sigmoid() >= 0.5
    truth = target >= 0.5
    intersection = (prediction & truth).sum(dim=(0, 2, 3)).float()
    union = (prediction | truth).sum(dim=(0, 2, 3)).float().clamp_min(1.0)
    iou = intersection / union
    exact = (prediction == truth).float().mean()
    nonempty = prediction[:, 0].flatten(1).any(dim=1).float().mean()
    return {
        "mean_iou": float(iou.mean().item()),
        "hull_iou": float(iou[0].item()),
        "pixel_accuracy": float(exact.item()),
        "nonempty_rate": float(nonempty.item()),
    }
