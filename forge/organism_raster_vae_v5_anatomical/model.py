from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..organism_raster_vae_v3.model import StructuredRasterVAE, VAEOutput, loss as base_loss
from .contract import MAX_TOKENS, TOKEN_FEATURES


@dataclass(slots=True)
class AnatomicalOutput(VAEOutput):
    attention12: Tensor
    attention24: Tensor


class AnatomicalGraphRasterVAE(StructuredRasterVAE):
    def __init__(self, config: RasterVAEV3Config = RasterVAEV3Config()) -> None:
        super().__init__(config)
        self.token_embed = nn.Sequential(
            nn.Linear(TOKEN_FEATURES, 320),
            nn.SiLU(),
            nn.Linear(320, 320),
            nn.LayerNorm(320),
        )
        self.token12 = nn.Linear(320, config.anatomy_width)
        self.token24 = nn.Linear(320, config.mid_width)
        self.position12 = nn.Linear(6, config.anatomy_width)
        self.position24 = nn.Linear(6, config.mid_width)
        self.attend12 = nn.MultiheadAttention(config.anatomy_width, 8, batch_first=True)
        self.attend24 = nn.MultiheadAttention(config.mid_width, 8, batch_first=True)
        self.gate12 = nn.Parameter(torch.zeros(()))
        self.gate24 = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _position(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        axis = torch.linspace(-1, 1, size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack(
            (
                xx,
                yy,
                torch.sin(torch.pi * xx),
                torch.cos(torch.pi * xx),
                torch.sin(torch.pi * yy),
                torch.cos(torch.pi * yy),
            ),
            -1,
        ).reshape(1, size * size, 6)

    def _attention(
        self,
        value: Tensor,
        tokens: Tensor,
        mask: Tensor,
        token_projection: nn.Linear,
        position_projection: nn.Linear,
        attention: nn.MultiheadAttention,
        gate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = value.shape
        query = value.flatten(2).transpose(1, 2)
        query = query + position_projection(self._position(height, value.device, value.dtype))
        keys = token_projection(tokens)
        attended, weights = attention(
            query,
            keys,
            keys,
            key_padding_mask=~mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )
        query = query + torch.tanh(gate) * attended
        return query.transpose(1, 2).reshape(batch, channels, height, width), weights

    def forward(
        self,
        living: Tensor,
        family: Tensor,
        traits: Tensor,
        phase: Tensor,
        tokens: Tensor,
        token_mask: Tensor,
        *,
        generator: torch.Generator | None = None,
        stochastic: bool = True,
    ) -> AnatomicalOutput:
        if tokens.shape[1:] != (MAX_TOKENS, TOKEN_FEATURES):
            raise ValueError("anatomical token geometry drifted")
        if token_mask.shape != tokens.shape[:2] or not bool(token_mask.any(1).all()):
            raise ValueError("anatomical token mask drifted")
        condition = self.condition_vector(family, traits, phase)
        token = self.token_embed(tokens.float())
        value = self.stem(living.float())
        for block in self.e48:
            value = block(value, condition)
        value = F.silu(self.d24(value))
        for block in self.e24:
            value = block(value, condition)
        fine_mean, fine_logvar = self.fmu(value), self.flv(value).clamp(-10, 5)
        fine = self.sample(fine_mean, fine_logvar, generator, stochastic)
        value = F.silu(self.d12(value))
        for block in self.e12:
            value = block(value, condition)
        anatomy_mean, anatomy_logvar = self.amu(value), self.alv(value).clamp(-10, 5)
        anatomy = self.sample(anatomy_mean, anatomy_logvar, generator, stochastic)
        value = F.silu(self.d6(value))
        for block in self.e6:
            value = block(value, condition)
        global_mean, global_logvar = self.gmu(value), self.glv(value).clamp(-10, 5)
        global_latent = self.sample(global_mean, global_logvar, generator, stochastic)
        value = self.gin(global_latent)
        for block in self.x6:
            value = block(value, condition)
        value = self.afuse(torch.cat((self.u12(value), anatomy), 1))
        value, weights12 = self._attention(
            value, token, token_mask, self.token12, self.position12, self.attend12, self.gate12
        )
        for block in self.x12:
            value = block(value, condition)
        value = self.ffuse(torch.cat((self.u24(value), fine), 1))
        value, weights24 = self._attention(
            value, token, token_mask, self.token24, self.position24, self.attend24, self.gate24
        )
        for block in self.x24:
            value = block(value, condition)
        value = self.u48(value)
        for block in self.x48:
            value = block(value, condition)
        occupancy = self.occupancy(value)
        tissue = self.tissue(value)
        rendered = self.render(self.u96(value))
        cell_alpha = F.interpolate(occupancy, size=(96, 96), mode="nearest")
        rgba = torch.cat(
            (torch.sigmoid(rendered[:, :3]), torch.sigmoid(rendered[:, 3:] + cell_alpha * 1.35)), 1
        )
        return AnatomicalOutput(
            rgba,
            occupancy,
            tissue,
            fine_mean,
            fine_logvar,
            fine,
            anatomy_mean,
            anatomy_logvar,
            anatomy,
            global_mean,
            global_logvar,
            global_latent,
            weights12,
            weights24,
        )


def _authority_nll(attention: Tensor, owner: Tensor) -> tuple[Tensor, Tensor, int]:
    target = owner[:, ::2, ::2].reshape(len(attention), -1)
    valid = target >= 0
    probability = attention.clamp_min(1e-7)
    selected = -probability.gather(2, target.clamp_min(0)[:, :, None]).squeeze(2).log()
    nll = selected[valid].mean() if bool(valid.any()) else selected.sum() * 0
    correct = ((attention.argmax(2) == target) & valid).sum()
    return nll, correct, int(valid.sum())


def loss(
    output: AnatomicalOutput,
    batch: dict[str, Tensor],
    config: RasterVAEV3Config,
    beta_scale: float,
) -> tuple[Tensor, dict[str, float]]:
    base, metrics = base_loss(output, batch, config, beta_scale)
    authority = {}
    total_nll = base.new_zeros(())
    weights = {"appendage": .35, "joint": .50, "organ": .55}
    for name in ("appendage", "joint", "organ"):
        nll, correct, count = _authority_nll(output.attention24, batch[f"{name}_owner"])
        total_nll = total_nll + weights[name] * nll
        authority[f"{name}_owner_nll"] = float(nll.detach())
        authority[f"{name}_owner_accuracy"] = float(correct.detach()) / max(count, 1)
    # Coarse and fine anatomical attention should agree after spatial pooling.
    coarse = output.attention12
    fine = output.attention24.reshape(len(output.rgba), 24, 24, MAX_TOKENS)
    fine = fine.reshape(len(output.rgba), 12, 2, 12, 2, MAX_TOKENS).mean((2, 4)).reshape_as(coarse)
    hierarchy = F.smooth_l1_loss(fine, coarse)
    total = base + total_nll + .25 * hierarchy
    metrics.update(authority)
    metrics.update({"attention_hierarchy": float(hierarchy.detach()), "loss": float(total.detach())})
    return total, metrics
