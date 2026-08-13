from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

TOKEN_COUNT = 9
MASK_TOKEN = TOKEN_COUNT


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = min(16, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, channels * 2)

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm1(inputs)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class CategoricalSpriteDiffusion(nn.Module):
    """Absorbing-state categorical diffusion over crisp semantic part tokens."""

    def __init__(
        self,
        token_count: int = TOKEN_COUNT,
        archetype_count: int = 4,
        gene_dim: int = 8,
        steps: int = 12,
        width: int = 64,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        if image_size < 16 or image_size % 4 != 0:
            raise ValueError("image_size must be at least 16 and divisible by four.")
        self.token_count = token_count
        self.mask_token = token_count
        self.archetype_count = archetype_count
        self.gene_dim = gene_dim
        self.steps = steps
        self.width = width
        self.image_size = image_size
        condition_dim = 128

        self.token_embedding = nn.Embedding(token_count + 1, width)
        self.archetype_embedding = nn.Embedding(archetype_count, 32)
        self.gene_embedding = nn.Sequential(
            nn.Linear(gene_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.time_embedding = nn.Embedding(steps + 1, 32)
        self.condition_mix = nn.Sequential(
            nn.Linear(128, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        self.input_projection = nn.Conv2d(width, width, 3, padding=1)
        self.down0 = nn.ModuleList(
            [FiLMResidualBlock(width, condition_dim) for _ in range(2)]
        )
        self.downsample0 = nn.Conv2d(width, 96, 4, stride=2, padding=1)
        self.down1 = nn.ModuleList(
            [FiLMResidualBlock(96, condition_dim) for _ in range(2)]
        )
        self.downsample1 = nn.Conv2d(96, 144, 4, stride=2, padding=1)
        self.middle = nn.ModuleList(
            [FiLMResidualBlock(144, condition_dim) for _ in range(3)]
        )
        self.up1_project = nn.Conv2d(144 + 96, 96, 3, padding=1)
        self.up1 = nn.ModuleList(
            [FiLMResidualBlock(96, condition_dim) for _ in range(2)]
        )
        self.up0_project = nn.Conv2d(96 + width, width, 3, padding=1)
        self.up0 = nn.ModuleList(
            [FiLMResidualBlock(width, condition_dim) for _ in range(2)]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(16, width),
            nn.SiLU(),
            nn.Conv2d(width, token_count, 1),
        )

    def architecture_config(self) -> dict[str, int | str]:
        """Return every constructor value needed to restore inference exactly."""
        return {
            "name": "categorical-absorbing-diffusion-unet",
            "token_count": self.token_count,
            "mask_token": self.mask_token,
            "archetype_count": self.archetype_count,
            "gene_dim": self.gene_dim,
            "steps": self.steps,
            "width": self.width,
            "image_size": self.image_size,
        }

    def build_condition(
        self, archetypes: Tensor, genes: Tensor, timesteps: Tensor
    ) -> Tensor:
        joined = torch.cat(
            (
                self.archetype_embedding(archetypes),
                self.gene_embedding(genes),
                self.time_embedding(timesteps),
            ),
            dim=1,
        )
        return self.condition_mix(joined)

    @staticmethod
    def _run_blocks(
        inputs: Tensor,
        condition: Tensor,
        blocks: nn.ModuleList,
    ) -> Tensor:
        for block in blocks:
            inputs = block(inputs, condition)
        return inputs

    def forward(
        self,
        tokens: Tensor,
        archetypes: Tensor,
        genes: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        condition = self.build_condition(archetypes, genes, timesteps)
        hidden = self.token_embedding(tokens).permute(0, 3, 1, 2)
        level0 = self._run_blocks(
            self.input_projection(hidden), condition, self.down0
        )
        level1 = self._run_blocks(self.downsample0(level0), condition, self.down1)
        middle = self._run_blocks(self.downsample1(level1), condition, self.middle)

        hidden = F.interpolate(middle, scale_factor=2.0, mode="nearest")
        hidden = self.up1_project(torch.cat((hidden, level1), dim=1))
        hidden = self._run_blocks(hidden, condition, self.up1)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="nearest")
        hidden = self.up0_project(torch.cat((hidden, level0), dim=1))
        hidden = self._run_blocks(hidden, condition, self.up0)
        return self.output(hidden)

    def mask_probability(self, timesteps: Tensor) -> Tensor:
        phase = timesteps.float() / float(self.steps)
        return torch.sin(phase * math.pi * 0.5).square()

    def corrupt(
        self,
        clean_tokens: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        probability = self.mask_probability(timesteps)[:, None, None]
        masked = torch.rand_like(clean_tokens, dtype=torch.float32) < probability
        empty = ~masked.flatten(1).any(dim=1)
        if empty.any():
            rows = torch.nonzero(empty, as_tuple=False).flatten()
            positions = torch.randint(
                0,
                clean_tokens.shape[-2] * clean_tokens.shape[-1],
                (rows.shape[0],),
                device=clean_tokens.device,
            )
            masked[rows, positions // clean_tokens.shape[-1], positions % clean_tokens.shape[-1]] = True
        corrupted = clean_tokens.masked_fill(masked, self.mask_token)
        return corrupted, masked

    @torch.no_grad()
    def sample(
        self,
        archetypes: Tensor,
        genes: Tensor,
        *,
        temperature: float = 0.92,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator] | None = None,
    ) -> Tensor:
        if generator is not None and generators is not None:
            raise ValueError("Pass either generator or per-sample generators, not both.")
        batch = archetypes.shape[0]
        if generators is not None and len(generators) != batch:
            raise ValueError(
                f"Expected {batch} per-sample generators, got {len(generators)}."
            )
        device = archetypes.device
        tokens = torch.full(
            (batch, self.image_size, self.image_size),
            self.mask_token,
            dtype=torch.long,
            device=device,
        )
        for step in range(self.steps, 0, -1):
            timesteps = torch.full(
                (batch,), step, dtype=torch.long, device=device
            )
            logits = self(tokens, archetypes, genes, timesteps) / max(temperature, 0.05)
            probabilities = logits.softmax(dim=1)
            if generators is None:
                flat_probability = probabilities.permute(0, 2, 3, 1).reshape(
                    -1, self.token_count
                )
                candidates = torch.multinomial(
                    flat_probability,
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).view(batch, self.image_size, self.image_size)
            else:
                sampled = []
                for batch_index in range(batch):
                    flat_probability = probabilities[batch_index].permute(
                        1, 2, 0
                    ).reshape(-1, self.token_count)
                    sampled.append(
                        torch.multinomial(
                            flat_probability,
                            num_samples=1,
                            replacement=True,
                            generator=generators[batch_index],
                        ).view(self.image_size, self.image_size)
                    )
                candidates = torch.stack(sampled, dim=0)
            confidence = probabilities.gather(
                1, candidates[:, None, :, :]
            ).squeeze(1)
            currently_masked = tokens == self.mask_token
            next_phase = float(step - 1) / float(self.steps)
            remaining_target = int(
                round(
                    self.image_size
                    * self.image_size
                    * math.sin(next_phase * math.pi * 0.5) ** 2
                )
            )
            for batch_index in range(batch):
                positions = torch.nonzero(
                    currently_masked[batch_index], as_tuple=False
                )
                fill_count = max(0, positions.shape[0] - remaining_target)
                if step == 1:
                    fill_count = positions.shape[0]
                if fill_count == 0:
                    continue
                position_confidence = confidence[
                    batch_index, positions[:, 0], positions[:, 1]
                ]
                selected = torch.topk(
                    position_confidence,
                    k=fill_count,
                    largest=True,
                ).indices
                chosen = positions[selected]
                tokens[batch_index, chosen[:, 0], chosen[:, 1]] = candidates[
                    batch_index, chosen[:, 0], chosen[:, 1]
                ]
        return tokens


@dataclass(slots=True)
class DiffusionLoss:
    loss: Tensor
    accuracy: Tensor
    masked_fraction: Tensor


def categorical_diffusion_loss(
    logits: Tensor,
    clean_tokens: Tensor,
    masked: Tensor,
    class_weight: Tensor | None = None,
) -> DiffusionLoss:
    per_pixel = F.cross_entropy(
        logits,
        clean_tokens,
        weight=class_weight,
        reduction="none",
    )
    weights = masked.float()
    loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)
    predictions = logits.argmax(dim=1)
    accuracy = ((predictions == clean_tokens) & masked).float().sum() / weights.sum().clamp_min(1.0)
    return DiffusionLoss(
        loss=loss,
        accuracy=accuracy,
        masked_fraction=weights.mean(),
    )
