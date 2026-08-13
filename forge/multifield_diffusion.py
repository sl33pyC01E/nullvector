from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .diffusion import FiLMResidualBlock


def seeded_generators(
    seeds: Sequence[int], device: torch.device | str
) -> list[torch.Generator]:
    """Construct independent device-correct generators for batch-stable replay."""
    target = torch.device(device)
    generator_device = target if target.type == "cuda" else torch.device("cpu")
    return [
        torch.Generator(device=generator_device).manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        for seed in seeds
    ]


def _validate_generator_device(
    generator: torch.Generator | None, device: torch.device
) -> None:
    if generator is None:
        return
    expected = device.type
    observed = torch.device(generator.device).type
    if observed != expected:
        raise ValueError(
            f"Generator device {observed!r} does not match tensor device {expected!r}."
        )


@dataclass(frozen=True, slots=True)
class MultiFieldVocabulary:
    part_count: int = 17
    material_count: int = 10
    emission_count: int = 4

    def to_dict(self) -> dict[str, int]:
        return {
            "part_count": self.part_count,
            "material_count": self.material_count,
            "emission_count": self.emission_count,
        }


@dataclass(slots=True)
class MultiFieldLogits:
    part: Tensor
    material: Tensor
    emission: Tensor


@dataclass(slots=True)
class MultiFieldLoss:
    loss: Tensor
    part_loss: Tensor
    material_loss: Tensor
    emission_loss: Tensor
    part_accuracy: Tensor
    material_accuracy: Tensor
    emission_accuracy: Tensor
    masked_fraction: Tensor


class MultiFieldSpriteDiffusion(nn.Module):
    """Graph-guided categorical diffusion over aligned sprite fields.

    Part ownership, material, and emission share one corruption mask and one
    spatial backbone. Committing all fields at the same confidence-ranked pixel
    keeps their boundaries aligned throughout the absorbing reverse process.
    """

    def __init__(
        self,
        *,
        vocabulary: MultiFieldVocabulary = MultiFieldVocabulary(),
        morphology_count: int = 5,
        subtype_count: int = 20,
        role_count: int = 8,
        gene_dim: int = 24,
        guide_channels: int = 8,
        steps: int = 16,
        width: int = 96,
        image_size: int = 48,
    ) -> None:
        super().__init__()
        if image_size < 16 or image_size % 4 != 0:
            raise ValueError("image_size must be at least 16 and divisible by four.")
        if width < 32 or width % 32 != 0:
            raise ValueError("width must be at least 32 and divisible by thirty-two.")
        if min(vocabulary.to_dict().values()) < 2:
            raise ValueError("Every categorical field needs at least two values.")
        self.vocabulary = vocabulary
        self.morphology_count = morphology_count
        self.subtype_count = subtype_count
        self.role_count = role_count
        self.gene_dim = gene_dim
        self.guide_channels = guide_channels
        self.steps = steps
        self.width = width
        self.image_size = image_size
        self.part_mask_token = vocabulary.part_count
        self.material_mask_token = vocabulary.material_count
        self.emission_mask_token = vocabulary.emission_count

        field_width = width // 2
        self.part_embedding = nn.Embedding(vocabulary.part_count + 1, field_width)
        self.material_embedding = nn.Embedding(
            vocabulary.material_count + 1, field_width
        )
        self.emission_embedding = nn.Embedding(
            vocabulary.emission_count + 1, field_width
        )
        self.guide_projection = nn.Conv2d(guide_channels, field_width, 3, padding=1)

        self.morphology_embedding = nn.Embedding(morphology_count, 24)
        self.subtype_embedding = nn.Embedding(subtype_count, 32)
        self.role_embedding = nn.Embedding(role_count, 24)
        self.gene_embedding = nn.Sequential(
            nn.Linear(gene_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.time_embedding = nn.Embedding(steps + 1, 32)
        condition_dim = 192
        self.condition_mix = nn.Sequential(
            nn.Linear(24 + 32 + 24 + 64 + 32, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        level1_width = width * 3 // 2
        middle_width = width * 2
        self.input_projection = nn.Conv2d(field_width * 4, width, 3, padding=1)
        self.down0 = nn.ModuleList(
            [FiLMResidualBlock(width, condition_dim) for _ in range(2)]
        )
        self.downsample0 = nn.Conv2d(width, level1_width, 4, stride=2, padding=1)
        self.down1 = nn.ModuleList(
            [FiLMResidualBlock(level1_width, condition_dim) for _ in range(2)]
        )
        self.downsample1 = nn.Conv2d(
            level1_width, middle_width, 4, stride=2, padding=1
        )
        self.middle = nn.ModuleList(
            [FiLMResidualBlock(middle_width, condition_dim) for _ in range(3)]
        )
        self.up1_projection = nn.Conv2d(
            middle_width + level1_width, level1_width, 3, padding=1
        )
        self.up1 = nn.ModuleList(
            [FiLMResidualBlock(level1_width, condition_dim) for _ in range(2)]
        )
        self.up0_projection = nn.Conv2d(
            level1_width + width, width, 3, padding=1
        )
        self.up0 = nn.ModuleList(
            [FiLMResidualBlock(width, condition_dim) for _ in range(2)]
        )
        self.output_norm = nn.Sequential(nn.GroupNorm(16, width), nn.SiLU())
        self.part_output = nn.Conv2d(width, vocabulary.part_count, 1)
        self.material_output = nn.Conv2d(width, vocabulary.material_count, 1)
        self.emission_output = nn.Conv2d(width, vocabulary.emission_count, 1)

    def architecture_config(self) -> dict[str, object]:
        return {
            "name": "graph-guided-multifield-categorical-diffusion-unet",
            "vocabulary": self.vocabulary.to_dict(),
            "morphology_count": self.morphology_count,
            "subtype_count": self.subtype_count,
            "role_count": self.role_count,
            "gene_dim": self.gene_dim,
            "guide_channels": self.guide_channels,
            "steps": self.steps,
            "width": self.width,
            "image_size": self.image_size,
        }

    @staticmethod
    def _run_blocks(inputs: Tensor, condition: Tensor, blocks: nn.ModuleList) -> Tensor:
        for block in blocks:
            inputs = block(inputs, condition)
        return inputs

    def _condition(
        self,
        morphologies: Tensor,
        subtypes: Tensor,
        roles: Tensor,
        genes: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        return self.condition_mix(
            torch.cat(
                (
                    self.morphology_embedding(morphologies),
                    self.subtype_embedding(subtypes),
                    self.role_embedding(roles),
                    self.gene_embedding(genes),
                    self.time_embedding(timesteps),
                ),
                dim=1,
            )
        )

    def forward(
        self,
        part_tokens: Tensor,
        material_tokens: Tensor,
        emission_tokens: Tensor,
        guide: Tensor,
        morphologies: Tensor,
        subtypes: Tensor,
        roles: Tensor,
        genes: Tensor,
        timesteps: Tensor,
    ) -> MultiFieldLogits:
        condition = self._condition(
            morphologies, subtypes, roles, genes, timesteps
        )
        embedded = torch.cat(
            (
                self.part_embedding(part_tokens).permute(0, 3, 1, 2),
                self.material_embedding(material_tokens).permute(0, 3, 1, 2),
                self.emission_embedding(emission_tokens).permute(0, 3, 1, 2),
                self.guide_projection(guide),
            ),
            dim=1,
        )
        level0 = self._run_blocks(
            self.input_projection(embedded), condition, self.down0
        )
        level1 = self._run_blocks(self.downsample0(level0), condition, self.down1)
        middle = self._run_blocks(self.downsample1(level1), condition, self.middle)
        hidden = F.interpolate(middle, scale_factor=2.0, mode="nearest")
        hidden = self.up1_projection(torch.cat((hidden, level1), dim=1))
        hidden = self._run_blocks(hidden, condition, self.up1)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="nearest")
        hidden = self.up0_projection(torch.cat((hidden, level0), dim=1))
        hidden = self._run_blocks(hidden, condition, self.up0)
        hidden = self.output_norm(hidden)
        return MultiFieldLogits(
            part=self.part_output(hidden),
            material=self.material_output(hidden),
            emission=self.emission_output(hidden),
        )

    def mask_probability(self, timesteps: Tensor) -> Tensor:
        phase = timesteps.float() / float(self.steps)
        return torch.sin(phase * math.pi * 0.5).square()

    def corrupt(
        self,
        part_tokens: Tensor,
        material_tokens: Tensor,
        emission_tokens: Tensor,
        timesteps: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _validate_generator_device(generator, part_tokens.device)
        if bool((timesteps < 0).any()) or bool((timesteps > self.steps).any()):
            raise ValueError(f"timesteps must be between 0 and {self.steps}.")
        probability = self.mask_probability(timesteps)[:, None, None]
        masked = (
            torch.rand(
                part_tokens.shape,
                dtype=torch.float32,
                device=part_tokens.device,
                generator=generator,
            )
            < probability
        )
        empty = (timesteps > 0) & ~masked.flatten(1).any(dim=1)
        if empty.any():
            rows = torch.nonzero(empty, as_tuple=False).flatten()
            positions = torch.randint(
                0,
                self.image_size * self.image_size,
                (rows.shape[0],),
                device=part_tokens.device,
                generator=generator,
            )
            masked[
                rows,
                positions // self.image_size,
                positions % self.image_size,
            ] = True
        return (
            part_tokens.masked_fill(masked, self.part_mask_token),
            material_tokens.masked_fill(masked, self.material_mask_token),
            emission_tokens.masked_fill(masked, self.emission_mask_token),
            masked,
        )

    @staticmethod
    def _sample_probabilities(
        probabilities: Tensor,
        generators: Sequence[torch.Generator],
    ) -> Tensor:
        batch, classes, height, width = probabilities.shape
        for generator in generators:
            _validate_generator_device(generator, probabilities.device)
        sampled = []
        for batch_index in range(batch):
            flattened = probabilities[batch_index].permute(1, 2, 0).reshape(
                -1, classes
            )
            sampled.append(
                torch.multinomial(
                    flattened,
                    num_samples=1,
                    replacement=True,
                    generator=generators[batch_index],
                ).view(height, width)
            )
        return torch.stack(sampled, dim=0)

    @torch.no_grad()
    def sample(
        self,
        guide: Tensor,
        morphologies: Tensor,
        subtypes: Tensor,
        roles: Tensor,
        genes: Tensor,
        *,
        temperature: float = 0.9,
        generators: Sequence[torch.Generator],
        legal_tuples: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = morphologies.shape[0]
        if len(generators) != batch:
            raise ValueError(f"Expected {batch} generators, got {len(generators)}.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if legal_tuples is not None:
            if legal_tuples.ndim != 2 or legal_tuples.shape[1] != 3:
                raise ValueError("legal_tuples must have shape [count, 3].")
            if legal_tuples.shape[0] == 0:
                raise ValueError("legal_tuples cannot be empty.")
            legal_tuples = legal_tuples.to(device=morphologies.device, dtype=torch.long)
            if (
                bool((legal_tuples[:, 0] < 0).any())
                or bool((legal_tuples[:, 0] >= self.vocabulary.part_count).any())
                or bool((legal_tuples[:, 1] < 0).any())
                or bool((legal_tuples[:, 1] >= self.vocabulary.material_count).any())
                or bool((legal_tuples[:, 2] < 0).any())
                or bool((legal_tuples[:, 2] >= self.vocabulary.emission_count).any())
            ):
                raise ValueError("legal_tuples contains an out-of-vocabulary value.")
        shape = (batch, self.image_size, self.image_size)
        device = morphologies.device
        part = torch.full(shape, self.part_mask_token, dtype=torch.long, device=device)
        material = torch.full(
            shape, self.material_mask_token, dtype=torch.long, device=device
        )
        emission = torch.full(
            shape, self.emission_mask_token, dtype=torch.long, device=device
        )
        for step in range(self.steps, 0, -1):
            timesteps = torch.full((batch,), step, dtype=torch.long, device=device)
            logits = self(
                part,
                material,
                emission,
                guide,
                morphologies,
                subtypes,
                roles,
                genes,
                timesteps,
            )
            divisor = max(float(temperature), 0.05)
            part_probability = (logits.part / divisor).softmax(dim=1)
            material_probability = (logits.material / divisor).softmax(dim=1)
            emission_probability = (logits.emission / divisor).softmax(dim=1)
            if legal_tuples is None:
                part_candidate = self._sample_probabilities(
                    part_probability, generators
                )
                material_candidate = self._sample_probabilities(
                    material_probability, generators
                )
                emission_candidate = self._sample_probabilities(
                    emission_probability, generators
                )
                confidence = (
                    part_probability.gather(1, part_candidate[:, None]).squeeze(1)
                    * material_probability.gather(
                        1, material_candidate[:, None]
                    ).squeeze(1)
                    * emission_probability.gather(
                        1, emission_candidate[:, None]
                    ).squeeze(1)
                ).clamp_min(1.0e-12).pow(1.0 / 3.0)
            else:
                tuple_probability = (
                    part_probability[:, legal_tuples[:, 0]]
                    * material_probability[:, legal_tuples[:, 1]]
                    * emission_probability[:, legal_tuples[:, 2]]
                )
                tuple_probability = tuple_probability / tuple_probability.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0e-12)
                tuple_indices = self._sample_probabilities(
                    tuple_probability, generators
                )
                flat_indices = tuple_indices.reshape(-1)
                chosen_tuples = legal_tuples[flat_indices].view(
                    batch, self.image_size, self.image_size, 3
                )
                part_candidate = chosen_tuples[..., 0]
                material_candidate = chosen_tuples[..., 1]
                emission_candidate = chosen_tuples[..., 2]
                confidence = tuple_probability.gather(
                    1, tuple_indices[:, None]
                ).squeeze(1)
            currently_masked = part == self.part_mask_token
            next_phase = float(step - 1) / float(self.steps)
            remaining_target = int(
                round(
                    self.image_size
                    * self.image_size
                    * math.sin(next_phase * math.pi * 0.5) ** 2
                )
            )
            for batch_index in range(batch):
                positions = torch.nonzero(currently_masked[batch_index], as_tuple=False)
                fill_count = max(0, positions.shape[0] - remaining_target)
                if step == 1:
                    fill_count = positions.shape[0]
                if fill_count == 0:
                    continue
                values = confidence[
                    batch_index, positions[:, 0], positions[:, 1]
                ]
                chosen = positions[
                    torch.topk(values, k=fill_count, largest=True).indices
                ]
                y, x = chosen[:, 0], chosen[:, 1]
                part[batch_index, y, x] = part_candidate[batch_index, y, x]
                material[batch_index, y, x] = material_candidate[
                    batch_index, y, x
                ]
                emission[batch_index, y, x] = emission_candidate[
                    batch_index, y, x
                ]
        return part, material, emission


def _masked_field_loss(
    logits: Tensor,
    target: Tensor,
    masked: Tensor,
    class_weight: Tensor | None,
) -> tuple[Tensor, Tensor]:
    per_pixel = F.cross_entropy(
        logits, target, weight=class_weight, reduction="none"
    )
    mask_weights = masked.float()
    if class_weight is None:
        loss_denominator = mask_weights.sum().clamp_min(1.0)
    else:
        loss_denominator = (
            class_weight[target] * mask_weights
        ).sum().clamp_min(1.0)
    accuracy_denominator = mask_weights.sum().clamp_min(1.0)
    loss = (per_pixel * mask_weights).sum() / loss_denominator
    prediction = logits.argmax(dim=1)
    accuracy = (
        ((prediction == target) & masked).float().sum() / accuracy_denominator
    )
    return loss, accuracy


def multifield_diffusion_loss(
    logits: MultiFieldLogits,
    part_target: Tensor,
    material_target: Tensor,
    emission_target: Tensor,
    masked: Tensor,
    *,
    class_weights: Mapping[str, Tensor] | None = None,
    field_weights: tuple[float, float, float] = (1.0, 0.65, 0.45),
) -> MultiFieldLoss:
    if len(field_weights) != 3:
        raise ValueError("field_weights must contain exactly three values.")
    if any(value < 0 for value in field_weights) or sum(field_weights) <= 0:
        raise ValueError("field_weights must be nonnegative with a positive sum.")
    class_weights = class_weights or {}
    part_loss, part_accuracy = _masked_field_loss(
        logits.part, part_target, masked, class_weights.get("part")
    )
    material_loss, material_accuracy = _masked_field_loss(
        logits.material, material_target, masked, class_weights.get("material")
    )
    emission_loss, emission_accuracy = _masked_field_loss(
        logits.emission, emission_target, masked, class_weights.get("emission")
    )
    total = (
        part_loss * field_weights[0]
        + material_loss * field_weights[1]
        + emission_loss * field_weights[2]
    ) / sum(field_weights)
    return MultiFieldLoss(
        loss=total,
        part_loss=part_loss,
        material_loss=material_loss,
        emission_loss=emission_loss,
        part_accuracy=part_accuracy,
        material_accuracy=material_accuracy,
        emission_accuracy=emission_accuracy,
        masked_fraction=masked.float().mean(),
    )
