from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, prod
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PART_COUNT = 17
MATERIAL_COUNT = 10
EMISSION_COUNT = 4
IMAGE_SIZE = 48


@dataclass(frozen=True, slots=True)
class SpriteLatentConfig:
    image_size: int = IMAGE_SIZE
    width: int = 64
    latent_levels: tuple[int, ...] = (8, 5, 5, 5)
    residual_depth: int = 2
    part_embedding_dim: int = 12
    material_embedding_dim: int = 8
    emission_embedding_dim: int = 4
    condition_dim: int = 64
    joint_tuple_weight: float = 0.6
    foreground_dice_weight: float = 0.2
    latent_usage_weight: float = 0.01
    soft_entropy_temperature: float = 0.35

    def __post_init__(self) -> None:
        if self.image_size != IMAGE_SIZE:
            raise ValueError("Sprite latent codec currently requires native 48px fields")
        if self.width < 8 or self.width % 8:
            raise ValueError("Sprite latent width must be >=8 and divisible by eight")
        if not 1 <= self.residual_depth <= 8:
            raise ValueError("residual_depth must be in [1, 8]")
        if not 2 <= len(self.latent_levels) <= 8:
            raise ValueError("FSQ needs between two and eight scalar dimensions")
        if any(level < 2 or level > 32 for level in self.latent_levels):
            raise ValueError("Every FSQ level count must be in [2, 32]")
        if prod(self.latent_levels) > 1_000_000:
            raise ValueError("Implicit FSQ vocabulary is unreasonably large")
        for value in (
            self.joint_tuple_weight,
            self.foreground_dice_weight,
            self.latent_usage_weight,
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError("Loss weights must be non-negative")
        if not isfinite(self.soft_entropy_temperature) or self.soft_entropy_temperature <= 0.0:
            raise ValueError("soft_entropy_temperature must be finite and positive")

    @property
    def latent_dim(self) -> int:
        return len(self.latent_levels)

    @property
    def implicit_code_count(self) -> int:
        return prod(self.latent_levels)

    @property
    def latent_grid_size(self) -> int:
        return self.image_size // 4

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "latent_dim": self.latent_dim,
                "implicit_code_count": self.implicit_code_count,
                "latent_grid_size": self.latent_grid_size,
                "quantizer": "finite-scalar-quantization-sigmoid-ste-v1",
                "usage_regularizer": "differentiable-soft-fsq-marginal-entropy-rbf-v1",
            }
        )
        result["latent_levels"] = list(self.latent_levels)
        return result


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = min(16, channels)
        while channels % groups:
            groups -= 1
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


class FSQQuantizer(nn.Module):
    """Fixed finite scalar quantizer with an implicit mixed-radix vocabulary.

    The bounded continuous representation is available for an autoencoder
    warm-up. Quantized training uses a straight-through estimator. Unlike a
    learned VQ codebook, the implicit vocabulary cannot suffer dead embeddings.
    """

    def __init__(self, levels: Sequence[int], *, soft_entropy_temperature: float = 0.35) -> None:
        super().__init__()
        values = tuple(int(level) for level in levels)
        if not values or any(level < 2 for level in values):
            raise ValueError("FSQ levels must contain integers >=2")
        if not isfinite(soft_entropy_temperature) or soft_entropy_temperature <= 0.0:
            raise ValueError("soft_entropy_temperature must be finite and positive")
        self.levels = values
        self.soft_entropy_temperature = float(soft_entropy_temperature)
        self.register_buffer("level_tensor", torch.tensor(values, dtype=torch.float32))
        basis = [1]
        for level in values[:-1]:
            basis.append(basis[-1] * level)
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))

    @property
    def dimension(self) -> int:
        return len(self.levels)

    @property
    def code_count(self) -> int:
        return prod(self.levels)

    def _continuous(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(inputs).mul(2.0).sub(1.0)

    def _digits_from_continuous(self, continuous: Tensor) -> Tensor:
        levels = self.level_tensor.to(device=continuous.device)
        return torch.round((continuous.add(1.0).mul(0.5)) * (levels - 1.0)).long()

    def digits_to_codes(self, digits: Tensor) -> Tensor:
        if digits.shape[-1] != self.dimension:
            raise ValueError("FSQ digit tensor has the wrong final dimension")
        levels = self.level_tensor.to(device=digits.device, dtype=torch.long)
        if bool(torch.any(digits < 0)) or bool(torch.any(digits >= levels)):
            raise ValueError("FSQ digits exceed their mixed-radix levels")
        return (digits * self.basis.to(device=digits.device)).sum(dim=-1)

    def codes_to_digits(self, codes: Tensor) -> Tensor:
        if bool(torch.any(codes < 0)) or bool(torch.any(codes >= self.code_count)):
            raise ValueError("FSQ code index is outside the implicit vocabulary")
        basis = self.basis.to(device=codes.device)
        levels = self.level_tensor.to(device=codes.device, dtype=torch.long)
        return torch.stack(((codes[..., None] // basis) % levels).unbind(dim=-1), dim=-1)

    def digits_to_normalized(self, digits: Tensor) -> Tensor:
        levels = self.level_tensor.to(device=digits.device, dtype=torch.float32)
        return digits.float().mul(2.0).div(levels - 1.0).sub(1.0)

    def forward(self, inputs: Tensor, *, quantize: bool = True) -> dict[str, Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != self.dimension:
            raise ValueError("FSQ inputs must have shape B,D,H,W")
        continuous = self._continuous(inputs).permute(0, 2, 3, 1).contiguous()
        digits = self._digits_from_continuous(continuous)
        discrete = self.digits_to_normalized(digits)
        quantized = continuous + (discrete - continuous).detach() if quantize else continuous
        codes = self.digits_to_codes(digits)
        counts = torch.bincount(codes.reshape(-1), minlength=self.code_count).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        active = probabilities > 0
        entropy = -(probabilities[active] * probabilities[active].log()).sum()
        perplexity = entropy.exp()
        marginal_entropies: list[Tensor] = []
        soft_marginal_entropies: list[Tensor] = []
        for dimension, level in enumerate(self.levels):
            marginal = torch.bincount(
                digits[..., dimension].reshape(-1), minlength=level
            ).float()
            marginal = marginal / marginal.sum().clamp_min(1.0)
            present = marginal > 0
            marginal_entropies.append(
                -(marginal[present] * marginal[present].log()).sum()
                / torch.log(torch.tensor(float(level), device=inputs.device))
            )
            # The hard histogram above is intentionally diagnostic: integer
            # digits and bincount do not carry gradients.  Train against a
            # smooth occupancy proxy instead.  Each bounded scalar votes for
            # every finite level with a radial-basis soft assignment, then the
            # entropy of the batch/spatial marginal is maximized.  Computing
            # the proxy in float32 keeps AMP numerically stable while the cast
            # back to ``continuous`` remains differentiable.
            position = (
                continuous[..., dimension].float().add(1.0).mul(0.5 * (level - 1))
            )
            centers = torch.arange(level, device=inputs.device, dtype=torch.float32)
            soft_logits = -(
                position[..., None] - centers
            ).square().div(self.soft_entropy_temperature)
            assignments = soft_logits.softmax(dim=-1)
            soft_marginal = assignments.reshape(-1, level).mean(dim=0)
            soft_entropy = -(
                soft_marginal * soft_marginal.clamp_min(torch.finfo(torch.float32).tiny).log()
            ).sum() / torch.log(torch.tensor(float(level), device=inputs.device))
            soft_marginal_entropies.append(soft_entropy)
        return {
            "quantized": quantized.permute(0, 3, 1, 2).contiguous(),
            "continuous": continuous.permute(0, 3, 1, 2).contiguous(),
            "digits": digits,
            "codes": codes,
            "perplexity": perplexity,
            "utilization": active.float().mean(),
            "marginal_entropy": torch.stack(marginal_entropies).mean(),
            "soft_marginal_entropy": torch.stack(soft_marginal_entropies).mean(),
        }


@dataclass(slots=True)
class CodecOutput:
    part_logits: Tensor
    material_logits: Tensor
    emission_logits: Tensor
    latent: Tensor
    continuous_latent: Tensor
    digits: Tensor
    codes: Tensor
    perplexity: Tensor
    utilization: Tensor
    marginal_entropy: Tensor
    soft_marginal_entropy: Tensor
    quantized: bool


class SemanticSpriteFSQ(nn.Module):
    """Condition-aware FSQ autoencoder for aligned sprite categorical fields."""

    def __init__(self, config: SpriteLatentConfig = SpriteLatentConfig()) -> None:
        super().__init__()
        self.config = config
        self.part_embedding = nn.Embedding(PART_COUNT, config.part_embedding_dim)
        self.material_embedding = nn.Embedding(MATERIAL_COUNT, config.material_embedding_dim)
        self.emission_embedding = nn.Embedding(EMISSION_COUNT, config.emission_embedding_dim)
        input_channels = (
            config.part_embedding_dim
            + config.material_embedding_dim
            + config.emission_embedding_dim
        )
        self.morphology_embedding = nn.Embedding(5, 16)
        self.subtype_embedding = nn.Embedding(20, 16)
        self.role_embedding = nn.Embedding(8, 16)
        self.gene_projection = nn.Sequential(nn.Linear(24, 32), nn.SiLU(), nn.Linear(32, 16))
        self.condition_projection = nn.Sequential(
            nn.Linear(64, config.condition_dim), nn.SiLU(), nn.Linear(config.condition_dim, config.condition_dim)
        )
        self.stem = nn.Conv2d(input_channels, config.width, 3, padding=1)
        self.encoder_blocks = nn.ModuleList(
            ResidualBlock(config.width, config.condition_dim)
            for _ in range(config.residual_depth)
        )
        self.down1 = nn.Conv2d(config.width, config.width, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(config.width, config.width, 4, stride=2, padding=1)
        self.to_latent = nn.Conv2d(config.width, config.latent_dim, 1)
        self.quantizer = FSQQuantizer(
            config.latent_levels,
            soft_entropy_temperature=config.soft_entropy_temperature,
        )
        self.from_latent = nn.Conv2d(config.latent_dim, config.width, 1)
        self.decoder_blocks = nn.ModuleList(
            ResidualBlock(config.width, config.condition_dim)
            for _ in range(config.residual_depth)
        )
        self.up1 = nn.Conv2d(config.width, config.width, 3, padding=1)
        self.up2 = nn.Conv2d(config.width, config.width, 3, padding=1)
        self.part_head = nn.Conv2d(config.width, PART_COUNT, 1)
        self.material_head = nn.Conv2d(config.width, MATERIAL_COUNT, 1)
        self.emission_head = nn.Conv2d(config.width, EMISSION_COUNT, 1)

    def condition_vector(
        self,
        morphology: Tensor,
        subtype: Tensor,
        role: Tensor,
        genes: Tensor,
    ) -> Tensor:
        if genes.ndim != 2 or genes.shape[1] != 24:
            raise ValueError("genes must have shape B,24")
        if not (morphology.shape == subtype.shape == role.shape == genes.shape[:1]):
            raise ValueError("condition batch dimensions disagree")
        raw = torch.cat(
            (
                self.morphology_embedding(morphology),
                self.subtype_embedding(subtype),
                self.role_embedding(role),
                self.gene_projection(genes.float()),
            ),
            dim=1,
        )
        return self.condition_projection(raw)

    def _embed_fields(self, part: Tensor, material: Tensor, emission: Tensor) -> Tensor:
        if part.shape != material.shape or part.shape != emission.shape or part.ndim != 3:
            raise ValueError("categorical fields must share shape B,H,W")
        if part.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("categorical fields must be native 48px")
        embeddings = (
            self.part_embedding(part),
            self.material_embedding(material),
            self.emission_embedding(emission),
        )
        return torch.cat(embeddings, dim=-1).permute(0, 3, 1, 2).contiguous()

    def encode(
        self,
        part: Tensor,
        material: Tensor,
        emission: Tensor,
        condition: Tensor,
        *,
        quantize: bool,
    ) -> dict[str, Tensor]:
        hidden = self.stem(self._embed_fields(part, material, emission))
        for block in self.encoder_blocks:
            hidden = block(hidden, condition)
        hidden = F.silu(self.down1(hidden))
        hidden = F.silu(self.down2(hidden))
        latent = self.to_latent(hidden)
        result = self.quantizer(latent, quantize=quantize)
        result["prequantized"] = latent
        return result

    def decode(self, latent: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if latent.ndim != 4 or latent.shape[1] != self.config.latent_dim:
            raise ValueError("latent must have shape B,D,H,W")
        hidden = self.from_latent(latent)
        for block in self.decoder_blocks:
            hidden = block(hidden, condition)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="nearest")
        hidden = F.silu(self.up1(hidden))
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="nearest")
        hidden = F.silu(self.up2(hidden))
        return self.part_head(hidden), self.material_head(hidden), self.emission_head(hidden)

    def forward(
        self,
        part: Tensor,
        material: Tensor,
        emission: Tensor,
        morphology: Tensor,
        subtype: Tensor,
        role: Tensor,
        genes: Tensor,
        *,
        quantize: bool = True,
    ) -> CodecOutput:
        condition = self.condition_vector(morphology, subtype, role, genes)
        encoded = self.encode(part, material, emission, condition, quantize=quantize)
        logits = self.decode(encoded["quantized"], condition)
        return CodecOutput(
            part_logits=logits[0],
            material_logits=logits[1],
            emission_logits=logits[2],
            latent=encoded["quantized"],
            continuous_latent=encoded["continuous"],
            digits=encoded["digits"],
            codes=encoded["codes"],
            perplexity=encoded["perplexity"],
            utilization=encoded["utilization"],
            marginal_entropy=encoded["marginal_entropy"],
            soft_marginal_entropy=encoded["soft_marginal_entropy"],
            quantized=quantize,
        )


def _validate_legal_tuples(legal_tuples: Tensor) -> Tensor:
    if not isinstance(legal_tuples, Tensor):
        raise TypeError("legal_tuples must be a torch Tensor")
    if legal_tuples.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("legal_tuples must use an integer dtype")
    values = legal_tuples.long()
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("legal_tuples must have shape K,3")
    if (
        bool(torch.any(values[:, 0] < 0))
        or bool(torch.any(values[:, 0] >= PART_COUNT))
        or bool(torch.any(values[:, 1] < 0))
        or bool(torch.any(values[:, 1] >= MATERIAL_COUNT))
        or bool(torch.any(values[:, 2] < 0))
        or bool(torch.any(values[:, 2] >= EMISSION_COUNT))
    ):
        raise ValueError("legal tuple table contains out-of-vocabulary values")
    codes = values[:, 0] * MATERIAL_COUNT * EMISSION_COUNT + values[:, 1] * EMISSION_COUNT + values[:, 2]
    if len(torch.unique(codes)) != len(values):
        raise ValueError("legal tuple table contains duplicates")
    if not bool(torch.all(codes[1:] > codes[:-1])):
        raise ValueError("legal tuple table must be sorted canonically")
    return values


def legal_tuple_scores(output: CodecOutput, legal_tuples: Tensor) -> Tensor:
    expected_shapes = (
        (output.part_logits, PART_COUNT),
        (output.material_logits, MATERIAL_COUNT),
        (output.emission_logits, EMISSION_COUNT),
    )
    spatial: tuple[int, int, int] | None = None
    for logits, classes in expected_shapes:
        if logits.ndim != 4 or logits.shape[1] != classes:
            raise ValueError("Codec output head has an invalid class dimension")
        observed = (int(logits.shape[0]), int(logits.shape[2]), int(logits.shape[3]))
        if spatial is not None and observed != spatial:
            raise ValueError("Codec output heads are not spatially aligned")
        spatial = observed
    values = _validate_legal_tuples(legal_tuples).to(output.part_logits.device)
    part = output.part_logits.permute(0, 2, 3, 1)[..., values[:, 0]]
    material = output.material_logits.permute(0, 2, 3, 1)[..., values[:, 1]]
    emission = output.emission_logits.permute(0, 2, 3, 1)[..., values[:, 2]]
    return part + material + emission


@torch.no_grad()
def project_legal_tuples(output: CodecOutput, legal_tuples: Tensor) -> dict[str, Tensor]:
    values = _validate_legal_tuples(legal_tuples).to(output.part_logits.device)
    selected = legal_tuple_scores(output, values).argmax(dim=-1)
    triples = values[selected]
    return {
        "part": triples[..., 0].contiguous(),
        "material": triples[..., 1].contiguous(),
        "emission": triples[..., 2].contiguous(),
        "legal_index": selected.contiguous(),
    }


def _target_legal_indices(
    part: Tensor, material: Tensor, emission: Tensor, legal_tuples: Tensor
) -> Tensor:
    if part.shape != material.shape or part.shape != emission.shape or part.ndim != 3:
        raise ValueError("Target categorical fields must be aligned B,H,W tensors")
    for name, target, classes in (
        ("part", part, PART_COUNT),
        ("material", material, MATERIAL_COUNT),
        ("emission", emission, EMISSION_COUNT),
    ):
        if target.dtype != torch.long:
            raise TypeError(f"{name} target must use torch.long")
        if bool(torch.any(target < 0)) or bool(torch.any(target >= classes)):
            raise ValueError(f"{name} target contains out-of-vocabulary values")
    values = _validate_legal_tuples(legal_tuples).to(part.device)
    table = torch.full(
        (PART_COUNT * MATERIAL_COUNT * EMISSION_COUNT,),
        -1,
        dtype=torch.long,
        device=part.device,
    )
    legal_codes = (
        values[:, 0] * MATERIAL_COUNT * EMISSION_COUNT
        + values[:, 1] * EMISSION_COUNT
        + values[:, 2]
    )
    table[legal_codes] = torch.arange(len(values), device=part.device)
    target_codes = part * MATERIAL_COUNT * EMISSION_COUNT + material * EMISSION_COUNT + emission
    targets = table[target_codes]
    if bool(torch.any(targets < 0)):
        raise ValueError("target fields contain tuples outside the legal table")
    return targets


def sprite_codec_loss(
    output: CodecOutput,
    part: Tensor,
    material: Tensor,
    emission: Tensor,
    legal_tuples: Tensor,
    *,
    config: SpriteLatentConfig,
    class_weights: Mapping[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = class_weights or {}
    part_loss = F.cross_entropy(
        output.part_logits,
        part,
        weight=weights.get("part", None).to(part.device) if "part" in weights else None,
    )
    material_loss = F.cross_entropy(
        output.material_logits,
        material,
        weight=weights.get("material", None).to(part.device) if "material" in weights else None,
    )
    emission_loss = F.cross_entropy(
        output.emission_logits,
        emission,
        weight=weights.get("emission", None).to(part.device) if "emission" in weights else None,
    )
    tuple_scores = legal_tuple_scores(output, legal_tuples)
    tuple_targets = _target_legal_indices(part, material, emission, legal_tuples)
    tuple_loss = F.cross_entropy(tuple_scores.permute(0, 3, 1, 2), tuple_targets)
    predicted_foreground = 1.0 - output.part_logits.softmax(dim=1)[:, 0]
    target_foreground = (part != 0).float()
    intersection = (predicted_foreground * target_foreground).sum(dim=(1, 2))
    denominator = predicted_foreground.sum(dim=(1, 2)) + target_foreground.sum(dim=(1, 2))
    foreground_dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    # Hard code counts are useful diagnostics but cannot regularize the
    # encoder.  The differentiable soft-FSQ marginal is the training signal.
    usage_penalty = 1.0 - output.soft_marginal_entropy
    reconstruction = part_loss + material_loss + emission_loss
    total = (
        reconstruction
        + config.joint_tuple_weight * tuple_loss
        + config.foreground_dice_weight * foreground_dice
        + (config.latent_usage_weight * usage_penalty if output.quantized else 0.0)
    )
    return total, {
        "loss": total.detach(),
        "reconstruction": reconstruction.detach(),
        "part_ce": part_loss.detach(),
        "material_ce": material_loss.detach(),
        "emission_ce": emission_loss.detach(),
        "joint_tuple_ce": tuple_loss.detach(),
        "foreground_dice_loss": foreground_dice.detach(),
        "usage_penalty": usage_penalty.detach(),
        "perplexity": output.perplexity.detach(),
        "utilization": output.utilization.detach(),
        "marginal_entropy": output.marginal_entropy.detach(),
        "soft_marginal_entropy": output.soft_marginal_entropy.detach(),
    }
