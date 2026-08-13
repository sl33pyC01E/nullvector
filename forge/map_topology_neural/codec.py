from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Final

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..maps.model import THEMES
from .contract import (
    FIELD_CLASS_COUNTS,
    GLOBAL_CONDITION_NAMES,
    PATCH_SCALE,
    POINT_CHANNELS,
    TopologyTensor,
)


CODEC_NAME: Final[str] = "nullvector-fully-convolutional-categorical-vq-codec"
CODEC_VERSION: Final[str] = "0.1.0-representation-smoke"


@dataclass(frozen=True, slots=True)
class CodecConfig:
    width: int = 32
    latent_dim: int = 32
    codebook_size: int = 128
    field_embedding_dim: int = 4
    residual_depth: int = 1
    ema_decay: float = 0.95
    ema_epsilon: float = 1.0e-5
    commitment_weight: float = 0.25

    def __post_init__(self) -> None:
        if not 4 <= self.width <= 256:
            raise ValueError("Codec width must be in [4, 256].")
        if not 4 <= self.latent_dim <= 256:
            raise ValueError("Codec latent_dim must be in [4, 256].")
        if not 4 <= self.codebook_size <= 4096:
            raise ValueError("Codec codebook_size must be in [4, 4096].")
        if not 2 <= self.field_embedding_dim <= 64:
            raise ValueError("Codec field_embedding_dim must be in [2, 64].")
        if not 0 <= self.residual_depth <= 4:
            raise ValueError("Codec residual_depth must be in [0, 4].")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("Codec ema_decay must be in (0, 1).")
        if not 0.0 < self.ema_epsilon <= 1.0:
            raise ValueError("Codec ema_epsilon must be in (0, 1].")
        if not math.isfinite(self.commitment_weight) or self.commitment_weight <= 0:
            raise ValueError("Codec commitment_weight must be finite and positive.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CodecConfig":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Codec config members are incomplete or unexpected.")
        return cls(**payload)  # type: ignore[arg-type]


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = min(8, width)
        while width % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


class EMAVectorQuantizer(nn.Module):
    """CPU-safe straight-through VQ with deterministic EMA codebook updates."""

    def __init__(
        self,
        codebook_size: int,
        dimension: int,
        *,
        decay: float,
        epsilon: float,
        commitment_weight: float,
    ) -> None:
        super().__init__()
        embeddings = torch.empty((codebook_size, dimension), dtype=torch.float32)
        nn.init.uniform_(embeddings, -1.0 / codebook_size, 1.0 / codebook_size)
        self.register_buffer("embeddings", embeddings)
        self.register_buffer("cluster_size", torch.zeros(codebook_size, dtype=torch.float32))
        self.register_buffer("embedding_sum", embeddings.clone())
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.commitment_weight = float(commitment_weight)
        self.codebook_size = int(codebook_size)
        self.dimension = int(dimension)

    def forward(self, latent: Tensor, *, update_ema: bool) -> dict[str, Tensor]:
        if latent.ndim != 4 or latent.shape[1] != self.dimension:
            raise ValueError("VQ latent must use B,D,H,W with the configured dimension.")
        flat = latent.permute(0, 2, 3, 1).contiguous().view(-1, self.dimension)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embeddings.t()
            + self.embeddings.square().sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(distances, dim=1)
        assignments = F.one_hot(indices, self.codebook_size).to(flat.dtype)
        if self.training and update_ema:
            with torch.no_grad():
                counts = assignments.sum(dim=0)
                sums = assignments.t() @ flat
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                self.embedding_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
                total = self.cluster_size.sum()
                normalized = (
                    (self.cluster_size + self.epsilon)
                    / (total + self.codebook_size * self.epsilon)
                    * total.clamp_min(1.0)
                )
                updated = self.embedding_sum / normalized.unsqueeze(1).clamp_min(self.epsilon)
                active = self.cluster_size > self.epsilon
                self.embeddings[active] = updated[active]
        quantized = F.embedding(indices, self.embeddings)
        quantized = quantized.view(latent.shape[0], latent.shape[2], latent.shape[3], self.dimension)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        commitment = self.commitment_weight * F.mse_loss(latent, quantized.detach())
        straight_through = latent + (quantized - latent).detach()
        probabilities = assignments.mean(dim=0)
        perplexity = torch.exp(-torch.sum(probabilities * torch.log(probabilities + 1.0e-10)))
        utilization = (probabilities > 0).to(torch.float32).mean()
        return {
            "quantized": straight_through,
            "indices": indices.view(latent.shape[0], latent.shape[2], latent.shape[3]),
            "commitment_loss": commitment,
            "perplexity": perplexity,
            "utilization": utilization,
        }


class CategoricalTopologyCodec(nn.Module):
    """Representation-only VQ codec. It does not define a generative prior."""

    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        self.config = config
        embedding_width = config.field_embedding_dim
        self.terrain_embedding = nn.Embedding(FIELD_CLASS_COUNTS["terrain"], embedding_width)
        self.hazard_embedding = nn.Embedding(FIELD_CLASS_COUNTS["hazard"], embedding_width)
        self.elevation_embedding = nn.Embedding(FIELD_CLASS_COUNTS["elevation"], embedding_width)
        input_width = embedding_width * 3 + len(POINT_CHANNELS) + 1
        self.input_conv = nn.Conv2d(input_width, config.width, 3, padding=1)
        self.theme_embedding = nn.Embedding(len(THEMES), config.width)
        self.global_projection = nn.Linear(len(GLOBAL_CONDITION_NAMES), config.width)
        self.encoder_residual = nn.Sequential(
            *(ResidualBlock(config.width) for _ in range(config.residual_depth))
        )
        self.down1 = nn.Conv2d(config.width, config.width, 4, stride=2, padding=1)
        self.middle_residual = nn.Sequential(
            *(ResidualBlock(config.width) for _ in range(config.residual_depth))
        )
        self.down2 = nn.Conv2d(config.width, config.width, 4, stride=2, padding=1)
        self.to_latent = nn.Conv2d(config.width, config.latent_dim, 1)
        self.quantizer = EMAVectorQuantizer(
            config.codebook_size,
            config.latent_dim,
            decay=config.ema_decay,
            epsilon=config.ema_epsilon,
            commitment_weight=config.commitment_weight,
        )
        self.from_latent = nn.Conv2d(config.latent_dim, config.width, 1)
        self.decoder_residual = nn.Sequential(
            *(ResidualBlock(config.width) for _ in range(config.residual_depth))
        )
        self.up1 = nn.ConvTranspose2d(config.width, config.width, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(config.width, config.width, 4, stride=2, padding=1)
        self.terrain_head = nn.Conv2d(config.width, FIELD_CLASS_COUNTS["terrain"], 1)
        self.hazard_head = nn.Conv2d(config.width, FIELD_CLASS_COUNTS["hazard"], 1)
        self.elevation_head = nn.Conv2d(config.width, FIELD_CLASS_COUNTS["elevation"], 1)

    def encode(self, batch: dict[str, Tensor], *, update_ema: bool) -> dict[str, Tensor]:
        categorical = batch["categorical"]
        points = batch["point_heatmaps"]
        valid = batch["valid_mask"]
        global_conditions = batch["global_conditions"]
        themes = batch["theme_index"]
        if categorical.ndim != 4 or categorical.shape[1] != 3:
            raise ValueError("categorical must have shape B,3,H,W.")
        if categorical.shape[-2] % PATCH_SCALE or categorical.shape[-1] % PATCH_SCALE:
            raise ValueError("Codec input dimensions must be exactly divisible by patch scale four.")
        if points.shape != (categorical.shape[0], len(POINT_CHANNELS), *categorical.shape[-2:]):
            raise ValueError("point_heatmaps shape disagrees with categorical input.")
        if valid.shape != (categorical.shape[0], 1, *categorical.shape[-2:]):
            raise ValueError("valid_mask shape disagrees with categorical input.")
        if global_conditions.shape != (categorical.shape[0], len(GLOBAL_CONDITION_NAMES)):
            raise ValueError("global_conditions shape disagrees with the tensor contract.")
        if themes.shape != (categorical.shape[0],):
            raise ValueError("theme_index shape disagrees with batch size.")
        embedded = torch.cat(
            (
                self.terrain_embedding(categorical[:, 0].long()).permute(0, 3, 1, 2),
                self.hazard_embedding(categorical[:, 1].long()).permute(0, 3, 1, 2),
                self.elevation_embedding(categorical[:, 2].long()).permute(0, 3, 1, 2),
                points.to(torch.float32),
                valid.to(torch.float32),
            ),
            dim=1,
        )
        hidden = self.input_conv(embedded)
        condition = self.theme_embedding(themes.long()) + self.global_projection(global_conditions.float())
        hidden = hidden + condition[:, :, None, None]
        hidden = self.encoder_residual(F.silu(hidden))
        hidden = self.middle_residual(F.silu(self.down1(hidden)))
        latent = self.to_latent(F.silu(self.down2(hidden)))
        quantized = self.quantizer(latent, update_ema=update_ema)
        quantized["latent"] = latent
        return quantized

    def decode(self, quantized: Tensor) -> dict[str, Tensor]:
        hidden = self.decoder_residual(F.silu(self.from_latent(quantized)))
        hidden = F.silu(self.up1(hidden))
        hidden = F.silu(self.up2(hidden))
        return {
            "terrain": self.terrain_head(hidden),
            "hazard": self.hazard_head(hidden),
            "elevation": self.elevation_head(hidden),
        }

    def forward(self, batch: dict[str, Tensor], *, update_ema: bool = False) -> dict[str, Any]:
        encoded = self.encode(batch, update_ema=update_ema)
        logits = self.decode(encoded["quantized"])
        if logits["terrain"].shape[-2:] != batch["categorical"].shape[-2:]:
            raise RuntimeError("Codec decoder violated exact pad/crop spatial identity.")
        return {"logits": logits, **encoded}


def build_codec(config: CodecConfig, *, init_seed: int) -> CategoricalTopologyCodec:
    if not 0 <= int(init_seed) < 1 << 63:
        raise ValueError("Codec initialization seed must be in [0, 2^63).")
    # Constructors draw only from the CPU default generator. Swap its state with a
    # dedicated CPU generator, then restore the caller's global state exactly.
    previous = torch.get_rng_state()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(init_seed))
    try:
        torch.set_rng_state(generator.get_state())
        model = CategoricalTopologyCodec(config)
    finally:
        torch.set_rng_state(previous)
    return model.to(device=torch.device("cpu"), dtype=torch.float32)


def collate_topology_tensors(tensors: list[TopologyTensor]) -> dict[str, Tensor]:
    if not tensors:
        raise ValueError("Cannot collate an empty topology tensor list.")
    height = max(item.padded_height for item in tensors)
    width = max(item.padded_width for item in tensors)
    if height % PATCH_SCALE or width % PATCH_SCALE:
        raise RuntimeError("Collated spatial bounds must remain divisible by patch scale.")
    batch = len(tensors)
    categorical = torch.zeros((batch, 3, height, width), dtype=torch.long, device="cpu")
    points = torch.zeros((batch, len(POINT_CHANNELS), height, width), dtype=torch.float32, device="cpu")
    valid = torch.zeros((batch, 1, height, width), dtype=torch.bool, device="cpu")
    global_conditions = torch.empty((batch, len(GLOBAL_CONDITION_NAMES)), dtype=torch.float32, device="cpu")
    themes = torch.empty((batch,), dtype=torch.long, device="cpu")
    for index, item in enumerate(tensors):
        categorical[index, :, : item.padded_height, : item.padded_width] = torch.from_numpy(item.categorical.copy()).long()
        points[index, :, : item.padded_height, : item.padded_width] = torch.from_numpy(item.point_heatmaps.copy())
        valid[index, :, : item.padded_height, : item.padded_width] = torch.from_numpy(item.valid_mask.copy()).bool()
        global_conditions[index] = torch.from_numpy(item.global_conditions.copy())
        themes[index] = item.theme_index
    return {
        "categorical": categorical,
        "point_heatmaps": points,
        "valid_mask": valid,
        "global_conditions": global_conditions,
        "theme_index": themes,
    }


def categorical_reconstruction_loss(output: dict[str, Any], batch: dict[str, Tensor]) -> dict[str, Tensor]:
    valid = batch["valid_mask"][:, 0].to(torch.float32)
    denominator = valid.sum().clamp_min(1.0)
    losses: dict[str, Tensor] = {}
    for field_index, name in enumerate(("terrain", "hazard", "elevation")):
        per_cell = F.cross_entropy(
            output["logits"][name],
            batch["categorical"][:, field_index].long(),
            reduction="none",
        )
        losses[name] = (per_cell * valid).sum() / denominator
    losses["commitment"] = output["commitment_loss"]
    losses["total"] = losses["terrain"] + losses["hazard"] + losses["elevation"] + losses["commitment"]
    return losses


def _ema_state_update(
    ema_state: dict[str, Tensor],
    model_state: dict[str, Tensor],
    *,
    decay: float,
) -> None:
    with torch.no_grad():
        for name, value in model_state.items():
            source = value.detach().cpu()
            if source.is_floating_point():
                ema_state[name].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                ema_state[name].copy_(source)


def train_cpu_smoke(
    model: CategoricalTopologyCodec,
    batch: dict[str, Tensor],
    *,
    steps: int = 2,
    learning_rate: float = 1.0e-3,
    training_seed: int = 0x544F504F,
    ema_decay: float = 0.99,
) -> dict[str, object]:
    if not 1 <= steps <= 2:
        raise ValueError("Foundation smoke is intentionally bounded to one or two CPU steps.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if any(tensor.device.type != "cpu" for tensor in batch.values()):
        raise ValueError("Topology codec smoke accepts CPU tensors only.")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_generator = torch.Generator(device="cpu")
    training_generator.manual_seed(int(training_seed))
    ema_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, float]] = []
    for step in range(steps):
        order = torch.randperm(batch["categorical"].shape[0], generator=training_generator)
        ordered = {name: value.index_select(0, order) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(ordered, update_ema=True)
        losses = categorical_reconstruction_loss(output, ordered)
        losses["total"].backward()
        optimizer.step()
        _ema_state_update(ema_state, model.state_dict(), decay=ema_decay)
        history.append(
            {
                "step": float(step + 1),
                "total": float(losses["total"].detach()),
                "terrain": float(losses["terrain"].detach()),
                "hazard": float(losses["hazard"].detach()),
                "elevation": float(losses["elevation"].detach()),
                "commitment": float(losses["commitment"].detach()),
                "codebook_perplexity": float(output["perplexity"].detach()),
                "codebook_utilization": float(output["utilization"].detach()),
            }
        )
    return {
        "format": "nullvector-topology-codec-cpu-smoke-state-v1",
        "authority": "representation_only_not_generative",
        "device": "cpu",
        "steps": steps,
        "history": history,
        "optimizer_state": optimizer.state_dict(),
        "ema_state": ema_state,
        "training_generator_state": training_generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
    }
