from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .corpus import (
    FROZEN_PRODUCTION_BASE_SEED,
    FROZEN_PRODUCTION_CORPUS_SHA256,
    FROZEN_PRODUCTION_LEGAL_TUPLE_COUNT,
    FROZEN_PRODUCTION_LEGAL_TUPLE_FINGERPRINT,
    FROZEN_PRODUCTION_SAMPLE_COUNT,
    FROZEN_PRODUCTION_SPLIT_FINGERPRINT,
    SemanticFieldCorpus,
    SemanticFieldDataset,
    compute_class_weights,
    compute_legal_tuples,
    legal_tuple_fingerprint,
    stratified_split,
)
from .codec import SemanticSpriteFSQ, SpriteLatentConfig, sprite_codec_loss


def canonical_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-sprite-fsq-state-v1\0")
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def load_production_training_contract(
    corpus_path: Path,
    *,
    split_seed: int = 0x5A17,
    validation_fraction: float = 0.08,
) -> dict[str, Any]:
    if split_seed != 0x5A17 or validation_fraction != 0.08:
        raise ValueError("Production sprite FSQ uses the frozen split seed and fraction")
    corpus = SemanticFieldCorpus.load(
        corpus_path,
        expected_file_sha256=FROZEN_PRODUCTION_CORPUS_SHA256,
    )
    if (
        corpus.count != FROZEN_PRODUCTION_SAMPLE_COUNT
        or corpus.base_seed != FROZEN_PRODUCTION_BASE_SEED
    ):
        raise ValueError("Production sprite FSQ corpus identity/count is not frozen")
    split = stratified_split(
        corpus,
        seed=split_seed,
        validation_fraction=validation_fraction,
    )
    if split.fingerprint != FROZEN_PRODUCTION_SPLIT_FINGERPRINT:
        raise ValueError("Production sprite FSQ split fingerprint drifted")
    legal = compute_legal_tuples(corpus, split.training)
    legal_fingerprint = legal_tuple_fingerprint(legal)
    if (
        len(legal) != FROZEN_PRODUCTION_LEGAL_TUPLE_COUNT
        or legal_fingerprint != FROZEN_PRODUCTION_LEGAL_TUPLE_FINGERPRINT
    ):
        raise ValueError("Production sprite FSQ train-only legal tuple contract drifted")
    weights = compute_class_weights(corpus, split.training)
    return {
        "corpus": corpus,
        "split": split,
        "legal_tuples": torch.from_numpy(legal.astype(np.int64, copy=True)),
        "legal_tuple_array": legal,
        "legal_tuple_fingerprint": legal_fingerprint,
        "class_weights": weights,
        "training_dataset": SemanticFieldDataset(corpus, split.training),
        "validation_dataset": SemanticFieldDataset(corpus, split.validation),
    }


def batch_to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    required = ("part", "material", "emission", "morphology", "subtype", "role", "genes")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"Sprite latent batch is missing fields: {missing}")
    return {name: batch[name].to(device=device, non_blocking=False) for name in required}


def training_step(
    model: SemanticSpriteFSQ,
    batch: Mapping[str, Tensor],
    legal_tuples: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    quantize: bool,
    class_weights: Mapping[str, Tensor] | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    if gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive")
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch["part"],
        batch["material"],
        batch["emission"],
        batch["morphology"],
        batch["subtype"],
        batch["role"],
        batch["genes"],
        quantize=quantize,
    )
    loss, pieces = sprite_codec_loss(
        output,
        batch["part"],
        batch["material"],
        batch["emission"],
        legal_tuples,
        config=model.config,
        class_weights=class_weights,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Sprite latent training loss became non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("Sprite latent gradient norm became non-finite")
    optimizer.step()
    result = {name: float(value) for name, value in pieces.items()}
    result["gradient_norm"] = float(gradient_norm)
    result["quantized"] = float(quantize)
    return result


@torch.no_grad()
def exact_reconstruction_metrics(
    model: SemanticSpriteFSQ,
    batch: Mapping[str, Tensor],
    legal_tuples: Tensor,
) -> dict[str, float]:
    from .codec import project_legal_tuples

    model.eval()
    output = model(
        batch["part"],
        batch["material"],
        batch["emission"],
        batch["morphology"],
        batch["subtype"],
        batch["role"],
        batch["genes"],
        quantize=True,
    )
    projected = project_legal_tuples(output, legal_tuples)
    aligned = (
        (projected["part"] == batch["part"])
        & (projected["material"] == batch["material"])
        & (projected["emission"] == batch["emission"])
    )
    target_visible = batch["part"] != 0
    predicted_visible = projected["part"] != 0
    intersection = (target_visible & predicted_visible).flatten(1).sum(dim=1).float()
    union = (target_visible | predicted_visible).flatten(1).sum(dim=1).float().clamp_min(1.0)
    return {
        "aligned_tuple_accuracy": float(aligned.float().mean()),
        "part_accuracy": float((projected["part"] == batch["part"]).float().mean()),
        "material_accuracy": float(
            (projected["material"] == batch["material"]).float().mean()
        ),
        "emission_accuracy": float(
            (projected["emission"] == batch["emission"]).float().mean()
        ),
        "visible_silhouette_iou": float((intersection / union).mean()),
        "legal_projection_fraction": 1.0,
        "code_perplexity": float(output.perplexity),
        "code_utilization": float(output.utilization),
        "marginal_entropy": float(output.marginal_entropy),
        "soft_marginal_entropy": float(output.soft_marginal_entropy),
    }
