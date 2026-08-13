from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ARCHETYPES, CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR, ForgeConfig
from .dataset import CachedSpriteDataset, build_corpus, split_indices
from .diffusion import (
    CategoricalSpriteDiffusion,
    TOKEN_COUNT,
    categorical_diffusion_loss,
)
from .grammar import compose_rgba, layers_to_tokens, tokens_to_layers
from .provenance import canonical_state_dict_hash, inference_source_hash
from .rig import postprocess_layers, structure_score
from .safety import require_disk_floor, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the categorical neural sprite constructor."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-corpus", action="store_true")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> ForgeConfig:
    config = ForgeConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.dataset_size is not None:
        config.dataset_size = args.dataset_size
    if args.validation_size is not None:
        config.validation_size = args.validation_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.seed is not None:
        config.seed = args.seed
    if args.smoke:
        config.dataset_size = 512
        config.validation_size = 64
        config.batch_size = 64
        config.epochs = 1
    return config


def layers_batch_to_tokens(layers: Tensor) -> Tensor:
    tokens = torch.zeros(
        (layers.shape[0], layers.shape[2], layers.shape[3]),
        dtype=torch.long,
        device=layers.device,
    )
    for index in range(layers.shape[1]):
        tokens = torch.where(layers[:, index] > 0.5, index + 1, tokens)
    return tokens


def compute_class_weights(corpus_path: Path) -> Tensor:
    with np.load(corpus_path) as payload:
        layers = payload["layers"]
        tokens = np.zeros((layers.shape[0], 32, 32), dtype=np.uint8)
        for index in range(layers.shape[1]):
            tokens[layers[:, index] > 0] = index + 1
    counts = np.bincount(tokens.reshape(-1), minlength=TOKEN_COUNT).astype(np.float64)
    frequency = counts / counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(frequency, 1.0e-6))
    weights /= weights.mean()
    weights = np.clip(weights, 0.25, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def update_ema(
    ema_model: CategoricalSpriteDiffusion,
    model: CategoricalSpriteDiffusion,
    decay: float,
) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def draw_preview_label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 10], fill=(4, 6, 12, 235))
    draw.text((3, 1), text, fill=(220, 240, 255, 255))


@torch.inference_mode()
def save_sample_preview(
    model: CategoricalSpriteDiffusion,
    device: torch.device,
    destination: Path,
    seed: int,
) -> dict[str, float]:
    model.eval()
    archetypes = torch.arange(len(ARCHETYPES), device=device).repeat_interleave(4)
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    genes = torch.rand((len(archetypes), 8), generator=cpu_generator).to(device)
    sample_generator = torch.Generator(device=device).manual_seed(seed ^ 0x5A17)
    tokens = model.sample(
        archetypes,
        genes,
        temperature=0.88,
        generator=sample_generator,
    ).cpu().numpy()

    scale = 5
    cell = 32 * scale
    sheet = Image.new("RGBA", (4 * cell, len(ARCHETYPES) * cell), (3, 4, 10, 255))
    valid = 0
    areas: list[int] = []
    for index, token_map in enumerate(tokens):
        raw_layers = tokens_to_layers(token_map)
        layers = postprocess_layers(raw_layers)
        body_area = int(np.maximum.reduce(layers[:6]).sum())
        areas.append(body_area)
        _, is_valid = structure_score(
            layers, int(archetypes[index].item())
        )
        if is_valid:
            valid += 1
        rgba = compose_rgba(
            layers,
            seed + index * 977,
            faction="player" if index < 4 else "hostile",
        )
        sprite = Image.fromarray(rgba).resize(
            (cell, cell), Image.Resampling.NEAREST
        )
        draw_preview_label(sprite, ARCHETYPES[int(archetypes[index].item())])
        sheet.alpha_composite(sprite, ((index % 4) * cell, (index // 4) * cell))

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return {
        "sample_valid_rate": valid / len(tokens),
        "sample_mean_hull_area": float(np.mean(areas)),
    }


@torch.inference_mode()
def evaluate(
    model: CategoricalSpriteDiffusion,
    loader: DataLoader,
    device: torch.device,
    class_weight: Tensor,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    accuracy_total = 0.0
    count = 0
    for layers, archetypes, _, genes in loader:
        layers = layers.to(device, non_blocking=True)
        archetypes = archetypes.to(device, non_blocking=True)
        genes = genes.to(device, non_blocking=True)
        tokens = layers_batch_to_tokens(layers)
        timesteps = torch.randint(
            1, model.steps + 1, (layers.shape[0],), device=device
        )
        corrupted, masked = model.corrupt(tokens, timesteps)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(corrupted, archetypes, genes, timesteps)
            result = categorical_diffusion_loss(
                logits, tokens, masked, class_weight
            )
        batch_size = layers.shape[0]
        loss_total += float(result.loss.item()) * batch_size
        accuracy_total += float(result.accuracy.item()) * batch_size
        count += batch_size
    return {
        "validation_loss": loss_total / max(count, 1),
        "validation_masked_accuracy": accuracy_total / max(count, 1),
    }


def save_checkpoint(
    destination: Path,
    model: CategoricalSpriteDiffusion,
    raw_model: CategoricalSpriteDiffusion,
    optimizer: torch.optim.Optimizer,
    config: ForgeConfig,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    torch.save(
        {
            "format": "neural-sprite-categorical-diffusion-v2",
            "model": state,
            "raw_model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
            "archetypes": ARCHETYPES,
            "architecture": model.architecture_config(),
            "canonical_model_hash": canonical_state_dict_hash(state),
            "inference_source_hash": inference_source_hash(),
            "training_objective": {
                "name": "masked-categorical-denoising",
                "mask_schedule": "sine-squared",
                "checkpoint_selection": "validation_masked_accuracy",
                "metric_scope": "partially-observed masked tokens; not unconditional generation",
            },
        },
        destination,
    )


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    require_disk_floor(
        CHECKPOINT_DIR,
        planned_bytes=max(config.dataset_size + config.validation_size, 1)
        * (8 * 32 * 32 + 64),
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_count = config.dataset_size + config.validation_size
    corpus_path = DATA_DIR / f"semantic_sprites_{total_count}_{config.seed}.npz"
    build_corpus(
        corpus_path,
        total_count,
        config.seed,
        force=args.force_corpus,
    )
    train_indices, validation_indices = split_indices(
        total_count, config.validation_size, config.seed ^ 0xC0FFEE
    )
    train_dataset = CachedSpriteDataset(corpus_path, train_indices)
    validation_dataset = CachedSpriteDataset(corpus_path, validation_indices)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    model = CategoricalSpriteDiffusion().to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs * len(train_loader), 1),
        eta_min=config.learning_rate * 0.08,
    )
    class_weight = compute_class_weights(corpus_path).to(device)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    global_step = 0
    best_score = -math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["raw_model"])
        ema_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint.get("history", []))

    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        model.train()
        rolling_loss = 0.0
        rolling_accuracy = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1:02d}/{config.epochs}")
        for layers, archetypes, _, genes in progress:
            layers = layers.to(device, non_blocking=True)
            archetypes = archetypes.to(device, non_blocking=True)
            genes = genes.to(device, non_blocking=True)
            clean_tokens = layers_batch_to_tokens(layers)
            timesteps = torch.randint(
                1, model.steps + 1, (layers.shape[0],), device=device
            )
            corrupted, masked = model.corrupt(clean_tokens, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(corrupted, archetypes, genes, timesteps)
                result = categorical_diffusion_loss(
                    logits,
                    clean_tokens,
                    masked,
                    class_weight,
                )
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            update_ema(ema_model, model, decay=min(0.999, 1.0 - 1.0 / (global_step + 1)))

            batch_size = layers.shape[0]
            rolling_loss += float(result.loss.item()) * batch_size
            rolling_accuracy += float(result.accuracy.item()) * batch_size
            sample_count += batch_size
            progress.set_postfix(
                loss=f"{rolling_loss / sample_count:.4f}",
                acc=f"{rolling_accuracy / sample_count:.3f}",
            )

        metrics = {
            "epoch": epoch + 1,
            "train_loss": rolling_loss / max(sample_count, 1),
            "train_masked_accuracy": rolling_accuracy / max(sample_count, 1),
            **evaluate(
                ema_model,
                validation_loader,
                device,
                class_weight,
            ),
        }
        if epoch == 0 or (epoch + 1) % 4 == 0 or epoch + 1 == config.epochs:
            metrics.update(
                save_sample_preview(
                    ema_model,
                    device,
                    OUTPUT_DIR / f"samples_epoch_{epoch + 1:03d}.png",
                    config.seed + epoch,
                )
            )
        history.append(metrics)
        score = metrics["validation_masked_accuracy"]
        save_checkpoint(
            CHECKPOINT_DIR / "latest.pt",
            ema_model,
            model,
            optimizer,
            config,
            epoch,
            global_step,
            history,
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                CHECKPOINT_DIR / "best.pt",
                ema_model,
                model,
                optimizer,
                config,
                epoch,
                global_step,
                history,
            )
        print(json.dumps(metrics, sort_keys=True))

    summary = {
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "epochs": config.epochs,
        "global_step": global_step,
        "best_validation_masked_accuracy": best_score,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        OUTPUT_DIR / "training_history.json",
        {"summary": summary, "history": history},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
