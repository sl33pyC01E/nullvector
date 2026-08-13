from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import ARCHETYPES, CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR, ForgeConfig
from .dataset import CachedSpriteDataset, build_corpus, split_indices
from .determinism import configure_deterministic_inference
from .grammar import genome_from_seed, genome_vector, stream_seed, tokens_to_layers
from .provenance import (
    architecture_from_state_dict,
    canonical_state_dict_hash,
    model_from_architecture,
)
from .rig import postprocess_layers, structure_score
from .train import layers_batch_to_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure neural sprite denoising, generation validity, and diversity."
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "model_benchmark.json",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    configure_deterministic_inference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ForgeConfig(**checkpoint["config"])
    state = checkpoint["model"]
    architecture = checkpoint.get("architecture") or architecture_from_state_dict(state)
    model = model_from_architecture(architecture).to(device)
    model.load_state_dict(state)
    model.eval()
    torch.manual_seed(0xC0DEC0DE)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0xC0DEC0DE)

    total_count = config.dataset_size + config.validation_size
    corpus_path = DATA_DIR / f"semantic_sprites_{total_count}_{config.seed}.npz"
    build_corpus(corpus_path, total_count, config.seed)
    _, validation_indices = split_indices(
        total_count, config.validation_size, config.seed ^ 0xC0FFEE
    )
    dataset = CachedSpriteDataset(corpus_path, validation_indices[:512])
    layers = torch.stack([dataset[index][0] for index in range(len(dataset))]).to(device)
    archetypes = torch.stack([dataset[index][1] for index in range(len(dataset))]).to(device)
    genes = torch.stack([dataset[index][3] for index in range(len(dataset))]).to(device)
    clean_tokens = layers_batch_to_tokens(layers)

    denoising = {}
    for timestep in (1, 3, 6, 9, 12):
        timesteps = torch.full(
            (len(dataset),), timestep, dtype=torch.long, device=device
        )
        corrupted, masked = model.corrupt(clean_tokens, timesteps)
        logits = model(corrupted, archetypes, genes, timesteps)
        prediction = logits.argmax(dim=1)
        masked_count = masked.sum().clamp_min(1)
        opaque = masked & (clean_tokens != 0)
        background = masked & (clean_tokens == 0)
        per_token_recall = {}
        per_token_iou = {}
        for token in range(model.token_count):
            target = masked & (clean_tokens == token)
            predicted = masked & (prediction == token)
            true_positive = (target & predicted).sum()
            per_token_recall[str(token)] = float(
                (true_positive / target.sum().clamp_min(1)).item()
            )
            per_token_iou[str(token)] = float(
                (
                    true_positive
                    / torch.logical_or(target, predicted).sum().clamp_min(1)
                ).item()
            )
        denoising[str(timestep)] = {
            "mask_fraction": float(masked.float().mean().item()),
            "masked_accuracy": float(
                (((prediction == clean_tokens) & masked).sum() / masked_count).item()
            ),
            "opaque_accuracy": float(
                (
                    ((prediction == clean_tokens) & opaque).sum()
                    / opaque.sum().clamp_min(1)
                ).item()
            ),
            "background_accuracy": float(
                (
                    ((prediction == clean_tokens) & background).sum()
                    / background.sum().clamp_min(1)
                ).item()
            ),
            "foreground_macro_iou": float(
                np.mean([per_token_iou[str(token)] for token in range(1, model.token_count)])
            ),
            "per_token_recall": per_token_recall,
            "per_token_iou": per_token_iou,
        }

    candidate_count = max(args.samples, len(ARCHETYPES))
    candidate_archetypes = torch.arange(
        candidate_count, device=device
    ) % len(ARCHETYPES)
    candidate_seeds = [91_117 + index * 4099 for index in range(candidate_count)]
    gene_array = np.stack(
        [
            genome_vector(
                genome_from_seed(seed, int(candidate_archetypes[index].item()))
            )
            for index, seed in enumerate(candidate_seeds)
        ]
    )
    candidate_genes = torch.from_numpy(gene_array).to(device)
    noise_seeds = [stream_seed(seed, 0xBADC) for seed in candidate_seeds]
    generators = [
        torch.Generator(device=device).manual_seed(seed) for seed in noise_seeds
    ]
    generated = model.sample(
        candidate_archetypes,
        candidate_genes,
        temperature=0.86,
        generators=generators,
    ).cpu().numpy()

    raw_valid = 0
    processed_valid = 0
    both_valid = 0
    scores = []
    fingerprints: set[bytes] = set()
    archetype_valid = {name: [0, 0] for name in ARCHETYPES}
    silhouettes = {name: [] for name in ARCHETYPES}
    for index, token_map in enumerate(generated):
        archetype = int(candidate_archetypes[index].item())
        raw_layers = tokens_to_layers(token_map)
        layers_np = postprocess_layers(raw_layers)
        _, is_raw_valid = structure_score(raw_layers, archetype)
        score, is_valid = structure_score(layers_np, archetype)
        scores.append(score)
        raw_valid += int(is_raw_valid)
        processed_valid += int(is_valid)
        accepted = is_raw_valid and is_valid
        both_valid += int(accepted)
        archetype_valid[ARCHETYPES[archetype]][0] += int(accepted)
        archetype_valid[ARCHETYPES[archetype]][1] += 1
        fingerprints.add(layers_np.tobytes())
        silhouettes[ARCHETYPES[archetype]].append(
            np.maximum.reduce(layers_np[:6]).astype(bool)
        )

    diversity = {}
    for name, masks in silhouettes.items():
        pairwise = []
        for first in range(len(masks)):
            for second in range(first + 1, len(masks)):
                union = np.logical_or(masks[first], masks[second]).sum()
                pairwise.append(
                    float(np.logical_and(masks[first], masks[second]).sum() / max(union, 1))
                )
        diversity[name] = {
            "mean_pairwise_silhouette_iou": float(np.mean(pairwise)) if pairwise else 0.0,
            "maximum_pairwise_silhouette_iou": float(np.max(pairwise)) if pairwise else 0.0,
        }

    report = {
        "checkpoint": str(args.checkpoint),
        "canonical_model_hash": canonical_state_dict_hash(state),
        "architecture": architecture,
        "epoch": int(checkpoint["epoch"]) + 1,
        "global_step": int(checkpoint["global_step"]),
        "device": str(device),
        "denoising_by_timestep": denoising,
        "generation": {
            "samples": candidate_count,
            "sampler_batch_size": candidate_count,
            "deterministic_cuda_kernels": True,
            "raw_structural_valid_rate": raw_valid / candidate_count,
            "processed_structural_valid_rate": processed_valid / candidate_count,
            "raw_and_processed_valid_rate": both_valid / candidate_count,
            "exact_semantic_unique_rate": len(fingerprints) / candidate_count,
            "mean_structure_score": float(np.mean(scores)),
            "archetype_valid_rate": {
                name: passed / max(total, 1)
                for name, (passed, total) in archetype_valid.items()
            },
            "diversity": diversity,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
