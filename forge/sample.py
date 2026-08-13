from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from .config import (
    ARCHETYPES,
    CHECKPOINT_DIR,
    GAME_GENERATED_DIR,
    LAYER_NAMES,
    OUTPUT_DIR,
)
from .determinism import configure_deterministic_inference
from .grammar import (
    compose_rgba,
    genome_from_seed,
    genome_vector,
    layers_to_tokens,
    palette_for_seed,
    stream_seed,
    tokens_to_layers,
)
from .provenance import (
    architecture_from_state_dict,
    canonical_state_dict_hash,
    checkpoint_file_hash,
    inference_source_hash,
    model_from_architecture,
)
from .rig import (
    POSTPROCESS_VERSION,
    RIG_VERSION,
    bake_animation_atlas,
    postprocess_layers,
    structure_score,
)
from .safety import require_disk_floor, write_json_atomic


SAMPLER_NAME = "absorbing-categorical-confidence-v2"
SAMPLER_SCHEDULE = "sine-squared"
SELECTION_VERSION = "validity-diversity-v3"
DIVERSITY_MAX_IOU = 0.92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample neural sprite genomes and bake Godot-ready atlases."
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--count-per-archetype", type=int, default=4)
    parser.add_argument("--seed", type=int, default=48_217)
    parser.add_argument("--temperature", type=float, default=0.86)
    parser.add_argument("--candidates-per-sprite", type=int, default=16)
    parser.add_argument("--destination", type=Path, default=GAME_GENERATED_DIR)
    return parser.parse_args()


def _silhouette(layers: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(layers[:6]).astype(bool)


def _silhouette_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def _select_diverse(
    indices: list[int],
    analyses: list[dict[str, object]],
    count: int,
) -> list[tuple[int, float, float]]:
    eligible = [
        index
        for index in indices
        if bool(analyses[index]["raw_valid"])
        and bool(analyses[index]["processed_valid"])
    ]
    unique: dict[bytes, int] = {}
    for index in eligible:
        fingerprint = analyses[index]["layers"].tobytes()  # type: ignore[union-attr]
        previous = unique.get(fingerprint)
        if previous is None or float(analyses[index]["processed_score"]) > float(
            analyses[previous]["processed_score"]
        ):
            unique[fingerprint] = index
    pool = list(unique.values())
    if len(pool) < count:
        raise RuntimeError(
            f"Only {len(pool)} raw+processed valid unique candidates are available; "
            f"need {count}. Increase --candidates-per-sprite."
        )

    selected: list[tuple[int, float, float]] = []
    while len(selected) < count:
        selected_indices = [item[0] for item in selected]
        scored: list[tuple[float, int, float]] = []
        for index in pool:
            if index in selected_indices:
                continue
            silhouette = analyses[index]["silhouette"]
            maximum_iou = max(
                (
                    _silhouette_iou(
                        silhouette, analyses[chosen]["silhouette"]  # type: ignore[arg-type]
                    )
                    for chosen in selected_indices
                ),
                default=0.0,
            )
            base = float(analyses[index]["processed_score"])
            diversity_adjusted = base - maximum_iou * 2.0
            scored.append((diversity_adjusted, index, maximum_iou))
        preferred = [
            candidate
            for candidate in scored
            if not selected_indices or candidate[2] < DIVERSITY_MAX_IOU
        ]
        candidates = preferred or scored
        best = max(candidates)
        assert best is not None
        selected.append((best[1], best[0], best[2]))
    return selected


def _save_token_artifacts(
    destination: Path,
    sprite_id: str,
    raw_tokens: np.ndarray,
    raw_layers: np.ndarray,
    processed_layers: np.ndarray,
) -> dict[str, str]:
    raw_name = f"{sprite_id}_raw_tokens.png"
    processed_name = f"{sprite_id}_processed_tokens.png"
    semantic_name = f"{sprite_id}_semantic_layers.npz"
    processed_tokens = layers_to_tokens(processed_layers)
    Image.fromarray(raw_tokens.astype(np.uint8)).save(destination / raw_name)
    Image.fromarray(processed_tokens.astype(np.uint8)).save(
        destination / processed_name
    )
    np.savez_compressed(
        destination / semantic_name,
        layers=processed_layers.astype(np.uint8),
        raw_layers=raw_layers.astype(np.uint8),
        raw_tokens=raw_tokens.astype(np.uint8),
        processed_tokens=processed_tokens.astype(np.uint8),
        layer_names=np.asarray(LAYER_NAMES),
    )
    return {
        "raw_tokens": raw_name,
        "processed_tokens": processed_name,
        "semantic_layers": semantic_name,
    }


def _new_sheet(count_per_archetype: int, title: str) -> tuple[Image.Image, int, int]:
    preview_scale = 6
    cell = 32 * preview_scale
    header = 28
    sheet = Image.new(
        "RGBA",
        (count_per_archetype * cell, header + len(ARCHETYPES) * cell),
        (3, 4, 10, 255),
    )
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, header - 1), fill=(7, 12, 24, 255))
    draw.text((8, 8), title, fill=(205, 241, 255, 255))
    return sheet, cell, header


def _draw_candidate(
    sheet: Image.Image,
    layers: np.ndarray,
    seed: int,
    faction: str,
    column: int,
    row: int,
    cell: int,
    header: int,
    label: str,
) -> None:
    rgba = compose_rgba(layers, seed, faction=faction)
    sprite = Image.fromarray(rgba).resize((cell, cell), Image.Resampling.NEAREST)
    x = column * cell
    y = header + row * cell
    sheet.alpha_composite(sprite, (x, y))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((x + 3, y + 3, x + cell - 3, y + 16), fill=(3, 5, 12, 205))
    draw.text((x + 6, y + 5), label, fill=(220, 240, 255, 255))


def main() -> None:
    args = parse_args()
    require_disk_floor(args.destination, planned_bytes=256 * 1024 * 1024)
    configure_deterministic_inference()
    if args.count_per_archetype < 1 or args.candidates_per_sprite < 1:
        raise ValueError("Sprite and candidate counts must be positive.")
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model"]
    architecture = dict(
        checkpoint.get("architecture") or architecture_from_state_dict(state)
    )
    model = model_from_architecture(architecture).to(device)
    model.load_state_dict(state)
    model.eval()

    candidates_per_archetype = args.count_per_archetype * args.candidates_per_sprite
    candidate_count = candidates_per_archetype * len(ARCHETYPES)
    archetypes = torch.arange(len(ARCHETYPES), device=device).repeat_interleave(
        candidates_per_archetype
    )
    seeds = [args.seed + index * 1009 for index in range(candidate_count)]
    genomes = [
        genome_from_seed(seed, int(archetypes[index].item()))
        for index, seed in enumerate(seeds)
    ]
    gene_array = np.stack([genome_vector(genome) for genome in genomes])
    genes = torch.from_numpy(gene_array).to(device)
    noise_seeds = [stream_seed(seed, 0xD1FF) for seed in seeds]
    generated_batches = []
    with torch.inference_mode():
        # Deliberately use batch size one. Some CUDA convolution algorithms can
        # make threshold-adjacent multinomial choices vary with batch shape;
        # one independent forward stream makes every exported manifest exactly
        # replayable in isolation.
        for index in tqdm(
            range(candidate_count), desc="sampling reproducible candidates"
        ):
            generator = torch.Generator(device=device).manual_seed(
                noise_seeds[index]
            )
            generated_batches.append(
                model.sample(
                    archetypes[index : index + 1],
                    genes[index : index + 1],
                    temperature=args.temperature,
                    generators=[generator],
                ).cpu()
            )
    tokens = torch.cat(generated_batches, dim=0).numpy()

    analyses: list[dict[str, object]] = []
    for index, token_map in enumerate(tokens):
        archetype = int(archetypes[index].item())
        raw_layers = tokens_to_layers(token_map)
        layers = postprocess_layers(raw_layers)
        raw_score, raw_valid = structure_score(raw_layers, archetype)
        processed_score, processed_valid = structure_score(layers, archetype)
        analyses.append(
            {
                "raw_layers": raw_layers,
                "layers": layers,
                "silhouette": _silhouette(layers),
                "raw_score": raw_score,
                "raw_valid": raw_valid,
                "processed_score": processed_score,
                "processed_valid": processed_valid,
            }
        )

    selected: list[tuple[int, float, float]] = []
    for archetype in range(len(ARCHETYPES)):
        start = archetype * candidates_per_archetype
        stop = start + candidates_per_archetype
        selected.extend(
            _select_diverse(list(range(start, stop)), analyses, args.count_per_archetype)
        )

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    canonical_hash = canonical_state_dict_hash(state)
    model_id = canonical_hash[:16]
    file_hash = checkpoint_file_hash(args.checkpoint)
    source_hash = inference_source_hash()
    curated_sheet, cell, header = _new_sheet(
        args.count_per_archetype,
        f"CURATED NEURAL OUTPUT // {len(selected)} OF {candidate_count} CANDIDATES",
    )
    random_sheet, random_cell, random_header = _new_sheet(
        args.count_per_archetype,
        "FIXED UNFILTERED CANDIDATES // POSTPROCESS APPLIED",
    )

    entries = []
    for output_index, (index, diversity_score, maximum_iou) in enumerate(selected):
        archetype = int(archetypes[index].item())
        seed = seeds[index]
        faction = "player" if output_index == 0 else "hostile"
        sprite_id = f"{ARCHETYPES[archetype]}_{seed:08x}"
        layers = analyses[index]["layers"]
        raw_layers = analyses[index]["raw_layers"]
        atlas_path = destination / f"{sprite_id}.png"
        manifest = bake_animation_atlas(
            layers,  # type: ignore[arg-type]
            seed,
            archetype,
            atlas_path,
            faction=faction,
        )
        artifact_names = _save_token_artifacts(
            destination,
            sprite_id,
            tokens[index],
            raw_layers,  # type: ignore[arg-type]
            layers,  # type: ignore[arg-type]
        )
        quality_rank = 1 + sum(
            float(analyses[other]["processed_score"])
            > float(analyses[index]["processed_score"])
            for other in range(
                archetype * candidates_per_archetype,
                (archetype + 1) * candidates_per_archetype,
            )
            if bool(analyses[other]["raw_valid"])
            and bool(analyses[other]["processed_valid"])
        )
        manifest.update(
            {
                "id": sprite_id,
                "model_hash": model_id,
                "faction": faction,
                "palette": {
                    key: list(value)
                    for key, value in palette_for_seed(seed, faction).items()
                },
                "genome": {
                    "seed": seed,
                    "genes": [round(float(value), 6) for value in gene_array[index]],
                    "recipe": genomes[index].to_dict(),
                },
                "generation": {
                    "sampler": SAMPLER_NAME,
                    "schedule": SAMPLER_SCHEDULE,
                    "temperature": args.temperature,
                    "noise_seed": noise_seeds[index],
                    "candidate_index": index,
                    "candidate_index_within_archetype": index
                    - archetype * candidates_per_archetype,
                    "candidate_batch_size": 1,
                    "candidates_per_archetype": candidates_per_archetype,
                    "torch_version": torch.__version__,
                    "device_type": device.type,
                },
                "selection": {
                    "version": SELECTION_VERSION,
                    "quality_rank": quality_rank,
                    "raw_score": round(float(analyses[index]["raw_score"]), 6),
                    "processed_score": round(
                        float(analyses[index]["processed_score"]), 6
                    ),
                    "diversity_adjusted_score": round(diversity_score, 6),
                    "maximum_prior_silhouette_iou": round(maximum_iou, 6),
                    "raw_valid": bool(analyses[index]["raw_valid"]),
                    "processed_valid": bool(analyses[index]["processed_valid"]),
                },
                "source": artifact_names,
                "inference_source_hash": source_hash[:16],
            }
        )
        entries.append(manifest)
        _draw_candidate(
            curated_sheet,
            layers,  # type: ignore[arg-type]
            seed,
            faction,
            output_index % args.count_per_archetype,
            output_index // args.count_per_archetype,
            cell,
            header,
            sprite_id,
        )

    random_rng = np.random.default_rng(args.seed ^ 0x51EE7)
    for archetype in range(len(ARCHETYPES)):
        start = archetype * candidates_per_archetype
        sampled = random_rng.choice(
            np.arange(start, start + candidates_per_archetype),
            size=args.count_per_archetype,
            replace=False,
        )
        for column, index_value in enumerate(sampled.tolist()):
            valid_mark = "OK" if analyses[index_value]["processed_valid"] else "INVALID"
            _draw_candidate(
                random_sheet,
                analyses[index_value]["layers"],  # type: ignore[arg-type]
                seeds[index_value],
                "hostile",
                column,
                archetype,
                random_cell,
                random_header,
                f"{ARCHETYPES[archetype]} #{index_value - start:02d} {valid_mark}",
            )

    both_valid = sum(
        bool(item["raw_valid"]) and bool(item["processed_valid"]) for item in analyses
    )
    acceptance = {
        "candidate_count": candidate_count,
        "selected_count": len(entries),
        "raw_valid_count": sum(bool(item["raw_valid"]) for item in analyses),
        "processed_valid_count": sum(
            bool(item["processed_valid"]) for item in analyses
        ),
        "raw_and_processed_valid_count": both_valid,
        "raw_and_processed_valid_rate": round(both_valid / candidate_count, 6),
    }
    sampler = {
        "name": SAMPLER_NAME,
        "schedule": SAMPLER_SCHEDULE,
        "temperature": args.temperature,
        "steps": int(architecture["steps"]),
        "confidence": "probability of sampled token",
        "accepted_tokens_are_absorbing": True,
        "noise_seed_derivation": "mix32(recipe_seed xor stream(0xD1FF))",
    }
    registry = {
        "format": "neural-sprite-registry-v2",
        "model_hash": model_id,
        "model": {
            "canonical_hash": canonical_hash,
            "checkpoint_file_hash": file_hash,
            "checkpoint": args.checkpoint.name,
            "checkpoint_format": checkpoint.get("format", "unknown"),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
            "global_step": int(checkpoint.get("global_step", -1)),
            "architecture": architecture,
            "training_source_hash": checkpoint.get("inference_source_hash"),
            "metric_note": (
                "Checkpoint selection used partially-observed masked-token accuracy; "
                "generation quality is reported separately."
            ),
        },
        "sampler": sampler,
        "pipeline": {
            "inference_source_hash": source_hash,
            "postprocess_version": POSTPROCESS_VERSION,
            "rig_version": RIG_VERSION,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "generation_seed": args.seed,
        "acceptance": acceptance,
        "sprite_count": len(entries),
        "sprites": entries,
    }
    registry_path = destination / "sprite_registry.json"
    write_json_atomic(registry_path, registry)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curated_path = OUTPUT_DIR / "neural_sprite_contact_sheet.png"
    random_path = OUTPUT_DIR / "neural_sprite_unfiltered_sheet.png"
    report_path = OUTPUT_DIR / "generation_report.json"
    curated_sheet.save(curated_path)
    random_sheet.save(random_path)
    write_json_atomic(
        report_path,
        {
            "model_hash": model_id,
            "source_hash": source_hash,
            "sampler": sampler,
            "acceptance": acceptance,
            "curated_sheet": str(curated_path),
            "unfiltered_sheet": str(random_path),
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "model_hash": model_id,
                "sprites": len(entries),
                "acceptance": acceptance,
                "registry": str(registry_path),
                "curated_preview": str(curated_path),
                "unfiltered_preview": str(random_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
