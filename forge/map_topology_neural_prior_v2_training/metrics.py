from __future__ import annotations

from contextlib import nullcontext
import hashlib
from typing import Any

import torch
import torch.nn.functional as F

from ..map_topology_neural_prior.masking import tensor_sha256
from ..map_topology_neural_prior_training.dataset import LatentRef, PriorTrainingDataset
from ..map_topology_neural_prior_v2.contract import CODEBOOK_SIZE
from ..map_topology_neural_prior_v2.masking import MASK_MODES_V2, mask_tokens_v2
from ..map_topology_neural_prior_v2.model import sample_parallel_v2
from .contract import PriorV2CalibrationConfig, canonical_json_bytes


def _context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def evaluate_masked(
    model: torch.nn.Module,
    dataset: PriorTrainingDataset,
    refs: tuple[LatentRef, ...],
    *, device: torch.device,
    config: PriorV2CalibrationConfig,
) -> dict[str, Any]:
    model.eval(); modes = {name: {"correct": 0, "count": 0, "loss": 0.0} for name in MASK_MODES_V2}
    registry = [ref.full_map_identity_sha256 for ref in refs]
    groups: dict[tuple[int, int], list[LatentRef]] = {}
    for ref in refs: groups.setdefault(ref.shape, []).append(ref)
    generator = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x4556414C)
    batch_counter = 0
    for shape in sorted(groups):
        values = groups[shape]
        for offset in range(0, len(values), config.maximum_batch_size):
            chunk = tuple(values[offset:offset + config.maximum_batch_size]); batch = dataset.collate(chunk)
            masked = mask_tokens_v2(batch["targets"], batch["valid_mask"], generator=generator, config=config.model_config(), step=batch_counter)
            inputs = {name: batch[name].to(device) for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")}
            inputs.update(tokens=masked["tokens"].to(device), mask_fraction=masked["mask_fraction"].to(device))
            with torch.inference_mode(), _context(device): logits = model(inputs)
            prediction = logits.float().argmax(dim=1).cpu(); per_cell = F.cross_entropy(logits.float(), batch["targets"].to(device), reduction="none").cpu()
            for index, mode in enumerate(masked["modes"]):
                local = masked["mask"][index]; count = int(local.sum()); modes[mode]["count"] += count
                modes[mode]["correct"] += int((prediction[index][local] == batch["targets"][index][local]).sum()); modes[mode]["loss"] += float(per_cell[index][local].sum())
            batch_counter += 1
    result_modes = {name: {"masked_cells": row["count"], "accuracy": row["correct"] / max(1, row["count"]), "loss": row["loss"] / max(1, row["count"])} for name, row in modes.items()}
    total = sum(row["masked_cells"] for row in result_modes.values()); correct = sum(modes[name]["correct"] for name in MASK_MODES_V2); loss = sum(modes[name]["loss"] for name in MASK_MODES_V2)
    return {
        "sample_count": len(refs), "sample_registry_sha256": hashlib.sha256(canonical_json_bytes(registry)).hexdigest(),
        "masked_cells": total, "accuracy": correct / max(1, total), "loss": loss / max(1, total),
        "macro_mode_accuracy": sum(row["accuracy"] for row in result_modes.values()) / len(result_modes),
        "full_mask_accuracy": result_modes["full"]["accuracy"], "full_mask_loss": result_modes["full"]["loss"],
        "modes": result_modes, "vocabulary_size": CODEBOOK_SIZE,
    }


def evaluate_free_generation(
    model: torch.nn.Module,
    dataset: PriorTrainingDataset,
    refs: tuple[LatentRef, ...],
    *, device: torch.device,
    config: PriorV2CalibrationConfig,
) -> dict[str, Any]:
    if len(refs) != 6 or len({ref.shape for ref in refs}) != 1:
        raise ValueError("Prior-v2 free-generation sentinel must contain six homogeneous theme samples.")
    batch = dataset.collate(refs); conditions = {name: batch[name].to(device) for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")}
    with _context(device): sampled = sample_parallel_v2(model, conditions, sampling_steps=config.sampling_steps)
    tokens = sampled["tokens"].cpu(); valid = batch["valid_mask"][:, 0]
    accuracy = float((tokens[valid] == batch["targets"][valid]).float().mean())
    return {
        "sample_count": 6,
        "sample_registry_sha256": hashlib.sha256(canonical_json_bytes([ref.full_map_identity_sha256 for ref in refs])).hexdigest(),
        "token_accuracy": accuracy, "unique_samples": len({tensor_sha256(tokens[index]) for index in range(6)}),
        "tokens_sha256": tensor_sha256(tokens), "uncertainty_sha256": tensor_sha256(sampled["uncertainty"].cpu()),
        "all_tokens_revealed": bool(((tokens >= 0) & (tokens < CODEBOOK_SIZE)).all()),
    }
