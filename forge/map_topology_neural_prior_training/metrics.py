from __future__ import annotations

from contextlib import nullcontext
import hashlib
from typing import Any

import torch
import torch.nn.functional as F

from ..map_topology_neural_prior.contract import CODEBOOK_SIZE
from ..map_topology_neural_prior.masking import MASK_MODES, mask_tokens
from .contract import PriorCalibrationConfig, canonical_json_bytes
from .dataset import LatentRef, PriorTrainingDataset


def evaluate_prior(
    model: torch.nn.Module,
    dataset: PriorTrainingDataset,
    refs: tuple[LatentRef, ...],
    *,
    device: torch.device,
    config: PriorCalibrationConfig,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    masked_cells = 0
    loss_sum = 0.0
    modes = {name: {"correct": 0, "count": 0} for name in MASK_MODES}
    generator = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x4556414C)
    groups: dict[tuple[int, int], list[LatentRef]] = {}
    for ref in refs:
        groups.setdefault(ref.shape, []).append(ref)
    batch_counter = 0
    for shape in sorted(groups):
        values = groups[shape]
        for offset in range(0, len(values), config.maximum_batch_size):
            chunk = tuple(values[offset : offset + config.maximum_batch_size])
            batch = dataset.collate(chunk)
            masked = mask_tokens(batch["targets"], batch["valid_mask"], generator=generator, config=config.model_config(), step=batch_counter)
            inputs = {
                "tokens": masked["tokens"].to(device),
                "valid_mask": batch["valid_mask"].to(device),
                "point_conditions": batch["point_conditions"].to(device),
                "global_conditions": batch["global_conditions"].to(device),
                "theme_index": batch["theme_index"].to(device),
                "mask_fraction": masked["mask_fraction"].to(device),
            }
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), context:
                logits = model(inputs)
            per_cell = F.cross_entropy(logits.float(), batch["targets"].to(device), reduction="none").cpu()
            prediction = logits.float().argmax(dim=1).cpu()
            mask = masked["mask"]
            correct += int((prediction[mask] == batch["targets"][mask]).sum())
            count = int(mask.sum())
            masked_cells += count
            loss_sum += float(per_cell[mask].sum())
            for index, mode in enumerate(masked["modes"]):
                local = mask[index]
                modes[mode]["correct"] += int((prediction[index][local] == batch["targets"][index][local]).sum())
                modes[mode]["count"] += int(local.sum())
            batch_counter += 1
    mode_metrics = {
        name: {"masked_cells": values["count"], "accuracy": values["correct"] / max(1, values["count"])}
        for name, values in modes.items()
    }
    return {
        "sample_count": len(refs),
        "sample_registry_sha256": hashlib.sha256(canonical_json_bytes([ref.full_map_identity_sha256 for ref in refs])).hexdigest(),
        "masked_cells": masked_cells,
        "loss": loss_sum / max(1, masked_cells),
        "accuracy": correct / max(1, masked_cells),
        "macro_mode_accuracy": sum(value["accuracy"] for value in mode_metrics.values()) / len(mode_metrics),
        "modes": mode_metrics,
        "vocabulary_size": CODEBOOK_SIZE,
    }

