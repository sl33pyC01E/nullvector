from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ..map_topology_neural.codec import CategoricalTopologyCodec
from ..map_topology_neural.contract import FIELD_CLASS_COUNTS
from ..maps.model import WALKABLE_TERRAIN
from .dataset import TopologyProductionDataset, TopologyRef, ref_registry_sha256


FIELDS = ("terrain", "hazard", "elevation")


def balanced_reconstruction_loss(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    valid = batch["valid_mask"][:, 0].bool()
    denominator = valid.sum().clamp_min(1).to(torch.float32)
    losses: dict[str, torch.Tensor] = {}
    for field_index, name in enumerate(FIELDS):
        target = batch["categorical"][:, field_index].long()
        classes = FIELD_CLASS_COUNTS[name]
        counts = torch.bincount(target[valid], minlength=classes).to(torch.float32)
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = torch.sqrt(denominator / (present.sum().clamp_min(1) * counts[present]))
        weights = weights.clamp(min=0.25, max=8.0)
        per_cell = F.cross_entropy(output["logits"][name].float(), target, weight=weights, reduction="none")
        losses[name] = (per_cell * valid).sum() / denominator
    losses["commitment"] = output["commitment_loss"].float()
    losses["total"] = sum(losses[name] for name in FIELDS) + losses["commitment"]
    return losses


def _field_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    confusion = confusion.to(torch.float64)
    target = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    true_positive = confusion.diag()
    union = target + predicted - true_positive
    recall = torch.where(target > 0, true_positive / target, torch.ones_like(target))
    iou = torch.where(union > 0, true_positive / union, torch.ones_like(union))
    return {
        "accuracy": float(true_positive.sum() / target.sum().clamp_min(1)),
        "macro_recall": float(recall.mean()),
        "macro_iou": float(iou.mean()),
        "per_class_recall": [float(value) for value in recall],
        "per_class_iou": [float(value) for value in iou],
        "target_count": [int(value) for value in target],
        "prediction_count": [int(value) for value in predicted],
        "confusion": [[int(value) for value in row] for row in confusion],
    }


def evaluate_codec(
    model: CategoricalTopologyCodec,
    dataset: TopologyProductionDataset,
    refs: tuple[TopologyRef, ...],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    confusion = {
        name: torch.zeros((FIELD_CLASS_COUNTS[name], FIELD_CLASS_COUNTS[name]), dtype=torch.int64)
        for name in FIELDS
    }
    used_codes = torch.zeros(model.config.codebook_size, dtype=torch.bool)
    walk_intersection = 0
    walk_union = 0
    valid_cells = 0
    for ref in refs:
        batch = dataset.collate((ref,), device)
        valid = batch["valid_mask"][:, 0].bool()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, update_ema=False)
        used_codes[output["indices"].detach().long().cpu().flatten().unique()] = True
        predictions = {name: output["logits"][name].argmax(dim=1) for name in FIELDS}
        for field_index, name in enumerate(FIELDS):
            target = batch["categorical"][:, field_index].long()
            encoded = target[valid] * FIELD_CLASS_COUNTS[name] + predictions[name][valid]
            confusion[name] += torch.bincount(
                encoded.detach().cpu(), minlength=FIELD_CLASS_COUNTS[name] ** 2
            ).reshape(FIELD_CLASS_COUNTS[name], FIELD_CLASS_COUNTS[name])
        target_terrain = batch["categorical"][:, 0].long()
        predicted_terrain = predictions["terrain"]
        target_walk = torch.zeros_like(valid)
        predicted_walk = torch.zeros_like(valid)
        for class_id in WALKABLE_TERRAIN:
            target_walk |= target_terrain == int(class_id)
            predicted_walk |= predicted_terrain == int(class_id)
        walk_intersection += int((target_walk & predicted_walk & valid).sum())
        walk_union += int(((target_walk | predicted_walk) & valid).sum())
        valid_cells += int(valid.sum())
    return {
        "sample_count": len(refs),
        "sample_registry_sha256": ref_registry_sha256(refs),
        "valid_cell_count": valid_cells,
        "fields": {name: _field_metrics(value) for name, value in confusion.items()},
        "walkability_iou": walk_intersection / max(walk_union, 1),
        "codebook": {
            "size": model.config.codebook_size,
            "used": int(used_codes.sum()),
            "utilization": float(used_codes.to(torch.float32).mean()),
        },
    }

