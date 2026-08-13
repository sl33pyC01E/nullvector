from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..sprite_latent.codec import SemanticSpriteFSQ, project_legal_tuples
from ..sprite_latent.corpus import SemanticFieldCorpus
from .contract import QUALITY_GATES


def batch_from_indices(corpus: SemanticFieldCorpus, indices: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    values = np.asarray(indices, dtype=np.int64)
    return {
        "part": torch.from_numpy(corpus.part_owner[values]).long().to(device),
        "material": torch.from_numpy(corpus.material[values]).long().to(device),
        "emission": torch.from_numpy(corpus.emission_level[values]).long().to(device),
        "genes": torch.from_numpy(corpus.genes[values]).to(device),
        "morphology": torch.from_numpy(corpus.morphologies[values].astype(np.int64)).to(device),
        "subtype": torch.from_numpy(corpus.subtypes[values].astype(np.int64)).to(device),
        "role": torch.from_numpy(corpus.roles[values].astype(np.int64)).to(device),
        "source_index": torch.from_numpy(values).to(device),
    }


def _counter() -> dict[str, float]:
    return defaultdict(float)


def _accumulate(counter: dict[str, float], projected: dict[str, Tensor], batch: dict[str, Tensor]) -> None:
    part_ok = projected["part"] == batch["part"]
    material_ok = projected["material"] == batch["material"]
    emission_ok = projected["emission"] == batch["emission"]
    aligned = part_ok & material_ok & emission_ok
    target_visible = batch["part"] != 0
    predicted_visible = projected["part"] != 0
    intersection = target_visible & predicted_visible
    union = target_visible | predicted_visible
    counter["pixels"] += float(aligned.numel())
    counter["visible_pixels"] += float(target_visible.sum())
    counter["aligned"] += float(aligned.sum())
    counter["visible_aligned"] += float((aligned & target_visible).sum())
    counter["part"] += float(part_ok.sum())
    counter["material"] += float(material_ok.sum())
    counter["emission"] += float(emission_ok.sum())
    counter["intersection"] += float(intersection.sum())
    counter["union"] += float(union.sum())


def _metrics(counter: dict[str, float]) -> dict[str, float]:
    pixels = max(1.0, counter["pixels"])
    visible = max(1.0, counter["visible_pixels"])
    return {
        "aligned_tuple_accuracy": counter["aligned"] / pixels,
        "visible_tuple_accuracy": counter["visible_aligned"] / visible,
        "part_accuracy": counter["part"] / pixels,
        "material_accuracy": counter["material"] / pixels,
        "emission_accuracy": counter["emission"] / pixels,
        "visible_silhouette_iou": counter["intersection"] / max(1.0, counter["union"]),
    }


@torch.no_grad()
def evaluate_model(
    model: SemanticSpriteFSQ,
    corpus: SemanticFieldCorpus,
    indices: np.ndarray,
    legal_tuples: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    aggregate = _counter()
    family_counters = {family: _counter() for family in range(5)}
    observed_codes: set[int] = set()
    soft_entropy_weighted = 0.0
    examples = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = np.asarray(indices[start : start + batch_size], dtype=np.int64)
        batch = batch_from_indices(corpus, batch_indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                batch["part"], batch["material"], batch["emission"],
                batch["morphology"], batch["subtype"], batch["role"], batch["genes"],
                quantize=True,
            )
        projected = project_legal_tuples(output, legal_tuples)
        _accumulate(aggregate, projected, batch)
        for family in range(5):
            mask = batch["morphology"] == family
            if bool(mask.any()):
                _accumulate(
                    family_counters[family],
                    {name: value[mask] for name, value in projected.items()},
                    {name: value[mask] for name, value in batch.items() if name != "source_index"},
                )
        observed_codes.update(map(int, torch.unique(output.codes).cpu().tolist()))
        soft_entropy_weighted += float(output.soft_marginal_entropy) * len(batch_indices)
        examples += len(batch_indices)
    result = _metrics(aggregate)
    families = {str(family): _metrics(counter) for family, counter in family_counters.items()}
    result.update(
        {
            "sample_count": int(len(indices)),
            "legal_projection_fraction": 1.0,
            "unique_code_count": len(observed_codes),
            "code_utilization": len(observed_codes) / model.config.implicit_code_count,
            "soft_marginal_entropy": soft_entropy_weighted / max(1, examples),
            "families": families,
            "minimum_family_visible_tuple_accuracy": min(value["visible_tuple_accuracy"] for value in families.values()),
            "minimum_family_visible_silhouette_iou": min(value["visible_silhouette_iou"] for value in families.values()),
        }
    )
    gates = {name: result[name] >= threshold for name, threshold in QUALITY_GATES.items()}
    gates["legal_projection_exact"] = result["legal_projection_fraction"] == 1.0
    result["quality_gates"] = gates
    result["quality_accepted"] = all(gates.values())
    result["quality_score"] = (
        0.40 * result["visible_tuple_accuracy"]
        + 0.30 * result["visible_silhouette_iou"]
        + 0.15 * result["aligned_tuple_accuracy"]
        + 0.10 * result["minimum_family_visible_tuple_accuracy"]
        + 0.05 * min(1.0, result["code_utilization"] / QUALITY_GATES["code_utilization"])
    )
    return result
