from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.legality import TorchLegalMasks
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_production.training import CorpusSampleRef
from ..map_decorator_production_v4.decoding import select_proposal_conditioned_argmax
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from ..map_decorator_production_v4.proposal import ProposalAuthority
from ..map_decorator_production_v4_training.dataset import ProposalTeacherSample, collate_proposal_samples
from .contract import OBJECT_METRICS


def _batches(items: Sequence[CorpusSampleRef], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _predict(
    model: ProposalConditionedDecoratorV4,
    batch: dict[str, object],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, bool]:
    features = batch["features"].to(device)  # type: ignore[union-attr]
    targets = {name: batch["targets"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    legal_masks = {name: batch["legal_masks"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    valid = batch["valid_cells"].to(device)  # type: ignore[union-attr]
    hard_empty = batch["hard_empty"].to(device)  # type: ignore[union-attr]
    theme = batch["theme_index"].to(device)  # type: ignore[union-attr]
    conditions = batch["global_conditions"].to(device)  # type: ignore[union-attr]
    proposals = {name: batch["proposals"][name].to(device) for name in ("decal", "prop")}  # type: ignore[index,union-attr]
    masked = {name: valid.clone() for name in HEAD_NAMES}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            features,
            targets,
            masked,
            theme,
            conditions,
            torch.ones((features.shape[0],), dtype=torch.float32, device=device),
            proposals,
        )
        prediction = select_proposal_conditioned_argmax(
            output,
            TorchLegalMasks(hard_empty=hard_empty, **legal_masks),
        )
    legal = True
    for name in HEAD_NAMES:
        selected_legal = legal_masks[name].gather(1, prediction[name].unsqueeze(1)).squeeze(1)
        legal = legal and bool(selected_legal[valid].all())
        if name != "variant":
            legal = legal and not bool((prediction[name][hard_empty & valid] != 0).any())
    legal = legal and not bool(((prediction["decal"] != 0) & (prediction["prop"] != 0) & valid).any())
    for name in ("decal", "prop"):
        proposed = proposals[name].gather(1, (prediction[name] - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        legal = legal and not bool(((prediction[name] != 0) & ~proposed & valid).any())
    return prediction, targets, valid, legal


def evaluate_full_split(
    model: ProposalConditionedDecoratorV4,
    authority: ProposalAuthority,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    refs = authority.authority.corpus.epoch_refs(split, 0, seed)
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    identities: list[str] = []
    valid_cells = 0
    hard_legal = True
    was_training = model.training
    model.eval()
    try:
        for group in _batches(refs, batch_size):
            samples = []
            for ref in group:
                sample, proposal = authority.sample_and_proposals(ref)
                samples.append(ProposalTeacherSample(sample, proposal))
            observed, truth, valid, batch_legal = _predict(
                model,
                collate_proposal_samples(samples),
                device=device,
            )
            hard_legal = hard_legal and batch_legal
            for name in HEAD_NAMES:
                predictions[name].append(observed[name][valid].detach().cpu())
                targets[name].append(truth[name][valid].detach().cpu())
            valid_cells += int(valid.sum().item())
            identities.extend(ref.sample_identity_sha256 for ref in group)
    finally:
        model.train(was_training)
    metrics = decoration_metrics(
        {name: torch.cat(values) for name, values in predictions.items()},
        {name: torch.cat(values) for name, values in targets.items()},
        torch.ones((valid_cells,), dtype=torch.bool),
    )
    metrics.update(
        {
            "split": split,
            "sample_count": len(identities),
            "sample_set_sha256": json_sha256(sorted(identities)),
            "full_split": len(identities) == len(authority.authority.corpus.refs_by_split[split]),
            "valid_cell_count": valid_cells,
            "hard_legality": 1.0 if hard_legal else 0.0,
            "immutable_semantic_changes": 0,
            "source_provenance_failures": 0,
        }
    )
    return metrics


def compare_to_baseline(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    records: dict[str, object] = {}
    passed = True
    strict_improvements = 0
    for split in ("validation", "test"):
        split_records: dict[str, object] = {}
        for head in ("decal", "prop"):
            head_records: dict[str, object] = {}
            for metric in OBJECT_METRICS:
                before = float(baseline[split]["heads"][head][metric])  # type: ignore[index]
                after = float(candidate[split]["heads"][head][metric])  # type: ignore[index]
                delta = after - before
                nonregressing = delta >= -tolerance
                passed = passed and nonregressing
                strict_improvements += int(delta > tolerance)
                head_records[metric] = {
                    "baseline": before,
                    "candidate": after,
                    "delta": delta,
                    "nonregressing": nonregressing,
                }
            split_records[head] = head_records
        records[split] = split_records
    safety = all(
        candidate[split].get("hard_legality") == 1.0
        and candidate[split].get("immutable_semantic_changes") == 0
        and candidate[split].get("source_provenance_failures") == 0
        and candidate[split].get("full_split") is True
        for split in ("validation", "test")
    )
    return {
        "tolerance": tolerance,
        "records": records,
        "strict_improvement_count": strict_improvements,
        "every_object_metric_nonregressing": passed,
        "at_least_one_strict_improvement": strict_improvements > 0,
        "hard_safety": safety,
        "passed": passed and strict_improvements > 0 and safety,
    }
