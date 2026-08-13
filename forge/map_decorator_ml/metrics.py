from __future__ import annotations

from collections.abc import Mapping

import torch

from .contract import HEAD_CLASS_COUNTS, HEAD_NAMES


def _confusion(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    classes: int,
) -> torch.Tensor:
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Prediction, target, and valid masks must share [B,H,W].")
    if prediction.dtype != torch.long or target.dtype != torch.long or valid.dtype != torch.bool:
        raise TypeError("Prediction/target must be int64 and valid must be bool.")
    pred = prediction[valid]
    truth = target[valid]
    if bool(((pred < 0) | (pred >= classes) | (truth < 0) | (truth >= classes)).any()):
        raise ValueError("Metric inputs contain an out-of-domain class.")
    bins = torch.bincount(truth * classes + pred, minlength=classes * classes)
    return bins.reshape(classes, classes).to(torch.float64)


def head_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    classes: int,
    empty_class: int | None,
) -> dict[str, object]:
    matrix = _confusion(prediction, target, valid, classes)
    true_positive = matrix.diag()
    target_count = matrix.sum(dim=1)
    prediction_count = matrix.sum(dim=0)
    union = target_count + prediction_count - true_positive
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    recall = torch.where(target_count > 0, true_positive / target_count, torch.nan)
    present = target_count > 0
    macro_iou = float(torch.nanmean(iou).item()) if bool(present.any()) else 0.0
    if empty_class is None:
        foreground_ids = torch.arange(classes, device=matrix.device)
    else:
        foreground_ids = torch.tensor(
            [index for index in range(classes) if index != empty_class], device=matrix.device
        )
    fg_truth = target_count[foreground_ids].sum()
    fg_prediction = prediction_count[foreground_ids].sum()
    fg_tp = true_positive[foreground_ids].sum()
    fg_precision = fg_tp / fg_prediction if fg_prediction > 0 else torch.tensor(0.0)
    fg_recall = fg_tp / fg_truth if fg_truth > 0 else torch.tensor(0.0)
    denominator = fg_precision + fg_recall
    foreground_f1 = 2 * fg_precision * fg_recall / denominator if denominator > 0 else torch.tensor(0.0)
    foreground_present = target_count[foreground_ids] > 0
    foreground_macro_iou = (
        torch.nanmean(iou[foreground_ids][foreground_present])
        if bool(foreground_present.any())
        else torch.tensor(0.0)
    )
    rare_mask = target_count > 0
    if empty_class is not None:
        rare_mask[empty_class] = False
    rare_candidates = torch.where(rare_mask)[0]
    if rare_candidates.numel():
        rare_order = rare_candidates[torch.argsort(target_count[rare_candidates], stable=True)]
        rare_count = max(1, int((rare_order.numel() + 3) // 4))
        rare_ids = rare_order[:rare_count]
        rare_recall = torch.nanmean(recall[rare_ids])
    else:
        rare_ids = torch.empty((0,), dtype=torch.long, device=matrix.device)
        rare_recall = torch.tensor(0.0)
    return {
        "macro_iou": macro_iou,
        "foreground_macro_iou": float(foreground_macro_iou.item()),
        "foreground_f1": float(foreground_f1.item()),
        "rare_class_recall": float(rare_recall.item()),
        "per_class_iou": [None if torch.isnan(value) else float(value.item()) for value in iou],
        "per_class_recall": [
            None if torch.isnan(value) else float(value.item()) for value in recall
        ],
        "target_count": [int(value.item()) for value in target_count],
        "prediction_count": [int(value.item()) for value in prediction_count],
        "rare_class_ids": [int(value.item()) for value in rare_ids],
        "confusion": [[int(value.item()) for value in row] for row in matrix],
    }


def decoration_metrics(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    valid_cells: torch.Tensor,
) -> dict[str, object]:
    heads: dict[str, object] = {}
    selection_terms: list[float] = []
    for name in HEAD_NAMES:
        result = head_metrics(
            predictions[name],
            targets[name],
            valid_cells,
            classes=HEAD_CLASS_COUNTS[name],
            empty_class=None if name == "variant" else 0,
        )
        heads[name] = result
        # Empty accuracy is intentionally absent from the selection score.
        selection_terms.extend(
            (
                float(result["foreground_macro_iou"]),
                float(result["foreground_f1"]),
                float(result["rare_class_recall"]),
            )
        )
    return {
        "heads": heads,
        "selection_score": sum(selection_terms) / len(selection_terms),
        "selection_components": "foreground_macro_iou+foreground_f1+rare_class_recall per head",
        "empty_accuracy_in_selection": False,
    }
