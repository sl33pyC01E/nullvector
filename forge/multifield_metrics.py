from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from .multifield_diffusion import MultiFieldLogits, MultiFieldVocabulary


def _confusion_matrix(target: Tensor, prediction: Tensor, count: int) -> np.ndarray:
    encoded = target.reshape(-1).to(torch.int64) * count + prediction.reshape(-1)
    matrix = torch.bincount(encoded, minlength=count * count).reshape(count, count)
    return matrix.detach().cpu().numpy().astype(np.int64, copy=False)


def _macro_iou(matrix: np.ndarray, *, exclude_background: bool) -> float:
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    intersection = np.diag(matrix)
    union = true_count + predicted_count - intersection
    eligible = true_count > 0
    if exclude_background and len(eligible):
        eligible[0] = False
    if not eligible.any():
        return 0.0
    return float(np.mean(intersection[eligible] / np.maximum(union[eligible], 1)))


def _accuracy(matrix: np.ndarray) -> float:
    return float(np.trace(matrix) / max(int(matrix.sum()), 1))


@dataclass(slots=True)
class MultiFieldMetricAccumulator:
    vocabulary: MultiFieldVocabulary
    legal_tuples: np.ndarray
    matrices: dict[str, np.ndarray] = field(init=False)
    silhouette_intersection: int = 0
    silhouette_union: int = 0
    foreground_pixels: int = 0
    foreground_correct: dict[str, int] = field(
        default_factory=lambda: {"part": 0, "material": 0, "emission": 0}
    )
    legal_pixels: int = 0
    legal_foreground_pixels: int = 0
    total_pixels: int = 0
    condition_examples: int = 0
    condition_preferred: int = 0
    condition_margin_sum: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.legal_tuples, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
            raise ValueError("legal_tuples must have shape [count, 3].")
        limits = (
            self.vocabulary.part_count,
            self.vocabulary.material_count,
            self.vocabulary.emission_count,
        )
        for column, limit in enumerate(limits):
            if int(values[:, column].min()) < 0 or int(values[:, column].max()) >= limit:
                raise ValueError("legal_tuples contains an out-of-vocabulary value.")
        self.legal_tuples = values
        self.matrices = {
            "part": np.zeros(
                (self.vocabulary.part_count, self.vocabulary.part_count),
                dtype=np.int64,
            ),
            "material": np.zeros(
                (self.vocabulary.material_count, self.vocabulary.material_count),
                dtype=np.int64,
            ),
            "emission": np.zeros(
                (self.vocabulary.emission_count, self.vocabulary.emission_count),
                dtype=np.int64,
            ),
        }

    def update(
        self,
        prediction: tuple[Tensor, Tensor, Tensor],
        target: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        predicted_part, predicted_material, predicted_emission = prediction
        target_part, target_material, target_emission = target
        tensors = (
            predicted_part,
            predicted_material,
            predicted_emission,
            target_part,
            target_material,
            target_emission,
        )
        if any(values.shape != target_part.shape for values in tensors):
            raise ValueError("All predicted and target fields must have identical shape.")

        fields = (
            ("part", predicted_part, target_part, self.vocabulary.part_count),
            (
                "material",
                predicted_material,
                target_material,
                self.vocabulary.material_count,
            ),
            (
                "emission",
                predicted_emission,
                target_emission,
                self.vocabulary.emission_count,
            ),
        )
        foreground = target_part != 0
        foreground_count = int(foreground.sum().item())
        self.foreground_pixels += foreground_count
        for name, predicted, expected, count in fields:
            self.matrices[name] += _confusion_matrix(expected, predicted, count)
            self.foreground_correct[name] += int(
                ((predicted == expected) & foreground).sum().item()
            )

        predicted_silhouette = predicted_part != 0
        target_silhouette = foreground
        self.silhouette_intersection += int(
            (predicted_silhouette & target_silhouette).sum().item()
        )
        self.silhouette_union += int(
            (predicted_silhouette | target_silhouette).sum().item()
        )

        material_count = self.vocabulary.material_count
        emission_count = self.vocabulary.emission_count
        codes = (
            predicted_part.to(torch.int64) * material_count * emission_count
            + predicted_material.to(torch.int64) * emission_count
            + predicted_emission.to(torch.int64)
        )
        legal_codes = torch.as_tensor(
            self.legal_tuples[:, 0] * material_count * emission_count
            + self.legal_tuples[:, 1] * emission_count
            + self.legal_tuples[:, 2],
            dtype=torch.int64,
            device=codes.device,
        )
        lookup = torch.zeros(
            self.vocabulary.part_count * material_count * emission_count,
            dtype=torch.bool,
            device=codes.device,
        )
        lookup[legal_codes] = True
        valid = lookup[codes]
        self.legal_pixels += int(valid.sum().item())
        self.legal_foreground_pixels += int((valid & foreground).sum().item())
        self.total_pixels += int(codes.numel())

    def update_logits(
        self,
        logits: MultiFieldLogits,
        target: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        self.update(
            (
                logits.part.argmax(dim=1),
                logits.material.argmax(dim=1),
                logits.emission.argmax(dim=1),
            ),
            target,
        )

    def update_condition_proxy(self, true_nll: Tensor, counterfactual_nll: Tensor) -> None:
        true_values = true_nll.detach().float().reshape(-1).cpu()
        wrong_values = counterfactual_nll.detach().float().reshape(-1).cpu()
        if true_values.shape != wrong_values.shape:
            raise ValueError("Condition NLL tensors must have identical shape.")
        margins = wrong_values - true_values
        self.condition_examples += int(len(margins))
        self.condition_preferred += int((margins > 0.0).sum().item())
        self.condition_margin_sum += float(margins.sum().item())

    def report(self, prefix: str) -> dict[str, float]:
        prefix = prefix.rstrip("_")
        foreground_denominator = max(self.foreground_pixels, 1)
        report = {
            f"{prefix}_silhouette_iou": self.silhouette_intersection
            / max(self.silhouette_union, 1),
            f"{prefix}_part_macro_iou": _macro_iou(
                self.matrices["part"], exclude_background=True
            ),
            f"{prefix}_material_macro_iou": _macro_iou(
                self.matrices["material"], exclude_background=True
            ),
            f"{prefix}_emission_macro_iou": _macro_iou(
                self.matrices["emission"], exclude_background=True
            ),
            f"{prefix}_part_accuracy": _accuracy(self.matrices["part"]),
            f"{prefix}_material_accuracy": _accuracy(self.matrices["material"]),
            f"{prefix}_emission_accuracy": _accuracy(self.matrices["emission"]),
            f"{prefix}_part_foreground_accuracy": self.foreground_correct["part"]
            / foreground_denominator,
            f"{prefix}_material_foreground_accuracy": self.foreground_correct[
                "material"
            ]
            / foreground_denominator,
            f"{prefix}_emission_foreground_accuracy": self.foreground_correct[
                "emission"
            ]
            / foreground_denominator,
            f"{prefix}_joint_tuple_validity": self.legal_pixels
            / max(self.total_pixels, 1),
            f"{prefix}_joint_tuple_foreground_validity": self.legal_foreground_pixels
            / foreground_denominator,
        }
        if self.condition_examples:
            report.update(
                {
                    f"{prefix}_condition_preference_rate": self.condition_preferred
                    / self.condition_examples,
                    f"{prefix}_condition_nll_margin": self.condition_margin_sum
                    / self.condition_examples,
                }
            )
        return {name: float(value) for name, value in report.items()}


def per_sample_unweighted_nll(
    logits: MultiFieldLogits,
    targets: tuple[Tensor, Tensor, Tensor],
    *,
    field_weights: tuple[float, float, float] = (1.0, 0.65, 0.45),
    pixel_mask: Tensor | None = None,
) -> Tensor:
    if len(field_weights) != 3 or sum(field_weights) <= 0.0:
        raise ValueError("field_weights must contain three values with a positive sum.")
    fields = (logits.part, logits.material, logits.emission)
    if pixel_mask is not None:
        if pixel_mask.shape != targets[0].shape:
            raise ValueError("pixel_mask must match the categorical target shape.")
        mask = pixel_mask.float()
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    losses = []
    for values, target in zip(fields, targets):
        per_pixel = torch.nn.functional.cross_entropy(values, target, reduction="none")
        if pixel_mask is None:
            losses.append(per_pixel.flatten(1).mean(dim=1))
        else:
            losses.append((per_pixel * mask).flatten(1).sum(dim=1) / denominator)
    return sum(weight * loss for weight, loss in zip(field_weights, losses)) / sum(
        field_weights
    )


def condition_preference_statistics(
    true_nll: Tensor, wrong_candidate_nll: Tensor
) -> tuple[Tensor, Tensor]:
    """Return true-condition preference and margin versus the best wrong candidate."""
    true_values = true_nll.reshape(-1)
    if wrong_candidate_nll.ndim != 2 or wrong_candidate_nll.shape[0] != len(
        true_values
    ):
        raise ValueError(
            "wrong_candidate_nll must have shape [batch, candidate_count]."
        )
    if wrong_candidate_nll.shape[1] == 0:
        raise ValueError("At least one wrong condition candidate is required.")
    margin = wrong_candidate_nll.min(dim=1).values - true_values
    return margin > 0.0, margin


def validation_selection_score(metrics: dict[str, float]) -> float:
    """Stable checkpoint score based only on every-epoch full-mask metrics."""
    weighted = (
        ("validation_silhouette_iou", 0.28),
        ("validation_part_macro_iou", 0.24),
        ("validation_material_foreground_accuracy", 0.12),
        ("validation_emission_foreground_accuracy", 0.10),
        ("validation_joint_tuple_validity", 0.12),
        ("validation_condition_preference_rate", 0.14),
    )
    missing = [name for name, _ in weighted if name not in metrics]
    if missing:
        raise KeyError(f"Selection metrics are missing: {missing}")
    return float(sum(metrics[name] * weight for name, weight in weighted))
