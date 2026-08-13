from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..morphology.constants import (
    CANVAS_SIZE,
    EMISSION_LEVEL_NAMES,
    FAMILIES,
    GUIDE_CHANNEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SUBTYPE_NAMES,
)
from ..multifield_data import SCAFFOLD_GUIDE_INDICES, legal_tuple_fingerprint
from .checkpoint import LoadedMultiFieldCheckpoint
from .conditions import ConditionRecord
from .rendering import aligned_fields_hash


VALIDATION_FORMAT = "nullvector-multifield-generation-validation-v2"
TEMPLATE_FORMAT = "nullvector-condition-template-bank-v2-diagnostic"
CALIBRATION_FORMAT = "nullvector-multifield-reference-calibration-v1"
STRUCTURAL_OWNER_MAX = PART_OWNER_NAMES.index("core")
STRUCTURAL_SAFETY_MARGIN = 3
VISIBLE_SAFETY_MARGIN = 2
MINIMUM_OCCUPANCY = 0.02
MAXIMUM_OCCUPANCY = 0.60
MINIMUM_SCAFFOLD_COVERAGE = 0.45
ESSENTIAL_OWNER_NAMES = ("core",)
HARD_GATE_NAMES = (
    "categorical_domains",
    "guide_contract",
    "target_contract",
    "condition_contract",
    "legal_table_contract",
    "nonempty",
    "occupancy",
    "visible_connected",
    "structural_margin",
    "visible_margin",
    "legal_tuples",
    "scaffold_coverage",
    "essential_owners",
)


def _component_sizes(mask: np.ndarray) -> list[int]:
    active = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(active)
    sizes: list[int] = []
    height, width = active.shape
    for y in range(height):
        for x in range(width):
            if not active[y, x] or seen[y, x]:
                continue
            size = 0
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                py, px = stack.pop()
                size += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = py + dy, px + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and active[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            sizes.append(size)
    return sorted(sizes, reverse=True)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius cannot be negative")
    result = np.asarray(mask, dtype=bool)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant")
        result = np.logical_or.reduce(
            [
                padded[y : y + result.shape[0], x : x + result.shape[1]]
                for y in range(3)
                for x in range(3)
            ]
        )
    return result


def _tuple_validity(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    legal_tuples: np.ndarray,
) -> tuple[float, np.ndarray]:
    material_count = len(MATERIAL_NAMES)
    emission_count = len(EMISSION_LEVEL_NAMES)
    codes = (
        part.astype(np.int64) * material_count * emission_count
        + material.astype(np.int64) * emission_count
        + emission.astype(np.int64)
    )
    legal_codes = (
        legal_tuples[:, 0].astype(np.int64) * material_count * emission_count
        + legal_tuples[:, 1].astype(np.int64) * emission_count
        + legal_tuples[:, 2].astype(np.int64)
    )
    lookup = np.zeros(
        len(PART_OWNER_NAMES) * material_count * emission_count, dtype=bool
    )
    lookup[legal_codes] = True
    valid = lookup[codes]
    return float(valid.mean()), valid


def _field_features_batch(
    part: np.ndarray, material: np.ndarray, emission: np.ndarray
) -> np.ndarray:
    part = np.asarray(part, dtype=np.uint8)
    material = np.asarray(material, dtype=np.uint8)
    emission = np.asarray(emission, dtype=np.uint8)
    if part.ndim == 2:
        part, material, emission = part[None], material[None], emission[None]
    if part.shape != material.shape or part.shape != emission.shape:
        raise ValueError("Aligned fields must have identical shapes")
    if part.ndim != 3 or part.shape[1:] != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError("Aligned fields must have shape [batch, 48, 48]")
    count = len(part)
    visible = part != 0
    visible_float = visible.astype(np.float32)
    denominator = visible_float.sum(axis=(1, 2)).clip(min=1.0)
    yy, xx = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
    x_mean = (visible_float * xx).sum(axis=(1, 2)) / denominator / (CANVAS_SIZE - 1)
    y_mean = (visible_float * yy).sum(axis=(1, 2)) / denominator / (CANVAS_SIZE - 1)
    x_var = (
        visible_float * (xx[None] / (CANVAS_SIZE - 1) - x_mean[:, None, None]) ** 2
    ).sum(axis=(1, 2)) / denominator
    y_var = (
        visible_float * (yy[None] / (CANVAS_SIZE - 1) - y_mean[:, None, None]) ** 2
    ).sum(axis=(1, 2)) / denominator
    pooled = visible_float.reshape(count, 6, 8, 6, 8).mean(axis=(2, 4))
    row_projection = visible_float.reshape(count, 6, 8, CANVAS_SIZE).mean(axis=(2, 3))
    column_projection = visible_float.reshape(count, CANVAS_SIZE, 6, 8).mean(axis=(1, 3))

    histograms = []
    for values, categories in (
        (part, len(PART_OWNER_NAMES)),
        (material, len(MATERIAL_NAMES)),
        (emission, len(EMISSION_LEVEL_NAMES)),
    ):
        histogram = np.stack(
            [(values == category).mean(axis=(1, 2)) for category in range(categories)],
            axis=1,
        )
        histograms.append(histogram)
    scalar = np.stack(
        (
            visible_float.mean(axis=(1, 2)),
            x_mean,
            y_mean,
            np.sqrt(x_var),
            np.sqrt(y_var),
        ),
        axis=1,
    )
    return np.concatenate(
        (
            pooled.reshape(count, -1),
            row_projection,
            column_projection,
            scalar,
            *histograms,
        ),
        axis=1,
    ).astype(np.float32)


@dataclass(slots=True)
class ConditionTemplateBank:
    mean: np.ndarray
    scale: np.ndarray
    family_centroids: np.ndarray
    family_thresholds: np.ndarray
    subtype_centroids: np.ndarray
    subtype_thresholds: np.ndarray
    role_centroids: np.ndarray
    role_thresholds: np.ndarray
    fingerprint: str

    @classmethod
    def build(
        cls, bundle: LoadedMultiFieldCheckpoint, *, quantile: float = 0.995
    ) -> "ConditionTemplateBank":
        if not 0.9 <= quantile < 1.0:
            raise ValueError("template quantile must be in [0.9, 1.0)")
        indices = bundle.training_indices
        features = _field_features_batch(
            bundle.corpus.part_owner[indices],
            bundle.corpus.material[indices],
            bundle.corpus.emission_level[indices],
        )
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1.0e-5] = 1.0
        normalized = (features - mean) / scale
        families = bundle.corpus.morphologies[indices].astype(np.int64)
        subtypes = bundle.corpus.subtypes[indices].astype(np.int64)
        roles = bundle.corpus.roles[indices].astype(np.int64)

        family_centroids = np.stack(
            [normalized[families == family].mean(axis=0) for family in range(len(FAMILIES))]
        )
        subtype_centroids = np.stack(
            [normalized[subtypes == subtype].mean(axis=0) for subtype in range(len(SUBTYPE_NAMES))]
        )
        role_centroids = np.stack(
            [
                np.stack(
                    [
                        normalized[(families == family) & (roles == role)].mean(axis=0)
                        for role in range(len(ROLE_NAMES))
                    ]
                )
                for family in range(len(FAMILIES))
            ]
        )

        def thresholds(labels: np.ndarray, centroids: np.ndarray, count: int) -> np.ndarray:
            output = np.empty(count, dtype=np.float32)
            for label in range(count):
                distances = np.linalg.norm(
                    normalized[labels == label] - centroids[label], axis=1
                )
                output[label] = max(float(np.quantile(distances, quantile)), 1.0e-5)
            return output

        family_thresholds = thresholds(families, family_centroids, len(FAMILIES))
        subtype_thresholds = thresholds(subtypes, subtype_centroids, len(SUBTYPE_NAMES))
        role_thresholds = np.empty((len(FAMILIES), len(ROLE_NAMES)), dtype=np.float32)
        for family in range(len(FAMILIES)):
            for role in range(len(ROLE_NAMES)):
                selected = (families == family) & (roles == role)
                distances = np.linalg.norm(
                    normalized[selected] - role_centroids[family, role], axis=1
                )
                role_thresholds[family, role] = max(
                    float(np.quantile(distances, quantile)), 1.0e-5
                )
        digest = hashlib.sha256()
        digest.update(b"nullvector-condition-template-bank-v2-diagnostic\0")
        digest.update(bundle.corpus.file_sha256.encode("ascii"))
        digest.update(bundle.payload["split"]["fingerprint"].encode("ascii"))
        digest.update(np.asarray([quantile], dtype=np.float64).tobytes())
        for values in (
            mean,
            scale,
            family_centroids,
            family_thresholds,
            subtype_centroids,
            subtype_thresholds,
            role_centroids,
            role_thresholds,
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
        return cls(
            mean=mean,
            scale=scale,
            family_centroids=family_centroids,
            family_thresholds=family_thresholds,
            subtype_centroids=subtype_centroids,
            subtype_thresholds=subtype_thresholds,
            role_centroids=role_centroids,
            role_thresholds=role_thresholds,
            fingerprint=digest.hexdigest(),
        )

    def classify(
        self,
        part: np.ndarray,
        material: np.ndarray,
        emission: np.ndarray,
        record: ConditionRecord,
    ) -> dict[str, Any]:
        feature = (_field_features_batch(part, material, emission)[0] - self.mean) / self.scale
        family_distances = np.linalg.norm(self.family_centroids - feature, axis=1)
        family_prediction = int(np.argmin(family_distances))
        subtype_candidates = np.arange(record.morphology * 4, record.morphology * 4 + 4)
        subtype_distances = np.linalg.norm(
            self.subtype_centroids[subtype_candidates] - feature, axis=1
        )
        subtype_prediction = int(subtype_candidates[int(np.argmin(subtype_distances))])
        role_distances = np.linalg.norm(
            self.role_centroids[record.morphology] - feature, axis=1
        )
        role_prediction = int(np.argmin(role_distances))
        family_distance = float(family_distances[record.morphology])
        subtype_distance = float(
            np.linalg.norm(self.subtype_centroids[record.subtype] - feature)
        )
        role_distance = float(role_distances[record.role])
        family_in_distribution = family_distance <= float(
            self.family_thresholds[record.morphology]
        )
        subtype_in_distribution = subtype_distance <= float(
            self.subtype_thresholds[record.subtype]
        )
        role_in_distribution = role_distance <= float(
            self.role_thresholds[record.morphology, record.role]
        )
        return {
            "template_format": TEMPLATE_FORMAT,
            "template_fingerprint": self.fingerprint,
            "predicted_morphology_id": family_prediction,
            "predicted_subtype_id": subtype_prediction,
            "predicted_role_id": role_prediction,
            "morphology_match": family_prediction == record.morphology,
            "subtype_match": subtype_prediction == record.subtype,
            "role_match": role_prediction == record.role,
            "morphology_distance": family_distance,
            "morphology_distance_threshold": float(
                self.family_thresholds[record.morphology]
            ),
            "subtype_distance": subtype_distance,
            "subtype_distance_threshold": float(
                self.subtype_thresholds[record.subtype]
            ),
            "role_distance": role_distance,
            "role_distance_threshold": float(
                self.role_thresholds[record.morphology, record.role]
            ),
            "morphology_in_distribution": family_in_distribution,
            "subtype_in_distribution": subtype_in_distribution,
            "role_in_distribution": role_in_distribution,
            "exact_condition_match": bool(
                family_prediction == record.morphology
                and subtype_prediction == record.subtype
                and role_prediction == record.role
            ),
            "all_axes_in_distribution": bool(
                family_in_distribution
                and subtype_in_distribution
                and role_in_distribution
            ),
        }


def validate_generated_fields(
    part: np.ndarray,
    material: np.ndarray,
    emission: np.ndarray,
    *,
    guide: np.ndarray,
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    record: ConditionRecord,
    legal_tuples: np.ndarray,
    templates: ConditionTemplateBank,
) -> dict[str, Any]:
    part = np.asarray(part)
    material = np.asarray(material)
    emission = np.asarray(emission)
    guide = np.asarray(guide)
    legal_tuples = np.asarray(legal_tuples)
    field_errors: list[str] = []
    guide_errors: list[str] = []
    target_errors: list[str] = []
    condition_errors: list[str] = []
    legal_table_errors: list[str] = []
    expected = (CANVAS_SIZE, CANVAS_SIZE)
    fields = (
        (part, len(PART_OWNER_NAMES), "part"),
        (material, len(MATERIAL_NAMES), "material"),
        (emission, len(EMISSION_LEVEL_NAMES), "emission"),
    )
    for values, count, name in fields:
        if values.shape != expected:
            field_errors.append(f"{name} shape is {values.shape}, expected {expected}")
        if values.dtype != np.uint8:
            field_errors.append(f"{name} dtype is {values.dtype}, expected uint8")
        if values.size and (int(values.min()) < 0 or int(values.max()) >= count):
            field_errors.append(f"{name} contains an out-of-vocabulary value")
    expected_guide = (len(GUIDE_CHANNEL_NAMES), CANVAS_SIZE, CANVAS_SIZE)
    if guide.shape != expected_guide:
        guide_errors.append(f"guide shape is {guide.shape}, expected {expected_guide}")
    if guide.dtype != np.float32:
        guide_errors.append(f"guide dtype is {guide.dtype}, expected float32")
    if guide.size and (
        not np.isfinite(guide).all()
        or float(guide.min()) < 0.0
        or float(guide.max()) > 1.0
    ):
        guide_errors.append("guide must contain finite values in [0, 1]")
    for values, count, name in (
        (np.asarray(target[0]), len(PART_OWNER_NAMES), "target_part"),
        (np.asarray(target[1]), len(MATERIAL_NAMES), "target_material"),
        (np.asarray(target[2]), len(EMISSION_LEVEL_NAMES), "target_emission"),
    ):
        if values.shape != expected or values.dtype != np.uint8:
            target_errors.append(f"{name} must be uint8 {expected}")
        elif int(values.min()) < 0 or int(values.max()) >= count:
            target_errors.append(f"{name} contains an out-of-vocabulary value")
    if not (
        0 <= int(record.morphology) < len(FAMILIES)
        and 0 <= int(record.subtype) < len(SUBTYPE_NAMES)
        and 0 <= int(record.role) < len(ROLE_NAMES)
        and int(record.subtype) // 4 == int(record.morphology)
    ):
        condition_errors.append(
            "condition ids must be in range and subtype must belong to morphology"
        )
    if legal_tuples.ndim != 2 or legal_tuples.shape[1:] != (3,):
        legal_table_errors.append("legal_tuples must have shape [count, 3]")
    elif legal_tuples.dtype != np.uint8 or len(legal_tuples) == 0:
        legal_table_errors.append("legal_tuples must be a non-empty uint8 table")
    else:
        for column, count, name in (
            (0, len(PART_OWNER_NAMES), "part"),
            (1, len(MATERIAL_NAMES), "material"),
            (2, len(EMISSION_LEVEL_NAMES), "emission"),
        ):
            values = legal_tuples[:, column]
            if int(values.min()) < 0 or int(values.max()) >= count:
                legal_table_errors.append(
                    f"legal_tuples {name} column contains an out-of-vocabulary value"
                )
        if len(np.unique(legal_tuples, axis=0)) != len(legal_tuples):
            legal_table_errors.append("legal_tuples contains duplicate rows")
    errors = (
        field_errors
        + guide_errors
        + target_errors
        + condition_errors
        + legal_table_errors
    )
    if errors:
        hard_gates = {name: False for name in HARD_GATE_NAMES}
        hard_gates.update(
            categorical_domains=not field_errors,
            guide_contract=not guide_errors,
            target_contract=not target_errors,
            condition_contract=not condition_errors,
            legal_table_contract=not legal_table_errors,
        )
        return {
            "format": VALIDATION_FORMAT,
            "sample_id": record.sample_id,
            "hard_gates": hard_gates,
            "hard_valid": False,
            "condition_exact_match": False,
            "condition_in_distribution": False,
            "accepted": False,
            "errors": errors,
        }

    visible = part != 0
    structural = (part > 0) & (part <= STRUCTURAL_OWNER_MAX)
    visible_components = _component_sizes(visible)
    structural_components = _component_sizes(structural)
    visible_pixels = int(visible.sum())
    structural_pixels = int(structural.sum())
    structural_unsafe = np.zeros_like(visible)
    structural_unsafe[:STRUCTURAL_SAFETY_MARGIN] = True
    structural_unsafe[-STRUCTURAL_SAFETY_MARGIN:] = True
    structural_unsafe[:, :STRUCTURAL_SAFETY_MARGIN] = True
    structural_unsafe[:, -STRUCTURAL_SAFETY_MARGIN:] = True
    visible_unsafe = np.zeros_like(visible)
    visible_unsafe[:VISIBLE_SAFETY_MARGIN] = True
    visible_unsafe[-VISIBLE_SAFETY_MARGIN:] = True
    visible_unsafe[:, :VISIBLE_SAFETY_MARGIN] = True
    visible_unsafe[:, -VISIBLE_SAFETY_MARGIN:] = True
    unsafe_structural_pixels = int((structural & structural_unsafe).sum())
    unsafe_visible_pixels = int((visible & visible_unsafe).sum())
    tuple_validity, valid_tuple_mask = _tuple_validity(
        part, material, emission, legal_tuples
    )
    scaffold = np.logical_or.reduce(
        guide[list(SCAFFOLD_GUIDE_INDICES)] > 0.0
    )
    scaffold_pixels = int(scaffold.sum())
    scaffold_coverage = float(
        (scaffold & _dilate(visible, 2)).sum() / max(scaffold_pixels, 1)
    )
    target_part, target_material, target_emission = target
    target_visible = target_part != 0
    intersection = int((visible & target_visible).sum())
    union = int((visible | target_visible).sum())
    source_metrics = {
        "silhouette_iou": intersection / max(union, 1),
        "part_accuracy": float((part == target_part).mean()),
        "material_accuracy": float((material == target_material).mean()),
        "emission_accuracy": float((emission == target_emission).mean()),
    }
    condition = templates.classify(part, material, emission, record)
    diagnostic_owner_ids = {
        name: PART_OWNER_NAMES.index(name) for name in ("body", "head", "core")
    }
    owner_presence = {
        name: bool((part == owner_id).any())
        for name, owner_id in diagnostic_owner_ids.items()
    }
    occupancy = visible_pixels / visible.size
    largest_visible_fraction = (
        visible_components[0] / max(visible_pixels, 1) if visible_components else 0.0
    )
    largest_structural_fraction = (
        structural_components[0] / max(structural_pixels, 1)
        if structural_components
        else 0.0
    )
    # Part ownership is a single categorical map. Joint, terminal, ornament,
    # and aura categories legitimately overwrite otherwise structural owner
    # labels. Connectivity therefore belongs to the complete visible union;
    # the structural-owner subset remains useful only for the stronger margin.
    hard_gates = {
        "categorical_domains": True,
        "guide_contract": True,
        "target_contract": True,
        "condition_contract": True,
        "legal_table_contract": True,
        "nonempty": visible_pixels > 0,
        "occupancy": MINIMUM_OCCUPANCY <= occupancy <= MAXIMUM_OCCUPANCY,
        "visible_connected": len(visible_components) == 1,
        "structural_margin": unsafe_structural_pixels == 0,
        "visible_margin": unsafe_visible_pixels == 0,
        "legal_tuples": tuple_validity == 1.0,
        "scaffold_coverage": scaffold_coverage >= MINIMUM_SCAFFOLD_COVERAGE,
        "essential_owners": all(
            owner_presence[name] for name in ESSENTIAL_OWNER_NAMES
        ),
    }
    hard_valid = bool(all(hard_gates.values()))
    return {
        "format": VALIDATION_FORMAT,
        "sample_id": record.sample_id,
        "raw_fields_sha256": aligned_fields_hash(part, material, emission),
        "errors": [],
        "topology": {
            "visible_pixels": visible_pixels,
            "structural_pixels": structural_pixels,
            "occupancy_fraction": occupancy,
            "visible_component_count": len(visible_components),
            "structural_component_count": len(structural_components),
            "largest_visible_fraction": largest_visible_fraction,
            "largest_structural_fraction": largest_structural_fraction,
            "scaffold_pixels": scaffold_pixels,
            "scaffold_coverage_radius_2": scaffold_coverage,
            "owner_presence": owner_presence,
            "essential_owner_names": list(ESSENTIAL_OWNER_NAMES),
        },
        "margins": {
            "structural_safety_margin": STRUCTURAL_SAFETY_MARGIN,
            "visible_safety_margin": VISIBLE_SAFETY_MARGIN,
            "unsafe_structural_pixels": unsafe_structural_pixels,
            "unsafe_visible_pixels": unsafe_visible_pixels,
            "safe": unsafe_structural_pixels == 0 and unsafe_visible_pixels == 0,
        },
        "tuples": {
            "valid_fraction": tuple_validity,
            "invalid_pixels": int((~valid_tuple_mask).sum()),
            "legal_tuple_count": int(len(legal_tuples)),
        },
        "source_similarity": source_metrics,
        "condition_adherence": condition,
        "hard_gates": hard_gates,
        "hard_valid": hard_valid,
        "condition_exact_match": bool(condition["exact_condition_match"]),
        "condition_in_distribution": bool(condition["all_axes_in_distribution"]),
        "accepted": hard_valid,
    }


def calibrate_reference_fields(
    bundle: LoadedMultiFieldCheckpoint,
    *,
    templates: ConditionTemplateBank | None = None,
    indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure the hard contract against untouched authoritative references.

    The reference population defaults to the checkpoint's complete held-out
    split. No neural output or target-derived guide channel is introduced: the
    exact corpus fields are validated against the checkpoint's sanitized guide
    policy, train-only legal table, and recorded condition labels.
    """

    selected = np.asarray(
        bundle.validation_indices if indices is None else indices,
        dtype=np.int64,
    )
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("Calibration indices must be a non-empty vector")
    if int(selected.min()) < 0 or int(selected.max()) >= bundle.corpus.count:
        raise IndexError("Calibration indices contain an out-of-range row")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Calibration indices must be unique")
    validation_lookup = set(map(int, bundle.validation_indices))
    if any(int(index) not in validation_lookup for index in selected):
        raise ValueError("Calibration may use only held-out validation rows")
    templates = templates or ConditionTemplateBank.build(bundle)

    from ..multifield_data import MorphologyCorpusDataset

    dataset = MorphologyCorpusDataset(
        bundle.corpus,
        selected,
        guide_policy=bundle.guide_policy,
    )
    hard_valid = 0
    exact_axis = {"morphology": 0, "subtype": 0, "role": 0, "all": 0}
    in_distribution = {"morphology": 0, "subtype": 0, "role": 0, "all": 0}
    failure_counts: dict[str, int] = {}
    gate_passes: dict[str, int] = {}
    anchor_counts = {name: 0 for name in ("body", "head", "core")}
    visible_component_counts: dict[str, int] = {}
    minimum_scaffold_coverage = 1.0
    occupancy_minimum = 1.0
    occupancy_maximum = 0.0
    for ordinal, source_index in enumerate(selected):
        source_index = int(source_index)
        sample = dataset[ordinal]
        record = ConditionRecord(
            ordinal=ordinal,
            grid_mode="fixed",
            source_index=source_index,
            variation=0,
            sample_seed=0,
            morphology=int(bundle.corpus.morphologies[source_index]),
            subtype=int(bundle.corpus.subtypes[source_index]),
            role=int(bundle.corpus.roles[source_index]),
        )
        target = (
            bundle.corpus.part_owner[source_index],
            bundle.corpus.material[source_index],
            bundle.corpus.emission_level[source_index],
        )
        report = validate_generated_fields(
            *target,
            guide=sample["guide"].numpy(),
            target=target,
            record=record,
            legal_tuples=bundle.legal_tuples,
            templates=templates,
        )
        hard_valid += int(report["hard_valid"])
        topology = report["topology"]
        margins = report["margins"]
        tuples = report["tuples"]
        components = str(topology["visible_component_count"])
        visible_component_counts[components] = (
            visible_component_counts.get(components, 0) + 1
        )
        for name, present in topology["owner_presence"].items():
            anchor_counts[name] += int(present)
        coverage = float(topology["scaffold_coverage_radius_2"])
        occupancy = float(topology["occupancy_fraction"])
        minimum_scaffold_coverage = min(minimum_scaffold_coverage, coverage)
        occupancy_minimum = min(occupancy_minimum, occupancy)
        occupancy_maximum = max(occupancy_maximum, occupancy)
        checks = report["hard_gates"]
        for name, passed in checks.items():
            gate_passes[name] = gate_passes.get(name, 0) + int(passed)
            if not passed:
                failure_counts[name] = failure_counts.get(name, 0) + 1
        condition = report["condition_adherence"]
        for axis in ("morphology", "subtype", "role"):
            exact_axis[axis] += int(condition[f"{axis}_match"])
            in_distribution[axis] += int(condition[f"{axis}_in_distribution"])
        exact_axis["all"] += int(condition["exact_condition_match"])
        in_distribution["all"] += int(condition["all_axes_in_distribution"])

    count = len(selected)
    digest = hashlib.sha256()
    digest.update(b"nullvector-multifield-reference-calibration-v1\0")
    digest.update(bundle.corpus.file_sha256.encode("ascii"))
    digest.update(bundle.payload["split"]["fingerprint"].encode("ascii"))
    digest.update(selected.tobytes())
    return {
        "format": CALIBRATION_FORMAT,
        "validation_format": VALIDATION_FORMAT,
        "template_format": TEMPLATE_FORMAT,
        "population": "held-out-authoritative-corpus-fields",
        "population_fingerprint": digest.hexdigest(),
        "samples": count,
        "hard_valid": hard_valid,
        "hard_valid_rate": hard_valid / count,
        "hard_gate_policy": {
            "categorical_domains_exact": True,
            "guide_contract_exact": True,
            "target_contract_exact": True,
            "condition_contract_exact": True,
            "legal_table_contract_exact": True,
            "legal_tuple_fingerprint": legal_tuple_fingerprint(
                np.asarray(bundle.legal_tuples, dtype=np.uint8)
            ),
            "visible_component_count": 1,
            "structural_safety_margin": STRUCTURAL_SAFETY_MARGIN,
            "visible_safety_margin": VISIBLE_SAFETY_MARGIN,
            "occupancy_range": [MINIMUM_OCCUPANCY, MAXIMUM_OCCUPANCY],
            "minimum_scaffold_coverage_radius_2": MINIMUM_SCAFFOLD_COVERAGE,
            "essential_owner_names": list(ESSENTIAL_OWNER_NAMES),
            "condition_templates_are_diagnostic_only": True,
        },
        "hard_gate_passes": gate_passes,
        "hard_gate_failures": dict(sorted(failure_counts.items())),
        "visible_component_counts": dict(sorted(visible_component_counts.items())),
        "owner_presence": {
            name: {
                "present": anchor_counts[name],
                "rate": anchor_counts[name] / count,
                "hard_required": name in ESSENTIAL_OWNER_NAMES,
            }
            for name in anchor_counts
        },
        "minimum_scaffold_coverage": minimum_scaffold_coverage,
        "occupancy_range": [occupancy_minimum, occupancy_maximum],
        "diagnostic_condition_adherence": {
            "exact_match": {
                axis: {"matched": value, "rate": value / count}
                for axis, value in exact_axis.items()
            },
            "in_distribution": {
                axis: {"passed": value, "rate": value / count}
                for axis, value in in_distribution.items()
            },
        },
    }


def validate_postprocess_delta(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
    processed: tuple[np.ndarray, np.ndarray, np.ndarray],
    report: Mapping[str, Any],
    legal_tuples: np.ndarray,
) -> dict[str, Any]:
    raw_part, raw_material, raw_emission = raw
    part, material, emission = processed
    changed = (
        (part != raw_part)
        | (material != raw_material)
        | (emission != raw_emission)
    )
    removed = changed & (part == 0) & (material == 0) & (emission == 0)
    additions = (raw_part == 0) & (part != 0)
    modified_survivors = changed & (part != 0)
    tuple_validity, _ = _tuple_validity(part, material, emission, legal_tuples)
    maximum = float(report["max_delta_fraction"])
    observed = float(changed.mean())
    errors = []
    if report.get("raw_fields_sha256") != aligned_fields_hash(*raw):
        errors.append("recorded raw field hash is incorrect")
    if report.get("processed_fields_sha256") != aligned_fields_hash(*processed):
        errors.append("recorded processed field hash is incorrect")
    if int(report.get("changed_pixels", -1)) != int(changed.sum()):
        errors.append("recorded changed pixel count is incorrect")
    if observed > maximum + 1.0e-12:
        errors.append("postprocess exceeded its delta fraction")
    if bool(additions.any()):
        errors.append("postprocess added foreground")
    if bool(modified_survivors.any()):
        errors.append("postprocess rewrote surviving tuples")
    if not np.array_equal(changed, removed):
        errors.append("postprocess made a change other than tuple removal")
    if tuple_validity != 1.0:
        errors.append("processed output contains an illegal tuple")
    return {
        "format": str(report.get("format")),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": observed,
        "maximum_changed_fraction": maximum,
        "tuple_validity": tuple_validity,
        "valid": not errors,
        "errors": errors,
    }


def diversity_report(
    fields: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    records: Sequence[ConditionRecord],
    *,
    max_global_pairs: int = 100_000,
) -> dict[str, Any]:
    if len(fields) != len(records) or not fields:
        raise ValueError("Diversity needs matching non-empty fields and records")
    if max_global_pairs <= 0:
        raise ValueError("max_global_pairs must be positive")
    hashes = [aligned_fields_hash(*item) for item in fields]
    silhouette_ious: list[float] = []
    categorical_hamming: list[float] = []
    same_condition_hamming: list[float] = []
    total_pairs = len(fields) * (len(fields) - 1) // 2
    if total_pairs <= max_global_pairs:
        pairs = [
            (first, second)
            for first in range(len(fields))
            for second in range(first + 1, len(fields))
        ]
        pair_policy = "all"
    else:
        generator = np.random.default_rng(0xD1A3E251)
        selected: set[tuple[int, int]] = set()
        while len(selected) < max_global_pairs:
            first = int(generator.integers(0, len(fields)))
            second = int(generator.integers(0, len(fields) - 1))
            if second >= first:
                second += 1
            selected.add((min(first, second), max(first, second)))
        pairs = sorted(selected)
        pair_policy = "deterministic-uniform-without-replacement"

    for first, second in pairs:
        first_part, first_material, first_emission = fields[first]
        first_visible = first_part != 0
        second_part, second_material, second_emission = fields[second]
        second_visible = second_part != 0
        intersection = int((first_visible & second_visible).sum())
        union = int((first_visible | second_visible).sum())
        silhouette_ious.append(intersection / max(union, 1))
        categorical_hamming.append(
            float(
                (
                    (first_part != second_part)
                    | (first_material != second_material)
                    | (first_emission != second_emission)
                ).mean()
            )
        )

    condition_groups: dict[tuple[int, int, int], list[int]] = {}
    for index, record in enumerate(records):
        condition_groups.setdefault(
            (record.morphology, record.subtype, record.role), []
        ).append(index)
    for group in condition_groups.values():
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                same_condition_hamming.append(
                    float(
                        (
                            (fields[first][0] != fields[second][0])
                            | (fields[first][1] != fields[second][1])
                            | (fields[first][2] != fields[second][2])
                        ).mean()
                    )
                )

    def statistics(values: Sequence[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "mean": 0.0, "p05": 0.0, "p95": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(len(array)),
            "mean": float(array.mean()),
            "p05": float(np.quantile(array, 0.05)),
            "p95": float(np.quantile(array, 0.95)),
        }

    return {
        "samples": len(fields),
        "exact_unique": len(set(hashes)),
        "exact_unique_fraction": len(set(hashes)) / len(hashes),
        "total_possible_pairs": total_pairs,
        "evaluated_global_pairs": len(pairs),
        "global_pair_policy": pair_policy,
        "pairwise_silhouette_iou": statistics(silhouette_ious),
        "pairwise_categorical_hamming": statistics(categorical_hamming),
        "same_condition_categorical_hamming": statistics(same_condition_hamming),
    }


def acceptance_breakdown(
    records: Sequence[ConditionRecord], reports: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(records) != len(reports):
        raise ValueError("Acceptance records and reports must have equal length")

    def group(key_values: Iterable[tuple[str, bool]]) -> dict[str, Any]:
        buckets: dict[str, list[bool]] = {}
        for key, accepted in key_values:
            buckets.setdefault(key, []).append(bool(accepted))
        return {
            key: {
                "accepted": int(sum(values)),
                "samples": len(values),
                "acceptance_rate": sum(values) / len(values),
            }
            for key, values in sorted(buckets.items())
        }

    return {
        "overall": {
            "accepted": int(sum(bool(report.get("accepted")) for report in reports)),
            "samples": len(reports),
            "acceptance_rate": (
                sum(bool(report.get("accepted")) for report in reports) / len(reports)
                if reports
                else 0.0
            ),
        },
        "per_family": group(
            (FAMILIES[record.morphology], bool(report.get("accepted")))
            for record, report in zip(records, reports)
        ),
        "per_role": group(
            (ROLE_NAMES[record.role], bool(report.get("accepted")))
            for record, report in zip(records, reports)
        ),
        "per_subtype": group(
            (SUBTYPE_NAMES[record.subtype], bool(report.get("accepted")))
            for record, report in zip(records, reports)
        ),
    }
