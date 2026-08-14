from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np

from .morphology.constants import FAMILIES, ROLE_NAMES, SUBTYPE_NAMES
from .multifield_style.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .multifield_style.source import PROJECT_ROOT, load_generation_bank
from .safety import require_disk_floor


FORMAT = "nullvector-neural-morphology-structural-diversity-v1"
DEFAULT_GENERATION_MANIFEST = (
    PROJECT_ROOT
    / "outputs/production_handoff_v2/final_best_stratified80_bank_attempt1/generation_manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/morphology_diversity_v1/morphology_diversity_report.json"
SCHEMA_PATH = PROJECT_ROOT / "shared/schema/morphology_diversity_report.schema.json"
SOURCE_FILES = (
    "forge/morphology_diversity.py",
    "shared/schema/morphology_diversity_report.schema.json",
)
BASIC_FEATURE_NAMES = (
    "solid_occupancy",
    "bbox_width",
    "bbox_height",
    "bbox_aspect",
    "bbox_fill",
    "centroid_x",
    "centroid_y",
    "perimeter_per_pixel",
    "horizontal_symmetry",
    "vertical_symmetry",
)
SPATIAL_FEATURE_NAMES = tuple(
    f"normalized_silhouette_y{row:02d}_x{column:02d}"
    for row in range(12)
    for column in range(12)
)
FEATURE_NAMES = BASIC_FEATURE_NAMES + SPATIAL_FEATURE_NAMES


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-morphology-diversity-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Morphology-diversity source member is missing: {relative}")
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _round(value: float) -> float:
    return round(float(value), 9)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Morphology-diversity summary must be finite and non-empty")
    return {
        "minimum": _round(array.min()),
        "median": _round(np.median(array)),
        "mean": _round(array.mean()),
        "maximum": _round(array.max()),
    }


def _project_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Morphology-diversity input must live under the project root") from error


def _resize_nearest(mask: np.ndarray, size: int = 12) -> np.ndarray:
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("Cannot normalize an empty silhouette")
    rows = np.minimum(
        ((np.arange(size, dtype=np.float64) + 0.5) * mask.shape[0] / size).astype(np.int64),
        mask.shape[0] - 1,
    )
    columns = np.minimum(
        ((np.arange(size, dtype=np.float64) + 0.5) * mask.shape[1] / size).astype(np.int64),
        mask.shape[1] - 1,
    )
    return np.asarray(mask[np.ix_(rows, columns)], dtype=np.float64)


def _feature_vector(sample: Any) -> tuple[np.ndarray, dict[str, Any]]:
    part = np.asarray(sample.fields.part, dtype=np.uint8)
    solid = (part != 0) & (part != 16)
    y_values, x_values = np.where(solid)
    if len(x_values) < 1:
        raise ValueError(f"Morphology sample has no solid silhouette: {sample.condition.sample_id}")
    left, right = int(x_values.min()), int(x_values.max())
    top, bottom = int(y_values.min()), int(y_values.max())
    width, height = right - left + 1, bottom - top + 1
    crop = solid[top : bottom + 1, left : right + 1]
    area = int(solid.sum())
    padded = np.pad(solid, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = sum(
        int(np.count_nonzero(center & ~neighbor))
        for neighbor in (
            padded[1:-1, 2:],
            padded[1:-1, :-2],
            padded[2:, 1:-1],
            padded[:-2, 1:-1],
        )
    )
    horizontal_symmetry = float(np.mean(crop == np.fliplr(crop)))
    vertical_symmetry = float(np.mean(crop == np.flipud(crop)))
    basic = np.asarray(
        [
            area / solid.size,
            width / solid.shape[1],
            height / solid.shape[0],
            width / height,
            area / (width * height),
            (float(x_values.mean()) - 23.5) / 48.0,
            (float(y_values.mean()) - 23.5) / 48.0,
            perimeter / area,
            horizontal_symmetry,
            vertical_symmetry,
        ],
        dtype=np.float64,
    )
    normalized = _resize_nearest(crop).reshape(-1)
    vector = np.concatenate((basic, normalized))
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Morphology-diversity feature vector is malformed")
    appendage_ids = np.isin(part, np.asarray([4, 5, 6, 7, 8, 9, 13, 14], dtype=np.uint8))
    left_arm, right_arm = int(np.count_nonzero(part == 4)), int(np.count_nonzero(part == 5))
    left_leg, right_leg = int(np.count_nonzero(part == 6)), int(np.count_nonzero(part == 7))
    pair_balance = lambda first, second: 1.0 - abs(first - second) / max(first + second, 1)
    metrics = {
        "solid_pixels": area,
        "bbox": [left, top, width, height],
        "horizontal_symmetry": _round(horizontal_symmetry),
        "vertical_symmetry": _round(vertical_symmetry),
        "appendage_fraction": _round(np.count_nonzero(appendage_ids & solid) / area),
        "arm_pair_balance": _round(pair_balance(left_arm, right_arm)),
        "leg_pair_balance": _round(pair_balance(left_leg, right_leg)),
    }
    return vector, metrics


def _coarse_signature(vector: np.ndarray) -> str:
    basic = np.rint(vector[: len(BASIC_FEATURE_NAMES)] * 50.0).astype(np.int16)
    spatial = np.asarray(vector[len(BASIC_FEATURE_NAMES) :] >= 0.5, dtype=np.uint8)
    payload = {
        "basic_quantization_step": 0.02,
        "normalized_silhouette": spatial.tolist(),
        "values": basic.tolist(),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _classify(
    matrix: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if matrix.shape[0] != labels.shape[0]:
        raise ValueError("Classification labels and features disagree")
    mean = matrix.mean(axis=0)
    deviation = matrix.std(axis=0)
    retained = deviation > 1e-7
    standardized = (matrix[:, retained] - mean[retained]) / deviation[retained]
    results: list[dict[str, Any]] = []
    margins: list[float] = []
    confusion: Counter[tuple[int, int]] = Counter()
    for index, vector in enumerate(standardized):
        distances: dict[int, float] = {}
        for class_id in range(len(class_names)):
            members = np.flatnonzero((labels == class_id) & (np.arange(len(labels)) != index))
            if members.size == 0:
                raise ValueError(f"Classification class has no leave-one-out members: {class_names[class_id]}")
            centroid = standardized[members].mean(axis=0)
            distances[class_id] = float(np.linalg.norm(vector - centroid) / np.sqrt(standardized.shape[1]))
        predicted = min(distances, key=lambda class_id: (distances[class_id], class_id))
        true_id = int(labels[index])
        other_distance = min(value for class_id, value in distances.items() if class_id != true_id)
        margin = other_distance - distances[true_id]
        margins.append(margin)
        confusion[(true_id, predicted)] += 1
        results.append(
            {
                "predicted_id": predicted,
                "predicted_name": class_names[predicted],
                "correct": predicted == true_id,
                "true_margin": _round(margin),
            }
        )
    per_class: list[dict[str, Any]] = []
    for class_id, name in enumerate(class_names):
        indices = np.flatnonzero(labels == class_id)
        correct = sum(bool(results[index]["correct"]) for index in indices)
        per_class.append(
            {
                "id": class_id,
                "name": name,
                "count": int(indices.size),
                "correct": int(correct),
                "recall": _round(correct / indices.size),
            }
        )
    confusion_rows = [
        {"true_id": true_id, "predicted_id": predicted_id, "count": count}
        for (true_id, predicted_id), count in sorted(confusion.items())
    ]
    aggregate = {
        "class_count": len(class_names),
        "retained_feature_count": int(retained.sum()),
        "leave_one_out_accuracy": _round(np.mean([row["correct"] for row in results])),
        "true_margin": _summary(margins),
        "per_class": per_class,
        "confusion": confusion_rows,
    }
    return results, aggregate


def audit_morphology_diversity(
    generation_manifest: Path = DEFAULT_GENERATION_MANIFEST,
) -> dict[str, Any]:
    manifest_path = Path(generation_manifest).resolve()
    bank = load_generation_bank(manifest_path)
    if len(bank.samples) != 80:
        raise ValueError(f"Morphology-diversity authority must contain 80 samples, found {len(bank.samples)}")
    vectors: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    family_labels: list[int] = []
    subtype_labels: list[int] = []
    role_labels: list[int] = []
    for sample in bank.samples:
        vector, metrics = _feature_vector(sample)
        condition = sample.condition
        vectors.append(vector)
        family_labels.append(condition.morphology_id)
        subtype_labels.append(condition.subtype_id)
        role_labels.append(condition.role_id)
        records.append(
            {
                "sample_id": condition.sample_id,
                "ordinal": condition.ordinal,
                "family_id": condition.morphology_id,
                "family": condition.morphology_name,
                "subtype_id": condition.subtype_id,
                "subtype": condition.subtype_name,
                "role_id": condition.role_id,
                "role": condition.role_name,
                "categorical_sha256": sample.fields.aligned_sha256,
                "coarse_chassis_sha256": _coarse_signature(vector),
                "feature_values": [_round(value) for value in vector],
                "metrics": metrics,
            }
        )
    matrix = np.stack(vectors)
    family_rows, family_classification = _classify(
        matrix, np.asarray(family_labels, dtype=np.int64), FAMILIES
    )
    subtype_rows, subtype_classification = _classify(
        matrix, np.asarray(subtype_labels, dtype=np.int64), SUBTYPE_NAMES
    )
    role_rows, role_classification = _classify(
        matrix, np.asarray(role_labels, dtype=np.int64), ROLE_NAMES
    )
    for record, family_row, subtype_row, role_row in zip(
        records, family_rows, subtype_rows, role_rows, strict=True
    ):
        record["classification"] = {
            "family": family_row,
            "subtype": subtype_row,
            "role": role_row,
        }
    coarse_count = len({record["coarse_chassis_sha256"] for record in records})
    per_family: list[dict[str, Any]] = []
    for family_id, family in enumerate(FAMILIES):
        selected = [record for record in records if record["family_id"] == family_id]
        per_family.append(
            {
                "id": family_id,
                "name": family,
                "count": len(selected),
                "horizontal_symmetry": _summary([row["metrics"]["horizontal_symmetry"] for row in selected]),
                "vertical_symmetry": _summary([row["metrics"]["vertical_symmetry"] for row in selected]),
                "appendage_fraction": _summary([row["metrics"]["appendage_fraction"] for row in selected]),
                "arm_pair_balance": _summary([row["metrics"]["arm_pair_balance"] for row in selected]),
                "leg_pair_balance": _summary([row["metrics"]["leg_pair_balance"] for row in selected]),
            }
        )
    aggregate = {
        "sample_count": len(records),
        "feature_count": len(FEATURE_NAMES),
        "unique_coarse_chassis_count": coarse_count,
        "coarse_chassis_unique_fraction": _round(coarse_count / len(records)),
        "horizontal_symmetry": _summary([row["metrics"]["horizontal_symmetry"] for row in records]),
        "vertical_symmetry": _summary([row["metrics"]["vertical_symmetry"] for row in records]),
        "appendage_fraction": _summary([row["metrics"]["appendage_fraction"] for row in records]),
        "per_family": per_family,
    }
    gates = {
        "exact_80_sample_authority": len(records) == 80,
        "every_family_has_16_samples": all(row["count"] == 16 for row in per_family),
        "coarse_chassis_unique_fraction_at_least_0_90": aggregate["coarse_chassis_unique_fraction"] >= 0.90,
        "family_shape_classification_accuracy_at_least_0_90": family_classification["leave_one_out_accuracy"] >= 0.90,
        "every_family_shape_recall_at_least_0_75": all(row["recall"] >= 0.75 for row in family_classification["per_class"]),
        "every_family_has_nonzero_appendage_fraction": all(row["appendage_fraction"]["mean"] > 0 for row in per_family),
        "every_subtype_has_structural_dispersion": all(
            len({record["coarse_chassis_sha256"] for record in records if record["subtype_id"] == subtype_id}) > 1
            for subtype_id in range(len(SUBTYPE_NAMES))
        ),
    }
    report: dict[str, Any] = {
        "format": FORMAT,
        "status": "ready" if all(gates.values()) else "rejected",
        "audit_source_sha256": source_sha256(),
        "input": {
            "generation_manifest_path": _project_relative(manifest_path),
            "generation_manifest_sha256": bank.manifest_sha256,
            "generation_manifest_bytes": bank.manifest_bytes,
        },
        "feature_contract": {
            "names": list(FEATURE_NAMES),
            "normalized_grid": [12, 12],
            "basic_quantization_step": 0.02,
            "sha256": sha256_bytes(canonical_json_bytes(list(FEATURE_NAMES))),
        },
        "records": records,
        "classification": {
            "family": family_classification,
            "subtype": subtype_classification,
            "role": role_classification,
        },
        "aggregate": aggregate,
        "gates": gates,
        "interpretation": {
            "hard_gate_scope": "silhouette-only family structure and coarse chassis diversity",
            "diagnostic_scope": "subtype, role, appendage balance, and soft bilateral symmetry",
            "symmetry_is_hard_requirement": False,
        },
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def _assert_report(report: Mapping[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: tuple(map(str, error.absolute_path)),
    )
    if errors:
        location = "/".join(map(str, errors[0].absolute_path)) or "<root>"
        raise ValueError(f"Morphology-diversity schema failed at {location}: {errors[0].message}")
    if report["audit_source_sha256"] != source_sha256():
        raise ValueError("Morphology-diversity source hash is stale")
    semantic = {key: value for key, value in report.items() if key != "report_sha256"}
    if report["report_sha256"] != sha256_bytes(canonical_json_bytes(semantic)):
        raise ValueError("Morphology-diversity report hash differs")
    if report["status"] != "ready" or not all(report["gates"].values()):
        raise ValueError("Morphology-diversity objective gates failed")


def validate_report(path: Path) -> dict[str, Any]:
    target = Path(path).resolve()
    raw = target.read_bytes()
    report = json.loads(raw)
    if raw != canonical_json_bytes(report):
        raise ValueError("Morphology-diversity report is not canonical JSON")
    _assert_report(report)
    manifest = PROJECT_ROOT / report["input"]["generation_manifest_path"]
    if (
        not manifest.is_file()
        or manifest.stat().st_size != report["input"]["generation_manifest_bytes"]
        or sha256_file(manifest) != report["input"]["generation_manifest_sha256"]
    ):
        raise ValueError("Morphology-diversity input authority drifted")
    rebuilt = audit_morphology_diversity(manifest)
    if rebuilt != report:
        raise ValueError("Morphology-diversity exact replay differs")
    return {
        "passed": True,
        "sample_count": report["aggregate"]["sample_count"],
        "coarse_chassis_unique_fraction": report["aggregate"]["coarse_chassis_unique_fraction"],
        "family_accuracy": report["classification"]["family"]["leave_one_out_accuracy"],
        "report_sha256": report["report_sha256"],
    }


def build_report(
    output: Path = DEFAULT_OUTPUT,
    *,
    generation_manifest: Path = DEFAULT_GENERATION_MANIFEST,
) -> dict[str, Any]:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    report = audit_morphology_diversity(generation_manifest)
    _assert_report(report)
    payload = canonical_json_bytes(report)
    require_disk_floor(target.parent, planned_bytes=len(payload) + 8 * 1024**2)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return validate_report(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit neural morphology structural diversity")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--generation-manifest", type=Path, default=DEFAULT_GENERATION_MANIFEST)
    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    result = build_report(args.output, generation_manifest=args.generation_manifest) if args.command == "build" else validate_report(args.report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
