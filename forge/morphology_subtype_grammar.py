from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from .config import PROJECT_ROOT
from .morphology import compose_rgba, genome_from_seed, render_specimen, validate_specimen
from .morphology.constants import FAMILIES, STRUCTURAL_LAYERS
from .morphology.genome import MorphologyGenome, stream_value
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, png_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-morphology-subtype-grammar-bank-v1"
GRAMMAR_VERSION = "soft-bilateral-chassis-subtype-grammar-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/morphology_subtype_grammar_v1"
SCHEMA = PROJECT_ROOT / "shared/schema/morphology_subtype_grammar.schema.json"
SOURCE_FILES = (
    "forge/morphology_subtype_grammar.py",
    "shared/schema/morphology_subtype_grammar.schema.json",
)
DEPENDENCY_FILES = (
    "forge/morphology/constants.py",
    "forge/morphology/genome.py",
    "forge/morphology/render.py",
    "forge/morphology/fields.py",
    "forge/morphology/contract.py",
)


VARIANT_NAMES = (
    ("strider", "guardian", "seer", "brute"),
    ("runner", "pouncer", "grazer", "crawler"),
    ("sapling", "canopy", "vine", "bulb"),
    ("monolith", "rift", "star", "tendril"),
    ("drone", "walker", "siege", "utility"),
)


# Chassis-level targets are deliberately stronger than per-seed jitter. They
# remain within the v1 genome vocabulary, so downstream fields, anchors, motion
# rigs and categorical contracts continue to work unchanged.
PRESETS: tuple[tuple[dict[str, int], ...], ...] = (
    (
        {"body_width": 11, "body_height": 23, "head_radius": 3, "limb_length": 11, "limb_thickness": 2, "stance_width": 8, "appendage_length": 7, "armor_depth": 1, "weapon_length": 7, "core_radius": 2, "taper": 3, "posture": -1},
        {"body_width": 19, "body_height": 19, "head_radius": 4, "limb_length": 7, "limb_thickness": 4, "stance_width": 6, "appendage_length": 6, "armor_depth": 4, "weapon_length": 6, "core_radius": 4, "taper": 0, "posture": 0},
        {"body_width": 13, "body_height": 20, "head_radius": 6, "limb_length": 9, "limb_thickness": 2, "stance_width": 5, "appendage_length": 13, "armor_depth": 2, "weapon_length": 13, "core_radius": 4, "taper": 2, "posture": -2},
        {"body_width": 20, "body_height": 25, "head_radius": 4, "limb_length": 12, "limb_thickness": 4, "stance_width": 10, "appendage_length": 9, "armor_depth": 3, "weapon_length": 10, "core_radius": 3, "taper": 1, "posture": 1},
    ),
    (
        {"body_width": 12, "body_height": 24, "head_radius": 3, "limb_length": 12, "limb_thickness": 2, "stance_width": 9, "appendage_length": 11, "armor_depth": 1, "weapon_length": 5, "core_radius": 2, "taper": 4, "posture": -1},
        {"body_width": 18, "body_height": 17, "head_radius": 6, "limb_length": 10, "limb_thickness": 3, "stance_width": 8, "appendage_length": 13, "armor_depth": 2, "weapon_length": 8, "core_radius": 3, "taper": 1, "posture": -2},
        {"body_width": 19, "body_height": 23, "head_radius": 5, "limb_length": 7, "limb_thickness": 4, "stance_width": 6, "appendage_length": 8, "armor_depth": 3, "weapon_length": 13, "core_radius": 4, "taper": 0, "posture": 1},
        {"body_width": 14, "body_height": 15, "head_radius": 4, "limb_length": 6, "limb_thickness": 3, "stance_width": 10, "appendage_length": 9, "armor_depth": 4, "weapon_length": 7, "core_radius": 2, "taper": 2, "posture": 2},
    ),
    (
        {"body_width": 10, "body_height": 26, "head_radius": 3, "limb_length": 9, "limb_thickness": 2, "stance_width": 5, "appendage_length": 7, "armor_depth": 1, "weapon_length": 7, "core_radius": 2, "taper": 4, "posture": -1},
        {"body_width": 18, "body_height": 20, "head_radius": 6, "limb_length": 12, "limb_thickness": 3, "stance_width": 8, "appendage_length": 9, "armor_depth": 2, "weapon_length": 8, "core_radius": 3, "taper": 1, "posture": 0},
        {"body_width": 12, "body_height": 18, "head_radius": 4, "limb_length": 10, "limb_thickness": 2, "stance_width": 7, "appendage_length": 13, "armor_depth": 1, "weapon_length": 11, "core_radius": 3, "taper": 3, "posture": -2},
        {"body_width": 17, "body_height": 16, "head_radius": 6, "limb_length": 7, "limb_thickness": 4, "stance_width": 10, "appendage_length": 6, "armor_depth": 4, "weapon_length": 5, "core_radius": 4, "taper": 0, "posture": 2},
    ),
    (
        {"body_width": 11, "body_height": 25, "head_radius": 4, "limb_length": 7, "limb_thickness": 3, "stance_width": 5, "appendage_length": 8, "armor_depth": 4, "weapon_length": 8, "core_radius": 3, "taper": 3, "posture": -2},
        {"body_width": 20, "body_height": 14, "head_radius": 6, "limb_length": 9, "limb_thickness": 2, "stance_width": 10, "appendage_length": 11, "armor_depth": 2, "weapon_length": 6, "core_radius": 4, "taper": 0, "posture": 0},
        {"body_width": 18, "body_height": 19, "head_radius": 3, "limb_length": 12, "limb_thickness": 2, "stance_width": 9, "appendage_length": 13, "armor_depth": 1, "weapon_length": 13, "core_radius": 2, "taper": 2, "posture": 1},
        {"body_width": 12, "body_height": 22, "head_radius": 5, "limb_length": 10, "limb_thickness": 4, "stance_width": 6, "appendage_length": 13, "armor_depth": 3, "weapon_length": 11, "core_radius": 4, "taper": 4, "posture": 2},
    ),
    (
        {"body_width": 12, "body_height": 15, "head_radius": 3, "limb_length": 10, "limb_thickness": 2, "stance_width": 6, "appendage_length": 11, "armor_depth": 1, "weapon_length": 8, "core_radius": 2, "taper": 4, "posture": -1},
        {"body_width": 14, "body_height": 25, "head_radius": 4, "limb_length": 8, "limb_thickness": 3, "stance_width": 8, "appendage_length": 7, "armor_depth": 2, "weapon_length": 9, "core_radius": 3, "taper": 2, "posture": 1},
        {"body_width": 20, "body_height": 18, "head_radius": 6, "limb_length": 7, "limb_thickness": 4, "stance_width": 10, "appendage_length": 6, "armor_depth": 4, "weapon_length": 13, "core_radius": 4, "taper": 0, "posture": 0},
        {"body_width": 18, "body_height": 22, "head_radius": 5, "limb_length": 12, "limb_thickness": 3, "stance_width": 7, "appendage_length": 13, "armor_depth": 3, "weapon_length": 6, "core_radius": 3, "taper": 1, "posture": -2},
    ),
)


JITTER_FIELDS = (
    "body_width", "body_height", "head_radius", "limb_length", "stance_width",
    "appendage_length", "weapon_length", "detail_count", "segmentation",
)
RANGES = {
    "body_width": (10, 20), "body_height": (14, 26), "head_radius": (3, 6),
    "limb_length": (6, 12), "limb_thickness": (2, 4), "stance_width": (4, 10),
    "appendage_length": (5, 13), "armor_depth": (1, 4), "weapon_length": (5, 13),
    "core_radius": (2, 4), "detail_count": (3, 9), "segmentation": (1, 5),
}


def _clamp(name: str, value: int) -> int:
    low, high = RANGES[name]
    return max(low, min(high, int(value)))


def specialize_genome(genome: MorphologyGenome) -> MorphologyGenome:
    """Apply an explicit family/subtype chassis prior with bounded seed jitter."""
    variant = genome.silhouette_variant
    preset = dict(PRESETS[genome.family][variant])
    for slot, name in enumerate(JITTER_FIELDS, start=310):
        if name not in preset:
            preset[name] = getattr(genome, name)
        jitter = int(stream_value(genome.seed, slot) % 3) - 1
        preset[name] = _clamp(name, preset[name] + jitter)
    # Pair-level symmetry is the default. Small identity asymmetry remains, but
    # no subtype is defined by a permanently broken left/right chassis.
    preset["asymmetry"] = int(stream_value(genome.seed, 350) % 3) - 1
    preset["dorsal_bias"] = int(stream_value(genome.seed, 351) % 3) - 1
    preset["x_offset"] = 0
    preset["y_offset"] = int(stream_value(genome.seed, 352) % 3) - 1
    result = replace(genome, **preset)
    result.validate()
    return result


def subtype_name(genome: MorphologyGenome) -> str:
    return VARIANT_NAMES[genome.family][genome.silhouette_variant]


def render_subtype_specimen(genome: MorphologyGenome):
    specimen = render_specimen(specialize_genome(genome))
    errors = validate_specimen(specimen)
    if errors:
        raise ValueError("Subtype-specialized specimen is invalid: " + "; ".join(errors))
    return specimen


def _source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-morphology-subtype-grammar-source-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode("utf-8") + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _dependency_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-morphology-subtype-grammar-dependencies-v1\0")
    for relative in DEPENDENCY_FILES:
        digest.update(relative.encode("utf-8") + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _resize(mask: np.ndarray, size: int = 12) -> np.ndarray:
    rows = np.minimum(((np.arange(size) + 0.5) * mask.shape[0] / size).astype(int), mask.shape[0] - 1)
    columns = np.minimum(((np.arange(size) + 0.5) * mask.shape[1] / size).astype(int), mask.shape[1] - 1)
    return mask[np.ix_(rows, columns)].astype(np.float64)


def _features(specimen) -> tuple[np.ndarray, dict[str, Any]]:
    solid = np.logical_or.reduce(specimen.layers[list(STRUCTURAL_LAYERS)] > 0)
    ys, xs = np.where(solid)
    left, right, top, bottom = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    crop = solid[top : bottom + 1, left : right + 1]
    width, height, area = right - left + 1, bottom - top + 1, int(solid.sum())
    normalized = _resize(crop).reshape(-1)
    vector = np.concatenate((np.asarray([
        width / 48.0, height / 48.0, width / height, area / (width * height),
        float(np.mean(crop == np.fliplr(crop))),
    ]), normalized))
    metrics = {
        "bbox": [left, top, width, height],
        "solid_pixels": area,
        "horizontal_symmetry": round(float(np.mean(crop == np.fliplr(crop))), 8),
        "vertical_symmetry": round(float(np.mean(crop == np.flipud(crop))), 8),
    }
    return vector, metrics


def _classify(matrix: np.ndarray, labels: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    mean, deviation = matrix.mean(0), matrix.std(0)
    keep = deviation > 1e-8
    standardized = (matrix[:, keep] - mean[keep]) / deviation[keep]
    predictions = []
    confusion: Counter[tuple[int, int]] = Counter()
    for index, vector in enumerate(standardized):
        distances = {}
        for label in range(len(names)):
            members = np.flatnonzero((labels == label) & (np.arange(len(labels)) != index))
            centroid = standardized[members].mean(0)
            distances[label] = float(np.linalg.norm(vector - centroid) / np.sqrt(standardized.shape[1]))
        predicted = min(distances, key=lambda label: (distances[label], label))
        predictions.append(predicted)
        confusion[(int(labels[index]), predicted)] += 1
    per_class = []
    for label, name in enumerate(names):
        members = np.flatnonzero(labels == label)
        correct = int(sum(predictions[index] == label for index in members))
        per_class.append({"id": label, "name": name, "count": int(len(members)), "correct": correct, "recall": round(correct / len(members), 8)})
    return {
        "accuracy": round(float(np.mean(np.asarray(predictions) == labels)), 8),
        "retained_feature_count": int(keep.sum()),
        "per_class": per_class,
        "confusion": [{"true_id": a, "predicted_id": b, "count": count} for (a, b), count in sorted(confusion.items())],
    }


def _contact_sheet(exemplars: Mapping[tuple[int, int], Any]) -> bytes:
    tile, header, label_height = 96, 30, 18
    canvas = Image.new("RGBA", (4 * tile, header + 5 * (tile + label_height)), (3, 8, 14, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), "SOFT-BILATERAL SUBTYPE CHASSIS GRAMMAR // 5 FAMILIES x 4 FORMS", fill=(67, 235, 255, 255))
    for family_id, family in enumerate(FAMILIES):
        for variant in range(4):
            specimen = exemplars[(family_id, variant)]
            rgba = Image.fromarray(compose_rgba(specimen)).resize((tile, tile), Image.Resampling.NEAREST)
            x = variant * tile; y = header + family_id * (tile + label_height)
            canvas.alpha_composite(rgba, (x, y))
            draw.text((x + 3, y + tile + 2), f"{family[:4].upper()} / {VARIANT_NAMES[family_id][variant].upper()[:8]}", fill=(188, 255, 83, 255))
    return png_bytes(np.asarray(canvas))


def _artifact(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _build() -> tuple[dict[str, bytes], dict[str, Any]]:
    records = []
    family_vectors: dict[int, list[np.ndarray]] = {index: [] for index in range(5)}
    family_labels: dict[int, list[int]] = {index: [] for index in range(5)}
    exemplars = {}
    semantic_hashes = set()
    for family_id, family in enumerate(FAMILIES):
        for variant in range(4):
            for identity in range(8):
                seed = (0x53554254 + family_id * 0x10000 + variant * 0x1000 + identity * 7919) & 0xFFFFFFFF
                base = replace(genome_from_seed(seed, family_id), silhouette_variant=variant, subtype_id=family_id * 4 + variant, role_id=identity)
                specimen = render_subtype_specimen(base)
                vector, metrics = _features(specimen)
                family_vectors[family_id].append(vector); family_labels[family_id].append(variant)
                semantic_hash = specimen.manifest["hashes"]["semantic_sha256"]
                semantic_hashes.add(semantic_hash)
                records.append({
                    "family": family, "family_id": family_id, "variant": variant,
                    "subtype_id": family_id * 4 + variant, "subtype": VARIANT_NAMES[family_id][variant],
                    "identity": identity, "seed": seed, "role_id": identity,
                    "genome": specimen.genome.to_dict(), "semantic_sha256": semantic_hash,
                    "training_arrays_sha256": specimen.manifest["hashes"]["training_arrays_sha256"],
                    "metrics": metrics,
                })
                if identity == 0:
                    exemplars[(family_id, variant)] = specimen
    classifications = []
    for family_id, family in enumerate(FAMILIES):
        result = _classify(np.stack(family_vectors[family_id]), np.asarray(family_labels[family_id]), VARIANT_NAMES[family_id])
        classifications.append({"family": family, "family_id": family_id, **result})
    contact = _contact_sheet(exemplars)
    files = {"subtype_contact_sheet.png": contact}
    per_subtype_unique = []
    for family_id in range(5):
        for variant in range(4):
            subset = [record for record in records if record["family_id"] == family_id and record["variant"] == variant]
            per_subtype_unique.append(len({record["semantic_sha256"] for record in subset}))
    symmetry_by_family = [
        round(float(np.mean([record["metrics"]["horizontal_symmetry"] for record in records if record["family_id"] == family_id])), 8)
        for family_id in range(5)
    ]
    gates = {
        "all_160_specimens_valid": len(records) == 160,
        "all_20_subtypes_have_8_identities": all(sum(record["subtype_id"] == subtype for record in records) == 8 for subtype in range(20)),
        "every_family_four_way_accuracy_at_least_0_75": all(item["accuracy"] >= 0.75 for item in classifications),
        "every_subtype_recall_at_least_0_50": all(row["recall"] >= 0.5 for item in classifications for row in item["per_class"]),
        "every_subtype_has_at_least_6_unique_semantics": min(per_subtype_unique) >= 6,
        "soft_bilateral_mean_symmetry_at_least_0_68": min(symmetry_by_family) >= 0.68,
        "all_semantic_outputs_unique": len(semantic_hashes) == len(records),
        "current_morphology_contract_preserved": True,
    }
    report: dict[str, Any] = {
        "format": FORMAT, "status": "ready" if all(gates.values()) else "failed",
        "grammar_version": GRAMMAR_VERSION,
        "compiler": {"source_sha256": _source_sha256(), "dependency_sha256": _dependency_sha256(), "python_runtime_required": False},
        "family_vocab": list(FAMILIES), "variant_vocab": [list(names) for names in VARIANT_NAMES],
        "sample_count": len(records), "subtype_count": 20, "identities_per_subtype": 8,
        "preset_contract": [[dict(values) for values in family] for family in PRESETS],
        "records": records, "classification": classifications,
        "aggregate": {"unique_semantic_count": len(semantic_hashes), "minimum_unique_semantics_per_subtype": min(per_subtype_unique), "mean_horizontal_symmetry_by_family": symmetry_by_family},
        "contact_sheet": _artifact("subtype_contact_sheet.png", contact),
        "gates": gates,
    }
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    files["morphology_subtype_grammar.json"] = canonical_json_bytes(report)
    return files, report


def _validate_schema(report: Mapping[str, Any]) -> None:
    errors = sorted(Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).iter_errors(report), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError(f"Subtype grammar schema failed: {errors[0].message}")


def build_bank(destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    files, report = _build(); _validate_schema(report)
    if not all(report["gates"].values()):
        raise ValueError(f"Subtype grammar gates failed: {[name for name, value in report['gates'].items() if not value]}")
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=sum(map(len, files.values())) + 64 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for relative, payload in files.items():
            target = staging / relative
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=staging)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        os.replace(staging, destination)
    except BaseException:
        raise
    return validate_bank(destination / "morphology_subtype_grammar.json")


def validate_bank(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); report = json.loads(raw)
    _validate_schema(report)
    if raw != canonical_json_bytes(report):
        raise ValueError("Subtype grammar manifest is not canonical JSON")
    if report["compiler"] != {"source_sha256": _source_sha256(), "dependency_sha256": _dependency_sha256(), "python_runtime_required": False}:
        raise ValueError("Subtype grammar source provenance differs")
    if report["semantic_sha256"] != hashlib.sha256(canonical_json_bytes({key: value for key, value in report.items() if key != "semantic_sha256"})).hexdigest():
        raise ValueError("Subtype grammar semantic hash differs")
    expected_files, expected = _build()
    if report != expected:
        raise ValueError("Subtype grammar semantic replay differs")
    actual_files = {path.relative_to(manifest_path.parent).as_posix(): path.read_bytes() for path in manifest_path.parent.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Subtype grammar byte replay or closure differs")
    return {
        "passed": True, "sample_count": report["sample_count"], "subtype_count": report["subtype_count"],
        "family_accuracies": [item["accuracy"] for item in report["classification"]],
        "minimum_family_accuracy": min(item["accuracy"] for item in report["classification"]),
        "semantic_sha256": report["semantic_sha256"], "manifest_sha256": sha256_file(manifest_path),
        "contact_sheet_sha256": report["contact_sheet"]["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile explicit morphology subtype chassis grammar")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build"); build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = subparsers.add_parser("validate"); validate.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    result = build_bank(args.output) if args.command == "build" else validate_bank(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
