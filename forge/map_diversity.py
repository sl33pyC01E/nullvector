from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np

from .config import PROJECT_ROOT
from .maps import MapConfig, THEMES, assert_valid, generate_map
from .maps.generator import splitmix64
from .maps.io import array_digest
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-map-structural-diversity-audit-v1"
BASE_SEED = 0x4449564552534954
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/map_diversity_v1/map_diversity_report.json"
SCHEMA_PATH = PROJECT_ROOT / "shared/schema/map_diversity_report.schema.json"
SOURCE_FILES = ("forge/map_diversity.py", "shared/schema/map_diversity_report.schema.json")
SCALAR_FEATURES = (
    "walkable_fraction", "hazard_per_walkable", "elevation_level_fraction",
    "zone_count_fraction", "protected_backbone_fraction", "required_clearance_fraction",
    "mean_cardinal_degree", "endpoint_fraction", "junction_fraction", "cycle_density",
    *(f"square_erosion_r{radius}_fraction" for radius in range(1, 10)),
)
SPATIAL_FEATURES = tuple(
    f"occupancy_y{row}_x{column}" for row in range(8) for column in range(8)
)
FEATURE_NAMES = SCALAR_FEATURES + SPATIAL_FEATURES


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-map-structural-diversity-source-v1\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode("utf-8") + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _square_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for offset_y in range(3):
        for offset_x in range(3):
            result &= padded[offset_y:offset_y + mask.shape[0], offset_x:offset_x + mask.shape[1]]
    return result


def _feature_vector(data: Any) -> np.ndarray:
    walkable = np.asarray(data.walkability, dtype=bool)
    height, width = walkable.shape; node_count = int(walkable.sum())
    degree = np.zeros(walkable.shape, dtype=np.uint8)
    horizontal = walkable[:, :-1] & walkable[:, 1:]
    vertical = walkable[:-1, :] & walkable[1:, :]
    degree[:, :-1] += horizontal; degree[:, 1:] += horizontal
    degree[:-1, :] += vertical; degree[1:, :] += vertical
    edge_count = int(horizontal.sum() + vertical.sum())
    erosion: list[float] = []; interior = walkable.copy()
    for _radius in range(1, 10):
        interior = _square_erode(interior)
        erosion.append(float(interior.sum()) / max(1, node_count))
    spatial: list[float] = []
    for row in range(8):
        top, bottom = row * height // 8, (row + 1) * height // 8
        for column in range(8):
            left, right = column * width // 8, (column + 1) * width // 8
            spatial.append(float(walkable[top:bottom, left:right].mean()))
    values = [
        node_count / (height * width),
        float(np.count_nonzero(data.hazard)) / max(1, node_count),
        len(np.unique(data.elevation[walkable])) / 16.0,
        len(np.unique(data.zone[walkable])) / 64.0,
        float(np.count_nonzero(data.protected_backbone)) / (height * width),
        float(np.count_nonzero(data.required_clearance)) / (height * width),
        float(degree[walkable].mean()) / 4.0,
        float(np.mean(degree[walkable] == 1)),
        float(np.mean(degree[walkable] >= 3)),
        max(0, edge_count - node_count + 1) / max(1, node_count),
        *erosion,
        *spatial,
    ]
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (83,) or not np.isfinite(result).all():
        raise ValueError("Map diversity feature vector is malformed")
    return result


def _coarse_signature(features: np.ndarray) -> str:
    scalar = np.rint(features[:len(SCALAR_FEATURES)] * 50.0).astype(np.int16)
    spatial = np.rint(features[len(SCALAR_FEATURES):] * 8.0).astype(np.int16)
    payload = {"scalar_step": 0.02, "spatial_step": 0.125, "values": np.concatenate((scalar, spatial)).tolist()}
    return sha256_bytes(canonical_json_bytes(payload))


def _classify(matrix: np.ndarray, labels: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mean = matrix.mean(axis=0); deviation = matrix.std(axis=0); retained = deviation > 1e-7
    standardized = (matrix[:, retained] - mean[retained]) / deviation[retained]
    centroids = {theme: standardized[[index for index, label in enumerate(labels) if label == theme]].mean(axis=0) for theme in THEMES}
    records: list[dict[str, Any]] = []; true_margins: list[float] = []
    for index, vector in enumerate(standardized):
        distances: dict[str, float] = {}
        for theme in THEMES:
            members = [row for row, label in enumerate(labels) if label == theme and row != index]
            centroid = standardized[members].mean(axis=0)
            distances[theme] = float(np.linalg.norm(vector - centroid) / np.sqrt(standardized.shape[1]))
        ordered = sorted(distances, key=lambda theme: (distances[theme], theme)); predicted = ordered[0]
        other = min(distances[theme] for theme in THEMES if theme != labels[index]); true_margin = other - distances[labels[index]]; true_margins.append(true_margin)
        records.append({
            "predicted_theme": predicted,
            "correct": predicted == labels[index],
            "true_theme_margin": round(true_margin, 9),
            "distances": {theme: round(distances[theme], 9) for theme in THEMES},
        })
    centroid_distances = [
        float(np.linalg.norm(centroids[left] - centroids[right]) / np.sqrt(standardized.shape[1]))
        for left_index, left in enumerate(THEMES) for right in THEMES[left_index + 1:]
    ]
    per_theme: dict[str, Any] = {}
    for theme in THEMES:
        indices = [index for index, label in enumerate(labels) if label == theme]
        dispersion = float(np.mean([np.linalg.norm(standardized[index] - centroids[theme]) / np.sqrt(standardized.shape[1]) for index in indices]))
        correct = sum(records[index]["correct"] for index in indices)
        per_theme[theme] = {"count": len(indices), "correct": correct, "recall": round(correct / len(indices), 9), "dispersion": round(dispersion, 9)}
    aggregate = {
        "retained_feature_count": int(retained.sum()),
        "leave_one_out_accuracy": round(sum(record["correct"] for record in records) / len(records), 9),
        "mean_true_theme_margin": round(float(np.mean(true_margins)), 9),
        "minimum_centroid_distance": round(min(centroid_distances), 9),
        "per_theme": per_theme,
    }
    return records, aggregate


def audit_map_diversity(*, samples_per_theme: int = 8, base_seed: int = BASE_SEED, config: MapConfig | None = None) -> dict[str, Any]:
    if not 3 <= samples_per_theme <= 64 or not 0 <= base_seed < 2**64:
        raise ValueError("Map diversity schedule is outside its bounded contract")
    cfg = config or MapConfig(width=48, height=48, objective_count=3, spawn_count=8, min_start_exit_distance=24)
    rows: list[np.ndarray] = []; labels: list[str] = []; records: list[dict[str, Any]] = []
    for theme_index, theme in enumerate(THEMES):
        for sample_index in range(samples_per_theme):
            seed = splitmix64(base_seed ^ ((theme_index + 1) << 48) ^ sample_index)
            data = generate_map(seed, theme, cfg); validation = assert_valid(data); features = _feature_vector(data)
            rows.append(features); labels.append(theme)
            records.append({
                "map_id": data.map_id, "theme": theme, "sample_index": sample_index,
                "seed": f"0x{seed:016x}", "semantic_sha256": array_digest(data.arrays()),
                "coarse_topology_sha256": _coarse_signature(features),
                "feature_values": [round(float(value), 9) for value in features],
                "validation_failure_count": len(validation["failures"]),
            })
    classifications, classification_aggregate = _classify(np.stack(rows), labels)
    for record, classification in zip(records, classifications, strict=True): record["classification"] = classification
    semantic_count = len({record["semantic_sha256"] for record in records}); coarse_count = len({record["coarse_topology_sha256"] for record in records}); count = len(records)
    aggregate = {
        "map_count": count, "unique_semantic_count": semantic_count, "unique_coarse_topology_count": coarse_count,
        "semantic_unique_fraction": round(semantic_count / count, 9), "coarse_topology_unique_fraction": round(coarse_count / count, 9),
        **classification_aggregate,
    }
    gates = {
        "all_maps_hard_valid": all(record["validation_failure_count"] == 0 for record in records),
        "all_semantic_maps_unique": semantic_count == count,
        "coarse_topology_unique_fraction_at_least_0_90": coarse_count / count >= 0.90,
        "theme_classification_accuracy_at_least_0_85": aggregate["leave_one_out_accuracy"] >= 0.85,
        "every_theme_recall_at_least_0_50": all(value["recall"] >= 0.50 for value in aggregate["per_theme"].values()),
        "mean_true_theme_margin_positive": aggregate["mean_true_theme_margin"] > 0.0,
        "every_theme_has_nonzero_dispersion": all(value["dispersion"] > 0.0 for value in aggregate["per_theme"].values()),
    }
    report: dict[str, Any] = {
        "format": FORMAT, "status": "ready" if all(gates.values()) else "rejected",
        "audit_source_sha256": source_sha256(),
        "schedule": {"base_seed": f"0x{base_seed:016x}", "samples_per_theme": samples_per_theme, "themes": list(THEMES)},
        "config": cfg.to_dict(),
        "feature_contract": {"names": list(FEATURE_NAMES), "scalar_quantization_step": 0.02, "spatial_quantization_step": 0.125, "sha256": sha256_bytes(canonical_json_bytes(list(FEATURE_NAMES)))},
        "records": records, "aggregate": aggregate, "gates": gates,
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def _assert_report(report: Mapping[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8")); errors = sorted(Draft202012Validator(schema).iter_errors(report), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Map diversity schema failed: {errors[0].message}")
    if report["audit_source_sha256"] != source_sha256(): raise ValueError("Map diversity source hash is stale")
    semantic = {key: value for key, value in report.items() if key != "report_sha256"}
    if report["report_sha256"] != sha256_bytes(canonical_json_bytes(semantic)): raise ValueError("Map diversity report hash differs")
    if report["status"] != "ready" or not all(report["gates"].values()): raise ValueError("Map diversity gates failed")


def validate_report(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(); raw = path.read_bytes(); report = json.loads(raw)
    if raw != canonical_json_bytes(report): raise ValueError("Map diversity report is not canonical JSON")
    _assert_report(report)
    cfg = MapConfig(**report["config"]); expected = audit_map_diversity(samples_per_theme=report["schedule"]["samples_per_theme"], base_seed=int(report["schedule"]["base_seed"], 16), config=cfg)
    if report != expected: raise ValueError("Map diversity exact replay differs")
    return {"passed": True, "map_count": report["aggregate"]["map_count"], "accuracy": report["aggregate"]["leave_one_out_accuracy"], "report_sha256": report["report_sha256"]}


def build_report(path: Path = DEFAULT_OUTPUT, *, samples_per_theme: int = 8, base_seed: int = BASE_SEED) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.exists(): raise FileExistsError(path)
    report = audit_map_diversity(samples_per_theme=samples_per_theme, base_seed=base_seed); _assert_report(report)
    payload = canonical_json_bytes(report); require_disk_floor(path.parent, planned_bytes=len(payload) + 16 * 1024**2); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    return validate_report(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit deterministic structural map diversity")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build"); build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); build.add_argument("--samples-per-theme", type=int, default=8); build.add_argument("--seed", type=lambda value: int(value, 0), default=BASE_SEED)
    validate = sub.add_parser("validate"); validate.add_argument("report", type=Path)
    args = parser.parse_args(argv); result = build_report(args.output, samples_per_theme=args.samples_per_theme, base_seed=args.seed) if args.command == "build" else validate_report(args.report)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
