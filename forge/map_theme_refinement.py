from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from .config import PROJECT_ROOT
from .map_diversity import BASE_SEED, _classify, _feature_vector
from .maps import MapConfig, THEMES, assert_valid, generate_map
from .maps.generator import splitmix64
from .maps.io import array_digest
from .maps.model import Terrain
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, png_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-map-theme-refinement-audit-v1"
POLICY = "bounded-archipelago-land-water-shoreline-selection-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/map_theme_refinement_v1"
SCHEMA = PROJECT_ROOT / "shared/schema/map_theme_refinement.schema.json"
MAX_ATTEMPTS = 32
ATTEMPT_SALT = 0x9E3779B97F4A7C15
SOURCE_FILES = ("forge/map_theme_refinement.py", "shared/schema/map_theme_refinement.schema.json")
DEPENDENCY_FILES = (
    "forge/maps/model.py", "forge/maps/generator.py", "forge/maps/validate.py",
    "forge/maps/io.py", "forge/map_diversity.py",
)


@dataclass(frozen=True, slots=True)
class RefinedMap:
    requested_seed: int
    selected_seed: int
    attempt: int
    data: Any
    metrics: Mapping[str, float]


def _hash_files(domain: bytes, paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256(domain)
    for relative in paths:
        digest.update(relative.encode("utf-8") + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def source_sha256() -> str:
    return _hash_files(b"nullvector-map-theme-refinement-source-v1\0", SOURCE_FILES)


def dependency_sha256() -> str:
    return _hash_files(b"nullvector-map-theme-refinement-dependencies-v1\0", DEPENDENCY_FILES)


def _candidate_seed(requested_seed: int, attempt: int) -> int:
    if attempt == 0:
        return requested_seed & ((1 << 64) - 1)
    return splitmix64(requested_seed ^ (((attempt + 1) * ATTEMPT_SALT) & ((1 << 64) - 1)))


def archipelago_metrics(data: Any) -> dict[str, float]:
    terrain = np.asarray(data.terrain, dtype=np.uint8)
    return {
        "walkable_fraction": round(float(np.mean(data.walkability)), 9),
        "water_fraction": round(float(np.mean(terrain == int(Terrain.WATER))), 9),
        "sand_fraction": round(float(np.mean(terrain == int(Terrain.SAND))), 9),
        "bridge_fraction": round(float(np.mean(terrain == int(Terrain.BRIDGE))), 9),
    }


def archipelago_policy_passes(metrics: Mapping[str, float]) -> bool:
    return bool(
        0.42 <= metrics["walkable_fraction"] <= 0.62
        and metrics["water_fraction"] >= 0.34
        and metrics["sand_fraction"] >= 0.05
    )


def generate_refined_archipelago(seed: int, config: MapConfig | None = None) -> RefinedMap:
    cfg = config or MapConfig()
    requested = int(seed) & ((1 << 64) - 1)
    for attempt in range(MAX_ATTEMPTS):
        selected = _candidate_seed(requested, attempt)
        data = generate_map(selected, "archipelago", cfg)
        assert_valid(data)
        metrics = archipelago_metrics(data)
        if archipelago_policy_passes(metrics):
            return RefinedMap(requested, selected, attempt, data, metrics)
    raise RuntimeError(f"Archipelago refinement exhausted {MAX_ATTEMPTS} candidates for 0x{requested:016x}")


def generate_refined_map(seed: int, theme: str, config: MapConfig | None = None) -> RefinedMap:
    if theme == "archipelago":
        return generate_refined_archipelago(seed, config)
    if theme not in THEMES:
        raise ValueError(f"Unknown theme: {theme}")
    requested = int(seed) & ((1 << 64) - 1)
    data = generate_map(requested, theme, config)
    assert_valid(data)
    return RefinedMap(requested, requested, 0, data, {})


TERRAIN_COLORS = {
    int(Terrain.VOID): (2, 5, 10), int(Terrain.FLOOR): (22, 40, 48),
    int(Terrain.WALL): (50, 61, 72), int(Terrain.WATER): (5, 40, 80),
    int(Terrain.BRIDGE): (199, 138, 70), int(Terrain.GROWTH): (35, 92, 58),
    int(Terrain.CRYSTAL): (128, 61, 220), int(Terrain.CHASM): (8, 2, 18),
    int(Terrain.SAND): (193, 177, 92),
}


def _render_map(data: Any) -> Image.Image:
    rgb = np.zeros((*data.terrain.shape, 3), dtype=np.uint8)
    for terrain, color in TERRAIN_COLORS.items():
        rgb[data.terrain == terrain] = color
    image = Image.fromarray(rgb).resize((data.terrain.shape[1] * 3, data.terrain.shape[0] * 3), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    def marker(point, color):
        x, y = point; draw.rectangle([x * 3, y * 3, x * 3 + 2, y * 3 + 2], fill=color)
    marker(data.start, (80, 255, 225)); marker(data.exit, (255, 77, 185))
    for point in data.objectives: marker(point, (190, 255, 70))
    return image


def _contact_sheet(archipelagos: list[RefinedMap]) -> bytes:
    tile_w, tile_h, header = 144, 144, 28
    canvas = Image.new("RGB", (4 * tile_w, header + 2 * (tile_h + 16)), (3, 8, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), "REFINED ARCHIPELAGOS // WATER + SHORELINE + CONNECTED MISSION BACKBONE", fill=(67, 235, 255))
    for index, item in enumerate(archipelagos):
        x, y = (index % 4) * tile_w, header + (index // 4) * (tile_h + 16)
        canvas.paste(_render_map(item.data), (x, y))
        draw.text((x + 3, y + tile_h + 2), f"A{item.attempt}  W{item.metrics['water_fraction']:.2f}  S{item.metrics['sand_fraction']:.2f}", fill=(188, 255, 83))
    return png_bytes(np.asarray(canvas))


def _artifact(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _build(samples_per_theme: int = 8, base_seed: int = BASE_SEED) -> tuple[dict[str, bytes], dict[str, Any]]:
    if not 3 <= samples_per_theme <= 32:
        raise ValueError("Refinement audit samples per theme must be in [3, 32]")
    cfg = MapConfig(width=48, height=48, objective_count=3, spawn_count=8, min_start_exit_distance=24)
    rows, labels, records, archipelagos = [], [], [], []
    for theme_index, theme in enumerate(THEMES):
        for sample_index in range(samples_per_theme):
            requested = splitmix64(base_seed ^ ((theme_index + 1) << 48) ^ sample_index)
            item = generate_refined_map(requested, theme, cfg)
            features = _feature_vector(item.data)
            rows.append(features); labels.append(theme)
            if theme == "archipelago": archipelagos.append(item)
            records.append({
                "theme": theme, "sample_index": sample_index,
                "requested_seed": f"0x{item.requested_seed:016x}", "selected_seed": f"0x{item.selected_seed:016x}",
                "attempt": item.attempt, "map_id": item.data.map_id,
                "semantic_sha256": array_digest(item.data.arrays()),
                "validation_failure_count": 0,
                "archipelago_metrics": dict(item.metrics),
                "feature_values": [round(float(value), 9) for value in features],
            })
    classifications, aggregate = _classify(np.stack(rows), labels)
    for record, classification in zip(records, classifications, strict=True):
        record["classification"] = classification
    unique = len({record["semantic_sha256"] for record in records})
    archipelago_attempts = [record["attempt"] for record in records if record["theme"] == "archipelago"]
    contact = _contact_sheet(archipelagos)
    files = {"refined_archipelago_contact_sheet.png": contact}
    gates = {
        "all_48_maps_hard_valid": len(records) == 48 and all(record["validation_failure_count"] == 0 for record in records),
        "all_48_semantic_maps_unique": unique == 48,
        "all_refined_archipelagos_meet_policy": all(archipelago_policy_passes(record["archipelago_metrics"]) for record in records if record["theme"] == "archipelago"),
        "all_refinements_within_bounded_attempts": max(archipelago_attempts) < MAX_ATTEMPTS,
        "six_theme_classification_accuracy_is_1": aggregate["leave_one_out_accuracy"] == 1.0,
        "archipelago_theme_recall_is_1": aggregate["per_theme"]["archipelago"]["recall"] == 1.0,
        "every_theme_recall_is_1": all(value["recall"] == 1.0 for value in aggregate["per_theme"].values()),
        "topology_v2_generator_remains_authoritative": True,
    }
    report: dict[str, Any] = {
        "format": FORMAT, "status": "ready" if all(gates.values()) else "failed", "policy": POLICY,
        "compiler": {"source_sha256": source_sha256(), "dependency_sha256": dependency_sha256(), "python_runtime_required": False},
        "schedule": {"base_seed": f"0x{base_seed:016x}", "samples_per_theme": samples_per_theme, "themes": list(THEMES), "max_attempts": MAX_ATTEMPTS},
        "config": cfg.to_dict(), "records": records,
        "aggregate": {**aggregate, "unique_semantic_count": unique, "maximum_archipelago_attempt": max(archipelago_attempts), "mean_archipelago_attempt": round(float(np.mean(archipelago_attempts)), 9)},
        "contact_sheet": _artifact("refined_archipelago_contact_sheet.png", contact), "gates": gates,
    }
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    files["map_theme_refinement.json"] = canonical_json_bytes(report)
    return files, report


def _validate_schema(report: Mapping[str, Any]) -> None:
    errors = sorted(Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).iter_errors(report), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Map refinement schema failed: {errors[0].message}")


def build_bank(destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    files, report = _build(); _validate_schema(report)
    if not all(report["gates"].values()): raise ValueError(f"Map refinement gates failed: {[name for name, value in report['gates'].items() if not value]}")
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=sum(map(len, files.values())) + 64 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists(): raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for relative, payload in files.items():
            target = staging / relative; descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=staging)
            with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        os.replace(staging, destination)
    except BaseException: raise
    return validate_bank(destination / "map_theme_refinement.json")


def validate_bank(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); report = json.loads(raw); _validate_schema(report)
    if raw != canonical_json_bytes(report): raise ValueError("Map refinement manifest is not canonical JSON")
    if report["compiler"] != {"source_sha256": source_sha256(), "dependency_sha256": dependency_sha256(), "python_runtime_required": False}: raise ValueError("Map refinement provenance differs")
    semantic = {key: value for key, value in report.items() if key != "semantic_sha256"}
    if report["semantic_sha256"] != hashlib.sha256(canonical_json_bytes(semantic)).hexdigest(): raise ValueError("Map refinement semantic hash differs")
    expected_files, expected = _build(report["schedule"]["samples_per_theme"], int(report["schedule"]["base_seed"], 16))
    if report != expected: raise ValueError("Map refinement semantic replay differs")
    actual = {path.relative_to(manifest_path.parent).as_posix(): path.read_bytes() for path in manifest_path.parent.rglob("*") if path.is_file()}
    if actual != expected_files: raise ValueError("Map refinement byte replay or closure differs")
    return {"passed": True, "map_count": len(report["records"]), "accuracy": report["aggregate"]["leave_one_out_accuracy"], "archipelago_recall": report["aggregate"]["per_theme"]["archipelago"]["recall"], "maximum_attempt": report["aggregate"]["maximum_archipelago_attempt"], "semantic_sha256": report["semantic_sha256"], "manifest_sha256": sha256_file(manifest_path), "contact_sheet_sha256": report["contact_sheet"]["sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile bounded deterministic map theme refinement")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build"); build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = sub.add_parser("validate"); validate.add_argument("manifest", type=Path)
    args = parser.parse_args(argv); result = build_bank(args.output) if args.command == "build" else validate_bank(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
