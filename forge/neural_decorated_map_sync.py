from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from .config import PROJECT_ROOT
from .maps.io import file_sha256
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .neural_decorated_maps.compiler import validate_bank
from .safety import require_disk_floor


FORMAT = "nullvector-neural-decorated-map-native-catalog/1.0.0"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs/neural_decorated_maps_v1_verified"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/neural_decorated_maps/v1_1"
SELECTION_AUDIT = PROJECT_ROOT / "outputs/map_decorator_production_v4_selection/protected_selection_audit_v1"
CALIBRATION = PROJECT_ROOT / "outputs/map_decorator_production_v4_calibration/calibration_100step_v1"
CORPUS = PROJECT_ROOT / "outputs/map_decorator_corpus_v1"
FOREGROUND_INDEX = PROJECT_ROOT / "outputs/map_decorator_production_v2/foreground_index_v2"
MAPS = PROJECT_ROOT / "outputs/maps_v2_forge_lab"


def _source_registry() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "forge/neural_decorated_maps/contract.py",
        PROJECT_ROOT / "forge/neural_decorated_maps/compiler.py",
        PROJECT_ROOT / "forge/neural_decorated_maps/renderer.py",
        PROJECT_ROOT / "game/NeuralDecoratedMapLab.tscn",
        PROJECT_ROOT / "game/scripts/neural_decorated_map_lab.gd",
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _validate_source(source: Path) -> dict[str, object]:
    report = validate_bank(
        source,
        selection_audit=SELECTION_AUDIT,
        calibration_root=CALIBRATION,
        corpus_root=CORPUS,
        index_root=FOREGROUND_INDEX,
        map_root=MAPS,
    )
    if report.get("status") != "passed" or not bool(report.get("visual", {}).get("visually_inspected")):
        raise ValueError("Native projection requires a visually inspected passing neural map bank.")
    for entry in report.get("maps", []):
        authority = entry.get("selection", {}).get("field_authority", {})
        expected = {
            "variant": "deterministic_semantic_teacher",
            "decal": "accepted_neural_protected_selector",
            "prop": "accepted_neural_protected_selector",
            "emission": "conditional_semantic_projection",
        }
        if authority != expected or entry.get("selection", {}).get("unsupported_neural_heads_cross_runtime_boundary") is not False:
            raise ValueError("Neural map field authority differs from the accepted hybrid contract.")
    return report


def project_runtime(source: Path = DEFAULT_SOURCE) -> dict[str, bytes]:
    source = Path(source).resolve()
    report = _validate_source(source)
    runtime_path = source / "runtime_index.json"
    atlas_path = source / "neural_map_atlas.png"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    registry = _source_registry()
    catalog: dict[str, object] = {
        "format": FORMAT,
        "status": "ready",
        "engine": "Godot 4.3",
        "source_bank_format": report["format"],
        "source_report_sha256": file_sha256(source / "bank_report.json"),
        "source_semantic_sha256": report["semantic_sha256"],
        "source_runtime_index_sha256": file_sha256(runtime_path),
        "compiler_source_sha256": report["compiler_source_sha256"],
        "contract_sha256": report["contract_sha256"],
        "selection_audit_sha256": report["selection_audit_sha256"],
        "ema_tensor_sha256": report["ema_tensor_sha256"],
        "visual_inspection_passed": True,
        "sync_source_manifest": registry,
        "sync_source_sha256": sha256_bytes(canonical_json_bytes(registry)),
        "themes": runtime["themes"],
        "layers": runtime["layers"],
        "theme_count": len(runtime["themes"]),
        "layer_count": len(runtime["layers"]),
        "atlas_frame_count": report["counts"]["atlas_frames"],
        "hazard_frames_per_theme": report["counts"]["hazard_frames_per_theme"],
        "atlas": {
            "path": "neural_map_atlas.png",
            "bytes": atlas_path.stat().st_size,
            "sha256": file_sha256(atlas_path),
            "size": runtime["atlas_size"],
            "columns": runtime["columns"],
            "rows": runtime["rows"],
            "cell_size": runtime["cell_size"],
        },
        "maps": runtime["maps"],
        "pixel_filter": "nearest",
        "runtime_asset_extensions": [".json", ".png"],
        "python_runtime_required": False,
        "checkpoint_shipped": False,
    }
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog))
    return {
        "catalog.json": canonical_json_bytes(catalog),
        "neural_map_atlas.png": atlas_path.read_bytes(),
    }


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=sum(map(len, files.values())) + 256 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    for relative, payload in sorted(files.items()):
        target = staging.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    os.replace(staging, destination)


def validate_runtime(destination: Path = DEFAULT_DESTINATION) -> dict[str, object]:
    destination = Path(destination).resolve()
    raw = (destination / "catalog.json").read_bytes()
    catalog = json.loads(raw)
    if raw != canonical_json_bytes(catalog) or catalog.get("format") != FORMAT or catalog.get("status") != "ready":
        raise ValueError("Neural decorated map native catalog header/canonical JSON differs.")
    registry = _source_registry()
    if catalog.get("sync_source_manifest") != registry or catalog.get("sync_source_sha256") != sha256_bytes(canonical_json_bytes(registry)):
        raise ValueError("Neural decorated map native source provenance differs.")
    if catalog.get("bundle_id") != sha256_bytes(canonical_json_bytes({key: value for key, value in catalog.items() if key != "bundle_id"})):
        raise ValueError("Neural decorated map native bundle identity differs.")
    if (catalog.get("theme_count"), catalog.get("layer_count"), catalog.get("atlas_frame_count")) != (6, 8, 90):
        raise ValueError("Neural decorated map native census differs.")
    if catalog.get("python_runtime_required") is not False or catalog.get("checkpoint_shipped") is not False:
        raise ValueError("Neural decorated map native runtime boundary differs.")
    expected = project_runtime(DEFAULT_SOURCE)
    actual_files = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual_files != set(expected):
        raise ValueError("Neural decorated map native output closure differs.")
    for relative, payload in expected.items():
        if (destination / relative).read_bytes() != payload:
            raise ValueError(f"Neural decorated map native artifact replay differs: {relative}")
    return {
        "passed": True,
        "theme_count": 6,
        "layer_count": 8,
        "atlas_frame_count": 90,
        "bundle_id": catalog["bundle_id"],
        "bytes": sum(map(len, expected.values())),
    }


def sync_runtime(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION) -> dict[str, object]:
    first = project_runtime(source)
    second = project_runtime(source)
    if first != second:
        raise ValueError("Neural decorated map native projection is not exact.")
    _publish(Path(destination), first)
    validation = validate_runtime(destination)
    tree = hashlib.sha256()
    for name, payload in sorted(first.items()):
        tree.update(name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    return {
        "passed": True,
        "source": str(Path(source).resolve()),
        "destination": str(Path(destination).resolve()),
        "file_count": len(first),
        "bytes": sum(map(len, first.values())),
        "repeat_exact": True,
        "tree_sha256": tree.hexdigest(),
        "runtime_validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project verified neural decorated maps into native Godot assets")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = sync_runtime(args.source, args.destination)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
