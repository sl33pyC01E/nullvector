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
from .neural_world_synthesis_v1 import validate_world_bank
from .safety import require_disk_floor


FORMAT = "nullvector-neural-world-native-bundle-v1/1.0.0"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs/neural_world_synthesis_v1/build_004"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/neural_world_synthesis/v1"


def _sync_source_registry() -> dict[str, str]:
    files = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "forge/neural_world_synthesis_v1/contract.py",
        PROJECT_ROOT / "forge/neural_world_synthesis_v1/build.py",
        PROJECT_ROOT / "forge/neural_world_synthesis_v1/map_pack.py",
        PROJECT_ROOT / "forge/neural_world_synthesis_v1/decorator.py",
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def project_runtime(source: Path = DEFAULT_SOURCE) -> dict[str, bytes]:
    source = Path(source).resolve()
    synthesis = validate_world_bank(source)
    if synthesis.get("status") != "experimental_ready" or not all(synthesis.get("gates", {}).values()):
        raise ValueError("Neural world synthesis is not safe to project.")
    decorated = source / "decorated_bank"
    runtime_path = decorated / "runtime_index.json"
    atlas_path = decorated / "neural_map_atlas.png"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("format") != "nullvector-composed-neural-map-runtime-v1/1.0.0":
        raise ValueError("Composed runtime index format differs.")
    if runtime.get("themes") != ["arena", "rooms", "caves", "archipelago", "garden", "anomaly"]:
        raise ValueError("Composed runtime theme order differs.")
    if runtime.get("python_runtime_required") is not False:
        raise ValueError("Composed runtime crossed the Python boundary.")
    registry = _sync_source_registry()
    catalog: dict[str, object] = {
        "format": FORMAT,
        "status": "experimental_ready",
        "engine": "Godot 4.3",
        "source_synthesis_manifest_sha256": file_sha256(source / "synthesis_manifest.json"),
        "source_synthesis_identity_sha256": synthesis["manifest_sha256"],
        "source_synthesis_source_sha256": synthesis["source_sha256"],
        "source_decorator_report_sha256": synthesis["decorator_report_sha256"],
        "prior_checkpoint_sha256": synthesis["prior_checkpoint_sha256"],
        "raw_required_reachable_rate": synthesis["aggregate"]["raw_required_reachable_rate"],
        "raw_radius_one_required_reachable_rate": synthesis["aggregate"]["raw_radius_one_required_reachable_rate"],
        "mean_repair_fraction": synthesis["aggregate"]["mean_repair_fraction"],
        "maximum_repair_fraction": synthesis["aggregate"]["maximum_repair_fraction"],
        "safety_compiler_authoritative": True,
        "runtime": synthesis["runtime"],
        "sync_source_manifest": registry,
        "sync_source_sha256": sha256_bytes(canonical_json_bytes(registry)),
        "themes": runtime["themes"],
        "layers": runtime["layers"],
        "maps": runtime["maps"],
        "theme_count": len(runtime["themes"]),
        "layer_count": len(runtime["layers"]),
        "atlas_frame_count": sum(sum(layer["frame_count"] for layer in entry["layers"]) for entry in runtime["maps"]),
        "atlas": {
            "path": "neural_world_atlas.png",
            "bytes": atlas_path.stat().st_size,
            "sha256": file_sha256(atlas_path),
            "size": runtime["atlas_size"],
            "columns": runtime["columns"],
            "rows": runtime["rows"],
            "cell_size": runtime["cell_size"],
        },
        "pixel_filter": "nearest",
        "world_synthesis_in_frame_loop": False,
        "python_runtime_required": False,
        "cuda_runtime_required": False,
        "checkpoint_shipped": False,
    }
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog))
    return {
        "catalog.json": canonical_json_bytes(catalog),
        "neural_world_atlas.png": atlas_path.read_bytes(),
    }


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=sum(map(len, files.values())) + 256 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for relative, payload in sorted(files.items()):
            target = staging.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        os.replace(staging, destination)
    except BaseException:
        raise


def validate_runtime(destination: Path = DEFAULT_DESTINATION, source: Path = DEFAULT_SOURCE) -> dict[str, object]:
    destination = Path(destination).resolve()
    raw = (destination / "catalog.json").read_bytes()
    catalog = json.loads(raw)
    if raw != canonical_json_bytes(catalog) or catalog.get("format") != FORMAT:
        raise ValueError("Neural world native catalog is not canonical or has the wrong format.")
    if catalog.get("bundle_id") != sha256_bytes(canonical_json_bytes({key: value for key, value in catalog.items() if key != "bundle_id"})):
        raise ValueError("Neural world native bundle identity differs.")
    if (catalog.get("theme_count"), catalog.get("layer_count"), catalog.get("atlas_frame_count")) != (6, 8, 90):
        raise ValueError("Neural world native census differs.")
    if catalog.get("world_synthesis_in_frame_loop") is not False or catalog.get("python_runtime_required") is not False or catalog.get("cuda_runtime_required") is not False:
        raise ValueError("Neural world native runtime boundary differs.")
    if catalog.get("runtime", {}).get("display_target_fps") != 30 or catalog.get("runtime", {}).get("embodied_motion_hz") != 30:
        raise ValueError("Neural world native cadence contract differs.")
    expected = project_runtime(source)
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual != set(expected):
        raise ValueError("Neural world native artifact closure differs.")
    for relative, payload in expected.items():
        if (destination / relative).read_bytes() != payload:
            raise ValueError(f"Neural world native artifact replay differs: {relative}")
    return {
        "passed": True,
        "bundle_id": catalog["bundle_id"],
        "theme_count": 6,
        "layer_count": 8,
        "atlas_frame_count": 90,
        "bytes": sum(map(len, expected.values())),
    }


def sync_runtime(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION) -> dict[str, object]:
    first = project_runtime(source)
    second = project_runtime(source)
    if first != second:
        raise ValueError("Neural world native projection is not exact.")
    _publish(Path(destination), first)
    validation = validate_runtime(destination, source)
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
    parser = argparse.ArgumentParser(description="Project composed neural worlds into native runtime assets")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    report = validate_runtime(args.destination, args.source) if args.validate else sync_runtime(args.source, args.destination)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
