from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

import numpy as np

from .cellular_ecology import validate_bank
from .cellular_ecology.compiler import _load_npz
from .cellular_ecology.contract import DEFAULT_OUTPUT
from .cellular_organism.contract import DISK_FLOOR_GIB
from .config import PROJECT_ROOT
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-cellular-ecology-native-catalog-v2"
DEFAULT_SOURCE = DEFAULT_OUTPUT / "cellular_ecology_manifest.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/cellular_ecology/v2"


def _source_registry() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(), PROJECT_ROOT / "forge/cellular_ecology/contract.py", PROJECT_ROOT / "forge/cellular_ecology/compiler.py",
        PROJECT_ROOT / "shared/schema/cellular_ecology_bank.schema.json", PROJECT_ROOT / "game/CellularEcologyLab.tscn",
        PROJECT_ROOT / "game/scripts/cellular_ecology_lab.gd", PROJECT_ROOT / "game/scripts/cellular_motion_lab.gd", PROJECT_ROOT / "game/scripts/cellular_organism_lab.gd",
    )
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _quantize(values: np.ndarray) -> list[int]:
    return np.asarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0), dtype=np.uint8).reshape(-1).tolist()


def project_runtime(source_manifest: Path) -> dict[str, bytes]:
    source_manifest = Path(source_manifest).resolve(); validation = validate_bank(source_manifest)
    source = json.loads(source_manifest.read_text(encoding="utf-8")); maps = []
    for record in source["maps"]:
        path = source_manifest.parent.joinpath(*PurePosixPath(record["artifact"]["path"]).parts); fields = _load_npz(path)
        maps.append({
            "map_id": record["map_id"], "theme": record["theme"], "dimensions": record["dimensions"],
            "source_semantic_sha256": record["source"]["semantic_array_sha256"], "ecology_array_sha256": record["ecology_array_sha256"],
            "statistics": record["statistics"], "family_habitats": record["family_habitats"], "resource_nodes": record["resource_nodes"],
            "fields_u8": {name: _quantize(fields[name]) for name in source["field_vocab"]},
            "family_suitability_u8": [_quantize(fields["family_suitability"][index]) for index in range(5)],
            "resource_type_u8": fields["resource_type"].reshape(-1).tolist(),
        })
    registry = _source_registry(); dependencies = {}
    for name, relative in {
        "organism": "game/generated/cellular_symmetry/v1/catalog.json",
        "motion": "game/generated/cellular_motion/v6/motion_catalog.json",
        "physiology": "game/generated/cellular_physiology/v5/catalog.json",
        "trauma": "game/generated/cellular_trauma/v2/catalog.json",
    }.items():
        path = PROJECT_ROOT / relative; data = json.loads(path.read_text(encoding="utf-8")); dependencies[name] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "bundle_id": data["bundle_id"]}
    catalog: dict[str, object] = {
        "format": FORMAT, "status": "ready", "bundle_version": 2,
        "source_manifest_sha256": sha256_file(source_manifest), "source_semantic_sha256": source["semantic_sha256"],
        "sync_source_manifest": registry, "sync_source_sha256": sha256_bytes(canonical_json_bytes(registry)),
        "map_count": 6, "resource_node_count": source["resource_node_count"], "family_vocab": source["family_vocab"],
        "resource_vocab": source["resource_vocab"], "field_vocab": source["field_vocab"], "maps": maps,
        "runtime_contract": source["runtime_contract"], "runtime_dependencies": dependencies, "validation": validation,
    }
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog))
    contact = (source_manifest.parent / source["contact_sheet"]["path"]).read_bytes()
    return {"ecology_catalog.json": canonical_json_bytes(catalog), "cellular_ecology_contact_sheet.png": contact}


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = destination.resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=sum(map(len, files.values())) + 256 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"; staging.mkdir(parents=True)
    for relative, payload in sorted(files.items()):
        target = staging.joinpath(*PurePosixPath(relative).parts); target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, target)
    os.replace(staging, destination)


def validate_runtime(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve(); path = destination / "ecology_catalog.json"; raw = path.read_bytes(); catalog = json.loads(raw)
    if raw != canonical_json_bytes(catalog) or catalog.get("format") != FORMAT or catalog.get("status") != "ready": raise ValueError("Ecology native catalog header/canonical JSON differs")
    registry = _source_registry()
    if catalog.get("sync_source_manifest") != registry or catalog.get("sync_source_sha256") != sha256_bytes(canonical_json_bytes(registry)): raise ValueError("Ecology native source provenance differs")
    if catalog.get("bundle_id") != sha256_bytes(canonical_json_bytes({key: value for key, value in catalog.items() if key != "bundle_id"})): raise ValueError("Ecology native bundle identity differs")
    if (catalog.get("map_count"), catalog.get("resource_node_count")) != (6, 120): raise ValueError("Ecology native census differs")
    for dependency in catalog.get("runtime_dependencies", {}).values():
        path = PROJECT_ROOT.joinpath(*PurePosixPath(dependency["path"]).parts)
        if path.stat().st_size != dependency["bytes"] or sha256_file(path) != dependency["sha256"] or json.loads(path.read_text(encoding="utf-8"))["bundle_id"] != dependency["bundle_id"]: raise ValueError("Ecology native dependency provenance differs")
    expected = project_runtime(DEFAULT_SOURCE)
    for relative, payload in expected.items():
        artifact = destination / relative
        if not artifact.is_file() or artifact.read_bytes() != payload: raise ValueError(f"Ecology native artifact replay differs: {relative}")
    actual = {item.relative_to(destination).as_posix() for item in destination.rglob("*") if item.is_file()}
    if actual != set(expected): raise ValueError("Ecology native output closure differs")
    return {"passed": True, "map_count": 6, "resource_node_count": 120, "bundle_id": catalog["bundle_id"], "bytes": sum(map(len, expected.values()))}


def sync_runtime(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION) -> dict[str, object]:
    first = project_runtime(source); second = project_runtime(source)
    if first != second: raise ValueError("Ecology native projection is not exact")
    _publish(Path(destination), first); validation = validate_runtime(destination)
    tree = hashlib.sha256()
    for name, payload in sorted(first.items()): tree.update(name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    return {"passed": True, "destination": str(Path(destination).resolve()), "file_count": len(first), "bytes": sum(map(len, first.values())), "repeat_exact": True, "tree_sha256": tree.hexdigest(), "runtime_validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project cellular ecology into native Godot JSON")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE); parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION); parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv); report = sync_runtime(args.source, args.destination)
    if args.report: args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
