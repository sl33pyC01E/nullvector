from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

import numpy as np

from .cellular_trauma import validate_bank
from .cellular_trauma.compiler import _load_trauma
from .cellular_trauma.contract import FORMAT as BANK_FORMAT
from .config import PROJECT_ROOT
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-cellular-trauma-runtime-v3"
CATALOG_FORMAT = "nullvector-cellular-trauma-native-catalog-v3"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs/cellular_trauma_v3/cellular_trauma_manifest.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/cellular_trauma/v7"


def _source_registry() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(), PROJECT_ROOT / "forge/cellular_trauma/contract.py", PROJECT_ROOT / "forge/cellular_trauma/compiler.py", PROJECT_ROOT / "forge/cellular_trauma/simulation.py", PROJECT_ROOT / "shared/schema/cellular_trauma_bank.schema.json", PROJECT_ROOT / "game/scripts/cellular_motion_lab.gd", PROJECT_ROOT / "game/scripts/cellular_organism_lab.gd", PROJECT_ROOT / "game/CellularMotionLab.tscn",
    )
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def project_runtime(source_manifest: Path = DEFAULT_SOURCE) -> dict[str, bytes]:
    source_manifest = Path(source_manifest).resolve(); validation = validate_bank(source_manifest); source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source["format"] != BANK_FORMAT: raise ValueError("Unexpected trauma bank format")
    files: dict[str, bytes] = {}; identities = []
    for record in source["identities"]:
        count = int(record["profile"]["cell_count"]); bonds = int(record["profile"]["bond_count"]); path = source_manifest.parent.joinpath(*PurePosixPath(record["arrays"]["path"]).parts); arrays = _load_trauma(path, count, bonds)
        runtime = {"format": FORMAT, "sample_id": record["sample_id"], "source_anatomy_sha256": record["source_anatomy_sha256"], "source_physiology_sha256": record["source_physiology_sha256"], "profile": record["profile"], "arrays": {name: arrays[name].astype(int).tolist() if arrays[name].dtype == np.uint8 else [round(float(value), 7) for value in arrays[name]] for name in sorted(arrays)}}
        relative = f"identities/{record['sample_id']}.json"; payload = canonical_json_bytes(runtime); files[relative] = payload; identities.append({"sample_id": record["sample_id"], "ordinal": record["ordinal"], "family": record["family"], "family_id": record["family_id"], "runtime": {"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)}})
    contact = (source_manifest.parent / source["contact_sheet"]["path"]).read_bytes(); files["cellular_trauma_contact_sheet.png"] = contact; registry = _source_registry()
    catalog: dict[str, object] = {"format": CATALOG_FORMAT, "status": "ready", "bundle_version": 3, "source_manifest_sha256": sha256_file(source_manifest), "source_semantic_sha256": source["semantic_sha256"], "sync_source_manifest": registry, "sync_source_sha256": sha256_bytes(canonical_json_bytes(registry)), "identity_count": 45, "total_cells": source["total_cells"], "total_bonds": source["total_bonds"], "identities": identities, "contact_sheet": {"path": "cellular_trauma_contact_sheet.png", "bytes": len(contact), "sha256": sha256_bytes(contact)}, "runtime_contract": source["runtime_contract"], "validation": validation}
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog)); files["catalog.json"] = canonical_json_bytes(catalog); return files


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = destination.resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=sum(map(len, files.values())) + 512 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"; staging.mkdir(parents=True)
    for relative, payload in sorted(files.items()):
        target = staging.joinpath(*PurePosixPath(relative).parts); target.parent.mkdir(parents=True, exist_ok=True); descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, target)
    os.replace(staging, destination)


def validate_runtime(destination: Path = DEFAULT_DESTINATION) -> dict[str, object]:
    destination = Path(destination).resolve(); raw = (destination / "catalog.json").read_bytes(); catalog = json.loads(raw)
    if raw != canonical_json_bytes(catalog) or catalog.get("format") != CATALOG_FORMAT or catalog.get("status") != "ready": raise ValueError("Trauma native catalog header differs")
    registry = _source_registry()
    if catalog["sync_source_manifest"] != registry or catalog["sync_source_sha256"] != sha256_bytes(canonical_json_bytes(registry)): raise ValueError("Trauma native source provenance differs")
    if catalog["bundle_id"] != sha256_bytes(canonical_json_bytes({key: value for key, value in catalog.items() if key != "bundle_id"})): raise ValueError("Trauma native bundle identity differs")
    if catalog["identity_count"] != 45: raise ValueError("Trauma native census differs")
    for identity in catalog["identities"]:
        artifact = identity["runtime"]; path = destination.joinpath(*PurePosixPath(artifact["path"]).parts).resolve(); path.relative_to(destination)
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]: raise ValueError("Trauma native identity integrity differs")
        data = json.loads(path.read_text(encoding="utf-8")); count = int(data["profile"]["cell_count"]); bonds = int(data["profile"]["bond_count"])
        if data["format"] != FORMAT or data["sample_id"] != identity["sample_id"] or len(data["arrays"]["heal_class"]) != count or len(data["arrays"]["bond_repair_weight"]) != bonds: raise ValueError("Trauma native identity contract differs")
    expected = project_runtime(DEFAULT_SOURCE)
    for relative, payload in expected.items():
        if not (destination / relative).is_file() or (destination / relative).read_bytes() != payload: raise ValueError(f"Trauma native exact replay differs: {relative}")
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual != set(expected): raise ValueError("Trauma native output closure differs")
    return {"passed": True, "identity_count": 45, "total_cells": catalog["total_cells"], "total_bonds": catalog["total_bonds"], "bundle_id": catalog["bundle_id"], "bytes": sum(map(len, expected.values()))}


def sync_runtime(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION, *, repeat_check: bool = True) -> dict[str, object]:
    first = project_runtime(source); second = project_runtime(source) if repeat_check else first
    if first != second: raise ValueError("Trauma native projection is not exact")
    _publish(Path(destination), first); validation = validate_runtime(destination); tree = hashlib.sha256()
    for name, payload in sorted(first.items()): tree.update(name.encode() + b"\0" + hashlib.sha256(payload).digest())
    return {"passed": True, "source": str(Path(source).resolve()), "destination": str(Path(destination).resolve()), "file_count": len(first), "bytes": sum(map(len, first.values())), "repeat_exact": True, "tree_sha256": tree.hexdigest(), "runtime_validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project cellular trauma into native Godot JSON"); parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE); parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION); parser.add_argument("--no-repeat-check", action="store_true"); parser.add_argument("--report", type=Path); args = parser.parse_args(argv)
    report = sync_runtime(args.source, args.destination, repeat_check=not args.no_repeat_check)
    if args.report: args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
