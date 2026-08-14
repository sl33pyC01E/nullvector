from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from .cellular_ontogeny import validate_bank
from .cellular_ontogeny.compiler import _load_program_arrays
from .cellular_ontogeny.contract import DEFAULT_OUTPUT
from .cellular_organism.contract import DISK_FLOOR_GIB
from .config import PROJECT_ROOT
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-cellular-ontogeny-native-catalog-v3"
DEFAULT_SOURCE = DEFAULT_OUTPUT / "cellular_ontogeny_manifest.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/cellular_ontogeny/v6"


def _source_registry() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(), PROJECT_ROOT / "forge/cellular_ontogeny/contract.py", PROJECT_ROOT / "forge/cellular_ontogeny/compiler.py",
        PROJECT_ROOT / "shared/schema/cellular_ontogeny_bank.schema.json", PROJECT_ROOT / "game/CellularOntogenyLab.tscn",
        PROJECT_ROOT / "game/scripts/cellular_ontogeny_lab.gd", PROJECT_ROOT / "game/scripts/cellular_ecology_lab.gd",
        PROJECT_ROOT / "game/scripts/cellular_motion_lab.gd",
    )
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def project_runtime(source_manifest: Path) -> dict[str, bytes]:
    source_manifest = Path(source_manifest).resolve(); validation = validate_bank(source_manifest); source = json.loads(source_manifest.read_text(encoding="utf-8")); programs = []
    for record in source["programs"]:
        path = source_manifest.parent.joinpath(*PurePosixPath(record["artifact"]["path"]).parts); arrays = _load_program_arrays(path, int(record["cell_count"]), int(record["bond_count"]))
        programs.append({
            "sample_id": record["sample_id"], "ordinal": record["ordinal"], "family": record["family"], "family_id": record["family_id"],
            "development_seconds": record["development_seconds"], "stages": record["stages"], "metrics": record["metrics"],
            "birth_order": arrays["birth_order"].tolist(), "activation_stage": arrays["activation_stage"].tolist(), "parent_cell": arrays["parent_cell"].tolist(),
            "lineage_id": arrays["lineage_id"].tolist(), "differentiation_time": arrays["differentiation_time"].round(8).tolist(),
            "bond_activation_stage": arrays["bond_activation_stage"].tolist(), "morphogen_lr": arrays["morphogen_lr"].round(8).tolist(),
            "morphogen_ap": arrays["morphogen_ap"].round(8).tolist(), "morphogen_core": arrays["morphogen_core"].round(8).tolist(),
            "array_sha256": record["artifact"]["array_sha256"],
        })
    registry = _source_registry(); catalog: dict[str, object] = {
        "format": FORMAT, "status": "ready", "bundle_version": 3, "source_manifest_sha256": sha256_file(source_manifest), "source_semantic_sha256": source["semantic_sha256"],
        "sync_source_manifest": registry, "sync_source_sha256": sha256_bytes(canonical_json_bytes(registry)), "program_count": 45, "stages": source["stages"], "lineages": source["lineages"],
        "programs": programs, "totals": source["totals"], "runtime_contract": source["runtime_contract"], "validation": validation,
    }
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog)); contact = (source_manifest.parent / source["contact_sheet"]["path"]).read_bytes()
    return {"ontogeny_catalog.json": canonical_json_bytes(catalog), "cellular_ontogeny_contact_sheet.png": contact}


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
    destination = Path(destination).resolve(); path = destination / "ontogeny_catalog.json"; raw = path.read_bytes(); catalog = json.loads(raw)
    if raw != canonical_json_bytes(catalog) or catalog.get("format") != FORMAT or catalog.get("status") != "ready": raise ValueError("Ontogeny native catalog header/canonical JSON differs")
    registry = _source_registry()
    if catalog.get("sync_source_manifest") != registry or catalog.get("sync_source_sha256") != sha256_bytes(canonical_json_bytes(registry)): raise ValueError("Ontogeny native source provenance differs")
    if catalog.get("bundle_id") != sha256_bytes(canonical_json_bytes({key: value for key, value in catalog.items() if key != "bundle_id"})): raise ValueError("Ontogeny native bundle differs")
    if catalog.get("program_count") != 45 or len(catalog.get("programs", [])) != 45: raise ValueError("Ontogeny native census differs")
    expected = project_runtime(DEFAULT_SOURCE)
    for relative, payload in expected.items():
        if not (destination / relative).is_file() or (destination / relative).read_bytes() != payload: raise ValueError(f"Ontogeny native replay differs: {relative}")
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual != set(expected): raise ValueError("Ontogeny native output closure differs")
    return {"passed": True, "program_count": 45, "bundle_id": catalog["bundle_id"], "bytes": sum(map(len, expected.values()))}


def sync_runtime(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION) -> dict[str, object]:
    first = project_runtime(source); second = project_runtime(source)
    if first != second: raise ValueError("Ontogeny native projection is not exact")
    _publish(Path(destination), first); validation = validate_runtime(destination); tree = hashlib.sha256()
    for name, payload in sorted(first.items()): tree.update(name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    return {"passed": True, "destination": str(Path(destination).resolve()), "file_count": len(first), "bytes": sum(map(len, first.values())), "repeat_exact": True, "tree_sha256": tree.hexdigest(), "runtime_validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project cellular ontogeny into native Godot JSON"); parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE); parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION); parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv); report = sync_runtime(args.source, args.destination)
    if args.report: args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
