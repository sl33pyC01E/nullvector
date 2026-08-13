from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

from .cellular_organism.compiler import _load_arrays
from .cellular_organism.contract import DISK_FLOOR_GIB, TISSUE_NAMES
from .cellular_organism_sync import CATALOG_FORMAT, RUNTIME_FORMAT, _runtime_species
from .config import PROJECT_ROOT
from .evolved_cellular_organism import validate_bank
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


DEFAULT_SOURCE = PROJECT_ROOT / "outputs/evolved_cellular_organism_v1/evolved_cellular_organism_manifest.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/evolved_cellular_organism/v1"


def _source_registry() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "forge/cellular_organism_sync.py",
        PROJECT_ROOT / "forge/cellular_organism/compiler.py",
        PROJECT_ROOT / "forge/cellular_organism/simulation.py",
        PROJECT_ROOT / "forge/evolved_cellular_organism/compiler.py",
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def project_runtime(source_manifest: Path) -> dict[str, bytes]:
    source_manifest = Path(source_manifest).resolve()
    validation = validate_bank(source_manifest)
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    root = source_manifest.parent
    files: dict[str, bytes] = {}
    catalog_species: list[dict[str, object]] = []
    for record in source["species"]:
        arrays_path = root.joinpath(*PurePosixPath(record["arrays"]["path"]).parts)
        arrays = _load_arrays(arrays_path)
        runtime = _runtime_species(record, arrays)
        runtime["lineage"] = record["lineage"]
        runtime["capabilities"] = record["capabilities"]
        relative = f"species/{int(record['ordinal']):04d}_{record['sample_id']}.json"
        payload = canonical_json_bytes(runtime)
        files[relative] = payload
        catalog_species.append(
            {
                "sample_id": record["sample_id"],
                "ordinal": record["ordinal"],
                "family": record["family"],
                "family_id": record["family_id"],
                "subtype": record["subtype"],
                "role": record["role"],
                "lineage": record["lineage"],
                "runtime": {"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)},
                "summary": record["summary"],
            }
        )
    contact = (root / source["contact_sheet"]["path"]).read_bytes()
    files["evolved_cellular_organism_contact_sheet.png"] = contact
    registry = _source_registry()
    catalog: dict[str, object] = {
        "format": CATALOG_FORMAT,
        "status": "ready",
        "bundle_version": 1,
        "bank_kind": "learned-latent-evolution-cellular",
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_semantic_sha256": source["semantic_sha256"],
        "source_evolution_sha256": source["source"]["evolution_sha256"],
        "sync_source_manifest": registry,
        "sync_source_sha256": sha256_bytes(canonical_json_bytes(registry)),
        "sample_count": len(catalog_species),
        "generation_counts": source["generation_counts"],
        "family_counts": source["family_counts"],
        "fusion_modes": source["fusion_modes"],
        "mutation_modes": source["mutation_modes"],
        "totals": source["totals"],
        "simulation": source["simulation"],
        "tissues": list(TISSUE_NAMES),
        "contact_sheet": {
            "path": "evolved_cellular_organism_contact_sheet.png",
            "bytes": len(contact),
            "sha256": sha256_bytes(contact),
        },
        "species": catalog_species,
        "validation": validation,
        "runtime_contract": {
            "python_required": False,
            "all_36_evolution_survivors_projected": True,
            "runtime_files_are_audit_bound_to_npz": True,
            "cell_and_bond_totals_exact": True,
            "neural_lineage_visible": True,
            "runtime_offspring_redecode": False,
        },
    }
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog))
    files["catalog.json"] = canonical_json_bytes(catalog)
    return files


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=sum(map(len, files.values())) + 512 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
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
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_runtime(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve()
    catalog_path = destination / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("format") != CATALOG_FORMAT or catalog.get("status") != "ready":
        raise ValueError("Evolved native catalog header is invalid")
    if catalog.get("bank_kind") != "learned-latent-evolution-cellular":
        raise ValueError("Evolved native catalog kind is invalid")
    if catalog.get("sample_count") != 36 or len(catalog.get("species", [])) != 36:
        raise ValueError("Evolved native species census differs")
    cells = bonds = organs = 0
    for entry in catalog["species"]:
        artifact = entry["runtime"]
        path = destination.joinpath(*PurePosixPath(artifact["path"]).parts).resolve()
        if not path.is_relative_to(destination) or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("Evolved native runtime artifact integrity differs")
        runtime = json.loads(path.read_text(encoding="utf-8"))
        if runtime.get("format") != RUNTIME_FORMAT or runtime.get("sample_id") != entry["sample_id"]:
            raise ValueError("Evolved native runtime identity differs")
        if runtime.get("lineage") != entry["lineage"] or runtime["genome"].get("neural_lineage", {}).get("lineage_sha256") != entry["lineage"]["lineage_sha256"]:
            raise ValueError("Evolved native lineage projection differs")
        count = len(runtime["arrays"]["position_xy"])
        edge_count = len(runtime["arrays"]["bond_ab"])
        if count != runtime["summary"]["physical_cell_count"] or edge_count != runtime["summary"]["bond_count"]:
            raise ValueError("Evolved native cell/bond census differs")
        cells += count
        bonds += edge_count
        organs += len(runtime["organs"])
    if cells != catalog["totals"]["physical_cells"] or bonds != catalog["totals"]["bonds"] or organs != catalog["totals"]["organs"]:
        raise ValueError("Evolved native aggregate census differs")
    return {"passed": True, "sample_count": 36, "cell_count": cells, "bond_count": bonds, "organ_count": organs, "bundle_id": catalog["bundle_id"]}


def sync_runtime(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION, *, repeat_check: bool = True) -> dict[str, object]:
    first = project_runtime(source)
    second = project_runtime(source) if repeat_check else first
    differences = [name for name in sorted(set(first) | set(second)) if first.get(name) != second.get(name)]
    if differences:
        raise ValueError(f"Evolved native projection is not exact: {differences[:5]}")
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
        "sample_count": 36,
        "bundle_id": validation["bundle_id"],
        "tree_sha256": tree.hexdigest(),
        "repeat_exact": not differences,
        "runtime_validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project evolved cellular descendants into native Godot JSON")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--no-repeat-check", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = sync_runtime(args.source, args.destination, repeat_check=not args.no_repeat_check)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
