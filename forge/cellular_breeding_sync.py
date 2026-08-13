from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from .cellular_breeding import validate_bank
from .cellular_organism.compiler import _load_arrays
from .cellular_organism.contract import DISK_FLOOR_GIB, TISSUE_NAMES
from .cellular_organism_sync import CATALOG_FORMAT, RUNTIME_FORMAT, _runtime_species
from .config import PROJECT_ROOT
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


DEFAULT_SOURCE = PROJECT_ROOT / "outputs/cellular_breeding_v1/cellular_breeding_manifest.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/cellular_breeding/v1"


def _source_registry() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "forge/cellular_breeding/contract.py",
        PROJECT_ROOT / "forge/cellular_breeding/compiler.py",
        PROJECT_ROOT / "forge/cellular_organism_sync.py",
        PROJECT_ROOT / "forge/cellular_organism/compiler.py",
        PROJECT_ROOT / "forge/cellular_organism/simulation.py",
        PROJECT_ROOT / "shared/schema/cellular_breeding_bank.schema.json",
        PROJECT_ROOT / "game/CellularBreedingLab.tscn",
        PROJECT_ROOT / "game/scripts/cellular_organism_lab.gd",
    )
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def project_runtime(source_manifest: Path) -> dict[str, bytes]:
    source_manifest = Path(source_manifest).resolve()
    validation = validate_bank(source_manifest)
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    root = source_manifest.parent
    files: dict[str, bytes] = {}
    species: list[dict[str, object]] = []
    for record in source["offspring"]:
        arrays_path = root.joinpath(*PurePosixPath(record["arrays"]["path"]).parts)
        arrays = _load_arrays(arrays_path)
        runtime = _runtime_species(record, arrays)
        runtime["lineage"] = record["lineage"]
        runtime["parents"] = record["parents"]
        runtime["breeding"] = record["breeding"]
        runtime["capabilities"] = record["capabilities"]
        relative = f"offspring/{int(record['ordinal']):04d}_{record['sample_id']}.json"
        payload = canonical_json_bytes(runtime)
        files[relative] = payload
        species.append(
            {
                "sample_id": record["sample_id"],
                "ordinal": record["ordinal"],
                "family": record["family"],
                "family_id": record["family_id"],
                "subtype": record["subtype"],
                "role": record["role"],
                "family_pair": record["family_pair"],
                "lineage": record["lineage"],
                "breeding": record["breeding"],
                "runtime": {"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)},
                "summary": record["summary"],
            }
        )
    contact = root.joinpath(*PurePosixPath(source["contact_sheet"]["path"]).parts).read_bytes()
    files["cellular_breeding_contact_sheet.png"] = contact
    registry = _source_registry()
    catalog: dict[str, object] = {
        "format": CATALOG_FORMAT,
        "status": "ready",
        "bundle_version": 1,
        "bank_kind": "two-parent-structural-cellular-breeding",
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_semantic_sha256": source["semantic_sha256"],
        "sync_source_manifest": registry,
        "sync_source_sha256": sha256_bytes(canonical_json_bytes(registry)),
        "sample_count": len(species),
        "family_counts": source["family_counts"],
        "family_pair_counts": source["family_pair_counts"],
        "crossover_modes": source["crossover_modes"],
        "mutation_modes": source["mutation_modes"],
        "totals": source["totals"],
        "simulation": source["simulation"],
        "tissues": list(TISSUE_NAMES),
        "contact_sheet": {"path": "cellular_breeding_contact_sheet.png", "bytes": len(contact), "sha256": sha256_bytes(contact)},
        "species": species,
        "validation": validation,
        "runtime_contract": {
            "python_required": False,
            "all_45_structural_offspring_projected": True,
            "two_parent_lineage_visible": True,
            "offspring_anatomy_is_forge_decoded": True,
            "cell_and_bond_totals_exact": True,
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
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        os.replace(staging, destination)
    except BaseException:
        raise


def validate_runtime(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve()
    catalog_path = destination / "catalog.json"
    raw_catalog = catalog_path.read_bytes()
    catalog = json.loads(raw_catalog)
    if raw_catalog != canonical_json_bytes(catalog):
        raise ValueError("Cellular breeding native catalog is not canonical JSON")
    if catalog.get("format") != CATALOG_FORMAT or catalog.get("status") != "ready" or catalog.get("bank_kind") != "two-parent-structural-cellular-breeding":
        raise ValueError("Cellular breeding native catalog header differs")
    if catalog.get("sample_count") != 45 or len(catalog.get("species", [])) != 45:
        raise ValueError("Cellular breeding native species census differs")
    registry = _source_registry()
    if catalog.get("sync_source_manifest") != registry or catalog.get("sync_source_sha256") != sha256_bytes(canonical_json_bytes(registry)):
        raise ValueError("Cellular breeding native source provenance differs")
    expected_bundle = sha256_bytes(canonical_json_bytes({key: value for key, value in catalog.items() if key != "bundle_id"}))
    if catalog.get("bundle_id") != expected_bundle:
        raise ValueError("Cellular breeding native bundle identity differs")
    contact = catalog.get("contact_sheet", {})
    contact_path = destination.joinpath(*PurePosixPath(str(contact.get("path", ""))).parts).resolve()
    if not contact_path.is_relative_to(destination) or not contact_path.is_file() or contact_path.stat().st_size != contact.get("bytes") or sha256_file(contact_path) != contact.get("sha256"):
        raise ValueError("Cellular breeding native contact-sheet integrity differs")
    cells = bonds = organs = 0
    family_pairs: set[str] = set()
    expected_files = {"catalog.json", str(contact["path"])}
    for entry in catalog["species"]:
        artifact = entry["runtime"]
        path = destination.joinpath(*PurePosixPath(artifact["path"]).parts).resolve()
        if not path.is_relative_to(destination) or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("Cellular breeding native artifact integrity differs")
        runtime = json.loads(path.read_text(encoding="utf-8"))
        if path.read_bytes() != canonical_json_bytes(runtime):
            raise ValueError("Cellular breeding native runtime JSON is not canonical")
        if runtime.get("format") != RUNTIME_FORMAT or runtime.get("sample_id") != entry["sample_id"]:
            raise ValueError("Cellular breeding native runtime identity differs")
        if runtime.get("lineage") != entry["lineage"] or runtime.get("breeding") != entry["breeding"]:
            raise ValueError("Cellular breeding native lineage differs")
        if runtime["genome"].get("structural_lineage", {}).get("parent_ids") != entry["lineage"]["parent_ids"]:
            raise ValueError("Cellular breeding native genome parent lineage differs")
        count = len(runtime["arrays"]["position_xy"]); edge_count = len(runtime["arrays"]["bond_ab"])
        if count != runtime["summary"]["physical_cell_count"] or edge_count != runtime["summary"]["bond_count"]:
            raise ValueError("Cellular breeding native cell/bond census differs")
        cells += count; bonds += edge_count; organs += len(runtime["organs"])
        family_pairs.add("+".join(entry["family_pair"]))
        expected_files.add(str(artifact["path"]))
    if (cells, bonds, organs) != (catalog["totals"]["physical_cells"], catalog["totals"]["bonds"], catalog["totals"]["organs"]):
        raise ValueError("Cellular breeding native aggregate census differs")
    if len(family_pairs) != 15:
        raise ValueError("Cellular breeding native family-pair coverage differs")
    actual_files = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Cellular breeding native output closure differs")
    return {"passed": True, "sample_count": 45, "cell_count": cells, "bond_count": bonds, "organ_count": organs, "family_pair_count": 15, "bundle_id": catalog["bundle_id"]}


def sync_runtime(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION, *, repeat_check: bool = True) -> dict[str, object]:
    first = project_runtime(source)
    second = project_runtime(source) if repeat_check else first
    differences = [name for name in sorted(set(first) | set(second)) if first.get(name) != second.get(name)]
    if differences:
        raise ValueError(f"Cellular breeding native projection is not exact: {differences[:5]}")
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
        "sample_count": 45,
        "bundle_id": validation["bundle_id"],
        "tree_sha256": tree.hexdigest(),
        "repeat_exact": not differences,
        "runtime_validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project structural cellular offspring into native Godot JSON")
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
