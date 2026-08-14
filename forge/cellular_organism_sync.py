from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

import numpy as np

from .cellular_organism.compiler import _load_arrays, validate_bank
from .cellular_organism.contract import DISK_FLOOR_GIB, FORMAT, TISSUE_NAMES
from .cellular_organism.orientation import orientation_manifest, top_down_simulation_defaults, validate_orientation
from .config import PROJECT_ROOT
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


RUNTIME_FORMAT = "nullvector-cellular-organism-runtime-v1"
CATALOG_FORMAT = "nullvector-cellular-organism-native-catalog-v1"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs/cellular_organism_v1/cellular_organism_manifest.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/cellular_organism/v2"


def _rounded(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


def _runtime_species(record: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        "format": RUNTIME_FORMAT,
        "sample_id": record["sample_id"],
        "ordinal": record["ordinal"],
        "family": record["family"],
        "family_id": record["family_id"],
        "subtype": record["subtype"],
        "role": record["role"],
        "anatomy_sha256": record["anatomy_sha256"],
        "source_arrays_sha256": record["arrays"]["sha256"],
        "fluid": record["fluid"],
        "genome": record["genome"],
        "organs": record["organs"],
        "palette": record["palette"],
        "summary": record["summary"],
        "arrays": {
            "position_xy": arrays["position_xy"].astype(int).tolist(),
            "part_owner": arrays["part_owner"].astype(int).tolist(),
            "material": arrays["material"].astype(int).tolist(),
            "emission": arrays["emission"].astype(int).tolist(),
            "tissue": arrays["tissue"].astype(int).tolist(),
            "organ_id": arrays["organ_id"].astype(int).tolist(),
            "cell_flags": arrays["cell_flags"].astype(int).tolist(),
            "max_health": _rounded(arrays["max_health"]),
            "fluid_capacity": _rounded(arrays["fluid_capacity"]),
            "fluid_initial": _rounded(arrays["fluid_initial"]),
            "nutrient_initial": _rounded(arrays["nutrient_initial"]),
            "energy_initial": _rounded(arrays["energy_initial"]),
            "mass": _rounded(arrays["mass"]),
            "stiffness": _rounded(arrays["stiffness"]),
            "bond_ab": arrays["bond_ab"].astype(int).tolist(),
            "bond_kind": arrays["bond_kind"].astype(int).tolist(),
            "bond_rest": _rounded(arrays["bond_rest"]),
            "bond_strength": _rounded(arrays["bond_strength"]),
            "bond_conductance": _rounded(arrays["bond_conductance"]),
        },
    }


def _source_manifest() -> dict[str, str]:
    paths = [
        Path(__file__),
        PROJECT_ROOT / "forge/cellular_organism/contract.py",
        PROJECT_ROOT / "forge/cellular_organism/compiler.py",
        PROJECT_ROOT / "forge/cellular_organism/simulation.py",
        PROJECT_ROOT / "forge/cellular_organism/orientation.py",
    ]
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def project_runtime(source_manifest: Path) -> dict[str, bytes]:
    validation = validate_bank(source_manifest)
    source = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    if source["format"] != FORMAT:
        raise ValueError("Unexpected cellular organism bank format.")
    root = Path(source_manifest).resolve().parent
    files: dict[str, bytes] = {}
    catalog_species: list[dict[str, object]] = []
    for record in source["species"]:
        arrays_path = root.joinpath(*PurePosixPath(record["arrays"]["path"]).parts)
        arrays = _load_arrays(arrays_path)
        runtime = _runtime_species(record, arrays)
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
                "runtime": {"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)},
                "summary": record["summary"],
            }
        )
    contact_path = root / source["contact_sheet"]["path"]
    contact = contact_path.read_bytes()
    files["cellular_organism_contact_sheet.png"] = contact
    sources = _source_manifest()
    catalog: dict[str, object] = {
        "format": CATALOG_FORMAT,
        "status": "ready",
        "bundle_version": 1,
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_semantic_sha256": source["semantic_sha256"],
        "sync_source_manifest": sources,
        "sync_source_sha256": sha256_bytes(canonical_json_bytes(sources)),
        "sample_count": len(catalog_species),
        "family_counts": source["family_counts"],
        "totals": source["totals"],
        "simulation": top_down_simulation_defaults(source["simulation"]),
        "orientation": orientation_manifest(),
        "tissues": list(TISSUE_NAMES),
        "contact_sheet": {
            "path": "cellular_organism_contact_sheet.png",
            "bytes": len(contact),
            "sha256": sha256_bytes(contact),
        },
        "species": catalog_species,
        "validation": validation,
        "runtime_contract": {
            "python_required": False,
            "all_80_species_projected": True,
            "runtime_files_are_audit_bound_to_npz": True,
            "cell_and_bond_totals_exact": True,
            "top_down_dorsal_projection": True,
            "uniform_screen_gravity_disabled": True,
            "external_fluid_is_surface_diffusion": True,
        },
    }
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog))
    files["catalog.json"] = canonical_json_bytes(catalog)
    return files


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=sum(map(len, files.values())) + 512 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
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


def validate_runtime(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve()
    catalog_path = destination / "catalog.json"
    if not catalog_path.is_file():
        raise ValueError("Native cellular catalog is missing.")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("format") != CATALOG_FORMAT or catalog.get("status") != "ready":
        raise ValueError("Native cellular catalog header is invalid.")
    if int(catalog.get("sample_count", -1)) != 80 or len(catalog.get("species", [])) != 80:
        raise ValueError("Native cellular catalog census is invalid.")
    validate_orientation(catalog.get("orientation", {}))
    if catalog.get("simulation", {}).get("gravity") != 0.0 or catalog.get("simulation", {}).get("legacy_scalar_gravity_disabled") is not True:
        raise ValueError("Native cellular simulation still exposes scalar gravity.")
    runtime_contract = catalog.get("runtime_contract", {})
    if not all(runtime_contract.get(key) is True for key in (
        "top_down_dorsal_projection",
        "uniform_screen_gravity_disabled",
        "external_fluid_is_surface_diffusion",
    )):
        raise ValueError("Native cellular top-down runtime gates differ.")
    total_cells = 0
    total_bonds = 0
    for entry in catalog["species"]:
        artifact = entry["runtime"]
        path = destination.joinpath(*PurePosixPath(artifact["path"]).parts).resolve()
        path.relative_to(destination)
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("Native species runtime hash/size mismatch.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != RUNTIME_FORMAT or data.get("sample_id") != entry["sample_id"]:
            raise ValueError("Native species runtime identity is invalid.")
        cells = len(data["arrays"]["position_xy"])
        bonds = len(data["arrays"]["bond_ab"])
        if cells != data["summary"]["physical_cell_count"] or bonds != data["summary"]["bond_count"]:
            raise ValueError("Native species array census drifted.")
        total_cells += cells
        total_bonds += bonds
    if total_cells != catalog["totals"]["physical_cells"] or total_bonds != catalog["totals"]["bonds"]:
        raise ValueError("Native cellular aggregate census drifted.")
    return {
        "passed": True,
        "sample_count": 80,
        "cell_count": total_cells,
        "bond_count": total_bonds,
        "bundle_id": catalog["bundle_id"],
    }


def sync_runtime(source: Path, destination: Path, *, repeat_check: bool = True) -> dict[str, object]:
    first = project_runtime(source)
    second = project_runtime(source) if repeat_check else first
    differences = [name for name in sorted(set(first) | set(second)) if first.get(name) != second.get(name)]
    if differences:
        raise ValueError(f"Cellular runtime projection is not exact: {differences[:5]}")
    _publish(Path(destination).resolve(), first)
    tree = hashlib.sha256()
    for name, payload in sorted(first.items()):
        tree.update(name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    catalog = json.loads(first["catalog.json"])
    runtime_validation = validate_runtime(destination)
    return {
        "passed": True,
        "source": str(Path(source).resolve()),
        "destination": str(Path(destination).resolve()),
        "file_count": len(first),
        "bytes": sum(map(len, first.values())),
        "sample_count": catalog["sample_count"],
        "cell_count": catalog["totals"]["physical_cells"],
        "bond_count": catalog["totals"]["bonds"],
        "bundle_id": catalog["bundle_id"],
        "tree_sha256": tree.hexdigest(),
        "repeat_exact": not differences,
        "runtime_validation": runtime_validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project cellular organism bank into native Godot JSON")
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
