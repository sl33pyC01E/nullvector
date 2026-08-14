from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from .cellular_motion import validate_bank
from .cellular_organism.contract import DISK_FLOOR_GIB
from .config import PROJECT_ROOT
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-cellular-neuromuscular-native-catalog-v7"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs/cellular_motion_v2/cellular_motion_manifest.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/cellular_motion/v12"


def _source_registry() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(), PROJECT_ROOT / "forge/cellular_motion/contract.py", PROJECT_ROOT / "forge/cellular_motion/compiler.py",
        PROJECT_ROOT / "shared/schema/cellular_motion_bank.schema.json", PROJECT_ROOT / "game/CellularMotionLab.tscn",
        PROJECT_ROOT / "game/scripts/cellular_motion_lab.gd", PROJECT_ROOT / "game/scripts/cellular_organism_lab.gd",
    )
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def project_runtime(source_manifest: Path) -> dict[str, bytes]:
    source_manifest = Path(source_manifest).resolve(); validation = validate_bank(source_manifest)
    source = json.loads(source_manifest.read_text(encoding="utf-8")); registry = _source_registry()
    catalog: dict[str, object] = {
        "format": FORMAT, "status": "ready", "bundle_version": 7,
        "source_manifest_sha256": sha256_file(source_manifest), "source_semantic_sha256": source["semantic_sha256"],
        "sync_source_manifest": registry, "sync_source_sha256": sha256_bytes(canonical_json_bytes(registry)),
        "identity_count": source["identity_count"], "family_count": source["family_count"], "motion_count": source["motion_count"],
        "facing_count": source["facing_count"], "clip_count": source["clip_count"], "frame_count": source["frame_count"],
        "drivers": source["drivers"], "facings": source["facings"], "motions": source["motions"],
        "programs": source["programs"], "identities": source["identities"], "runtime_contract": source["runtime_contract"],
        "validation": validation,
    }
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog)); payload = canonical_json_bytes(catalog)
    contact_record = source["contact_sheet"]; contact_path = source_manifest.parent.joinpath(*PurePosixPath(contact_record["path"]).parts); contact = contact_path.read_bytes()
    return {"motion_catalog.json": payload, "cellular_motion_contact_sheet.png": contact}


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = destination.resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=sum(map(len, files.values())) + 512 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"; staging.mkdir(parents=True)
    for relative, payload in sorted(files.items()):
        target = staging.joinpath(*PurePosixPath(relative).parts); target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    os.replace(staging, destination)


def validate_runtime(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve(); path = destination / "motion_catalog.json"; raw = path.read_bytes(); catalog = json.loads(raw)
    if raw != canonical_json_bytes(catalog) or catalog.get("format") != FORMAT or catalog.get("status") != "ready": raise ValueError("Cellular motion native catalog header/canonical JSON differs")
    registry = _source_registry()
    if catalog.get("sync_source_manifest") != registry or catalog.get("sync_source_sha256") != sha256_bytes(canonical_json_bytes(registry)): raise ValueError("Cellular motion native source provenance differs")
    if catalog.get("bundle_id") != sha256_bytes(canonical_json_bytes({key: value for key, value in catalog.items() if key != "bundle_id"})): raise ValueError("Cellular motion native bundle identity differs")
    if (catalog.get("identity_count"), catalog.get("clip_count"), catalog.get("frame_count")) != (45, 520, 4720): raise ValueError("Cellular motion native census differs")
    source_path = DEFAULT_SOURCE.resolve()
    if catalog.get("source_manifest_sha256") != sha256_file(source_path): raise ValueError("Cellular motion native source manifest differs")
    validate_bank(source_path)
    expected = project_runtime(source_path)
    for relative, payload in expected.items():
        artifact = destination / relative
        if not artifact.is_file() or artifact.read_bytes() != payload: raise ValueError(f"Cellular motion native artifact replay differs: {relative}")
    actual = {item.relative_to(destination).as_posix() for item in destination.rglob("*") if item.is_file() and item.suffix.lower() != ".import"}
    if actual != set(expected): raise ValueError("Cellular motion native output closure differs")
    return {"passed": True, "identity_count": 45, "clip_count": 520, "frame_count": 4720, "bundle_id": catalog["bundle_id"], "bytes": sum(map(len, expected.values()))}


def sync_runtime(source: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION, *, repeat_check: bool = True) -> dict[str, object]:
    first = project_runtime(source); second = project_runtime(source) if repeat_check else first
    if first != second: raise ValueError("Cellular motion native projection is not exact")
    _publish(Path(destination), first); validation = validate_runtime(destination); tree = hashlib.sha256()
    for name, payload in sorted(first.items()): tree.update(name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    return {"passed": True, "source": str(Path(source).resolve()), "destination": str(Path(destination).resolve()), "file_count": len(first), "bytes": sum(map(len, first.values())), "repeat_exact": True, "tree_sha256": tree.hexdigest(), "runtime_validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project cellular motion programs into native Godot JSON")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE); parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--no-repeat-check", action="store_true"); parser.add_argument("--report", type=Path); args = parser.parse_args(argv)
    report = sync_runtime(args.source, args.destination, repeat_check=not args.no_repeat_check)
    if args.report: args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
