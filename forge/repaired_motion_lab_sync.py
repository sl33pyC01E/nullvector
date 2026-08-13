from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import tempfile
from typing import Any, Mapping

from .config import PROJECT_ROOT
from .multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from .neural_rig_repair_style.production import _source_hash, validate_shard
from .safety import require_disk_floor


FORMAT = "nullvector-repaired-motion-native-catalog-v1"
SOURCE_FORMAT = "nullvector-neural-rig-repair-style-bank-v1"
IDENTITY_FORMAT = "nullvector-neural-rig-repair-style-identity-v1"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs/neural_rig_repair_style_all80_v1"
DEFAULT_DESTINATION = PROJECT_ROOT / "game/generated/repaired_motion_lab/v1"
LAYERS = ("base", "outline", "emission_core", "aura", "bloom_r1", "bloom_r2", "composite")
MOTIONS = (
    "idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear",
    "confused", "sleep", "taunt", "attack", "cast", "hit", "death",
)
FACINGS = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
DISK_FLOOR_GIB = 100
MAX_JSON_BYTES = 8 * 1024 * 1024


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe JSON file {path}")
    if path.stat().st_size > maximum:
        raise ValueError(f"oversized JSON file {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON {token}")),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    if canonical_json_bytes(value) != path.read_bytes():
        raise ValueError(f"JSON is not canonical: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, record: Mapping[str, Any]) -> Path:
    relative = PurePosixPath(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("artifact path is not canonical relative POSIX")
    path = root.joinpath(*relative.parts).resolve()
    path.relative_to(root.resolve())
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"artifact is missing or unsafe: {relative}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"artifact byte count drifted: {relative}")
    if _sha256_file(path) != record.get("sha256"):
        raise ValueError(f"artifact hash drifted: {relative}")
    return path


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def _source_files() -> dict[str, str]:
    paths = (Path(__file__), PROJECT_ROOT / "forge/neural_rig_repair_style/production.py")
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path) for path in paths}


def _identity_record(source_root: Path, shard_root: Path, entry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = _resolve(shard_root, entry["artifact"])
    identity = _load_json(manifest_path)
    if identity.get("format") != IDENTITY_FORMAT or identity.get("status") != "ready" or identity.get("neural_output") is not True:
        raise ValueError("repaired identity header is invalid")
    if identity.get("sample_id") != entry.get("sample_id") or identity.get("ordinal") != entry.get("ordinal"):
        raise ValueError("repaired identity linkage drifted")
    if any(value is not True for value in identity.get("gates", {}).values()):
        raise ValueError("repaired identity gate failed")
    unsigned = dict(identity)
    recorded_hash = unsigned.pop("identity_sha256", None)
    if sha256_bytes(canonical_json_bytes(unsigned)) != recorded_hash or recorded_hash != entry.get("identity_sha256"):
        raise ValueError("repaired identity self hash drifted")
    layout = identity.get("layout", {})
    if layout != {
        "cell_size": 48,
        "clip_count": 104,
        "columns": 16,
        "frame_count": 944,
        "layers": list(LAYERS),
        "rows": 59,
    }:
        raise ValueError("repaired identity atlas layout drifted")
    clips = identity.get("clips", [])
    if len(clips) != 104 or sum(int(clip["frame_count"]) for clip in clips) != 944:
        raise ValueError("repaired identity clip census drifted")
    if {(clip["motion"], clip["facing"]) for clip in clips} != {(motion, facing) for motion in MOTIONS for facing in FACINGS}:
        raise ValueError("repaired identity motion matrix drifted")
    source_paths: dict[str, Path] = {}
    atlas_records: dict[str, dict[str, Any]] = {}
    for layer in LAYERS:
        artifact = identity["artifacts"]["layers"][layer]
        source = _resolve(shard_root, artifact)
        if _png_size(source) != (768, 2832):
            raise ValueError(f"repaired {layer} atlas dimensions drifted")
        relative = f"atlases/{int(identity['ordinal']):04d}_{identity['sample_id']}/{layer}.png"
        source_paths[relative] = source
        atlas_records[layer] = {
            "path": relative,
            "bytes": source.stat().st_size,
            "sha256": artifact["sha256"],
        }
    condition = identity["condition"]
    return {
        "sample_id": identity["sample_id"],
        "ordinal": identity["ordinal"],
        "family": identity["family"],
        "subtype": condition["subtype_name"],
        "subtype_id": condition["subtype_id"],
        "role": condition["role_name"],
        "role_id": condition["role_id"],
        "sample_seed": str(condition["sample_seed"]),
        "identity_sha256": recorded_hash,
        "layout": layout,
        "atlases": atlas_records,
        "clips": [
            {
                "motion": clip["motion"], "facing": clip["facing"],
                "start_cell": clip["start_cell"], "frame_count": clip["frame_count"],
                "fps": clip["fps"], "loop": clip["loop"],
                "clip_sha256": clip["repair_style_clip_sha256"],
            }
            for clip in clips
        ],
    }, source_paths


def project_runtime(source_root: Path = DEFAULT_SOURCE) -> tuple[dict[str, bytes], dict[str, Path]]:
    source_root = Path(source_root).resolve()
    manifest_path = source_root / "style_bank_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("format") != SOURCE_FORMAT or manifest.get("status") != "ready" or manifest.get("neural_output") is not True:
        raise ValueError("all-80 repaired style bank header is invalid")
    if any(value is not True for value in manifest.get("gates", {}).values()):
        raise ValueError("all-80 repaired style bank gate failed")
    if manifest.get("compiler", {}).get("source_sha256") != _source_hash():
        raise ValueError("all-80 repaired style compiler source drifted")
    unsigned = dict(manifest)
    recorded_bank_hash = unsigned.pop("bank_sha256", None)
    if sha256_bytes(canonical_json_bytes(unsigned)) != recorded_bank_hash:
        raise ValueError("all-80 repaired style bank self hash drifted")
    expected_counts = {"identity_count": 80, "clip_count": 8320, "frame_count": 75520, "layer_atlas_count": 560, "shard_count": 16}
    if any(int(manifest.get("counts", {}).get(key, -1)) != value for key, value in expected_counts.items()):
        raise ValueError("all-80 repaired style bank census drifted")

    identities: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}
    for index in range(16):
        primary_root = source_root / "primary/shards" / f"shard_{index:02d}"
        replay_root = source_root / "replay/shards" / f"shard_{index:02d}"
        primary = validate_shard(primary_root, expected_index=index)
        replay = validate_shard(replay_root, expected_index=index)
        if primary != replay:
            raise ValueError(f"primary/replay shard manifest differs at {index}")
        for entry in primary["identities"]:
            runtime, artifacts = _identity_record(source_root, primary_root, entry)
            replay_entry = next(item for item in replay["identities"] if item["ordinal"] == entry["ordinal"])
            replay_manifest = _resolve(replay_root, replay_entry["artifact"])
            if replay_manifest.read_bytes() != _resolve(primary_root, entry["artifact"]).read_bytes():
                raise ValueError("identity manifest replay bytes differ")
            for relative, primary_path in artifacts.items():
                replay_source = replay_root / entry["artifact"]["path"]
                replay_source = replay_source.parent / Path(relative).name
                if not replay_source.is_file() or _sha256_file(replay_source) != _sha256_file(primary_path):
                    raise ValueError(f"atlas replay bytes differ: {relative}")
                if relative in sources:
                    raise ValueError(f"duplicate runtime artifact path {relative}")
                sources[relative] = primary_path
            identities.append(runtime)
    identities.sort(key=lambda value: int(value["ordinal"]))
    if [entry["ordinal"] for entry in identities] != list(range(80)):
        raise ValueError("runtime identities do not cover all 80 ordinals once")
    family_counts = {family: sum(entry["family"] == family for entry in identities) for family in ("humanoid", "animalian", "plantlike", "anomaly", "machine")}
    if set(family_counts.values()) != {16}:
        raise ValueError("runtime family balance drifted")
    sync_sources = _source_files()
    catalog: dict[str, Any] = {
        "format": FORMAT,
        "status": "ready",
        "neural_output": True,
        "source": {
            "style_bank_sha256": recorded_bank_hash,
            "style_bank_file_sha256": _sha256_file(manifest_path),
            "repair_bank_sha256": manifest["authority"]["repair_bank_sha256"],
            "compiler_source_sha256": manifest["compiler"]["source_sha256"],
            "primary_tree_sha256": manifest["primary_tree_sha256"],
        },
        "sync_source_manifest": sync_sources,
        "sync_source_sha256": sha256_bytes(canonical_json_bytes(sync_sources)),
        "counts": {"identity_count": 80, "family_count": 5, "motion_count": 13, "facing_count": 8, "clip_count": 8320, "frame_count": 75520, "atlas_count": 560},
        "family_counts": family_counts,
        "layers": list(LAYERS),
        "motions": list(MOTIONS),
        "facings": list(FACINGS),
        "identities": identities,
        "runtime_contract": {
            "python_required": False,
            "all_80_identities_animated": True,
            "all_8320_clips_addressable": True,
            "all_75520_frames_addressable": True,
            "all_560_atlases_hash_bound": True,
            "primary_replay_bytes_exact": True,
        },
    }
    catalog["bundle_id"] = sha256_bytes(canonical_json_bytes(catalog))
    return {"catalog.json": canonical_json_bytes(catalog)}, sources


def _tree(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, _sha256_file(path))
        for path in root.rglob("*") if path.is_file()
    }


def _publish(destination: Path, inline: Mapping[str, bytes], sources: Mapping[str, Path]) -> None:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    planned = sum(map(len, inline.values())) + sum(path.stat().st_size for path in sources.values())
    require_disk_floor(destination.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=planned + 512 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for relative, payload in sorted(inline.items()):
            target = staging.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        for relative, source in sorted(sources.items()):
            target = staging.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
            os.close(descriptor)
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_runtime(destination: Path) -> dict[str, Any]:
    destination = Path(destination).resolve()
    catalog = _load_json(destination / "catalog.json")
    if catalog.get("format") != FORMAT or catalog.get("status") != "ready" or catalog.get("neural_output") is not True:
        raise ValueError("native repaired-motion catalog header is invalid")
    unsigned = dict(catalog)
    bundle_id = unsigned.pop("bundle_id", None)
    if sha256_bytes(canonical_json_bytes(unsigned)) != bundle_id:
        raise ValueError("native repaired-motion bundle ID drifted")
    if len(catalog.get("identities", [])) != 80:
        raise ValueError("native repaired-motion identity census drifted")
    seen: set[int] = set()
    artifact_count = 0
    for identity in catalog["identities"]:
        ordinal = int(identity["ordinal"])
        if ordinal in seen:
            raise ValueError("duplicate native repaired-motion ordinal")
        seen.add(ordinal)
        if len(identity.get("clips", [])) != 104:
            raise ValueError("native repaired-motion clip census drifted")
        for layer in LAYERS:
            path = _resolve(destination, identity["atlases"][layer])
            if _png_size(path) != (768, 2832):
                raise ValueError("native repaired-motion atlas dimensions drifted")
            artifact_count += 1
    if seen != set(range(80)) or artifact_count != 560:
        raise ValueError("native repaired-motion coverage drifted")
    return {"passed": True, "identity_count": 80, "clip_count": 8320, "frame_count": 75520, "atlas_count": 560, "bundle_id": bundle_id}


def sync_runtime(source: Path, destination: Path, *, repeat_check: bool = True) -> dict[str, Any]:
    first_inline, first_sources = project_runtime(source)
    if repeat_check:
        second_inline, second_sources = project_runtime(source)
        if first_inline != second_inline or set(first_sources) != set(second_sources):
            raise ValueError("repaired-motion runtime projection is not deterministic")
        for name in first_sources:
            if _sha256_file(first_sources[name]) != _sha256_file(second_sources[name]):
                raise ValueError(f"repaired-motion source changed between projections: {name}")
    _publish(destination, first_inline, first_sources)
    validation = validate_runtime(destination)
    tree = _tree(Path(destination).resolve())
    tree_sha = sha256_bytes(canonical_json_bytes({name: {"bytes": size, "sha256": digest} for name, (size, digest) in tree.items()}))
    return {"passed": True, "source": str(Path(source).resolve()), "destination": str(Path(destination).resolve()), "file_count": len(tree), "bytes": sum(size for size, _ in tree.values()), "tree_sha256": tree_sha, "repeat_exact": repeat_check, "runtime_validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project all-80 repaired neural motion into a native Godot lab")
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
