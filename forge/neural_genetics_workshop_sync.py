from __future__ import annotations

"""Compile neural fusion, latent fusion, and evolution into a native Godot bank."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSION_SOURCE = PROJECT_ROOT / "outputs" / "neural_fusion_pilot_v2"
DEFAULT_LATENT_SOURCE = PROJECT_ROOT / "outputs" / "neural_fusion_production_v1_run2"
DEFAULT_EVOLUTION_SOURCE = PROJECT_ROOT / "outputs" / "neural_fusion_production_evolution_v1_run2"
DEFAULT_DESTINATION = PROJECT_ROOT / "game" / "generated" / "neural_genetics" / "v3"

FORMAT = "nullvector-neural-genetics-workshop-assets-v3"
LAYERS = ("base", "outline", "emission_core", "aura", "bloom_r1", "bloom_r2", "composite")
FUSION_FORMAT = "nullvector-neural-fusion-pilot-v1"
LATENT_FORMAT = "nullvector-production-neural-latent-fusion-v1"
EVOLUTION_FORMAT = "nullvector-production-neural-latent-evolution-v1"
FUSION_COUNT = 10
LATENT_COUNT = 12
EVOLUTION_COUNT = 36
MIN_FREE_BYTES = 100 * 1024**3
PLANNED_BYTES = 256 * 1024**2


class GeneticsContractError(ValueError):
    pass


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, *, label: str) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if not text or "\\" in text or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise GeneticsContractError(f"unsafe {label} path: {text!r}")
    return path


def _load_manifest(path: Path, *, expected_format: str, status: str, hash_key: str) -> dict[str, Any]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or raw != _canonical(manifest):
        raise GeneticsContractError(f"manifest is not a canonical object: {path}")
    unsigned = dict(manifest)
    stored = unsigned.pop(hash_key, None)
    if stored != _sha_bytes(_canonical(unsigned)):
        raise GeneticsContractError(f"manifest self-hash mismatch: {path}")
    if manifest.get("format") != expected_format or manifest.get("status") != status:
        raise GeneticsContractError(f"manifest authority mismatch: {path}")
    if any(value is not True for value in manifest.get("gates", {}).values()):
        raise GeneticsContractError(f"manifest has a failed gate: {path}")
    return manifest


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _copy_artifact(source_root: Path, destination: Path, record: dict[str, Any]) -> dict[str, Any]:
    relative = _safe_relative(record["path"], label="source artifact")
    source = (source_root / relative).resolve()
    if not source.is_relative_to(source_root.resolve()) or not source.is_file():
        raise GeneticsContractError(f"missing/escaping source artifact: {relative}")
    payload = source.read_bytes()
    if len(payload) != int(record["bytes"]) or _sha_bytes(payload) != record["sha256"]:
        raise GeneticsContractError(f"source artifact hash mismatch: {relative}")
    with Image.open(source) as image:
        image.verify()
    _write_exact(destination, payload)
    return {
        "path": destination.relative_to(DEFAULT_DESTINATION if destination.is_relative_to(DEFAULT_DESTINATION) else destination.parents[3]).as_posix(),
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _copy_to_root(source_root: Path, runtime_root: Path, relative: str, record: dict[str, Any]) -> dict[str, Any]:
    destination = runtime_root / PurePosixPath(relative)
    copied = _copy_artifact(source_root, destination, record)
    copied["path"] = destination.relative_to(runtime_root).as_posix()
    return copied


def _clip_payload(clip: dict[str, Any]) -> dict[str, Any]:
    return {
        "motion": str(clip["motion"]),
        "facing": str(clip["facing"]),
        "fps": int(clip["fps"]),
        "frame_count": int(clip["frame_count"]),
        "start_cell": int(clip["start_cell"]),
        "loop": bool(clip["loop"]),
    }


def _motion_bank(source_root: Path, runtime_root: Path, manifest: dict[str, Any], *, latent: bool) -> dict[str, Any]:
    records = []
    for source_record in manifest["specimens"]:
        specimen_id = str(source_record["specimen_id"])
        layout = dict(source_record.get("layout", {}))
        clips = [_clip_payload(clip) for clip in source_record["clips"]]
        frame_count = sum(clip["frame_count"] for clip in clips)
        if latent:
            layout = {"cell_size": 48, "columns": 16, "rows": (frame_count + 15) // 16, "frame_count": frame_count}
        if (
            int(layout.get("cell_size", -1)) != 48
            or int(layout.get("columns", -1)) != 16
            or int(layout.get("frame_count", -1)) != frame_count
            or int(layout.get("rows", -1)) * 16 < frame_count
        ):
            raise GeneticsContractError(f"invalid motion layout: {specimen_id}")
        cursor = 0
        for clip in clips:
            if clip["start_cell"] != cursor:
                raise GeneticsContractError(f"non-contiguous clip atlas: {specimen_id}")
            cursor += clip["frame_count"]
        runtime_layers = {}
        for layer in LAYERS:
            runtime_layers[layer] = _copy_to_root(
                source_root,
                runtime_root,
                f"{'latent' if latent else 'fusion'}/{specimen_id}/{layer}.png",
                source_record["artifacts"][layer],
            )
            with Image.open(runtime_root / runtime_layers[layer]["path"]) as image:
                if image.mode != "RGBA" or image.size != (layout["columns"] * 48, layout["rows"] * 48):
                    raise GeneticsContractError(f"atlas dimensions mismatch: {specimen_id}/{layer}")
        if latent:
            record = {
                "sample_id": specimen_id,
                "family": str(source_record.get("family", "cross-family latent hybrid")),
                "mode": str(source_record["fusion_mode"]),
                "mutation_mode": str(source_record["mutation_mode"]),
                "mutation_strength": float(source_record["mutation_strength"]),
                "alpha": float(source_record["alpha"]),
                "quality_tier": str(manifest["quality_tier"]),
                "parents": [str(source_record["parent_a"]), str(source_record["parent_b"])],
                "binding_sha256": str(source_record["binding_sha256"]),
            }
        else:
            record = {
                "sample_id": specimen_id,
                "family": str(source_record["condition"]["morphology_name"]),
                "mode": str(source_record["fusion_mode"]),
                "mutation_mode": str(source_record["mutation_mode"]),
                "mutation_strength": int(source_record["mutation_strength"]),
                "parents": [dict(source_record["parent_a"]), dict(source_record["parent_b"])],
            }
        record.update(
            {
                "lineage_sha256": str(source_record["lineage_sha256"]),
                "fields_sha256": str(source_record["fields_sha256"]),
                "metrics": dict(source_record["metrics"]),
                "layout": layout,
                "layers": runtime_layers,
                "clips": clips,
            }
        )
        records.append(record)
    return {
        "status": "ready",
        "truth_label": "production-ema-fsq-latent-genetics" if latent else "verified-categorical-fusion",
        "specimen_count": len(records),
        "clip_count": sum(len(record["clips"]) for record in records),
        "frame_count": sum(record["layout"]["frame_count"] for record in records),
        "layers": list(LAYERS),
        "specimens": records,
    }


def _evolution_bank(source_root: Path, runtime_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    records = []
    for source_record in manifest["selected"]:
        specimen_id = str(source_record["specimen_id"])
        layout = dict(source_record["layout"])
        clips = [_clip_payload(clip) for clip in source_record["clips"]]
        frame_count = sum(clip["frame_count"] for clip in clips)
        if layout != {"cell_size": 48, "columns": 16, "rows": (frame_count + 15) // 16, "frame_count": frame_count}:
            raise GeneticsContractError(f"evolution layout mismatch: {specimen_id}")
        cursor = 0
        for clip in clips:
            if clip["start_cell"] != cursor:
                raise GeneticsContractError(f"evolution clip cursor mismatch: {specimen_id}")
            cursor += clip["frame_count"]
        layers = {}
        for layer in LAYERS:
            layers[layer] = _copy_to_root(
                source_root, runtime_root,
                f"evolution/g{int(source_record['generation'])}/{int(source_record['rank']):02d}_{specimen_id}/{layer}.png",
                source_record["artifacts"][layer],
            )
            with Image.open(runtime_root / layers[layer]["path"]) as decoded:
                if decoded.mode != "RGBA" or decoded.size != (layout["columns"] * 48, layout["rows"] * 48):
                    raise GeneticsContractError(f"evolution atlas mismatch: {specimen_id}/{layer}")
        records.append(
            {
                "sample_id": specimen_id,
                "generation": int(source_record["generation"]),
                "rank": int(source_record["rank"]),
                "family": str(source_record["family"]),
                "fusion_mode": str(source_record["fusion_mode"]),
                "mutation_mode": str(source_record["mutation_mode"]),
                "parents": list(source_record["parent_ids"]),
                "lineage_sha256": str(source_record["lineage_sha256"]),
                "fields_sha256": str(source_record["fields_sha256"]),
                "binding_sha256": str(source_record["binding_sha256"]),
                "alpha": float(source_record["alpha"]),
                "mutation_strength": int(source_record["mutation_strength"]),
                "score": dict(source_record["score"]),
                "metrics": dict(source_record["metrics"]),
                "layout": layout,
                "layers": layers,
                "clips": clips,
            }
        )
    return {
        "status": "ready",
        "truth_label": "production-ema-fsq-three-generation-selection",
        "generation_count": 3,
        "selected_count": len(records),
        "specimen_count": len(records),
        "clip_count": sum(len(record["clips"]) for record in records),
        "frame_count": sum(record["layout"]["frame_count"] for record in records),
        "layers": list(LAYERS),
        "selection_policy": dict(manifest["selection_policy"]),
        "specimens": records,
    }


def _inventory(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.name != "asset_index.json"):
        payload = path.read_bytes()
        result.append({"path": path.relative_to(root).as_posix(), "bytes": len(payload), "sha256": _sha_bytes(payload)})
    return result


def _bundle_id(fusion: dict[str, Any], latent: dict[str, Any], evolution: dict[str, Any], inventory: list[dict[str, Any]]) -> str:
    return _sha_bytes(_canonical({"fusion": fusion, "latent": latent, "evolution": evolution, "inventory": inventory}))


def sync_genetics_workshop(
    destination: Path = DEFAULT_DESTINATION,
    *,
    fusion_source: Path = DEFAULT_FUSION_SOURCE,
    latent_source: Path = DEFAULT_LATENT_SOURCE,
    evolution_source: Path = DEFAULT_EVOLUTION_SOURCE,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    usage = shutil.disk_usage(destination.parent if destination.parent.exists() else PROJECT_ROOT)
    if usage.free - PLANNED_BYTES < MIN_FREE_BYTES:
        raise RuntimeError("neural genetics runtime sync would breach the 100 GiB disk floor")
    fusion_manifest_path = Path(fusion_source) / "fusion_manifest.json"
    latent_manifest_path = Path(latent_source) / "production_fusion_manifest.json"
    evolution_manifest_path = Path(evolution_source) / "production_evolution_manifest.json"
    fusion_manifest = _load_manifest(fusion_manifest_path, expected_format=FUSION_FORMAT, status="ready", hash_key="bank_sha256")
    latent_manifest = _load_manifest(latent_manifest_path, expected_format=LATENT_FORMAT, status="ready", hash_key="bank_sha256")
    evolution_manifest = _load_manifest(evolution_manifest_path, expected_format=EVOLUTION_FORMAT, status="ready", hash_key="evolution_sha256")
    if fusion_manifest["counts"]["specimen_count"] != FUSION_COUNT or latent_manifest["counts"]["specimens"] != LATENT_COUNT or evolution_manifest["counts"]["selected"] != EVOLUTION_COUNT:
        raise GeneticsContractError("neural genetics source census mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    fusion = _motion_bank(Path(fusion_source), destination, fusion_manifest, latent=False)
    latent = _motion_bank(Path(latent_source), destination, latent_manifest, latent=True)
    evolution = _evolution_bank(Path(evolution_source), destination, evolution_manifest)
    inventory = _inventory(destination)
    index = {
        "format": FORMAT,
        "schema_version": "3.0.0",
        "status": "ready",
        "engine": "Godot 4.3",
        "pixel_filter": "nearest",
        "python_runtime_required": False,
        "disk_budget": {"minimum_free_bytes": MIN_FREE_BYTES, "planned_bytes": PLANNED_BYTES, "guard_passed": True},
        "sources": {
            "fusion": {"path": str(fusion_manifest_path.resolve()), "sha256": _sha_file(fusion_manifest_path), "bank_sha256": fusion_manifest["bank_sha256"]},
            "latent": {"path": str(latent_manifest_path.resolve()), "sha256": _sha_file(latent_manifest_path), "bank_sha256": latent_manifest["bank_sha256"]},
            "evolution": {"path": str(evolution_manifest_path.resolve()), "sha256": _sha_file(evolution_manifest_path), "evolution_sha256": evolution_manifest["evolution_sha256"]},
        },
        "fusion": fusion,
        "latent": latent,
        "evolution": evolution,
        "inventory": inventory,
        "asset_count": len(inventory) + 1,
        "generator_sha256": _sha_file(Path(__file__)),
        "gates": {
            "categorical_fusion_ready": True,
            "production_latent_authority_ready": True,
            "production_latent_all_fusion_modes_present": latent_manifest["counts"]["fusion_modes"] == 6,
            "production_latent_all_mutation_modes_present": latent_manifest["counts"]["mutation_modes"] == 6,
            "evolution_selection_ready": True,
            "all_motion_atlases_exact": True,
            "production_evolution_ready": True,
            "all_evolution_motion_atlases_exact": True,
            "runtime_png_json_only": True,
            "disk_floor_preserved": True,
        },
    }
    index["bundle_id"] = _bundle_id(fusion, latent, evolution, inventory)
    _write_exact(destination / "asset_index.json", _canonical(index))
    errors = validate_genetics_workshop(destination / "asset_index.json")
    if errors:
        raise GeneticsContractError("; ".join(errors))
    return index


def validate_genetics_workshop(index_path: Path = DEFAULT_DESTINATION / "asset_index.json") -> list[str]:
    index_path = Path(index_path)
    if not index_path.is_file():
        return ["missing index"]
    try:
        raw = index_path.read_bytes()
        index = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return ["invalid index JSON"]
    errors = []
    if raw != _canonical(index): errors.append("index canonical JSON")
    if index.get("format") != FORMAT or index.get("status") != "ready": errors.append("index authority")
    if index.get("pixel_filter") != "nearest" or index.get("python_runtime_required") is not False: errors.append("runtime contract")
    if any(value is not True for value in index.get("gates", {}).values()): errors.append("index gates")
    root = index_path.parent
    inventory = index.get("inventory", [])
    if int(index.get("asset_count", -1)) != len(inventory) + 1: errors.append("asset count")
    for record in inventory:
        try:
            relative = _safe_relative(record["path"], label="inventory")
            path = root / relative
            if not path.is_file() or path.stat().st_size != int(record["bytes"]) or _sha_file(path) != record["sha256"]:
                errors.append(f"inventory {relative}")
        except (KeyError, TypeError, ValueError, GeneticsContractError):
            errors.append("inventory record")
    expected = (("fusion", FUSION_COUNT, "ready"), ("latent", LATENT_COUNT, "ready"))
    for bank_name, count, status in expected:
        bank = index.get(bank_name, {})
        if bank.get("status") != status or int(bank.get("specimen_count", -1)) != count or tuple(bank.get("layers", [])) != LAYERS:
            errors.append(f"{bank_name} census")
        for specimen in bank.get("specimens", []):
            layout = specimen.get("layout", {})
            for layer in LAYERS:
                record = specimen.get("layers", {}).get(layer, {})
                path = root / PurePosixPath(str(record.get("path", "")))
                try:
                    with Image.open(path) as image:
                        if image.mode != "RGBA" or image.size != (int(layout["columns"]) * 48, int(layout["rows"]) * 48):
                            errors.append(f"{bank_name} atlas {specimen.get('sample_id')}/{layer}")
                except (OSError, KeyError, TypeError, ValueError):
                    errors.append(f"{bank_name} atlas decode")
            cursor = 0
            for clip in specimen.get("clips", []):
                if int(clip.get("start_cell", -1)) != cursor: errors.append(f"{bank_name} clip cursor")
                cursor += int(clip.get("frame_count", 0))
            if cursor != int(layout.get("frame_count", -1)): errors.append(f"{bank_name} frame census")
    evolution = index.get("evolution", {})
    if evolution.get("status") != "ready" or int(evolution.get("selected_count", -1)) != EVOLUTION_COUNT or tuple(evolution.get("layers", [])) != LAYERS: errors.append("evolution census")
    for generation in (1, 2, 3):
        entries = [value for value in evolution.get("specimens", []) if int(value.get("generation", -1)) == generation]
        if len(entries) != 12 or len({value.get("family") for value in entries}) != 5: errors.append(f"evolution generation {generation}")
    for specimen in evolution.get("specimens", []):
        layout = specimen.get("layout", {}); cursor = 0
        for clip in specimen.get("clips", []):
            if int(clip.get("start_cell", -1)) != cursor: errors.append("evolution clip cursor")
            cursor += int(clip.get("frame_count", 0))
        if cursor != int(layout.get("frame_count", -1)): errors.append("evolution frame census")
        for layer in LAYERS:
            record = specimen.get("layers", {}).get(layer, {}); path = root / PurePosixPath(str(record.get("path", "")))
            try:
                with Image.open(path) as image:
                    if image.mode != "RGBA" or image.size != (int(layout["columns"]) * 48, int(layout["rows"]) * 48): errors.append("evolution atlas")
            except (OSError, KeyError, TypeError, ValueError): errors.append("evolution atlas decode")
    expected_bundle = _bundle_id(index.get("fusion", {}), index.get("latent", {}), evolution, inventory)
    if index.get("bundle_id") != expected_bundle: errors.append("bundle id")
    if index.get("generator_sha256") != _sha_file(Path(__file__)): errors.append("generator hash")
    return errors


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big")); digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    return digest.hexdigest()


def repeat_check() -> dict[str, Any]:
    work = PROJECT_ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="genetics-workshop-", dir=work) as temporary:
        root = Path(temporary)
        sync_genetics_workshop(root / "a")
        sync_genetics_workshop(root / "b")
        first = _tree_digest(root / "a"); second = _tree_digest(root / "b")
        return {"passed": first == second, "first_tree_sha256": first, "second_tree_sha256": second}


def main(arguments: Iterable[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repeat-check", action="store_true")
    args = parser.parse_args(arguments)
    index = sync_genetics_workshop(args.destination)
    replay = repeat_check() if args.repeat_check else {"passed": None}
    errors = validate_genetics_workshop(Path(args.destination) / "asset_index.json")
    report = {
        "format": "nullvector-neural-genetics-workshop-sync-report-v1",
        "status": "passed" if not errors and replay.get("passed") is not False else "failed",
        "passed": not errors and replay.get("passed") is not False,
        "bundle_id": index["bundle_id"],
        "asset_count": index["asset_count"],
        "validation_errors": errors,
        "repeat_check": replay,
        "counts": {"fusion": FUSION_COUNT, "latent": LATENT_COUNT, "evolution": EVOLUTION_COUNT},
        "disk": {"free_bytes": shutil.disk_usage(Path(args.destination)).free, "minimum_free_bytes": MIN_FREE_BYTES},
    }
    if args.report: _write_exact(args.report, _canonical(report))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
