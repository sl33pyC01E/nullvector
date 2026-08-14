from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np

from .config import PROJECT_ROOT
from .morphology import FACING_NAMES, MOTION_NAMES, generate_motion_clip
from .morphology.constants import EMISSION, LAYER_NAMES
from .morphology_subtype_grammar import DEFAULT_OUTPUT as GRAMMAR_OUTPUT, VARIANT_NAMES
from .morphology_subtype_motion import DEFAULT_OUTPUT as MOTION_OUTPUT, _specimen
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import canonical_json_bytes, png_bytes
from .safety import require_disk_floor


FORMAT = "nullvector-native-morphology-subtype-runtime-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "game/generated/morphology_subtype_lab/v1"
SCHEMA = PROJECT_ROOT / "shared/schema/morphology_subtype_runtime.schema.json"
LAYERS = ("composite", "semantic", "emission")
ATLAS_COLUMNS = 16
ATLAS_ROWS = 12
ATLAS_WIDTH = 768
ATLAS_HEIGHT = 576
CELL_SIZE = 48
SOURCE_FILES = (
    "forge/morphology_subtype_runtime_sync.py",
    "shared/schema/morphology_subtype_runtime.schema.json",
)
DEPENDENCY_FILES = (
    "forge/morphology_subtype_grammar.py",
    "forge/morphology_subtype_motion.py",
    "forge/morphology/motion.py",
    "forge/morphology/render.py",
)

SEMANTIC_COLORS = np.asarray([
    (0, 0, 0, 0), (44, 220, 255, 255), (130, 90, 255, 255), (255, 92, 187, 255),
    (82, 255, 133, 255), (255, 151, 69, 255), (120, 180, 255, 255), (255, 230, 77, 255),
    (178, 255, 83, 255), (255, 75, 75, 255), (246, 250, 255, 255), (123, 255, 232, 255),
    (222, 95, 255, 255),
], dtype=np.uint8)


def _hash_files(domain: bytes, paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256(domain)
    for relative in paths:
        digest.update(relative.encode("utf-8") + b"\0" + (PROJECT_ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def source_sha256() -> str:
    return _hash_files(b"nullvector-morphology-subtype-runtime-source-v1\0", SOURCE_FILES)


def dependency_sha256() -> str:
    return _hash_files(b"nullvector-morphology-subtype-runtime-dependencies-v1\0", DEPENDENCY_FILES)


def _artifact(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _clip_schedule() -> list[tuple[str, str]]:
    result = []
    for motion in MOTION_NAMES:
        facings = FACING_NAMES if motion == "locomote" else ("north",)
        result.extend((motion, facing) for facing in facings)
    return result


def _frame_layers(frame) -> dict[str, np.ndarray]:
    semantic = SEMANTIC_COLORS[np.asarray(frame.tokens, dtype=np.uint8)]
    emission = np.zeros_like(frame.rgba)
    mask = frame.layers[EMISSION] > 0
    emission[mask, :3] = np.asarray((120, 248, 255), dtype=np.uint8)
    emission[mask, 3] = np.uint8(255)
    return {"composite": frame.rgba, "semantic": semantic, "emission": emission}


def _build_files() -> tuple[dict[str, bytes], dict[str, Any]]:
    grammar_manifest = GRAMMAR_OUTPUT / "morphology_subtype_grammar.json"
    motion_manifest = MOTION_OUTPUT / "morphology_subtype_motion.json"
    if not grammar_manifest.is_file() or not motion_manifest.is_file():
        raise FileNotFoundError("Published subtype grammar and motion authorities are required")
    grammar = json.loads(grammar_manifest.read_text(encoding="utf-8")); motion_authority = json.loads(motion_manifest.read_text(encoding="utf-8"))
    if grammar.get("status") != "ready" or motion_authority.get("status") != "ready":
        raise ValueError("Subtype source authority is not ready")
    files: dict[str, bytes] = {}
    identities = []
    total_clips = total_frames = 0
    schedule = _clip_schedule()
    for family_id in range(5):
        for variant in range(4):
            specimen = _specimen(family_id, variant)
            atlases = {layer: np.zeros((ATLAS_HEIGHT, ATLAS_WIDTH, 4), dtype=np.uint8) for layer in LAYERS}
            clips = []
            cell = 0
            for motion, facing in schedule:
                clip = generate_motion_clip(specimen, motion, facing=facing)
                start = cell
                for frame in clip.frames:
                    if cell >= ATLAS_COLUMNS * ATLAS_ROWS:
                        raise ValueError("Subtype runtime atlas capacity exceeded")
                    x, y = (cell % ATLAS_COLUMNS) * CELL_SIZE, (cell // ATLAS_COLUMNS) * CELL_SIZE
                    for layer, pixels in _frame_layers(frame).items():
                        atlases[layer][y:y + CELL_SIZE, x:x + CELL_SIZE] = pixels
                    cell += 1
                clips.append({
                    "motion": motion, "facing": facing, "start_cell": start,
                    "frame_count": len(clip.frames), "fps": clip.fps, "loop": clip.loop,
                    "clip_sha256": clip.sha256,
                })
            if cell != 181 or len(clips) != 20:
                raise ValueError("Subtype runtime clip/frame schedule differs")
            subtype_id = family_id * 4 + variant
            atlas_records = {}
            for layer, pixels in atlases.items():
                relative = f"atlases/{subtype_id:02d}_{layer}.png"
                payload = png_bytes(pixels); files[relative] = payload; atlas_records[layer] = _artifact(relative, payload)
            identities.append({
                "subtype_id": subtype_id, "family_id": family_id, "family": specimen.genome.family_name,
                "variant": variant, "subtype": VARIANT_NAMES[family_id][variant], "seed": specimen.genome.seed,
                "specimen_semantic_sha256": specimen.manifest["hashes"]["semantic_sha256"],
                "clip_count": len(clips), "frame_count": cell, "clips": clips, "atlases": atlas_records,
            })
            total_clips += len(clips); total_frames += cell
    catalog: dict[str, Any] = {
        "format": FORMAT, "status": "ready", "neural_output": False, "source_kind": "procedural-subtype-reference",
        "compiler": {"source_sha256": source_sha256(), "dependency_sha256": dependency_sha256(), "python_runtime_required": False},
        "authority": {
            "grammar_manifest": grammar_manifest.relative_to(PROJECT_ROOT).as_posix(), "grammar_manifest_sha256": sha256_file(grammar_manifest), "grammar_semantic_sha256": grammar["semantic_sha256"],
            "motion_manifest": motion_manifest.relative_to(PROJECT_ROOT).as_posix(), "motion_manifest_sha256": sha256_file(motion_manifest), "motion_semantic_sha256": motion_authority["semantic_sha256"],
        },
        "layout": {"cell_size": CELL_SIZE, "columns": ATLAS_COLUMNS, "rows": ATLAS_ROWS, "width": ATLAS_WIDTH, "height": ATLAS_HEIGHT},
        "layers": list(LAYERS), "motions": list(MOTION_NAMES), "facings": list(FACING_NAMES),
        "counts": {"identity_count": 20, "clip_count": total_clips, "frame_count": total_frames, "atlas_count": 60},
        "identities": identities,
        "runtime_contract": {"native_nearest_filter": True, "loop_terminal_frame_excluded_from_playback": True, "all_actions_north_facing": True, "locomotion_has_eight_facings": True, "procedural_reference_labeled": True, "python_runtime_required": False},
        "gates": {"all_20_subtypes": len(identities) == 20, "all_400_clips": total_clips == 400, "all_3620_frames": total_frames == 3620, "all_60_atlases": len(files) == 60, "source_authorities_ready": True},
    }
    catalog["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(catalog)).hexdigest()
    files["catalog.json"] = canonical_json_bytes(catalog)
    return files, catalog


def _validate_schema(catalog: Mapping[str, Any]) -> None:
    errors = sorted(Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).iter_errors(catalog), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Subtype runtime schema failed: {errors[0].message}")


def validate_runtime(root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    root = Path(root).resolve(); catalog_path = root / "catalog.json"; raw = catalog_path.read_bytes(); catalog = json.loads(raw); _validate_schema(catalog)
    if raw != canonical_json_bytes(catalog): raise ValueError("Subtype runtime catalog is not canonical JSON")
    if catalog["compiler"] != {"source_sha256": source_sha256(), "dependency_sha256": dependency_sha256(), "python_runtime_required": False}: raise ValueError("Subtype runtime provenance differs")
    semantic = {key: value for key, value in catalog.items() if key != "semantic_sha256"}
    if catalog["semantic_sha256"] != hashlib.sha256(canonical_json_bytes(semantic)).hexdigest(): raise ValueError("Subtype runtime semantic hash differs")
    for key in ("grammar_manifest", "motion_manifest"):
        source = PROJECT_ROOT.joinpath(*PurePosixPath(catalog["authority"][key]).parts)
        if sha256_file(source) != catalog["authority"][key + "_sha256"]: raise ValueError("Subtype runtime authority hash differs")
    expected_paths = {"catalog.json"}; bytes_total = len(raw)
    from PIL import Image
    for identity in catalog["identities"]:
        if len(identity["clips"]) != 20 or identity["frame_count"] != 181: raise ValueError("Subtype runtime identity counts differ")
        for artifact in identity["atlases"].values():
            expected_paths.add(artifact["path"]); path = root.joinpath(*PurePosixPath(artifact["path"]).parts)
            if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]: raise ValueError("Subtype runtime atlas integrity differs")
            with Image.open(path) as image:
                if image.mode != "RGBA" or image.size != (ATLAS_WIDTH, ATLAS_HEIGHT): raise ValueError("Subtype runtime atlas shape differs")
            bytes_total += artifact["bytes"]
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_paths != expected_paths: raise ValueError("Subtype runtime output closure differs")
    return {"passed": True, **catalog["counts"], "bytes": bytes_total, "semantic_sha256": catalog["semantic_sha256"], "catalog_sha256": sha256_file(catalog_path)}


def sync_runtime(destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    files, catalog = _build_files(); _validate_schema(catalog)
    if not all(catalog["gates"].values()): raise ValueError("Subtype runtime gates failed")
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=sum(map(len, files.values())) + 128 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists(): raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for relative, payload in files.items():
            target = staging.joinpath(*PurePosixPath(relative).parts); target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
            with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        os.replace(staging, destination)
    except BaseException: raise
    return validate_runtime(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync morphology subtype motion into native runtime atlases")
    parser.add_argument("--destination", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv); result = validate_runtime(args.destination) if args.validate_only else sync_runtime(args.destination)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
