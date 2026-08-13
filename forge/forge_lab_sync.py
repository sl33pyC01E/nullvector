from __future__ import annotations

"""Build the compact, engine-facing ForgeLab asset bank.

The source for the lab remains the replayable morphology and map-art outputs.
This module intentionally exports only PNGs and JSON into ``res://``: no model
checkpoints, corpora, or NumPy archives are copied into the Godot project.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .morphology import (
    FACING_NAMES,
    MOTION_RENDERER_VERSION,
    MOTION_NAMES,
    RENDERER_VERSION,
    MorphologyGenome,
    assert_valid_motion_clip,
    generate_motion_clip,
    render_specimen,
)
from .morphology.constants import ROLE_NAMES
from .map_art.provenance import source_hash as map_art_source_hash
from .map_art.validate import validate_art_pack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTION_SOURCE = PROJECT_ROOT / "outputs" / "morphology_motion"
MAP_ART_SOURCE = PROJECT_ROOT / "outputs" / "map_art"
DEFAULT_DESTINATION = PROJECT_ROOT / "game" / "generated" / "v2"
MIN_FREE_BYTES = 100 * 1024**3
MOTION_COLUMNS = 16
MOTION_CELL_SIZE = 48
MAP_CELL_SIZE = 384
MAP_COLUMNS = 4
MAP_THEMES = ("arena", "rooms", "caves", "archipelago", "garden", "anomaly")
MAP_LAYERS = (
    "composite",
    "base_color",
    "emissive",
    "collision",
    "occlusion",
    "autotile",
    "elevation_edges",
    "objects",
    "hazard",
)
EXPECTED_FAMILIES = ("humanoid", "animalian", "plantlike", "anomaly", "machine")
ASSET_INDEX_FORMAT = "nullvector-forge-lab-assets-v1"


@dataclass(frozen=True, slots=True)
class SyncResult:
    destination: Path
    index_path: Path
    index: dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_if_changed(destination: Path, payload: bytes) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_bytes() == payload:
        return False
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return True


def _write_json(destination: Path, payload: Any) -> bool:
    return _write_if_changed(destination, _canonical_json(payload))


def _write_png(destination: Path, image: Image.Image) -> tuple[str, bool]:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    payload = buffer.getvalue()
    return _sha256_bytes(payload), _write_if_changed(destination, payload)


def _copy_if_changed(source: Path, destination: Path) -> bool:
    if destination.is_file() and _sha256_file(source) == _sha256_file(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return True


def _guard_disk(destination: Path, planned_bytes: int = 128 * 1024**2) -> dict[str, int | bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(destination.parent)
    projected = usage.free - planned_bytes
    if projected < MIN_FREE_BYTES:
        raise RuntimeError(
            "ForgeLab sync would breach the 100 GiB free-space floor: "
            f"free={usage.free}, planned={planned_bytes}, projected={projected}"
        )
    # The engine-facing index is a content manifest and therefore must not
    # depend on ambient disk usage.  The live free-space measurement is still
    # enforced above and is emitted by the CLI audit report instead.
    return {
        "guard_passed": True,
        "planned_bytes": planned_bytes,
        "minimum_free_bytes": MIN_FREE_BYTES,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _motion_lookup(bank: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in bank.get("clips", []):
        key = (str(entry["family"]), str(entry["motion"]), str(entry["facing"]))
        if key in lookup:
            raise ValueError(f"Duplicate source motion clip {key}")
        lookup[key] = entry
    return lookup


def _validate_motion_source_contract(
    bank: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, dict[str, Any]]]:
    if bank.get("format") != "neural-morphology-motion-bank-v1":
        raise ValueError("Unsupported morphology motion bank format")
    families = tuple(str(value) for value in bank.get("families", []))
    motions = tuple(str(value) for value in bank.get("motion_names", []))
    if families != EXPECTED_FAMILIES:
        raise ValueError(f"Unexpected morphology family order: {families}")
    if motions != tuple(MOTION_NAMES):
        raise ValueError(f"Unexpected motion order: {motions}")
    if str(bank.get("facing", "")) != "north":
        raise ValueError("Source motion bank must contain the canonical north-facing clips")

    source_entries = list(bank.get("sources", []))
    sources = {str(entry.get("family", "")): entry for entry in source_entries}
    if len(source_entries) != len(sources) or tuple(sources) != families:
        raise ValueError("Motion source specimens do not cover every family exactly once")
    for family in families:
        source = sources[family]
        if str(source.get("renderer_version", "")) != RENDERER_VERSION:
            raise ValueError(
                f"Stale morphology renderer for {family}: "
                f"{source.get('renderer_version')!r}, expected {RENDERER_VERSION!r}"
            )
        genome = dict(source.get("genome", {}))
        training = dict(source.get("training_contract", {}))
        role_id = int(genome.get("role_id", -1))
        if not 0 <= role_id < len(ROLE_NAMES):
            raise ValueError(f"Invalid role id for {family}: {role_id}")
        if int(training.get("role_id", -1)) != role_id:
            raise ValueError(f"Role conditioning contract mismatch for {family}")
        if str(training.get("role_name", "")) != ROLE_NAMES[role_id]:
            raise ValueError(f"Role name contract mismatch for {family}")

    lookup = _motion_lookup(bank)
    expected_keys = {
        (family, motion, "north")
        for family in families
        for motion in motions
    }
    if set(lookup) != expected_keys:
        missing = sorted(expected_keys - set(lookup))
        unexpected = sorted(set(lookup) - expected_keys)
        raise ValueError(
            f"Canonical motion clip matrix mismatch: missing={missing}, unexpected={unexpected}"
        )
    renderer_versions = {
        str(entry.get("motion_renderer_version", ""))
        for entry in lookup.values()
    }
    if renderer_versions != {MOTION_RENDERER_VERSION}:
        raise ValueError(
            "Motion renderer contract mismatch: "
            f"observed={sorted(renderer_versions)}, expected={MOTION_RENDERER_VERSION!r}"
        )
    if int(bank.get("clip_count", -1)) != len(expected_keys):
        raise ValueError("Source motion clip count does not match the canonical matrix")
    return families, motions, sources


def _sync_motion_bank(destination: Path) -> dict[str, Any]:
    manifest_path = MOTION_SOURCE / "morphology_motion_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    bank = _load_json(manifest_path)
    families, motions, sources = _validate_motion_source_contract(bank)
    source_lookup = _motion_lookup(bank)

    motion_destination = destination / "motion"
    _copy_if_changed(manifest_path, motion_destination / "source_manifest.json")
    atlas_entries: list[dict[str, Any]] = []
    all_clips: list[dict[str, Any]] = []

    for family in families:
        source = sources[family]
        genome = MorphologyGenome.from_dict(dict(source["genome"]))
        specimen = render_specimen(genome)
        observed_semantic = str(specimen.manifest["hashes"]["semantic_sha256"])
        expected_semantic = str(source["hashes"]["semantic_sha256"])
        if observed_semantic != expected_semantic:
            raise ValueError(f"Replayed {family} source specimen hash does not match the bank")

        clips = []
        frame_count = 0
        for motion in motions:
            for facing in FACING_NAMES:
                clip = generate_motion_clip(specimen, motion, facing=facing)
                assert_valid_motion_clip(clip)
                if facing == "north":
                    source_clip = source_lookup.get((family, motion, facing))
                    if source_clip is None:
                        raise ValueError(f"Source bank is missing {(family, motion, facing)}")
                    source_hash = str(source_clip["hashes"]["clip_sha256"])
                    if clip.sha256 != source_hash:
                        raise ValueError(f"Replayed clip hash mismatch for {(family, motion, facing)}")
                clips.append(clip)
                frame_count += len(clip.frames)

        rows = (frame_count + MOTION_COLUMNS - 1) // MOTION_COLUMNS
        atlas = Image.new(
            "RGBA",
            (MOTION_COLUMNS * MOTION_CELL_SIZE, rows * MOTION_CELL_SIZE),
            (0, 0, 0, 0),
        )
        cursor = 0
        for clip in clips:
            start_cell = cursor
            for frame in clip.frames:
                x = (cursor % MOTION_COLUMNS) * MOTION_CELL_SIZE
                y = (cursor // MOTION_COLUMNS) * MOTION_CELL_SIZE
                atlas.paste(Image.fromarray(frame.rgba), (x, y))
                cursor += 1
            all_clips.append(
                {
                    "id": clip.manifest["id"],
                    "family": family,
                    "motion": clip.motion,
                    "facing": clip.facing,
                    "fps": clip.fps,
                    "loop": clip.loop,
                    "frame_count": len(clip.frames),
                    "start_cell": start_cell,
                    "atlas_columns": MOTION_COLUMNS,
                    "cell_size": MOTION_CELL_SIZE,
                    "clip_sha256": clip.sha256,
                    "frame_sha256": list(clip.manifest["frame_sha256"]),
                    "events": list(clip.manifest["events"]),
                    "metrics": dict(clip.manifest["metrics"]),
                }
            )

        relative_atlas = f"motion/{family}_motion_atlas.png"
        atlas_path = destination / relative_atlas
        atlas_hash, _ = _write_png(atlas_path, atlas)
        atlas_entries.append(
            {
                "family": family,
                "source_id": source["id"],
                "source_seed": source["seed"],
                "source_semantic_sha256": expected_semantic,
                "source_genome_sha256": source["hashes"]["genome_sha256"],
                "source_renderer_version": source["renderer_version"],
                "source_role_id": source["training_contract"]["role_id"],
                "source_role_name": source["training_contract"]["role_name"],
                "atlas": relative_atlas,
                "atlas_sha256": atlas_hash,
                "atlas_size": list(atlas.size),
                "columns": MOTION_COLUMNS,
                "rows": rows,
                "cell_size": MOTION_CELL_SIZE,
                "frame_count": frame_count,
                "palette": source.get("palette", {}),
                "genome": source.get("genome", {}),
            }
        )

    return {
        "status": "ready",
        "source_manifest": "motion/source_manifest.json",
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_format": bank["format"],
        "renderer": MOTION_RENDERER_VERSION,
        "source_morphology_renderer": RENDERER_VERSION,
        "families": list(families),
        "motions": list(motions),
        "facings": list(FACING_NAMES),
        "atlases": atlas_entries,
        "clips": all_clips,
        "clip_count": len(all_clips),
        "frame_count": sum(entry["frame_count"] for entry in atlas_entries),
    }


def _rgb(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(array, dtype=np.uint8))


def _upscale_debug(array: np.ndarray) -> Image.Image:
    return _rgb(array).resize((MAP_CELL_SIZE, MAP_CELL_SIZE), Image.Resampling.NEAREST)


def _value_palette(values: np.ndarray, palette: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.int64), 0, len(palette) - 1)
    return palette[clipped]


def _map_debug_layers(semantics: dict[str, np.ndarray]) -> dict[str, Image.Image]:
    collision = np.asarray(semantics["collision"], dtype=np.uint8)
    occlusion = np.asarray(semantics["occlusion"], dtype=np.uint8)
    autotile = np.asarray(semantics["autotile_mask"], dtype=np.uint8)
    elevation = np.asarray(semantics["elevation_edge_mask"], dtype=np.uint8)
    prop_id = np.asarray(semantics["prop_id"])
    decal_id = np.asarray(semantics["decal_id"])
    expected = (48, 48)
    for name, value in (
        ("collision", collision),
        ("occlusion", occlusion),
        ("autotile_mask", autotile),
        ("elevation_edge_mask", elevation),
        ("prop_id", prop_id),
        ("decal_id", decal_id),
    ):
        if value.shape != expected:
            raise ValueError(f"Map-art semantic {name} has shape {value.shape}, expected {expected}")

    collision_rgb = np.zeros((*expected, 3), dtype=np.uint8)
    collision_rgb[:] = (5, 10, 20)
    collision_rgb[collision > 0] = (255, 43, 116)
    occlusion_palette = np.asarray(
        [(4, 8, 18), (25, 95, 118), (122, 61, 218), (243, 252, 255)], dtype=np.uint8
    )
    bit_palette = np.asarray(
        [
            (4, 8, 18), (25, 221, 255), (65, 255, 157), (60, 166, 255),
            (255, 74, 149), (134, 91, 255), (255, 185, 64), (193, 105, 255),
            (252, 84, 84), (74, 247, 219), (177, 255, 78), (98, 141, 255),
            (255, 92, 220), (255, 147, 72), (196, 246, 255), (243, 252, 255),
        ],
        dtype=np.uint8,
    )
    object_rgb = np.zeros((*expected, 3), dtype=np.uint8)
    object_rgb[:] = (4, 8, 18)
    object_rgb[decal_id >= 0] = (176, 55, 255)
    object_rgb[prop_id >= 0] = (45, 239, 255)
    object_rgb[(prop_id >= 0) & (decal_id >= 0)] = (243, 252, 255)
    return {
        "collision": _upscale_debug(collision_rgb),
        "occlusion": _upscale_debug(_value_palette(occlusion, occlusion_palette)),
        "autotile": _upscale_debug(_value_palette(autotile, bit_palette)),
        "elevation_edges": _upscale_debug(_value_palette(elevation, bit_palette)),
        "objects": _upscale_debug(object_rgb),
    }


def _select_map_packs() -> dict[str, Path]:
    pack_root = MAP_ART_SOURCE / "packs"
    selected: dict[str, Path] = {}
    active_source_hash = map_art_source_hash()
    for theme in MAP_THEMES:
        manifest_paths = sorted(
            path
            for path in pack_root.glob(f"{theme}-*/manifest.json")
            if path.is_file()
        )
        if not manifest_paths:
            raise FileNotFoundError(f"No map-art pack found for theme {theme!r}")
        # Ignore packs from superseded renderer sources instead of letting an
        # old lexically-first directory silently enter the native bank.
        candidates = []
        for manifest_path in manifest_paths:
            manifest = _load_json(manifest_path)
            renderer = dict(manifest.get("renderer", {}))
            if (
                renderer.get("name") == "nullvector-map-art-forge"
                and renderer.get("source_sha256") == active_source_hash
                and bool(renderer.get("deterministic"))
            ):
                candidates.append(manifest_path.parent)
        if not candidates:
            raise ValueError(
                f"No current-renderer map-art pack found for theme {theme!r}; "
                f"expected source SHA-256 {active_source_hash}"
            )
        # Stable lexical selection prevents filesystem enumeration order from
        # changing which current pack enters the engine bank.
        selected[theme] = candidates[0]
    return selected


def _load_hazard_frames(pack: Path, manifest: dict[str, Any]) -> list[Image.Image]:
    source = Image.open(pack / "hazard_frames.png").convert("RGBA")
    animation = manifest["animation"]
    frame_count = int(animation["frame_count"])
    columns = int(animation["grid"]["columns"])
    rows = int(animation["grid"]["rows"])
    expected_size = (columns * MAP_CELL_SIZE, rows * MAP_CELL_SIZE)
    if source.size != expected_size:
        raise ValueError(
            f"Hazard frame sheet has size {source.size}, expected {expected_size}: {pack}"
        )
    if frame_count < 1 or frame_count > columns * rows:
        raise ValueError(f"Invalid hazard frame count {frame_count}: {pack}")
    frames = []
    for index in range(frame_count):
        x = (index % columns) * MAP_CELL_SIZE
        y = (index // columns) * MAP_CELL_SIZE
        frames.append(source.crop((x, y, x + MAP_CELL_SIZE, y + MAP_CELL_SIZE)))
    return frames


def _sync_map_bank(destination: Path) -> dict[str, Any]:
    packs = _select_map_packs()
    active_source_hash = map_art_source_hash()
    entries: list[dict[str, Any]] = []
    for theme in MAP_THEMES:
        pack = packs[theme]
        manifest_path = pack / "manifest.json"
        manifest = _load_json(manifest_path)
        if manifest.get("theme") != theme:
            raise ValueError(f"Map pack theme mismatch in {manifest_path}")
        renderer = dict(manifest.get("renderer", {}))
        if renderer.get("source_sha256") != active_source_hash:
            raise ValueError(f"Map pack was produced by a stale renderer: {manifest_path}")
        pack_report = validate_art_pack(pack)
        if not bool(pack_report.get("passed")):
            raise ValueError(f"Map art pack failed strict validation: {pack_report}")
        if any(not bool(check.get("passed")) for check in manifest.get("validation", [])):
            raise ValueError(f"Map pack contains failed validation checks: {manifest_path}")
        source_manifest_relative = f"maps/{theme}_source_manifest.json"
        _copy_if_changed(manifest_path, destination / source_manifest_relative)

        with np.load(pack / "art_semantics.npz", allow_pickle=False) as archive:
            semantics = {name: np.asarray(archive[name]) for name in archive.files}
        debug = _map_debug_layers(semantics)
        base = Image.open(pack / "base_color.png").convert("RGB")
        preview = Image.open(pack / "preview.png").convert("RGB")
        emissive = Image.open(pack / "emissive.png").convert("RGB")
        for name, image in (("base_color", base), ("preview", preview), ("emissive", emissive)):
            if image.size != (MAP_CELL_SIZE, MAP_CELL_SIZE):
                raise ValueError(f"{theme} {name} has unexpected image size {image.size}")

        hazard_frames = _load_hazard_frames(pack, manifest)
        frame_count = len(hazard_frames)
        atlas_cells = 8 + frame_count
        rows = (atlas_cells + MAP_COLUMNS - 1) // MAP_COLUMNS
        atlas = Image.new("RGB", (MAP_COLUMNS * MAP_CELL_SIZE, rows * MAP_CELL_SIZE), (3, 6, 14))
        stills: list[tuple[str, Image.Image]] = [
            ("composite", preview),
            ("base_color", base),
            ("emissive", emissive),
            ("collision", debug["collision"]),
            ("occlusion", debug["occlusion"]),
            ("autotile", debug["autotile"]),
            ("elevation_edges", debug["elevation_edges"]),
            ("objects", debug["objects"]),
        ]
        layer_entries: list[dict[str, Any]] = []
        for cell, (name, image) in enumerate(stills):
            atlas.paste(image, ((cell % MAP_COLUMNS) * MAP_CELL_SIZE, (cell // MAP_COLUMNS) * MAP_CELL_SIZE))
            layer_entries.append({"name": name, "start_cell": cell, "frame_count": 1, "fps": 0})
        hazard_start = len(stills)
        dim_base = Image.blend(Image.new("RGB", base.size, (3, 6, 14)), base, 0.28).convert("RGBA")
        for offset, hazard in enumerate(hazard_frames):
            composed = Image.alpha_composite(dim_base, hazard).convert("RGB")
            cell = hazard_start + offset
            atlas.paste(composed, ((cell % MAP_COLUMNS) * MAP_CELL_SIZE, (cell // MAP_COLUMNS) * MAP_CELL_SIZE))
        duration_ms = int(manifest["animation"]["duration_ms"])
        layer_entries.append(
            {
                "name": "hazard",
                "start_cell": hazard_start,
                "frame_count": frame_count,
                "fps": round(1000.0 / duration_ms, 6),
            }
        )
        if tuple(entry["name"] for entry in layer_entries) != MAP_LAYERS:
            raise AssertionError("Map layer packing order drifted from the public contract")

        atlas_relative = f"maps/{theme}_map_atlas.png"
        atlas_hash, _ = _write_png(destination / atlas_relative, atlas)
        entries.append(
            {
                "theme": theme,
                "art_id": manifest["art_id"],
                "map_id": manifest["map_id"],
                "seed": manifest["seed"],
                "atlas": atlas_relative,
                "atlas_sha256": atlas_hash,
                "atlas_size": list(atlas.size),
                "columns": MAP_COLUMNS,
                "rows": rows,
                "cell_size": MAP_CELL_SIZE,
                "source_manifest": source_manifest_relative,
                "source_manifest_sha256": _sha256_file(manifest_path),
                "source_semantic_sha256": manifest["source"]["semantic_array_sha256"],
                "renderer": manifest["renderer"],
                "statistics": manifest["statistics"],
                "layers": layer_entries,
            }
        )
    return {
        "status": "ready",
        "renderer_source_sha256": active_source_hash,
        "themes": list(MAP_THEMES),
        "layers": list(MAP_LAYERS),
        "maps": entries,
        "map_count": len(entries),
    }


def sync_forge_lab_assets(
    destination: Path = DEFAULT_DESTINATION,
    *,
    allow_partial: bool = False,
) -> SyncResult:
    destination = destination.resolve()
    disk_budget = _guard_disk(destination)
    errors: list[dict[str, str]] = []
    try:
        motion = _sync_motion_bank(destination)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        if not allow_partial:
            raise
        motion = {"status": "unavailable", "families": [], "motions": [], "facings": [], "clips": []}
        errors.append({"subsystem": "motion", "error": str(exc)})
    try:
        maps = _sync_map_bank(destination)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        if not allow_partial:
            raise
        maps = {"status": "unavailable", "themes": [], "layers": [], "maps": []}
        errors.append({"subsystem": "maps", "error": str(exc)})

    index = {
        "format": ASSET_INDEX_FORMAT,
        "engine": "Godot 4.3",
        "pixel_filter": "nearest",
        "python_runtime_required": False,
        "runtime_asset_extensions": [".json", ".png"],
        "coordinate_system": "atlas cells row-major from top-left",
        "source_root": "outputs/",
        "generator": {
            "module": "forge.forge_lab_sync",
            "source_sha256": _sha256_file(Path(__file__)),
            "deterministic": True,
        },
        "disk_budget": disk_budget,
        "motion": motion,
        "maps": maps,
        "errors": errors,
    }
    index_path = destination / "asset_index.json"
    _write_json(index_path, index)
    return SyncResult(destination=destination, index_path=index_path, index=index)


def validate_synced_assets(index_path: Path) -> list[str]:
    """Validate the complete engine bank without invoking either renderer.

    This is intentionally stricter than Godot's resource loader: it catches a
    stale-but-loadable atlas, a partial sync, out-of-bounds atlas cells, and
    unexpected files left in the v2 bank.
    """

    errors: list[str] = []
    index_path = index_path.resolve()
    root = index_path.parent
    try:
        index = _load_json(index_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"index: {exc}"]
    if index.get("format") != ASSET_INDEX_FORMAT:
        errors.append("index format mismatch")
    if index.get("engine") != "Godot 4.3":
        errors.append("engine contract mismatch")
    if index.get("pixel_filter") != "nearest":
        errors.append("pixel filter is not nearest")
    if index.get("errors"):
        errors.append("asset index contains subsystem errors")
    generator = dict(index.get("generator", {}))
    if generator.get("module") != "forge.forge_lab_sync" or not bool(generator.get("deterministic")):
        errors.append("generator contract mismatch")
    if generator.get("source_sha256") != _sha256_file(Path(__file__)):
        errors.append("asset index was produced by stale sync source")
    disk_budget = dict(index.get("disk_budget", {}))
    if disk_budget != {
        "guard_passed": True,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "planned_bytes": 128 * 1024**2,
    }:
        errors.append("disk guard contract is missing or non-deterministic")

    expected_files = {"asset_index.json"}

    def asset_path(relative: Any, label: str) -> Path | None:
        text = str(relative).replace("\\", "/")
        candidate = (root / text).resolve()
        if not candidate.is_relative_to(root):
            errors.append(f"{label} escapes asset root: {text}")
            return None
        expected_files.add(candidate.relative_to(root).as_posix())
        return candidate

    def validate_hashed_file(relative: Any, expected_hash: Any, label: str) -> Path | None:
        path = asset_path(relative, label)
        if path is None:
            return None
        if not path.is_file():
            errors.append(f"{label} missing: {path}")
        elif _sha256_file(path) != str(expected_hash):
            errors.append(f"{label} hash mismatch: {path}")
        return path

    def validate_png(
        relative: Any,
        expected_hash: Any,
        expected_size: Any,
        label: str,
    ) -> Path | None:
        path = validate_hashed_file(relative, expected_hash, label)
        if path is None or not path.is_file():
            return path
        try:
            with Image.open(path) as image:
                observed_size = image.size
                image.verify()
            if list(observed_size) != list(expected_size):
                errors.append(f"{label} size mismatch: {observed_size} != {expected_size}")
        except (OSError, ValueError) as exc:
            errors.append(f"{label} is not a valid PNG: {exc}")
        return path

    motion = dict(index.get("motion", {}))
    if motion.get("status") != "ready":
        errors.append("motion bank is not ready")
    else:
        families = tuple(motion.get("families", []))
        motions = tuple(motion.get("motions", []))
        facings = tuple(motion.get("facings", []))
        if families != EXPECTED_FAMILIES:
            errors.append("motion families mismatch")
        if motions != tuple(MOTION_NAMES):
            errors.append("motion names mismatch")
        if facings != tuple(FACING_NAMES):
            errors.append("motion facings mismatch")
        if motion.get("renderer") != MOTION_RENDERER_VERSION:
            errors.append("motion renderer version mismatch")
        if motion.get("source_morphology_renderer") != RENDERER_VERSION:
            errors.append("motion bank uses a stale morphology renderer")
        validate_hashed_file(
            motion.get("source_manifest", ""),
            motion.get("source_manifest_sha256", ""),
            "motion source manifest",
        )

        clip_entries = list(motion.get("clips", []))
        expected_keys = {
            (family, motion_name, facing)
            for family in EXPECTED_FAMILIES
            for motion_name in MOTION_NAMES
            for facing in FACING_NAMES
        }
        observed_keys: set[tuple[str, str, str]] = set()
        clip_frames_by_family = {family: 0 for family in EXPECTED_FAMILIES}
        clip_ids: set[str] = set()
        atlas_by_family = {
            str(entry.get("family", "")): entry
            for entry in motion.get("atlases", [])
        }
        if tuple(atlas_by_family) != EXPECTED_FAMILIES:
            errors.append("motion atlas family order or coverage mismatch")
        for family in EXPECTED_FAMILIES:
            atlas = dict(atlas_by_family.get(family, {}))
            if not atlas:
                continue
            if atlas.get("source_renderer_version") != RENDERER_VERSION:
                errors.append(f"{family} atlas uses a stale morphology renderer")
            role_id = int(atlas.get("source_role_id", -1))
            if not 0 <= role_id < len(ROLE_NAMES):
                errors.append(f"{family} atlas role id is invalid")
            elif atlas.get("source_role_name") != ROLE_NAMES[role_id]:
                errors.append(f"{family} atlas role name mismatch")
            validate_png(
                atlas.get("atlas", ""),
                atlas.get("atlas_sha256", ""),
                atlas.get("atlas_size", []),
                f"{family} motion atlas",
            )
            expected_width = int(atlas.get("columns", 0)) * int(atlas.get("cell_size", 0))
            expected_height = int(atlas.get("rows", 0)) * int(atlas.get("cell_size", 0))
            if list(atlas.get("atlas_size", [])) != [expected_width, expected_height]:
                errors.append(f"{family} motion atlas grid size mismatch")

        for clip in clip_entries:
            family = str(clip.get("family", ""))
            key = (family, str(clip.get("motion", "")), str(clip.get("facing", "")))
            if key in observed_keys:
                errors.append(f"duplicate motion clip: {key}")
            observed_keys.add(key)
            clip_id = str(clip.get("id", ""))
            if not clip_id or clip_id in clip_ids:
                errors.append(f"duplicate or missing motion clip id: {clip_id!r}")
            clip_ids.add(clip_id)
            frame_count = int(clip.get("frame_count", 0))
            if frame_count < 1 or float(clip.get("fps", 0)) <= 0:
                errors.append(f"invalid motion timing: {key}")
            if len(clip.get("frame_sha256", [])) != frame_count:
                errors.append(f"motion frame hash count mismatch: {key}")
            atlas = dict(atlas_by_family.get(family, {}))
            capacity = int(atlas.get("columns", 0)) * int(atlas.get("rows", 0))
            start = int(clip.get("start_cell", -1))
            if (
                int(clip.get("cell_size", -1)) != int(atlas.get("cell_size", 0))
                or int(clip.get("atlas_columns", -1)) != int(atlas.get("columns", 0))
                or start < 0
                or start + frame_count > capacity
            ):
                errors.append(f"motion atlas region is out of bounds: {key}")
            if family in clip_frames_by_family:
                clip_frames_by_family[family] += frame_count
        if observed_keys != expected_keys:
            errors.append("motion clip matrix is incomplete or contains unexpected entries")
        if int(motion.get("clip_count", -1)) != len(expected_keys) or len(clip_entries) != len(expected_keys):
            errors.append("motion clip count mismatch")
        total_motion_frames = sum(clip_frames_by_family.values())
        if int(motion.get("frame_count", -1)) != total_motion_frames:
            errors.append("motion frame count mismatch")
        for family, observed in clip_frames_by_family.items():
            if int(atlas_by_family.get(family, {}).get("frame_count", -1)) != observed:
                errors.append(f"{family} atlas frame count mismatch")

    maps = dict(index.get("maps", {}))
    if maps.get("status") != "ready":
        errors.append("map bank is not ready")
    else:
        if tuple(maps.get("themes", [])) != MAP_THEMES:
            errors.append("map themes mismatch")
        if tuple(maps.get("layers", [])) != MAP_LAYERS:
            errors.append("map layers mismatch")
        if maps.get("renderer_source_sha256") != map_art_source_hash():
            errors.append("map bank uses a stale renderer source")
        map_entries = list(maps.get("maps", []))
        if int(maps.get("map_count", -1)) != len(MAP_THEMES) or len(map_entries) != len(MAP_THEMES):
            errors.append("map count mismatch")
        if tuple(str(entry.get("theme", "")) for entry in map_entries) != MAP_THEMES:
            errors.append("map entry order or coverage mismatch")
        for entry in map_entries:
            theme = str(entry.get("theme", ""))
            validate_hashed_file(
                entry.get("source_manifest", ""),
                entry.get("source_manifest_sha256", ""),
                f"{theme} source manifest",
            )
            validate_png(
                entry.get("atlas", ""),
                entry.get("atlas_sha256", ""),
                entry.get("atlas_size", []),
                f"{theme} map atlas",
            )
            renderer = dict(entry.get("renderer", {}))
            if renderer.get("source_sha256") != map_art_source_hash():
                errors.append(f"{theme} map renderer source mismatch")
            columns = int(entry.get("columns", 0))
            rows = int(entry.get("rows", 0))
            cell_size = int(entry.get("cell_size", 0))
            if list(entry.get("atlas_size", [])) != [columns * cell_size, rows * cell_size]:
                errors.append(f"{theme} map atlas grid size mismatch")
            capacity = columns * rows
            layer_entries = list(entry.get("layers", []))
            names = tuple(str(layer.get("name", "")) for layer in layer_entries)
            if names != MAP_LAYERS:
                errors.append(f"map layer order mismatch: {theme}")
            for layer in layer_entries:
                name = str(layer.get("name", ""))
                frame_count = int(layer.get("frame_count", 0))
                start = int(layer.get("start_cell", -1))
                fps = float(layer.get("fps", 0))
                if frame_count < 1 or start < 0 or start + frame_count > capacity:
                    errors.append(f"map atlas region is out of bounds: {theme}/{name}")
                if (name == "hazard" and fps <= 0) or (name != "hazard" and fps != 0):
                    errors.append(f"map layer timing mismatch: {theme}/{name}")

    # Godot writes adjacent ``.import`` sidecars during editor import.  They
    # are engine cache metadata rather than bank source assets, so preserve
    # and ignore them here; every other unexpected file remains an error.
    ignored_engine_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".import"
    }
    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() != ".import"
    }
    missing_files = sorted(expected_files - observed_files)
    unexpected_files = sorted(observed_files - expected_files)
    if missing_files:
        errors.append(f"asset bank is missing files: {missing_files}")
    if unexpected_files:
        errors.append(f"asset bank contains stale or unexpected files: {unexpected_files}")
    unsupported = sorted(relative for relative in observed_files if Path(relative).suffix.lower() not in {".json", ".png"})
    if unsupported:
        errors.append(f"asset bank contains runtime-incompatible files: {unsupported}")
    return errors


def asset_inventory(index_path: Path) -> list[dict[str, Any]]:
    """Return stable file records used for determinism and deployment audits."""

    index_path = index_path.resolve()
    root = index_path.parent
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix())
        if path.is_file() and path.suffix.lower() in {".json", ".png"}
    ]


def _parse_args(arguments: Iterable[str] | None = None) -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--repeat-check", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = _parse_args(arguments)
    index_path = args.destination / "asset_index.json"
    if not args.verify_only:
        result = sync_forge_lab_assets(args.destination, allow_partial=args.allow_partial)
        index_path = result.index_path
    errors = validate_synced_assets(index_path)
    inventory = asset_inventory(index_path) if index_path.is_file() else []
    inventory_hash = _sha256_bytes(_canonical_json(inventory))
    repeat_check: dict[str, Any] = {"requested": bool(args.repeat_check), "passed": None}
    if args.repeat_check:
        if args.verify_only:
            errors.append("--repeat-check cannot be combined with --verify-only")
            repeat_check["passed"] = False
        else:
            first_index_hash = _sha256_file(index_path)
            first_inventory = inventory
            first_inventory_hash = inventory_hash
            second = sync_forge_lab_assets(args.destination, allow_partial=args.allow_partial)
            second_errors = validate_synced_assets(second.index_path)
            second_inventory = asset_inventory(second.index_path)
            second_inventory_hash = _sha256_bytes(_canonical_json(second_inventory))
            second_index_hash = _sha256_file(second.index_path)
            exact = (
                not second_errors
                and first_index_hash == second_index_hash
                and first_inventory_hash == second_inventory_hash
                and first_inventory == second_inventory
            )
            repeat_check = {
                "requested": True,
                "passed": exact,
                "first_index_sha256": first_index_hash,
                "second_index_sha256": second_index_hash,
                "first_asset_tree_sha256": first_inventory_hash,
                "second_asset_tree_sha256": second_inventory_hash,
                "byte_identical_asset_count": len(second_inventory) if first_inventory == second_inventory else 0,
                "errors": second_errors,
            }
            if not exact:
                errors.append("repeat sync was not byte-identical")
                errors.extend(f"repeat: {error}" for error in second_errors)
            inventory = second_inventory
            inventory_hash = second_inventory_hash
    report = {
        "passed": not errors,
        "index": str(index_path.resolve()),
        "index_sha256": _sha256_file(index_path) if index_path.is_file() else None,
        "asset_tree_sha256": inventory_hash,
        "asset_count": len(inventory),
        "asset_bytes": sum(int(entry["bytes"]) for entry in inventory),
        "asset_extensions": sorted({Path(entry["path"]).suffix.lower() for entry in inventory}),
        "python_runtime_required": False,
        "free_bytes_after": shutil.disk_usage(index_path.parent).free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "renderers": {
            "morphology": RENDERER_VERSION,
            "motion": MOTION_RENDERER_VERSION,
            "map_source_sha256": map_art_source_hash(),
        },
        "coverage": {
            "families": len(EXPECTED_FAMILIES),
            "motions": len(MOTION_NAMES),
            "facings": len(FACING_NAMES),
            "motion_clips": len(EXPECTED_FAMILIES) * len(MOTION_NAMES) * len(FACING_NAMES),
            "map_themes": len(MAP_THEMES),
            "map_layers": len(MAP_LAYERS),
        },
        "repeat_check": repeat_check,
        "assets": inventory,
        "errors": errors,
    }
    if args.report is not None:
        _write_json(args.report.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
