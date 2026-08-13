from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from .backgrounds import background_catalog, load_background_crops
from .contact_sheet import (
    build_hero_contact_sheet,
    build_layer_contact_sheet,
    build_map_context_contact_sheet,
)
from .hashing import aligned_fields_hash, artifact_record
from .io import prepare_immutable_destination, require_disk_floor, write_json_new, write_png_new
from .metrics import METRICS_FORMAT, evaluate_style
from .model import LoadedGenerationBank, LoadedSourceSample, RenderedLayers
from .palette import PALETTE_FORMAT
from .rendering import LAYER_FORMAT, render_layers
from .schema import STYLE_BANK_SCHEMA, validate_schema
from .source import PROJECT_ROOT, load_generation_bank


STYLE_BANK_FORMAT = "nullvector-multifield-style-bank-v1"
STYLE_REPLAY_FORMAT = "nullvector-multifield-style-replay-v1"
COMPILER_ID = "deterministic-perceptual-categorical-presentation-v1"
COMPILER_SOURCE_FILES = (
    "forge/multifield_style/__init__.py",
    "forge/multifield_style/backgrounds.py",
    "forge/multifield_style/cli.py",
    "forge/multifield_style/color.py",
    "forge/multifield_style/compiler.py",
    "forge/multifield_style/contact_sheet.py",
    "forge/multifield_style/hashing.py",
    "forge/multifield_style/io.py",
    "forge/multifield_style/metrics.py",
    "forge/multifield_style/model.py",
    "forge/multifield_style/__main__.py",
    "forge/multifield_style/palette.py",
    "forge/multifield_style/procedural.py",
    "forge/multifield_style/rendering.py",
    "forge/multifield_style/replay.py",
    "forge/multifield_style/schema.py",
    "forge/multifield_style/source.py",
    "shared/schema/multifield_style_bank.schema.json",
    "shared/schema/multifield_style_procedural_reference.schema.json",
)

PRESENTATION_ARTIFACT_NAMES = (
    "base",
    "outline",
    "emission_core",
    "aura",
    "bloom_r1",
    "bloom_r2",
    "composite",
    "palette",
    "metrics",
)


def compiler_source_hash(project_root: Path = PROJECT_ROOT) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    digest.update(b"nullvector-multifield-style-source-v1\0")
    for relative in COMPILER_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Style compiler source member is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Style provenance input must live under project root: {path}") from error


def _write_sample(
    sample: LoadedSourceSample,
    destination: Path,
    backgrounds: tuple,
) -> tuple[dict[str, Any], RenderedLayers]:
    rendered = render_layers(sample.fields, sample.condition)
    fields_after = aligned_fields_hash(
        sample.fields.part,
        sample.fields.material,
        sample.fields.emission,
    )
    metrics = evaluate_style(
        sample.fields,
        sample.condition,
        rendered,
        backgrounds,
        fields_hash_after_render=fields_after,
    )
    if not metrics["passed"]:
        failed = [name for name, passed in metrics["gates"].items() if not passed]
        raise ValueError(
            f"Style sample {sample.condition.sample_id} failed objective gates: {failed}"
        )

    sample_root = destination / "sprites" / sample.condition.sample_id
    layers = {
        "base": rendered.base,
        "outline": rendered.outline,
        "emission_core": rendered.emission_core,
        "aura": rendered.aura,
        "bloom_r1": rendered.bloom_r1,
        "bloom_r2": rendered.bloom_r2,
        "composite": rendered.composite,
    }
    artifact_paths: dict[str, Path] = {}
    for name, pixels in layers.items():
        path = sample_root / f"{name}.png"
        write_png_new(path, pixels)
        artifact_paths[name] = path
    palette_path = sample_root / "palette.json"
    metrics_path = sample_root / "metrics.json"
    write_json_new(palette_path, rendered.palette)
    write_json_new(metrics_path, metrics)
    artifact_paths["palette"] = palette_path
    artifact_paths["metrics"] = metrics_path

    source_fields_path = (
        sample.fields_artifact.get("path")
        if isinstance(sample.fields_artifact.get("path"), str)
        else ""
    )
    record = {
        "condition": sample.condition.as_dict(),
        "source": {
            "compiled_fields_path": source_fields_path,
            "compiled_fields_bytes": int(sample.fields_artifact["bytes"]),
            "compiled_fields_artifact_sha256": str(sample.fields_artifact["sha256"]),
            "raw_fields_sha256": sample.raw_fields_sha256,
            "compiled_fields_sha256": sample.fields.aligned_sha256,
        },
        "presentation": {
            "layer_format": LAYER_FORMAT,
            "palette_format": PALETTE_FORMAT,
            "metrics_format": METRICS_FORMAT,
            "metrics_passed": True,
            "artifacts": {
                name: artifact_record(artifact_paths[name], destination)
                for name in PRESENTATION_ARTIFACT_NAMES
            },
        },
    }
    return record, rendered


def _parent_record(bank: LoadedGenerationBank) -> dict[str, Any]:
    provenance = bank.manifest["provenance"]
    return {
        "manifest_path": _project_relative(bank.manifest_path),
        "manifest_bytes": bank.manifest_bytes,
        "manifest_sha256": bank.manifest_sha256,
        "format": bank.manifest["format"],
        "status": bank.manifest["status"],
        "evaluation_source_hash": bank.manifest["evaluation_source_hash"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "canonical_ema_hash": provenance["canonical_ema_hash"],
        "training_source_hash": provenance["training_source_hash"],
        "legal_tuple_fingerprint": provenance["legal_tuple_fingerprint"],
    }


def compile_generation_bank(
    generation_manifest: Path,
    destination: Path,
    *,
    map_art_root: Path | None = None,
) -> dict[str, Any]:
    """Compile one immutable neural generation bank into derived style assets."""

    bank = load_generation_bank(generation_manifest)
    destination = prepare_immutable_destination(
        destination,
        planned_bytes=max(32 * 1024 * 1024, len(bank.samples) * 2 * 1024 * 1024),
    )
    background_root = (
        Path(map_art_root).resolve()
        if map_art_root is not None
        else PROJECT_ROOT / "outputs" / "map_art" / "packs"
    )
    backgrounds = load_background_crops(background_root)
    require_disk_floor(destination, planned_bytes=len(bank.samples) * 2 * 1024 * 1024)
    records: list[dict[str, Any]] = []
    rendered_samples: list[RenderedLayers] = []
    for sample in bank.samples:
        record, rendered = _write_sample(sample, destination, backgrounds)
        records.append(record)
        rendered_samples.append(rendered)

    conditions = [sample.condition for sample in bank.samples]
    layer_sheet_path = destination / "layer_contact_sheet.png"
    hero_sheet_path = destination / "composite_contact_sheet.png"
    map_sheet_path = destination / "map_context_contact_sheet.png"
    write_png_new(
        layer_sheet_path,
        build_layer_contact_sheet(
            conditions,
            rendered_samples,
            title="NEURAL STYLE BANK · CALIBRATED ACCEPTED FIELDS",
        ),
    )
    write_png_new(
        hero_sheet_path,
        build_hero_contact_sheet(
            conditions,
            rendered_samples,
            title="NEURAL PRESENTATION SMOKE BANK",
        ),
    )
    write_png_new(
        map_sheet_path,
        build_map_context_contact_sheet(
            conditions,
            rendered_samples,
            backgrounds,
            title="NEURAL PRESENTATION · MAP CONTRAST CONTEXT",
        ),
    )
    require_disk_floor(destination)
    manifest = {
        "format": STYLE_BANK_FORMAT,
        "status": "ready",
        "source_kind": "neural-generation-bank",
        "compiler": {
            "id": COMPILER_ID,
            "source_sha256": compiler_source_hash(),
            "image_size": 48,
            "resampling": "none-native-48px",
            "color_space": "bounded-oklch-to-srgb8",
        },
        "authority": {
            "raw_categorical_fields_remain_authoritative": True,
            "presentation_is_derived_only": True,
            "rig_authority_modified": False,
            "collision_authority_modified": False,
            "aura_is_effect_not_body": True,
        },
        "parent": _parent_record(bank),
        "background_catalog": background_catalog(backgrounds, PROJECT_ROOT),
        "disk_guard": {"floor_bytes": 100 * 1024**3, "enforced": True},
        "contact_sheets": {
            "layers": artifact_record(layer_sheet_path, destination),
            "composites": artifact_record(hero_sheet_path, destination),
            "map_context": artifact_record(map_sheet_path, destination),
        },
        "sample_count": len(records),
        "all_metrics_passed": True,
        "samples": records,
    }
    validate_schema(manifest, STYLE_BANK_SCHEMA)
    write_json_new(destination / "style_manifest.json", manifest)
    return manifest


def load_style_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.name != "style_manifest.json" or not path.is_file() or path.is_symlink():
        raise ValueError("Style manifest must be the canonical regular style_manifest.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(payload, allow_nan=False)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Style manifest is not strict UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Style manifest root must be an object")
    validate_schema(payload, STYLE_BANK_SCHEMA)
    return payload
