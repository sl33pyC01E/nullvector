from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
import zipfile

import numpy as np

from ..morphology.constants import (
    FAMILIES,
    LAYER_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SUBTYPE_NAMES,
)
from .backgrounds import background_catalog, load_background_crops
from .compiler import COMPILER_ID, PRESENTATION_ARTIFACT_NAMES, compiler_source_hash
from .contact_sheet import (
    build_hero_contact_sheet,
    build_layer_contact_sheet,
    build_map_context_contact_sheet,
)
from .hashing import aligned_fields_hash, artifact_record, canonical_json_bytes, sha256_file
from .io import prepare_immutable_destination, require_disk_floor, write_json_new, write_png_new
from .metrics import METRICS_FORMAT, evaluate_style
from .model import CategoricalFields, RenderedLayers, StyleCondition
from .palette import PALETTE_FORMAT
from .rendering import LAYER_FORMAT, render_layers
from .schema import PROCEDURAL_REFERENCE_SCHEMA, validate_schema
from .source import PROJECT_ROOT


PROCEDURAL_REFERENCE_FORMAT = "nullvector-multifield-style-procedural-reference-v1"
PROCEDURAL_PARENT_FORMAT = "broad-morphology-prototype-bank-v1"
PROCEDURAL_ARCHIVE_KEYS = {
    "layers",
    "tokens",
    "seeds",
    "families",
    "layer_names",
    "guide",
    "part_owner",
    "material",
    "emission_level",
    "morphologies",
    "subtypes",
    "roles",
    "genes",
}
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024


def _project_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Procedural source must live under project root: {path}") from error


def _resolve_local(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Procedural archive path must be a string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("Unsafe procedural archive path")
    if "\\" in relative:
        raise ValueError("Procedural archive paths must use POSIX separators")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Procedural archive escapes its source bank") from error
    if not target.is_file() or target.is_symlink():
        raise ValueError("Procedural archive must be a regular non-symlink file")
    return target


def _validate_archive_container(path: Path) -> None:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Procedural archive exceeds the compressed size bound")
    expected_members = {f"{name}.npy" for name in PROCEDURAL_ARCHIVE_KEYS}
    with zipfile.ZipFile(path, "r") as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or set(names) != expected_members:
            raise ValueError("Procedural archive ZIP members do not exactly match the contract")
        total = 0
        for entry in entries:
            if PurePosixPath(entry.filename).name != entry.filename:
                raise ValueError("Procedural archive contains nested or unsafe ZIP members")
            if entry.file_size < 0 or entry.file_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Procedural archive member exceeds the uncompressed bound")
            total += entry.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Procedural archive exceeds the total uncompressed bound")


def _training_arrays_hash(arrays: dict[str, np.ndarray], index: int) -> str:
    digest = hashlib.sha256()
    for name in ("guide", "part_owner", "material", "emission_level", "genes"):
        values = np.ascontiguousarray(arrays[name][index])
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    digest.update(
        bytes(
            (
                int(arrays["morphologies"][index]),
                int(arrays["subtypes"][index]),
                int(arrays["roles"][index]),
            )
        )
    )
    return digest.hexdigest()


def _load_procedural_source(
    manifest_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, np.ndarray]]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Procedural parent manifest must be a regular non-symlink file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        json.dumps(manifest, allow_nan=False)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Procedural manifest is not strict UTF-8 JSON: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != PROCEDURAL_PARENT_FORMAT:
        raise ValueError("Unsupported procedural parent manifest format")
    count = manifest.get("sprite_count")
    sprites = manifest.get("sprites")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Procedural sprite_count must be a positive integer")
    if not isinstance(sprites, list) or len(sprites) != count:
        raise ValueError("Procedural sprite entries disagree with sprite_count")
    if manifest.get("families") != list(FAMILIES):
        raise ValueError("Procedural family vocabulary/order mismatch")
    archive_path = _resolve_local(manifest_path.parent, manifest.get("semantic_archive"))
    _validate_archive_container(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != PROCEDURAL_ARCHIVE_KEYS:
            raise ValueError("Procedural archive array keys do not exactly match the contract")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected_shapes = {
        "layers": (count, 12, 48, 48),
        "tokens": (count, 48, 48),
        "seeds": (count,),
        "families": (count,),
        "layer_names": (12,),
        "guide": (count, 8, 48, 48),
        "part_owner": (count, 48, 48),
        "material": (count, 48, 48),
        "emission_level": (count, 48, 48),
        "morphologies": (count,),
        "subtypes": (count,),
        "roles": (count,),
        "genes": (count, 24),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Procedural archive {name} shape mismatch")
    if arrays["layer_names"].tolist() != list(LAYER_NAMES):
        raise ValueError("Procedural layer vocabulary/order mismatch")
    for name in ("layers", "tokens", "part_owner", "material", "emission_level", "morphologies", "subtypes", "roles"):
        if arrays[name].dtype != np.uint8:
            raise ValueError(f"Procedural archive {name} must be uint8")
    if arrays["seeds"].dtype != np.uint32 or arrays["guide"].dtype != np.float32 or arrays["genes"].dtype != np.float32:
        raise ValueError("Procedural numeric dtypes do not match the authoritative contract")
    if int(arrays["part_owner"].max(initial=0)) >= len(PART_OWNER_NAMES):
        raise ValueError("Procedural part owner exceeds its vocabulary")
    if int(arrays["material"].max(initial=0)) >= len(MATERIAL_NAMES):
        raise ValueError("Procedural material exceeds its vocabulary")
    if int(arrays["emission_level"].max(initial=0)) > 3:
        raise ValueError("Procedural emission exceeds its vocabulary")

    for index, sprite in enumerate(sprites):
        if not isinstance(sprite, dict):
            raise ValueError("Procedural sprite manifest entries must be objects")
        training = sprite.get("training_contract", {})
        hashes = sprite.get("hashes", {})
        morphology_id = int(arrays["morphologies"][index])
        subtype_id = int(arrays["subtypes"][index])
        role_id = int(arrays["roles"][index])
        if (
            sprite.get("seed") != int(arrays["seeds"][index])
            or sprite.get("family") != str(arrays["families"][index])
            or morphology_id >= len(FAMILIES)
            or str(arrays["families"][index]) != FAMILIES[morphology_id]
            or training.get("morphology_index") != morphology_id
            or training.get("subtype_id") != subtype_id
            or training.get("subtype_name") != SUBTYPE_NAMES[subtype_id]
            or training.get("role_id") != role_id
            or training.get("role_name") != ROLE_NAMES[role_id]
        ):
            raise ValueError(f"Procedural condition binding mismatch at index {index}")
        training_hash = _training_arrays_hash(arrays, index)
        semantic_hash = hashlib.sha256(
            arrays["layers"][index].tobytes() + arrays["tokens"][index].tobytes()
        ).hexdigest()
        if training.get("arrays_sha256") != training_hash or hashes.get("training_arrays_sha256") != training_hash:
            raise ValueError(f"Procedural training-array hash mismatch at index {index}")
        if hashes.get("semantic_sha256") != semantic_hash:
            raise ValueError(f"Procedural semantic hash mismatch at index {index}")
    return manifest, archive_path, arrays


def _select_family_references(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[int]:
    selected: list[int] = []
    for family_id in range(len(FAMILIES)):
        target_role = family_id
        candidates = [
            index
            for index in range(len(manifest["sprites"]))
            if int(arrays["morphologies"][index]) == family_id
            and int(arrays["roles"][index]) == target_role
        ]
        if not candidates:
            candidates = [
                index
                for index in range(len(manifest["sprites"]))
                if int(arrays["morphologies"][index]) == family_id
            ]
        if not candidates:
            raise ValueError(f"Procedural parent is missing family {FAMILIES[family_id]}")
        selected.append(min(candidates))
    return selected


def compile_procedural_reference_bank(
    procedural_manifest: Path,
    destination: Path,
    *,
    map_art_root: Path | None = None,
) -> dict[str, Any]:
    """Compile a clearly labeled, non-neural five-family reference bank."""

    parent, archive_path, arrays = _load_procedural_source(procedural_manifest)
    selected = _select_family_references(parent, arrays)
    destination = prepare_immutable_destination(destination, planned_bytes=32 * 1024 * 1024)
    background_root = (
        Path(map_art_root).resolve()
        if map_art_root is not None
        else PROJECT_ROOT / "outputs" / "map_art" / "packs"
    )
    backgrounds = load_background_crops(background_root)
    require_disk_floor(destination, planned_bytes=16 * 1024 * 1024)
    records: list[dict[str, Any]] = []
    conditions: list[StyleCondition] = []
    rendered_samples: list[RenderedLayers] = []
    for ordinal, index in enumerate(selected):
        morphology_id = int(arrays["morphologies"][index])
        subtype_id = int(arrays["subtypes"][index])
        role_id = int(arrays["roles"][index])
        condition = StyleCondition(
            sample_id=f"procedural_ref_{ordinal}_{FAMILIES[morphology_id]}",
            ordinal=ordinal,
            sample_seed=int(arrays["seeds"][index]),
            morphology_id=morphology_id,
            morphology_name=FAMILIES[morphology_id],
            subtype_id=subtype_id,
            subtype_name=SUBTYPE_NAMES[subtype_id],
            role_id=role_id,
            role_name=ROLE_NAMES[role_id],
        )
        condition.validate()
        fields_hash = aligned_fields_hash(
            arrays["part_owner"][index],
            arrays["material"][index],
            arrays["emission_level"][index],
        )
        fields = CategoricalFields(
            part=np.array(arrays["part_owner"][index], copy=True),
            material=np.array(arrays["material"][index], copy=True),
            emission=np.array(arrays["emission_level"][index], copy=True),
            aligned_sha256=fields_hash,
        )
        rendered = render_layers(fields, condition)
        metrics = evaluate_style(
            fields,
            condition,
            rendered,
            backgrounds,
            fields_hash_after_render=aligned_fields_hash(fields.part, fields.material, fields.emission),
        )
        if not metrics["passed"]:
            failed = [name for name, value in metrics["gates"].items() if not value]
            raise ValueError(f"Procedural reference {condition.sample_id} failed gates: {failed}")
        sample_root = destination / "sprites" / condition.sample_id
        layer_values = {
            "base": rendered.base,
            "outline": rendered.outline,
            "emission_core": rendered.emission_core,
            "aura": rendered.aura,
            "bloom_r1": rendered.bloom_r1,
            "bloom_r2": rendered.bloom_r2,
            "composite": rendered.composite,
        }
        artifact_paths: dict[str, Path] = {}
        for name, pixels in layer_values.items():
            path = sample_root / f"{name}.png"
            write_png_new(path, pixels)
            artifact_paths[name] = path
        palette_path, metrics_path = sample_root / "palette.json", sample_root / "metrics.json"
        write_json_new(palette_path, rendered.palette)
        write_json_new(metrics_path, metrics)
        artifact_paths.update({"palette": palette_path, "metrics": metrics_path})
        sprite = parent["sprites"][index]
        records.append(
            {
                "condition": condition.as_dict(),
                "source": {
                    "procedural_sprite_id": sprite["id"],
                    "procedural_index": index,
                    "training_arrays_sha256": sprite["hashes"]["training_arrays_sha256"],
                    "semantic_sha256": sprite["hashes"]["semantic_sha256"],
                    "aligned_fields_sha256": fields_hash,
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
        )
        conditions.append(condition)
        rendered_samples.append(rendered)

    layer_sheet_path = destination / "layer_contact_sheet.png"
    hero_sheet_path = destination / "five_family_reference.png"
    map_sheet_path = destination / "map_context_contact_sheet.png"
    write_png_new(
        layer_sheet_path,
        build_layer_contact_sheet(
            conditions,
            rendered_samples,
            title="PROCEDURAL REFERENCE · NOT NEURAL OUTPUT",
        ),
    )
    write_png_new(
        hero_sheet_path,
        build_hero_contact_sheet(
            conditions,
            rendered_samples,
            title="FIVE-FAMILY PROCEDURAL STYLE REFERENCE · NOT NEURAL",
        ),
    )
    write_png_new(
        map_sheet_path,
        build_map_context_contact_sheet(
            conditions,
            rendered_samples,
            backgrounds,
            title="PROCEDURAL FIVE-FAMILY REFERENCE · MAP CONTEXT · NOT NEURAL",
        ),
    )
    require_disk_floor(destination)
    manifest_path = Path(procedural_manifest).resolve()
    result = {
        "format": PROCEDURAL_REFERENCE_FORMAT,
        "status": "ready",
        "source_kind": "authoritative-procedural-reference",
        "neural_output": False,
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
            "procedural_reference_only": True,
            "neural_output_claimed": False,
        },
        "parent": {
            "manifest_path": _project_relative(manifest_path),
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": sha256_file(manifest_path),
            "format": parent["format"],
            "semantic_archive_path": _project_relative(archive_path),
            "semantic_archive_bytes": archive_path.stat().st_size,
            "semantic_archive_sha256": sha256_file(archive_path),
        },
        "background_catalog": background_catalog(backgrounds, PROJECT_ROOT),
        "disk_guard": {"floor_bytes": 100 * 1024**3, "enforced": True},
        "contact_sheets": {
            "layers": artifact_record(layer_sheet_path, destination),
            "five_family_reference": artifact_record(hero_sheet_path, destination),
            "map_context": artifact_record(map_sheet_path, destination),
        },
        "sample_count": len(records),
        "family_coverage": list(FAMILIES),
        "all_metrics_passed": True,
        "samples": records,
    }
    validate_schema(result, PROCEDURAL_REFERENCE_SCHEMA)
    write_json_new(destination / "procedural_reference_manifest.json", result)
    return result


def load_procedural_reference_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if (
        path.name != "procedural_reference_manifest.json"
        or not path.is_file()
        or path.is_symlink()
    ):
        raise ValueError(
            "Procedural reference manifest must be the canonical regular "
            "procedural_reference_manifest.json"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(payload, allow_nan=False)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Procedural reference manifest is not strict JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Procedural reference manifest root must be an object")
    validate_schema(payload, PROCEDURAL_REFERENCE_SCHEMA)
    return payload


def _safe_project_path(relative: object) -> Path:
    return _resolve_local(PROJECT_ROOT, relative)


def _replay_procedural_reference(path: Path, map_art_root: Path | None) -> dict[str, Any]:
    # Imported lazily to keep the generation-bank replay independent of this
    # explicitly non-neural reference adapter.
    from .replay import _artifact_bytes, _png_bytes

    manifest_path = Path(path).resolve()
    manifest = load_procedural_reference_manifest(manifest_path)
    root = manifest_path.parent.resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    current_source_hash = compiler_source_hash()
    checks["compiler_source_hash_exact"] = current_source_hash == manifest["compiler"]["source_sha256"]
    if not checks["compiler_source_hash_exact"]:
        errors.append("compiler source hash mismatch")
    parent_path = _safe_project_path(manifest["parent"]["manifest_path"])
    parent, archive_path, arrays = _load_procedural_source(parent_path)
    expected_parent = {
        "manifest_path": _project_relative(parent_path),
        "manifest_bytes": parent_path.stat().st_size,
        "manifest_sha256": sha256_file(parent_path),
        "format": parent["format"],
        "semantic_archive_path": _project_relative(archive_path),
        "semantic_archive_bytes": archive_path.stat().st_size,
        "semantic_archive_sha256": sha256_file(archive_path),
    }
    checks["parent_exact"] = expected_parent == manifest["parent"]
    if not checks["parent_exact"]:
        errors.append("procedural parent binding mismatch")
    background_root = (
        Path(map_art_root).resolve()
        if map_art_root is not None
        else PROJECT_ROOT / "outputs" / "map_art" / "packs"
    )
    backgrounds = load_background_crops(background_root)
    expected_catalog = background_catalog(backgrounds, PROJECT_ROOT)
    checks["background_catalog_exact"] = expected_catalog == manifest["background_catalog"]
    if not checks["background_catalog_exact"]:
        errors.append("map background catalog mismatch")
    selected = _select_family_references(parent, arrays)
    if len(manifest["samples"]) != len(selected) or manifest["sample_count"] != len(selected):
        raise ValueError("Procedural reference sample count mismatch")

    conditions: list[StyleCondition] = []
    renders: list[RenderedLayers] = []
    sample_reports: list[dict[str, Any]] = []
    for ordinal, (index, entry) in enumerate(zip(selected, manifest["samples"])):
        sample_errors: list[str] = []
        morphology_id = int(arrays["morphologies"][index])
        subtype_id = int(arrays["subtypes"][index])
        role_id = int(arrays["roles"][index])
        condition = StyleCondition(
            sample_id=f"procedural_ref_{ordinal}_{FAMILIES[morphology_id]}",
            ordinal=ordinal,
            sample_seed=int(arrays["seeds"][index]),
            morphology_id=morphology_id,
            morphology_name=FAMILIES[morphology_id],
            subtype_id=subtype_id,
            subtype_name=SUBTYPE_NAMES[subtype_id],
            role_id=role_id,
            role_name=ROLE_NAMES[role_id],
        )
        condition.validate()
        if condition.as_dict() != entry["condition"]:
            sample_errors.append("condition mismatch")
        fields_hash = aligned_fields_hash(
            arrays["part_owner"][index],
            arrays["material"][index],
            arrays["emission_level"][index],
        )
        sprite = parent["sprites"][index]
        expected_source = {
            "procedural_sprite_id": sprite["id"],
            "procedural_index": index,
            "training_arrays_sha256": sprite["hashes"]["training_arrays_sha256"],
            "semantic_sha256": sprite["hashes"]["semantic_sha256"],
            "aligned_fields_sha256": fields_hash,
        }
        if expected_source != entry["source"]:
            sample_errors.append("categorical procedural source binding mismatch")
        fields = CategoricalFields(
            part=np.array(arrays["part_owner"][index], copy=True),
            material=np.array(arrays["material"][index], copy=True),
            emission=np.array(arrays["emission_level"][index], copy=True),
            aligned_sha256=fields_hash,
        )
        rendered = render_layers(fields, condition)
        metrics = evaluate_style(
            fields,
            condition,
            rendered,
            backgrounds,
            fields_hash_after_render=aligned_fields_hash(fields.part, fields.material, fields.emission),
        )
        expected_payloads = {
            "base": _png_bytes(rendered.base),
            "outline": _png_bytes(rendered.outline),
            "emission_core": _png_bytes(rendered.emission_core),
            "aura": _png_bytes(rendered.aura),
            "bloom_r1": _png_bytes(rendered.bloom_r1),
            "bloom_r2": _png_bytes(rendered.bloom_r2),
            "composite": _png_bytes(rendered.composite),
            "palette": canonical_json_bytes(rendered.palette),
            "metrics": canonical_json_bytes(metrics),
        }
        artifact_checks: dict[str, bool] = {}
        for name in PRESENTATION_ARTIFACT_NAMES:
            actual = _artifact_bytes(root, entry["presentation"]["artifacts"][name])
            exact = actual == expected_payloads[name]
            artifact_checks[name] = exact
            if not exact:
                sample_errors.append(f"recomputed {name} bytes mismatch")
        if not metrics["passed"]:
            sample_errors.append("recomputed metrics gates failed")
        sample_reports.append(
            {
                "sample_id": condition.sample_id,
                "artifact_bytes_exact": artifact_checks,
                "metrics_recomputed_passed": bool(metrics["passed"]),
                "errors": sample_errors,
                "passed": not sample_errors,
            }
        )
        errors.extend(f"{condition.sample_id}: {error}" for error in sample_errors)
        conditions.append(condition)
        renders.append(rendered)

    expected_sheets = {
        "layers": _png_bytes(
            build_layer_contact_sheet(
                conditions,
                renders,
                title="PROCEDURAL REFERENCE · NOT NEURAL OUTPUT",
            )
        ),
        "five_family_reference": _png_bytes(
            build_hero_contact_sheet(
                conditions,
                renders,
                title="FIVE-FAMILY PROCEDURAL STYLE REFERENCE · NOT NEURAL",
            )
        ),
        "map_context": _png_bytes(
            build_map_context_contact_sheet(
                conditions,
                renders,
                backgrounds,
                title="PROCEDURAL FIVE-FAMILY REFERENCE · MAP CONTEXT · NOT NEURAL",
            )
        ),
    }
    contact_checks: dict[str, bool] = {}
    for name, expected in expected_sheets.items():
        actual = _artifact_bytes(root, manifest["contact_sheets"][name])
        contact_checks[name] = actual == expected
        if actual != expected:
            errors.append(f"recomputed {name} contact sheet mismatch")
    checks["contact_sheets_exact"] = contact_checks
    checks["all_samples_exact"] = all(record["passed"] for record in sample_reports)
    return {
        "format": "nullvector-multifield-style-procedural-replay-v1",
        "procedural_reference_manifest_path": str(manifest_path),
        "procedural_reference_manifest_sha256": sha256_file(manifest_path),
        "compiler_source_sha256": current_source_hash,
        "checks": checks,
        "samples": sample_reports,
        "errors": errors,
        "passed": not errors and checks["all_samples_exact"] and all(contact_checks.values()),
    }


def replay_procedural_reference_bank(
    path: Path,
    *,
    map_art_root: Path | None = None,
) -> dict[str, Any]:
    try:
        return _replay_procedural_reference(path, map_art_root)
    except Exception as error:
        manifest_path = Path(path).resolve()
        return {
            "format": "nullvector-multifield-style-procedural-replay-v1",
            "procedural_reference_manifest_path": str(manifest_path),
            "procedural_reference_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
            "compiler_source_sha256": compiler_source_hash(),
            "checks": {},
            "samples": [],
            "errors": [f"{type(error).__name__}: {error}"],
            "passed": False,
        }
