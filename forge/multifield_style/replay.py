from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .backgrounds import background_catalog, load_background_crops
from .compiler import (
    PRESENTATION_ARTIFACT_NAMES,
    STYLE_REPLAY_FORMAT,
    _parent_record,
    compiler_source_hash,
    load_style_manifest,
)
from .contact_sheet import (
    build_hero_contact_sheet,
    build_layer_contact_sheet,
    build_map_context_contact_sheet,
)
from .hashing import aligned_fields_hash, canonical_json_bytes, sha256_bytes, sha256_file
from .metrics import evaluate_style
from .rendering import render_layers
from .source import PROJECT_ROOT, load_generation_bank


def _safe_resolve(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Artifact path must be a string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe artifact path: {relative!r}")
    if "\\" in relative:
        raise ValueError("Artifact paths must use canonical POSIX separators")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact escapes its root: {relative}") from error
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"Artifact must be a regular non-symlink file: {relative}")
    return target


def _artifact_bytes(root: Path, record: Mapping[str, Any]) -> bytes:
    path = _safe_resolve(root, record.get("path"))
    payload = path.read_bytes()
    if len(payload) != record.get("bytes"):
        raise ValueError(f"Artifact byte count mismatch: {record.get('path')}")
    if sha256_bytes(payload) != record.get("sha256"):
        raise ValueError(f"Artifact SHA-256 mismatch: {record.get('path')}")
    return payload


def _png_bytes(pixels: np.ndarray) -> bytes:
    values = np.asarray(pixels)
    buffer = BytesIO()
    Image.fromarray(values).save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


def _record_check(name: str, actual: object, expected: object, errors: list[str]) -> bool:
    exact = actual == expected
    if not exact:
        errors.append(f"{name} mismatch")
    return exact


def _replay(style_manifest_path: Path, map_art_root: Path | None) -> dict[str, Any]:
    manifest_path = Path(style_manifest_path).resolve()
    manifest = load_style_manifest(manifest_path)
    root = manifest_path.parent.resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    checks["compiler_source_hash_exact"] = _record_check(
        "compiler source hash",
        compiler_source_hash(),
        manifest["compiler"]["source_sha256"],
        errors,
    )
    parent_path = _safe_resolve(PROJECT_ROOT, manifest["parent"]["manifest_path"])
    parent = load_generation_bank(parent_path)
    checks["parent_manifest_exact"] = _record_check(
        "parent generation manifest",
        _parent_record(parent),
        manifest["parent"],
        errors,
    )
    background_root = (
        Path(map_art_root).resolve()
        if map_art_root is not None
        else PROJECT_ROOT / "outputs" / "map_art" / "packs"
    )
    backgrounds = load_background_crops(background_root)
    checks["background_catalog_exact"] = _record_check(
        "map background catalog",
        background_catalog(backgrounds, PROJECT_ROOT),
        manifest["background_catalog"],
        errors,
    )
    checks["sample_count_exact"] = _record_check(
        "style sample count",
        len(parent.samples),
        manifest["sample_count"],
        errors,
    ) and _record_check(
        "style sample entry count",
        len(manifest["samples"]),
        manifest["sample_count"],
        errors,
    )
    if len(manifest["samples"]) != len(parent.samples):
        raise ValueError("Cannot replay a style manifest with a mismatched sample count")

    rendered_samples = []
    sample_reports: list[dict[str, Any]] = []
    for source_sample, entry in zip(parent.samples, manifest["samples"]):
        sample_errors: list[str] = []
        _record_check(
            "condition",
            source_sample.condition.as_dict(),
            entry["condition"],
            sample_errors,
        )
        expected_source = {
            "compiled_fields_path": str(source_sample.fields_artifact["path"]),
            "compiled_fields_bytes": int(source_sample.fields_artifact["bytes"]),
            "compiled_fields_artifact_sha256": str(source_sample.fields_artifact["sha256"]),
            "raw_fields_sha256": source_sample.raw_fields_sha256,
            "compiled_fields_sha256": source_sample.fields.aligned_sha256,
        }
        _record_check("categorical source binding", expected_source, entry["source"], sample_errors)
        rendered = render_layers(source_sample.fields, source_sample.condition)
        rendered_samples.append(rendered)
        metrics = evaluate_style(
            source_sample.fields,
            source_sample.condition,
            rendered,
            backgrounds,
            fields_hash_after_render=aligned_fields_hash(
                source_sample.fields.part,
                source_sample.fields.material,
                source_sample.fields.emission,
            ),
        )
        expected_payloads: dict[str, bytes] = {
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
        passed = not sample_errors and metrics["passed"]
        sample_reports.append(
            {
                "sample_id": source_sample.condition.sample_id,
                "artifact_bytes_exact": artifact_checks,
                "metrics_recomputed_passed": bool(metrics["passed"]),
                "errors": sample_errors,
                "passed": passed,
            }
        )
        errors.extend(
            f"{source_sample.condition.sample_id}: {error}" for error in sample_errors
        )

    conditions = [sample.condition for sample in parent.samples]
    expected_sheets = {
        "layers": _png_bytes(
            build_layer_contact_sheet(
                conditions,
                rendered_samples,
                title="NEURAL STYLE BANK · CALIBRATED ACCEPTED FIELDS",
            )
        ),
        "composites": _png_bytes(
            build_hero_contact_sheet(
                conditions,
                rendered_samples,
                title="NEURAL PRESENTATION SMOKE BANK",
            )
        ),
        "map_context": _png_bytes(
            build_map_context_contact_sheet(
                conditions,
                rendered_samples,
                backgrounds,
                title="NEURAL PRESENTATION · MAP CONTRAST CONTEXT",
            )
        ),
    }
    contact_checks: dict[str, bool] = {}
    for name, expected in expected_sheets.items():
        actual = _artifact_bytes(root, manifest["contact_sheets"][name])
        exact = actual == expected
        contact_checks[name] = exact
        if not exact:
            errors.append(f"recomputed {name} contact sheet bytes mismatch")
    checks["contact_sheets_exact"] = contact_checks
    checks["all_samples_exact"] = all(sample["passed"] for sample in sample_reports)
    return {
        "format": STYLE_REPLAY_FORMAT,
        "style_manifest_path": str(manifest_path),
        "style_manifest_sha256": sha256_file(manifest_path),
        "compiler_source_sha256": compiler_source_hash(),
        "checks": checks,
        "samples": sample_reports,
        "errors": errors,
        "passed": not errors and checks["all_samples_exact"] and all(contact_checks.values()),
    }


def replay_style_bank(
    style_manifest_path: Path,
    *,
    map_art_root: Path | None = None,
) -> dict[str, Any]:
    """Strictly replay a derived style bank, returning a fail-closed report."""

    try:
        return _replay(style_manifest_path, map_art_root)
    except Exception as error:
        path = Path(style_manifest_path).resolve()
        return {
            "format": STYLE_REPLAY_FORMAT,
            "style_manifest_path": str(path),
            "style_manifest_sha256": sha256_file(path) if path.is_file() else None,
            "compiler_source_sha256": compiler_source_hash(),
            "checks": {},
            "samples": [],
            "errors": [f"{type(error).__name__}: {error}"],
            "passed": False,
        }
