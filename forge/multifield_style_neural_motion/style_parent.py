from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..multifield_style import compiler_source_hash as presentation_source_hash
from ..multifield_style.schema import STYLE_BANK_SCHEMA, validate_schema
from .model import NeuralMotionSource, NeuralStyleParent
from .source import _safe_record_path, _sha256_file


PRESENTATION_ARTIFACTS = (
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


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Neural style parent JSON must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(payload, allow_nan=False)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Invalid neural style parent JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Neural style parent JSON root must be an object")
    return payload


def load_neural_style_parent(
    style_manifest: Path,
    source: NeuralMotionSource,
) -> NeuralStyleParent:
    path = Path(style_manifest).resolve()
    manifest = _strict_json(path)
    validate_schema(manifest, STYLE_BANK_SCHEMA)
    if (
        manifest["format"] != "nullvector-multifield-style-bank-v1"
        or manifest["status"] != "ready"
        or manifest["source_kind"] != "neural-generation-bank"
        or manifest["sample_count"] != len(source.bank.samples)
        or manifest["all_metrics_passed"] is not True
        or manifest["parent"]["manifest_sha256"] != source.bank.manifest_sha256
        or manifest["parent"]["manifest_bytes"] != source.bank.manifest_bytes
        or manifest["compiler"]["source_sha256"] != presentation_source_hash()
    ):
        raise ValueError("Neural style parent does not bind the active production bank/compiler")
    root = path.parent.resolve()
    source_by_id = {sample.condition.sample_id: sample for sample in source.bank.samples}
    palettes: dict[str, dict[str, Any]] = {}
    palette_artifacts: dict[str, dict[str, Any]] = {}
    expected_order = [sample.condition.sample_id for sample in source.bank.samples]
    observed_order: list[str] = []
    for record in manifest["samples"]:
        condition = record["condition"]
        sample_id = condition["sample_id"]
        observed_order.append(sample_id)
        sample = source_by_id.get(sample_id)
        if sample is None or condition != sample.condition.as_dict():
            raise ValueError(f"Neural style parent condition mismatch: {sample_id}")
        if (
            record["source"]["raw_fields_sha256"] != sample.raw_fields_sha256
            or record["source"]["compiled_fields_sha256"] != sample.fields.aligned_sha256
            or record["presentation"]["metrics_passed"] is not True
        ):
            raise ValueError(f"Neural style parent field/metrics mismatch: {sample_id}")
        artifacts = record["presentation"]["artifacts"]
        for artifact_name in PRESENTATION_ARTIFACTS:
            _safe_record_path(root, artifacts[artifact_name], f"style {sample_id}/{artifact_name}")
        palette_path = _safe_record_path(root, artifacts["palette"], f"style {sample_id}/palette")
        palette = _strict_json(palette_path)
        if palette.get("format") != "nullvector-perceptual-palette-v1":
            raise ValueError(f"Neural style palette format mismatch: {sample_id}")
        palettes[sample_id] = palette
        palette_artifacts[sample_id] = dict(artifacts["palette"])
    if observed_order != expected_order:
        raise ValueError("Neural style parent sample ordering mismatch")
    return NeuralStyleParent(
        root=root,
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        manifest_bytes=path.stat().st_size,
        manifest=manifest,
        palettes=palettes,
        palette_artifacts=palette_artifacts,
    )
