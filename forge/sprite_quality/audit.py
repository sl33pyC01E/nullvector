from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image

from ..morphology import FACING_NAMES, FAMILIES, MOTION_NAMES
from ..morphology.constants import (
    EMISSION_LEVEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SUBTYPE_NAMES,
)
from ..multifield_style.backgrounds import load_background_crops
from ..multifield_style.compiler import load_style_manifest
from ..multifield_style.hashing import (
    aligned_fields_hash,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from ..multifield_style.metrics import evaluate_style
from ..multifield_style.rendering import render_layers
from ..multifield_style.schema import STYLE_BANK_SCHEMA, validate_schema as validate_style_schema
from ..multifield_style.source import PROJECT_ROOT, load_generation_bank
from ..multifield_style_motion.io import verify_artifact
from ..multifield_style_neural_motion.schema import BANK_SCHEMA as MOTION_BANK_SCHEMA
from ..multifield_style_neural_motion.schema import validate_schema as validate_motion_schema
from ..multifield_style_neural_motion.validation import (
    load_verified_identity_manifest,
    strict_json_file,
)
from .io import (
    artifact_record,
    prepare_immutable_destination,
    write_bytes_new,
    write_json_new,
)
from .render import render_motion_energy_heatmap
from .schema import validate_schema


REPORT_FORMAT = "nullvector-neural-sprite-quality-report-v1"
REPLAY_FORMAT = "nullvector-neural-sprite-quality-replay-v1"
REPORT_FILENAME = "sprite_quality_report.json"
HEATMAP_FILENAME = "motion_energy_heatmap.png"
DEFAULT_STATIC_MANIFEST = (
    PROJECT_ROOT / "outputs" / "multifield_style" / "final_best_stratified80_v3" / "style_manifest.json"
)
DEFAULT_MOTION_MANIFEST = (
    PROJECT_ROOT / "outputs" / "multifield_style_neural_motion" / "motion_style_neural_manifest.json"
)
DEFAULT_MOTION_REPLAY = (
    PROJECT_ROOT / "outputs" / "multifield_style_neural_motion" / "verification_report.json"
)
DEFAULT_MAP_ART_ROOT = PROJECT_ROOT / "outputs" / "map_art" / "packs"
SOURCE_FILES = (
    "forge/sprite_quality/__init__.py",
    "forge/sprite_quality/__main__.py",
    "forge/sprite_quality/audit.py",
    "forge/sprite_quality/io.py",
    "forge/sprite_quality/render.py",
    "forge/sprite_quality/schema.py",
    "shared/schema/sprite_quality_report.schema.json",
)
MOTION_REPLAY_SCHEMA = (
    PROJECT_ROOT / "shared" / "schema" / "multifield_style_neural_motion_replay.schema.json"
)
LAYER_NAMES = (
    "base",
    "outline",
    "emission_core",
    "aura",
    "bloom_r1",
    "bloom_r2",
    "composite",
)


def audit_source_hash(project_root: Path = PROJECT_ROOT) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    digest.update(b"nullvector-neural-sprite-quality-source-v1\0")
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Sprite-quality source member is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Sprite-quality provenance must live under project root: {path}") from error


def _round(value: float) -> float:
    return round(float(value), 9)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty diagnostic series")
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "minimum": _round(ordered.min()),
        "median": _round(np.median(ordered)),
        "mean": _round(ordered.mean()),
        "maximum": _round(ordered.max()),
    }


def _integer_summary(values: Sequence[int]) -> dict[str, float | int]:
    summary = _summary([float(value) for value in values])
    return {
        "minimum": int(min(values)),
        "median": summary["median"],
        "mean": summary["mean"],
        "maximum": int(max(values)),
    }


def _counter_rows(names: Sequence[str], counts: Counter[str]) -> list[dict[str, Any]]:
    return [{"name": name, "count": int(counts[name])} for name in names]


def _load_canonical_json(path: Path, *, maximum_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file() or target.is_symlink() or target.stat().st_size > maximum_bytes:
        raise ValueError(f"JSON must be a bounded regular file: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        json.dumps(payload, allow_nan=False)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid strict JSON at {target}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {target}")
    if canonical_json_bytes(payload) != target.read_bytes():
        raise ValueError(f"JSON is not canonical: {target}")
    return payload


def _validate_motion_replay(payload: Mapping[str, Any]) -> None:
    schema = json.loads(MOTION_REPLAY_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(dict(payload)))
    if errors:
        error = sorted(errors, key=lambda item: tuple(map(str, item.absolute_path)))[0]
        location = "/".join(map(str, error.absolute_path)) or "<root>"
        raise ValueError(f"Motion replay schema failure at {location}: {error.message}")


def _verify_static_artifacts(
    style_root: Path,
    generation_bank: Any,
    style_manifest: Mapping[str, Any],
    backgrounds: tuple,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if style_manifest["sample_count"] != len(generation_bank.samples):
        raise ValueError("Static style/generation sample counts disagree")
    by_id = {sample.condition.sample_id: sample for sample in generation_bank.samples}
    if len(by_id) != len(generation_bank.samples):
        raise ValueError("Generation sample IDs are not unique")
    diagnostics: list[dict[str, Any]] = []
    artifact_count = 0
    byte_count = 0
    expected_ids: list[str] = []
    for expected_ordinal, record in enumerate(style_manifest["samples"]):
        condition = record["condition"]
        sample_id = condition["sample_id"]
        expected_ids.append(sample_id)
        if condition["ordinal"] != expected_ordinal or sample_id not in by_id:
            raise ValueError("Static style sample order diverges from the generation bank")
        sample = by_id[sample_id]
        if condition != sample.condition.as_dict():
            raise ValueError(f"Static style condition mismatch: {sample_id}")
        source = record["source"]
        if (
            source["raw_fields_sha256"] != sample.raw_fields_sha256
            or source["compiled_fields_sha256"] != sample.fields.aligned_sha256
            or source["compiled_fields_artifact_sha256"] != sample.fields_artifact["sha256"]
            or source["compiled_fields_bytes"] != sample.fields_artifact["bytes"]
        ):
            raise ValueError(f"Static style categorical provenance mismatch: {sample_id}")
        loaded_artifacts: dict[str, Path] = {}
        for name, artifact in record["presentation"]["artifacts"].items():
            path = verify_artifact(style_root, artifact)
            loaded_artifacts[name] = path
            artifact_count += 1
            byte_count += path.stat().st_size
        metrics = _load_canonical_json(loaded_artifacts["metrics"])
        palette = _load_canonical_json(loaded_artifacts["palette"])
        if not metrics.get("passed") or not all(metrics.get("gates", {}).values()):
            raise ValueError(f"Static style metrics are not fully passed: {sample_id}")
        rendered = render_layers(sample.fields, sample.condition)
        recomputed = evaluate_style(
            sample.fields,
            sample.condition,
            rendered,
            backgrounds,
            fields_hash_after_render=aligned_fields_hash(
                sample.fields.part, sample.fields.material, sample.fields.emission
            ),
        )
        if recomputed != metrics or rendered.palette != palette:
            raise ValueError(f"Static style metrics/palette replay mismatch: {sample_id}")
        for name in LAYER_NAMES:
            with Image.open(loaded_artifacts[name]) as image:
                stored = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            if not np.array_equal(stored, getattr(rendered, name)):
                raise ValueError(f"Static style layer replay mismatch: {sample_id}/{name}")
        measurements = metrics["measurements"]
        part = sample.fields.part
        material = sample.fields.material
        emission = sample.fields.emission
        body = (part != 0) & (part != 16)
        diagnostics.append(
            {
                "sample_id": sample_id,
                "family": condition["morphology_name"],
                "subtype": condition["subtype_name"],
                "role": condition["role_name"],
                "variation": expected_ordinal % 2,
                "categorical_sha256": sample.fields.aligned_sha256,
                "silhouette_sha256": sha256_bytes(np.ascontiguousarray(part != 0).tobytes()),
                "solid_body_sha256": sha256_bytes(np.ascontiguousarray(body).tobytes()),
                "occupancy_fraction": _round(np.count_nonzero(part) / part.size),
                "solid_body_fraction": _round(np.count_nonzero(body) / body.size),
                "visible_part_class_count": int(np.count_nonzero(np.bincount(part.ravel(), minlength=17))),
                "visible_material_class_count": int(np.count_nonzero(np.bincount(material.ravel(), minlength=10))),
                "visible_emission_class_count": int(np.count_nonzero(np.bincount(emission.ravel(), minlength=4))),
                "body_pixels": int(measurements["body_pixels"]),
                "categorical_aura_pixels": int(measurements["categorical_aura_pixels"]),
                "emission_core_pixels": int(measurements["emission_core_pixels"]),
                "outline_pixels": int(measurements["outline_pixels"]),
                "aura_pixels": int(measurements["aura_pixels"]),
                "bloom_radius_1_pixels": int(measurements["bloom_radius_1_pixels"]),
                "bloom_radius_2_pixels": int(measurements["bloom_radius_2_pixels"]),
                "base_palette_color_count": int(measurements["base_palette_color_count"]),
                "composite_palette_color_count": int(measurements["composite_palette_color_count"]),
                "clipped_white_fraction": float(measurements["clipped_white_fraction"]),
                "map_contrast_p10": float(
                    measurements["map_background_contrast"]["global_p10_delta_e_oklab"]
                ),
                "minimum_theme_median_contrast": float(
                    measurements["map_background_contrast"][
                        "minimum_theme_median_delta_e_oklab"
                    ]
                ),
            }
        )
    if expected_ids != [sample.condition.sample_id for sample in generation_bank.samples]:
        raise ValueError("Static style and generation bank order mismatch")
    return diagnostics, {
        "artifact_count": artifact_count,
        "artifact_bytes": byte_count,
        "exact_static_presentation_replay": True,
    }


def _coverage(static_rows: Sequence[Mapping[str, Any]], motion_manifest: Mapping[str, Any]) -> dict[str, Any]:
    family = Counter(str(row["family"]) for row in static_rows)
    subtype = Counter(str(row["subtype"]) for row in static_rows)
    role = Counter(str(row["role"]) for row in static_rows)
    family_role = Counter((str(row["family"]), str(row["role"])) for row in static_rows)
    return {
        "static_sample_count": len(static_rows),
        "family_counts": _counter_rows(FAMILIES, family),
        "subtype_counts": _counter_rows(SUBTYPE_NAMES, subtype),
        "role_counts": _counter_rows(ROLE_NAMES, role),
        "family_role_cells": [
            {"family": family_name, "role": role_name, "count": family_role[(family_name, role_name)]}
            for family_name in FAMILIES
            for role_name in ROLE_NAMES
        ],
        "variant_count_per_family_role_cell": 2,
        "motion_identity_count": int(motion_manifest["identity_count"]),
        "motion_identity_scope": "one actual neural representative per family",
        "all_80_motion_claimed": False,
        "bindable_source_count": int(motion_manifest["source_census"]["bindable_count"]),
        "rejected_source_count": int(motion_manifest["source_census"]["rejected_count"]),
    }


def _categorical_quality(samples: Sequence[Any]) -> dict[str, Any]:
    field_hashes = {sample.fields.aligned_sha256 for sample in samples}
    silhouette_hashes = {
        sha256_bytes(np.ascontiguousarray(sample.fields.part != 0).tobytes()) for sample in samples
    }
    solid_hashes = {
        sha256_bytes(
            np.ascontiguousarray((sample.fields.part != 0) & (sample.fields.part != 16)).tobytes()
        )
        for sample in samples
    }
    pair_iou: list[float] = []
    within_iou: list[float] = []
    between_iou: list[float] = []
    hamming: list[float] = []
    for left, right in itertools.combinations(samples, 2):
        left_mask = left.fields.part != 0
        right_mask = right.fields.part != 0
        union = int(np.count_nonzero(left_mask | right_mask))
        iou = np.count_nonzero(left_mask & right_mask) / max(union, 1)
        pair_iou.append(float(iou))
        (within_iou if left.condition.morphology_id == right.condition.morphology_id else between_iou).append(
            float(iou)
        )
        hamming.append(
            float(
                np.mean(
                    (left.fields.part != right.fields.part)
                    | (left.fields.material != right.fields.material)
                    | (left.fields.emission != right.fields.emission)
                )
            )
        )
    vocabularies = []
    for name, names in (
        ("part", PART_OWNER_NAMES),
        ("material", MATERIAL_NAMES),
        ("emission", EMISSION_LEVEL_NAMES),
    ):
        pixel_counts = np.zeros(len(names), dtype=np.int64)
        sample_counts = np.zeros(len(names), dtype=np.int64)
        for sample in samples:
            values = getattr(sample.fields, name)
            pixel_counts += np.bincount(values.ravel(), minlength=len(names))
            sample_counts[np.unique(values)] += 1
        vocabularies.append(
            {
                "field": name,
                "class_count": len(names),
                "observed_class_count": int(np.count_nonzero(pixel_counts)),
                "classes": [
                    {
                        "id": index,
                        "name": class_name,
                        "pixel_count": int(pixel_counts[index]),
                        "sample_count": int(sample_counts[index]),
                    }
                    for index, class_name in enumerate(names)
                ],
            }
        )
    occupancy = [float(np.mean(sample.fields.part != 0)) for sample in samples]
    return {
        "sample_count": len(samples),
        "unique_categorical_field_count": len(field_hashes),
        "unique_visible_silhouette_count": len(silhouette_hashes),
        "unique_solid_body_count": len(solid_hashes),
        "categorical_duplicate_pair_count": len(samples) - len(field_hashes),
        "visible_silhouette_duplicate_count": len(samples) - len(silhouette_hashes),
        "solid_body_duplicate_count": len(samples) - len(solid_hashes),
        "occupancy_fraction": _summary(occupancy),
        "pairwise_visible_silhouette_iou": _summary(pair_iou),
        "within_family_visible_silhouette_iou": _summary(within_iou),
        "between_family_visible_silhouette_iou": _summary(between_iou),
        "pairwise_aligned_categorical_hamming": _summary(hamming),
        "vocabularies": vocabularies,
    }


def _presentation_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def floats(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    def integers(name: str) -> list[int]:
        return [int(row[name]) for row in rows]

    body = np.asarray(integers("body_pixels"), dtype=np.float64)
    aura = np.asarray(integers("categorical_aura_pixels"), dtype=np.float64)
    emission = np.asarray(integers("emission_core_pixels"), dtype=np.float64)
    outline = np.asarray(integers("outline_pixels"), dtype=np.float64)
    bloom1 = np.asarray(integers("bloom_radius_1_pixels"), dtype=np.float64)
    bloom2 = np.asarray(integers("bloom_radius_2_pixels"), dtype=np.float64)
    aggregate = {
        "base_palette_color_count": _integer_summary(integers("base_palette_color_count")),
        "composite_palette_color_count": _integer_summary(
            integers("composite_palette_color_count")
        ),
        "body_pixels": _integer_summary(integers("body_pixels")),
        "categorical_aura_fraction_of_visible": _summary(
            list(aura / np.maximum(body + aura, 1.0))
        ),
        "emission_core_fraction_of_body": _summary(list(emission / np.maximum(body, 1.0))),
        "outline_fraction_of_body": _summary(list(outline / np.maximum(body, 1.0))),
        "bloom_radius_1_fraction_of_body": _summary(list(bloom1 / np.maximum(body, 1.0))),
        "bloom_radius_2_fraction_of_body": _summary(list(bloom2 / np.maximum(body, 1.0))),
        "clipped_white_fraction": _summary(floats("clipped_white_fraction")),
        "map_contrast_p10": _summary(floats("map_contrast_p10")),
        "minimum_theme_median_contrast": _summary(
            floats("minimum_theme_median_contrast")
        ),
    }
    by_family = []
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        by_family.append(
            {
                "family": family,
                "sample_count": len(selected),
                "body_pixels": _integer_summary([int(row["body_pixels"]) for row in selected]),
                "emission_core_fraction_of_body": _summary(
                    [
                        int(row["emission_core_pixels"]) / max(int(row["body_pixels"]), 1)
                        for row in selected
                    ]
                ),
                "map_contrast_p10": _summary(
                    [float(row["map_contrast_p10"]) for row in selected]
                ),
            }
        )
    return {"aggregate": aggregate, "by_family": by_family}


def _load_atlas(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    expected = (59 * 48, 16 * 48, 4)
    if values.shape != expected:
        raise ValueError(f"Motion atlas shape mismatch: {path} {values.shape} != {expected}")
    return values


def _cell(atlas: np.ndarray, index: int) -> np.ndarray:
    row, column = divmod(index, 16)
    return atlas[row * 48 : (row + 1) * 48, column * 48 : (column + 1) * 48]


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    points = np.argwhere(mask)
    if not len(points):
        return (23.5, 23.5)
    return (float(points[:, 1].mean()), float(points[:, 0].mean()))


def _motion_quality(
    motion_root: Path,
    manifest: Mapping[str, Any],
    generation_manifest_sha256: str,
    style_manifest_sha256: str,
    style_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    static_by_id = {
        record["condition"]["sample_id"]: record for record in style_manifest["samples"]
    }
    clip_rows: list[dict[str, Any]] = []
    manifest_bytes = 0
    for identity in manifest["identities"]:
        static = static_by_id[identity["sample_id"]]
        # Pillow 12 deprecates the explicit ``mode`` argument used by the
        # already-frozen motion validator.  Its 4,720-cell strict validation
        # would otherwise flood this independent audit with 84,960 warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'mode' parameter is deprecated.*",
                category=DeprecationWarning,
            )
            verified = load_verified_identity_manifest(
                motion_root,
                identity["family"],
                sample_id=identity["sample_id"],
                compiler=manifest["compiler"],
                generation_manifest_sha256=generation_manifest_sha256,
                style_manifest_sha256=style_manifest_sha256,
                static_palette_artifact=static["presentation"]["artifacts"]["palette"],
            )
        manifest_bytes += int(identity["manifest"]["bytes"])
        atlases = {
            name: _load_atlas(
                verify_artifact(motion_root, verified["artifacts"]["layers"][name])
            )
            for name in ("base", "composite")
        }
        for clip in verified["clips"]:
            frames = [
                {name: _cell(atlas, index) for name, atlas in atlases.items()}
                for index in range(
                    int(clip["start_cell"]),
                    int(clip["start_cell"]) + int(clip["frame_count"]),
                )
            ]
            endpoint_exact = all(
                np.array_equal(frames[0][name], frames[-1][name]) for name in atlases
            )
            if clip["loop"] and not endpoint_exact:
                raise ValueError(f"Motion loop endpoint contract mismatch: {clip['id']}")
            effective = frames[:-1] if clip["loop"] else frames
            if not effective:
                raise ValueError(f"Motion clip has no effective frames: {clip['id']}")
            unique_base = len(
                {sha256_bytes(np.ascontiguousarray(frame["base"]).tobytes()) for frame in effective}
            )
            unique_composite = len(
                {
                    sha256_bytes(np.ascontiguousarray(frame["composite"]).tobytes())
                    for frame in effective
                }
            )
            pairs = (
                [(effective[index], effective[(index + 1) % len(effective)]) for index in range(len(effective))]
                if clip["loop"]
                else [
                    (effective[index], effective[index + 1])
                    for index in range(len(effective) - 1)
                ]
            )
            silhouette_changes: list[float] = []
            composite_changes: list[float] = []
            centroid_steps: list[float] = []
            for left, right in pairs:
                left_body = left["base"][..., 3] > 0
                right_body = right["base"][..., 3] > 0
                silhouette_union = np.count_nonzero(left_body | right_body)
                silhouette_changes.append(
                    np.count_nonzero(left_body ^ right_body) / max(int(silhouette_union), 1)
                )
                left_visible = left["composite"][..., 3] > 0
                right_visible = right["composite"][..., 3] > 0
                visible_union = np.count_nonzero(left_visible | right_visible)
                changed = np.any(left["composite"] != right["composite"], axis=2)
                composite_changes.append(
                    np.count_nonzero(changed) / max(int(visible_union), 1)
                )
                left_center = _centroid(left_body)
                right_center = _centroid(right_body)
                centroid_steps.append(
                    float(np.hypot(right_center[0] - left_center[0], right_center[1] - left_center[1]))
                )
            areas = [int(np.count_nonzero(frame["base"][..., 3])) for frame in effective]
            clip_rows.append(
                {
                    "id": clip["id"],
                    "family": identity["family"],
                    "sample_id": identity["sample_id"],
                    "motion": clip["motion"],
                    "facing": clip["facing"],
                    "loop": bool(clip["loop"]),
                    "stored_frame_count": int(clip["frame_count"]),
                    "effective_frame_count": len(effective),
                    "unique_base_frame_count": unique_base,
                    "unique_composite_frame_count": unique_composite,
                    "silhouette_change": _summary(silhouette_changes),
                    "composite_change": _summary(composite_changes),
                    "centroid_step_pixels": _summary(centroid_steps),
                    "body_area_pixels": _integer_summary(areas),
                    "loop_endpoint_exact": endpoint_exact if clip["loop"] else True,
                }
            )
    if len(clip_rows) != 520:
        raise ValueError(f"Motion quality expected 520 clips, found {len(clip_rows)}")
    by_family_motion: list[dict[str, Any]] = []
    for family in FAMILIES:
        for motion in MOTION_NAMES:
            rows = [
                row for row in clip_rows if row["family"] == family and row["motion"] == motion
            ]
            if len(rows) != len(FACING_NAMES):
                raise ValueError(f"Motion family/action matrix incomplete: {family}/{motion}")
            by_family_motion.append(
                {
                    "family": family,
                    "motion": motion,
                    "clip_count": len(rows),
                    "minimum_unique_base_frames": min(
                        int(row["unique_base_frame_count"]) for row in rows
                    ),
                    "minimum_unique_composite_frames": min(
                        int(row["unique_composite_frame_count"]) for row in rows
                    ),
                    "silhouette_change": _summary(
                        [float(row["silhouette_change"]["mean"]) for row in rows]
                    ),
                    "composite_change": _summary(
                        [float(row["composite_change"]["mean"]) for row in rows]
                    ),
                    "centroid_step_pixels": _summary(
                        [float(row["centroid_step_pixels"]["mean"]) for row in rows]
                    ),
                }
            )
    result = {
        "identity_count": len(manifest["identities"]),
        "clip_count": len(clip_rows),
        "stored_frame_count": int(sum(row["stored_frame_count"] for row in clip_rows)),
        "effective_frame_count": int(sum(row["effective_frame_count"] for row in clip_rows)),
        "loop_clip_count": int(sum(bool(row["loop"]) for row in clip_rows)),
        "nonloop_clip_count": int(sum(not bool(row["loop"]) for row in clip_rows)),
        "collapsed_base_clip_count": int(
            sum(int(row["unique_base_frame_count"]) <= 1 for row in clip_rows)
        ),
        "collapsed_composite_clip_count": int(
            sum(int(row["unique_composite_frame_count"]) <= 1 for row in clip_rows)
        ),
        "minimum_unique_base_frames": min(
            int(row["unique_base_frame_count"]) for row in clip_rows
        ),
        "minimum_unique_composite_frames": min(
            int(row["unique_composite_frame_count"]) for row in clip_rows
        ),
        "all_loop_endpoints_exact": all(bool(row["loop_endpoint_exact"]) for row in clip_rows),
        "silhouette_change": _summary(
            [float(row["silhouette_change"]["mean"]) for row in clip_rows]
        ),
        "composite_change": _summary(
            [float(row["composite_change"]["mean"]) for row in clip_rows]
        ),
        "centroid_step_pixels": _summary(
            [float(row["centroid_step_pixels"]["mean"]) for row in clip_rows]
        ),
        "by_family_motion": by_family_motion,
        "clips": clip_rows,
    }
    return result, {
        "identity_manifest_bytes": manifest_bytes,
        "strict_identity_count": len(manifest["identities"]),
        "strict_motion_atlas_cell_count": 4720,
    }


def _source_record(path: Path) -> dict[str, Any]:
    target = Path(path).resolve()
    return {
        "path": _project_relative(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _load_motion_bank(path: Path) -> dict[str, Any]:
    payload = _load_canonical_json(path)
    validate_motion_schema(payload, MOTION_BANK_SCHEMA)
    return payload


def _load_replay(path: Path, motion_manifest: Path) -> dict[str, Any]:
    payload = _load_canonical_json(path)
    _validate_motion_replay(payload)
    if (
        payload["manifest"]["sha256"] != sha256_file(motion_manifest)
        or payload["manifest"]["bytes"] != motion_manifest.stat().st_size
        or not payload["exact_identity_replay"]
        or not payload["exact_showcase_replay"]
        or not payload["all_gates_passed"]
    ):
        raise ValueError("Motion replay report does not prove this exact motion bank")
    return payload


def build_sprite_quality_report(
    *,
    static_manifest_path: Path = DEFAULT_STATIC_MANIFEST,
    motion_manifest_path: Path = DEFAULT_MOTION_MANIFEST,
    motion_replay_path: Path = DEFAULT_MOTION_REPLAY,
    map_art_root: Path = DEFAULT_MAP_ART_ROOT,
) -> dict[str, Any]:
    static_path = Path(static_manifest_path).resolve()
    motion_path = Path(motion_manifest_path).resolve()
    replay_path = Path(motion_replay_path).resolve()
    style_root, motion_root = static_path.parent, motion_path.parent
    style_manifest = load_style_manifest(static_path)
    if canonical_json_bytes(style_manifest) != static_path.read_bytes():
        raise ValueError("Static style manifest is not canonical JSON")
    validate_style_schema(style_manifest, STYLE_BANK_SCHEMA)
    generation_path = PROJECT_ROOT / Path(*style_manifest["parent"]["manifest_path"].split("/"))
    generation_bank = load_generation_bank(generation_path)
    if (
        generation_bank.manifest_sha256 != style_manifest["parent"]["manifest_sha256"]
        or generation_bank.manifest_bytes != style_manifest["parent"]["manifest_bytes"]
    ):
        raise ValueError("Static style parent generation provenance mismatch")
    motion_manifest = _load_motion_bank(motion_path)
    if (
        motion_manifest["parent"]["generation_manifest_sha256"]
        != generation_bank.manifest_sha256
        or motion_manifest["parent"]["style_manifest_sha256"] != sha256_file(static_path)
    ):
        raise ValueError("Static and motion banks do not share the same neural authority")
    replay = _load_replay(replay_path, motion_path)
    backgrounds = load_background_crops(Path(map_art_root).resolve())
    static_rows, static_proof = _verify_static_artifacts(
        style_root, generation_bank, style_manifest, backgrounds
    )
    categorical = _categorical_quality(generation_bank.samples)
    presentation = _presentation_quality(static_rows)
    motion, motion_proof = _motion_quality(
        motion_root,
        motion_manifest,
        generation_bank.manifest_sha256,
        sha256_file(static_path),
        style_manifest,
    )
    coverage = _coverage(static_rows, motion_manifest)
    gates = {
        "shared_neural_generation_authority": True,
        "static_manifest_schema_and_canonical_json": True,
        "static_all_80_artifacts_and_metrics_exact": True,
        "static_family_subtype_role_grid_exact": (
            coverage["static_sample_count"] == 80
            and all(row["count"] == 16 for row in coverage["family_counts"])
            and all(row["count"] == 4 for row in coverage["subtype_counts"])
            and all(row["count"] == 10 for row in coverage["role_counts"])
            and all(row["count"] == 2 for row in coverage["family_role_cells"])
        ),
        "categorical_fields_all_unique": categorical["unique_categorical_field_count"] == 80,
        "all_part_material_emission_classes_observed": all(
            row["observed_class_count"] == row["class_count"] for row in categorical["vocabularies"]
        ),
        "motion_manifest_schema_and_canonical_json": True,
        "motion_replay_proves_exact_bank": True,
        "motion_identity_atlas_validation_exact": True,
        "motion_matrix_5x13x8_exact": motion["identity_count"] == 5
        and motion["clip_count"] == 520
        and motion["stored_frame_count"] == 4720,
        "motion_no_collapsed_composite_clips": motion["collapsed_composite_clip_count"] == 0,
        "motion_loop_endpoints_exact": motion["all_loop_endpoints_exact"],
        "all_80_motion_scope_not_overclaimed": (
            not coverage["all_80_motion_claimed"]
            and coverage["motion_identity_count"] == 5
            and motion_manifest["source_census"]["animation_bank_scope"]["all_80_animated"] is False
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("Sprite-quality objective gates failed: " + ", ".join(failed))
    report = {
        "format": REPORT_FORMAT,
        "status": "passed",
        "scope": {
            "static_neural_samples": 80,
            "animated_neural_identities": 5,
            "animated_identity_policy": "one actual neural representative per family",
            "all_80_animated": False,
            "integrity_gates_are_not_aesthetic_approval": True,
            "known_visual_limitation": (
                "Humanoid and machine representatives are stylized/abstract rather than "
                "conventional readable bipeds; human visual review remains required."
            ),
        },
        "auditor": {
            "id": "read-only-neural-sprite-quality-audit-v1",
            "source_sha256": audit_source_hash(),
        },
        "sources": {
            "generation_manifest": _source_record(generation_path),
            "static_style_manifest": _source_record(static_path),
            "motion_manifest": _source_record(motion_path),
            "motion_replay": _source_record(replay_path),
        },
        "coverage": coverage,
        "categorical_quality": categorical,
        "presentation_quality": presentation,
        "motion_quality": motion,
        "proof": {
            **static_proof,
            **motion_proof,
            "motion_replay_artifact_count": int(replay["artifact_count_compared"]),
            "motion_replay_bytes": int(replay["bytes_compared"]),
            "motion_replay_exact_identity": bool(replay["exact_identity_replay"]),
            "motion_replay_exact_showcase": bool(replay["exact_showcase_replay"]),
        },
        "gates": gates,
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    validate_schema(report)
    return report


def assert_valid_sprite_quality_report(report: Mapping[str, Any]) -> None:
    validate_schema(report)
    base = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_bytes(canonical_json_bytes(base)):
        raise ValueError("Sprite-quality report hash is not canonical")
    if report.get("auditor", {}).get("source_sha256") != audit_source_hash():
        raise ValueError("Sprite-quality auditor source hash drifted")
    if not all(report.get("gates", {}).values()):
        raise ValueError("Sprite-quality report contains a failed objective gate")


def compile_sprite_quality_audit(
    destination: Path,
    *,
    static_manifest_path: Path | None = None,
    motion_manifest_path: Path | None = None,
    motion_replay_path: Path | None = None,
    map_art_root: Path | None = None,
) -> dict[str, Any]:
    output = prepare_immutable_destination(destination)
    report = build_sprite_quality_report(
        static_manifest_path=static_manifest_path or DEFAULT_STATIC_MANIFEST,
        motion_manifest_path=motion_manifest_path or DEFAULT_MOTION_MANIFEST,
        motion_replay_path=motion_replay_path or DEFAULT_MOTION_REPLAY,
        map_art_root=map_art_root or DEFAULT_MAP_ART_ROOT,
    )
    heatmap = render_motion_energy_heatmap(report)
    write_bytes_new(output / HEATMAP_FILENAME, heatmap)
    report = dict(report)
    report["artifacts"] = {
        "motion_energy_heatmap": artifact_record(output / HEATMAP_FILENAME, output)
    }
    report.pop("report_sha256")
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    validate_schema(report)
    write_json_new(output / REPORT_FILENAME, report)
    return report


def assert_exact_sprite_quality_replay(report_path: Path) -> dict[str, Any]:
    path = Path(report_path).resolve()
    report = _load_canonical_json(path, maximum_bytes=64 * 1024 * 1024)
    assert_valid_sprite_quality_report(report)
    stored_artifact = report["artifacts"]["motion_energy_heatmap"]
    heatmap_path = verify_artifact(path.parent, stored_artifact)
    rebuilt = build_sprite_quality_report(
        static_manifest_path=PROJECT_ROOT / Path(*report["sources"]["static_style_manifest"]["path"].split("/")),
        motion_manifest_path=PROJECT_ROOT / Path(*report["sources"]["motion_manifest"]["path"].split("/")),
        motion_replay_path=PROJECT_ROOT / Path(*report["sources"]["motion_replay"]["path"].split("/")),
        map_art_root=DEFAULT_MAP_ART_ROOT,
    )
    rebuilt["artifacts"] = report["artifacts"]
    rebuilt.pop("report_sha256")
    rebuilt["report_sha256"] = sha256_bytes(canonical_json_bytes(rebuilt))
    if rebuilt != report:
        raise ValueError("Sprite-quality report exact replay mismatch")
    expected_heatmap = render_motion_energy_heatmap(report)
    if heatmap_path.read_bytes() != expected_heatmap:
        raise ValueError("Sprite-quality heatmap exact replay mismatch")
    return {
        "format": REPLAY_FORMAT,
        "status": "passed",
        "report_sha256": sha256_file(path),
        "semantic_report_sha256": report["report_sha256"],
        "heatmap_sha256": sha256_file(heatmap_path),
        "exact_report": True,
        "exact_heatmap": True,
    }
