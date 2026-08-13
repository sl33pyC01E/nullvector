from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    ARCHETYPES,
    GAME_GENERATED_DIR,
    LAYER_NAMES,
    OUTPUT_DIR,
    PROJECT_ROOT,
)
from .grammar import layers_to_tokens, tokens_to_layers
from .rig import ANIMATIONS, postprocess_layers, structure_score


REGISTRY_FORMATS = {"neural-sprite-registry-v1", "neural-sprite-registry-v2"}
STRICT_REGISTRY_FORMAT = "neural-sprite-registry-v2"
SPRITE_SCHEMA_PATH = PROJECT_ROOT / "shared" / "schema" / "sprite_manifest.schema.json"
FRAME_SIZE = 32
REQUIRED_SOCKETS = ("muzzle", "core", "left", "right")
MODEL_HASH_PATTERN = re.compile(r"^[a-f0-9]{16}$")
CANONICAL_HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")
DIVERSITY_WARNING_IOU = 0.92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate neural sprite atlases and their native-game contract."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=GAME_GENERATED_DIR / "sprite_registry.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_DIR / "validation_report.json",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict",
        dest="strict",
        action="store_const",
        const=True,
        help="Require the complete v2 provenance and semantic-artifact contract.",
    )
    mode.add_argument(
        "--allow-legacy",
        dest="strict",
        action="store_const",
        const=False,
        help="Read a v1 registry and report missing v2 fields as warnings.",
    )
    parser.set_defaults(strict=None)
    return parser.parse_args()


def _check(condition: bool, message: str, failures: list[str]) -> bool:
    if not condition:
        failures.append(message)
    return condition


def _strict_check(
    condition: bool,
    message: str,
    strict: bool,
    failures: list[str],
    warnings: list[str],
) -> bool:
    if not condition:
        (failures if strict else warnings).append(message)
    return condition


def _is_strict_registry(registry: dict[str, Any]) -> bool:
    if registry.get("format") == STRICT_REGISTRY_FORMAT:
        return True
    model = registry.get("model")
    if isinstance(model, dict) and model.get("canonical_hash"):
        return True
    sprites = registry.get("sprites")
    return bool(
        isinstance(sprites, list)
        and any(
            isinstance(sprite, dict)
            and ("source" in sprite or "selection" in sprite)
            for sprite in sprites
        )
    )


def _legacy_schema_errors(manifest: dict[str, Any]) -> list[str]:
    """Validate the stable v1 subset without requiring v2 provenance fields."""
    required = {
        "id",
        "seed",
        "archetype",
        "atlas",
        "emission_atlas",
        "frame_size",
        "pivot",
        "sockets",
        "animations",
        "model_hash",
        "genome",
    }
    errors = [f"required property {name!r} is missing" for name in sorted(required.difference(manifest))]
    if manifest.get("archetype") not in ARCHETYPES:
        errors.append("archetype is unsupported")
    if manifest.get("frame_size") != [FRAME_SIZE, FRAME_SIZE]:
        errors.append("frame_size must be [32, 32]")
    if not isinstance(manifest.get("animations"), dict):
        errors.append("animations must be an object")
    if not isinstance(manifest.get("sockets"), dict):
        errors.append("sockets must be an object")
    genome = manifest.get("genome")
    if not isinstance(genome, dict):
        errors.append("genome must be an object")
    else:
        genes = genome.get("genes")
        if not isinstance(genes, list) or len(genes) != 8:
            errors.append("genome/genes must contain eight values")
    return errors


def _schema_errors(
    manifest: dict[str, Any], schema_path: Path, *, strict: bool
) -> list[str]:
    """Validate the shared JSON schema, with a small dependency-free fallback."""
    if not strict:
        return _legacy_schema_errors(manifest)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"sprite schema could not be read: {error}"]

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        required = schema.get("required", [])
        errors = [f"required property {name!r} is missing" for name in required if name not in manifest]
        if not isinstance(manifest.get("animations"), dict):
            errors.append("'animations' must be an object")
        if not isinstance(manifest.get("genome"), dict):
            errors.append("'genome' must be an object")
        return errors

    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(manifest),
        key=lambda item: "/".join(str(part) for part in item.absolute_path),
    )
    output = []
    for error in errors:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        output.append(f"{location}: {error.message}")
    return output


def _resolve_artifact(
    root: Path,
    value: Any,
    label: str,
    failures: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        failures.append(f"{label}: artifact path is missing or is not a string.")
        return None
    root = root.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        failures.append(f"{label}: artifact path escapes the registry directory.")
        return None
    return candidate


def _load_rgba(
    path: Path | None,
    label: str,
    failures: list[str],
) -> tuple[np.ndarray | None, str | None]:
    if path is None:
        return None, None
    if not path.is_file():
        failures.append(f"{label}: file is missing.")
        return None, None
    try:
        with Image.open(path) as image:
            mode = image.mode
            pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except OSError as error:
        failures.append(f"{label}: image could not be decoded: {error}")
        return None, None
    return pixels, mode


def _load_token_image(
    path: Path | None,
    label: str,
    failures: list[str],
) -> np.ndarray | None:
    if path is None:
        return None
    if not path.is_file():
        failures.append(f"{label}: file is missing.")
        return None
    try:
        with Image.open(path) as image:
            mode = image.mode
            values = np.asarray(image)
    except OSError as error:
        failures.append(f"{label}: token image could not be decoded: {error}")
        return None
    if not _check(mode == "L" and values.ndim == 2, f"{label}: token PNG must be a single-channel 8-bit image.", failures):
        return None
    if not _check(values.shape == (FRAME_SIZE, FRAME_SIZE), f"{label}: expected a 32x32 token map.", failures):
        return None
    if not _check(np.issubdtype(values.dtype, np.integer), f"{label}: tokens must be integers.", failures):
        return None
    if not _check(bool(np.isin(values, np.arange(len(LAYER_NAMES) + 1)).all()), f"{label}: token values must be in [0, {len(LAYER_NAMES)}].", failures):
        return None
    return values.astype(np.uint8, copy=False)


def _socket_coordinates(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        return None
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y) or not x.is_integer() or not y.is_integer():
        return None
    return int(x), int(y)


def _validate_sockets(
    sockets: Any,
    label: str,
    alpha: np.ndarray | None,
    strict: bool,
    failures: list[str],
    warnings: list[str],
) -> dict[str, list[int]]:
    if not _strict_check(isinstance(sockets, dict), f"{label}: socket map is missing.", strict, failures, warnings):
        return {}
    normalized: dict[str, list[int]] = {}
    for name in REQUIRED_SOCKETS:
        if not _strict_check(name in sockets, f"{label}: required socket {name!r} is missing.", strict, failures, warnings):
            continue
        point = _socket_coordinates(sockets[name])
        if not _strict_check(point is not None, f"{label}/{name}: socket must be an integer [x, y] pair.", strict, failures, warnings):
            continue
        assert point is not None
        x, y = point
        in_bounds = 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE
        if not _strict_check(in_bounds, f"{label}/{name}: socket is outside the 32x32 frame.", strict, failures, warnings):
            continue
        normalized[name] = [x, y]
        if alpha is not None:
            _strict_check(
                bool(alpha[y, x] > 0),
                f"{label}/{name}: socket is not attached to an opaque sprite pixel.",
                strict,
                failures,
                warnings,
            )
    return normalized


def _transparent_margin(alpha: np.ndarray) -> bool:
    return not bool(
        alpha[0].any()
        or alpha[-1].any()
        or alpha[:, 0].any()
        or alpha[:, -1].any()
    )


def _silhouette_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def _validate_semantic_source(
    sprite: dict[str, Any],
    root: Path,
    archetype_index: int | None,
    strict: bool,
    failures: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], np.ndarray | None]:
    sprite_id = str(sprite.get("id", "missing"))
    source = sprite.get("source")
    if not _strict_check(isinstance(source, dict), f"{sprite_id}: semantic source metadata is missing.", strict, failures, warnings):
        return {}, None
    assert isinstance(source, dict)

    paths = {
        name: _resolve_artifact(root, source.get(name), f"{sprite_id}/source/{name}", failures)
        for name in ("raw_tokens", "processed_tokens", "semantic_layers")
    }
    raw_png = _load_token_image(paths["raw_tokens"], f"{sprite_id}/raw_tokens", failures)
    processed_png = _load_token_image(paths["processed_tokens"], f"{sprite_id}/processed_tokens", failures)
    semantic_path = paths["semantic_layers"]
    if semantic_path is None or not semantic_path.is_file():
        if semantic_path is not None:
            failures.append(f"{sprite_id}/semantic_layers: file is missing.")
        return {}, None

    try:
        with np.load(semantic_path, allow_pickle=False) as payload:
            required = {"layers", "raw_layers", "raw_tokens", "processed_tokens", "layer_names"}
            missing = sorted(required.difference(payload.files))
            _check(not missing, f"{sprite_id}/semantic_layers: missing arrays {missing}.", failures)
            if missing:
                return {}, None
            layers = np.asarray(payload["layers"])
            raw_layers = np.asarray(payload["raw_layers"])
            raw_tokens = np.asarray(payload["raw_tokens"])
            processed_tokens = np.asarray(payload["processed_tokens"])
            layer_names = [str(value) for value in np.asarray(payload["layer_names"]).tolist()]
    except (OSError, ValueError) as error:
        failures.append(f"{sprite_id}/semantic_layers: NPZ could not be decoded: {error}")
        return {}, None

    expected_layer_shape = (len(LAYER_NAMES), FRAME_SIZE, FRAME_SIZE)
    layer_arrays = (("layers", layers), ("raw_layers", raw_layers))
    for name, values in layer_arrays:
        _check(values.shape == expected_layer_shape, f"{sprite_id}/semantic_layers/{name}: expected shape {expected_layer_shape}.", failures)
        _check(values.dtype == np.uint8, f"{sprite_id}/semantic_layers/{name}: dtype must be uint8.", failures)
        _check(bool(np.isin(values, (0, 1)).all()), f"{sprite_id}/semantic_layers/{name}: layers must be binary.", failures)
    for name, values in (("raw_tokens", raw_tokens), ("processed_tokens", processed_tokens)):
        _check(values.shape == (FRAME_SIZE, FRAME_SIZE), f"{sprite_id}/semantic_layers/{name}: expected shape (32, 32).", failures)
        _check(values.dtype == np.uint8, f"{sprite_id}/semantic_layers/{name}: dtype must be uint8.", failures)
        _check(bool(np.isin(values, np.arange(len(LAYER_NAMES) + 1)).all()), f"{sprite_id}/semantic_layers/{name}: token values are out of range.", failures)
    _check(np.asarray(layer_names).dtype != object, f"{sprite_id}/semantic_layers: layer_names may not use object dtype.", failures)
    _check(layer_names == list(LAYER_NAMES), f"{sprite_id}/semantic_layers: layer_names do not match the runtime contract.", failures)

    if any(values.shape != expected_layer_shape for _, values in layer_arrays) or raw_tokens.shape != (FRAME_SIZE, FRAME_SIZE) or processed_tokens.shape != (FRAME_SIZE, FRAME_SIZE):
        return {}, None

    raw_tokens = raw_tokens.astype(np.uint8, copy=False)
    processed_tokens = processed_tokens.astype(np.uint8, copy=False)
    layers = layers.astype(np.uint8, copy=False)
    raw_layers = raw_layers.astype(np.uint8, copy=False)
    _check(raw_png is not None and np.array_equal(raw_png, raw_tokens), f"{sprite_id}: raw token PNG and semantic NPZ disagree.", failures)
    _check(processed_png is not None and np.array_equal(processed_png, processed_tokens), f"{sprite_id}: processed token PNG and semantic NPZ disagree.", failures)
    _check(np.array_equal(tokens_to_layers(raw_tokens), raw_layers), f"{sprite_id}: raw layers do not decode from raw_tokens.", failures)
    _check(np.array_equal(layers_to_tokens(layers), processed_tokens), f"{sprite_id}: processed_tokens do not encode semantic layers.", failures)
    _check(np.array_equal(postprocess_layers(raw_layers), layers), f"{sprite_id}: semantic layers do not match the current postprocess result.", failures)

    semantic_report: dict[str, Any] = {
        "present": True,
        "raw_valid": None,
        "processed_valid": None,
        "raw_score": None,
        "processed_score": None,
    }
    if archetype_index is not None:
        raw_score, raw_valid = structure_score(raw_layers, archetype_index)
        processed_score, processed_valid = structure_score(layers, archetype_index)
        semantic_report.update(
            {
                "raw_valid": bool(raw_valid),
                "processed_valid": bool(processed_valid),
                "raw_score": float(raw_score),
                "processed_score": float(processed_score),
            }
        )
        selection = sprite.get("selection")
        if _strict_check(isinstance(selection, dict), f"{sprite_id}: selection metadata is missing.", strict, failures, warnings):
            assert isinstance(selection, dict)
            for name, actual in (("raw_valid", raw_valid), ("processed_valid", processed_valid)):
                _check(isinstance(selection.get(name), bool), f"{sprite_id}/selection/{name}: boolean metadata is required.", failures)
                if isinstance(selection.get(name), bool):
                    _check(selection[name] is bool(actual), f"{sprite_id}/selection/{name}: metadata disagrees with semantic artifacts.", failures)
            for name, actual in (("raw_score", raw_score), ("processed_score", processed_score)):
                value = selection.get(name)
                _check(isinstance(value, Real) and not isinstance(value, bool), f"{sprite_id}/selection/{name}: numeric metadata is required.", failures)
                if isinstance(value, Real) and not isinstance(value, bool):
                    _check(math.isclose(float(value), float(actual), abs_tol=1.0e-5), f"{sprite_id}/selection/{name}: metadata disagrees with semantic artifacts.", failures)
        _strict_check(bool(raw_valid), f"{sprite_id}: selected raw neural output is structurally invalid.", strict, failures, warnings)
        _strict_check(bool(processed_valid), f"{sprite_id}: selected processed output is structurally invalid.", strict, failures, warnings)

    silhouette = np.maximum.reduce(layers[:6]).astype(bool)
    return semantic_report, silhouette


def _validate_animation_contract(
    sprite: dict[str, Any],
    atlas: np.ndarray,
    emission: np.ndarray,
    strict: bool,
    failures: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray]:
    sprite_id = str(sprite.get("id", "missing"))
    animations = sprite.get("animations")
    if not _check(isinstance(animations, dict), f"{sprite_id}: animations must be an object.", failures):
        return {}, None, np.zeros(atlas.shape[:2], dtype=bool)
    assert isinstance(animations, dict)
    animation_reports: dict[str, Any] = {}
    declared = np.zeros(atlas.shape[:2], dtype=bool)
    idle_silhouette: np.ndarray | None = None
    seen_rects: set[tuple[int, int, int, int]] = set()

    expected_names = set(ANIMATIONS)
    _check(expected_names.issubset(animations), f"{sprite_id}: required animations are missing: {sorted(expected_names.difference(animations))}.", failures)
    extras = sorted(set(animations).difference(expected_names))
    if extras:
        warnings.append(f"{sprite_id}: undeclared extra animations are present: {extras}.")

    for row, (animation_name, poses) in enumerate(ANIMATIONS.items()):
        animation = animations.get(animation_name)
        if not isinstance(animation, dict):
            continue
        expected_loop = animation_name in {"idle", "move"}
        _check(animation.get("loop") is expected_loop, f"{sprite_id}/{animation_name}: loop flag is incorrect.", failures)
        frames = animation.get("frames")
        if not _check(isinstance(frames, list), f"{sprite_id}/{animation_name}: frames must be an array.", failures):
            continue
        assert isinstance(frames, list)
        _check(len(frames) == len(poses), f"{sprite_id}/{animation_name}: expected {len(poses)} frames, got {len(frames)}.", failures)
        frame_payloads: list[bytes] = []
        clear_margin_frames = 0
        attached_socket_count = 0
        checked_socket_count = 0
        for column, pose in enumerate(poses):
            if column >= len(frames) or not isinstance(frames[column], dict):
                continue
            frame = frames[column]
            label = f"{sprite_id}/{animation_name}/{pose.name}"
            _check(frame.get("name") == pose.name, f"{label}: frame name is incorrect.", failures)
            _check(frame.get("duration_ms") == pose.duration_ms, f"{label}: duration_ms must be {pose.duration_ms}.", failures)
            expected_event = "fire" if animation_name == "attack" and column == 2 else None
            _check(frame.get("event") == expected_event, f"{label}: event must be {expected_event!r}.", failures)
            if strict:
                _check("event" in frame, f"{label}: explicit event metadata is required.", failures)

            rect = frame.get("rect")
            valid_rect = (
                isinstance(rect, list)
                and len(rect) == 4
                and all(isinstance(value, int) and not isinstance(value, bool) for value in rect)
            )
            if not _check(valid_rect, f"{label}: rect must contain four integers.", failures):
                continue
            x, y, width, height = rect
            expected_rect = [column * FRAME_SIZE, row * FRAME_SIZE, FRAME_SIZE, FRAME_SIZE]
            _check(rect == expected_rect, f"{label}: rect must be {expected_rect}.", failures)
            in_bounds = x >= 0 and y >= 0 and width == FRAME_SIZE and height == FRAME_SIZE and x + width <= atlas.shape[1] and y + height <= atlas.shape[0]
            if not _check(in_bounds, f"{label}: frame rectangle is outside the atlas.", failures):
                continue
            rect_key = (x, y, width, height)
            _check(rect_key not in seen_rects, f"{label}: frame rectangle overlaps another declared frame.", failures)
            seen_rects.add(rect_key)
            declared[y : y + height, x : x + width] = True
            pixels = atlas[y : y + height, x : x + width]
            glow = emission[y : y + height, x : x + width]
            alpha = pixels[..., 3]
            _check(bool((alpha > 0).any()), f"{label}: frame is empty.", failures)
            _check(not bool(((glow[..., 3] > 0) & (alpha == 0)).any()), f"{label}: emission escapes the base silhouette.", failures)
            margin_clear = _transparent_margin(alpha)
            clear_margin_frames += int(margin_clear)
            _strict_check(margin_clear, f"{label}: frame lacks a one-pixel transparent safety margin.", strict, failures, warnings)
            normalized = _validate_sockets(frame.get("sockets"), f"{label}/sockets", alpha, strict, failures, warnings)
            checked_socket_count += len(normalized)
            attached_socket_count += sum(bool(alpha[point[1], point[0]] > 0) for point in normalized.values())
            frame_payloads.append(pixels.tobytes())
            if animation_name == "idle" and column == 0:
                idle_silhouette = alpha > 0

        unique_frames = len(set(frame_payloads))
        if len(frame_payloads) > 1:
            _check(unique_frames > 1, f"{sprite_id}/{animation_name}: animation is completely static.", failures)
        animation_reports[animation_name] = {
            "frames": len(frames),
            "unique_frames": unique_frames,
            "transparent_margin_frames": clear_margin_frames,
            "attached_sockets": attached_socket_count,
            "checked_sockets": checked_socket_count,
        }
    return animation_reports, idle_silhouette, declared


def _diversity_report(
    silhouettes: dict[str, list[tuple[str, np.ndarray]]],
    warnings: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for archetype in ARCHETYPES:
        entries = silhouettes.get(archetype, [])
        pairs = []
        for index, (first_id, first) in enumerate(entries):
            for second_id, second in entries[:index]:
                pairs.append((first_id, second_id, _silhouette_iou(first, second)))
        values = [item[2] for item in pairs]
        near = [
            {"first": first, "second": second, "iou": round(iou, 6)}
            for first, second, iou in pairs
            if iou >= DIVERSITY_WARNING_IOU
        ]
        if near:
            warnings.append(
                f"{archetype}: {len(near)} silhouette pair(s) have IoU >= {DIVERSITY_WARNING_IOU:.2f}."
            )
        report[archetype] = {
            "sprites": len(entries),
            "pairs": len(pairs),
            "mean_pairwise_iou": float(np.mean(values)) if values else None,
            "maximum_pairwise_iou": max(values) if values else None,
            "near_duplicate_pairs": near,
        }
    return report


def validate_registry(
    registry_path: Path,
    *,
    strict: bool | None = None,
    schema_path: Path = SPRITE_SCHEMA_PATH,
) -> dict[str, Any]:
    registry_path = Path(registry_path)
    failures: list[str] = []
    warnings: list[str] = []
    try:
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "passed": False,
            "strict": bool(strict),
            "failures": [f"Registry could not be read: {error}"],
            "warnings": [],
            "sprites": [],
            "diversity": {},
        }
    if not isinstance(registry_payload, dict):
        return {
            "passed": False,
            "strict": bool(strict),
            "failures": ["Registry root must be a JSON object."],
            "warnings": [],
            "sprites": [],
            "diversity": {},
        }
    registry: dict[str, Any] = registry_payload
    if strict is None:
        strict = _is_strict_registry(registry)
    root = registry_path.parent.resolve()

    registry_format = registry.get("format")
    if registry_format == STRICT_REGISTRY_FORMAT:
        strict = True
    _check(registry_format in REGISTRY_FORMATS, "Registry format identifier is missing or unsupported.", failures)
    if strict:
        _check(registry_format == STRICT_REGISTRY_FORMAT, f"Strict artifacts must use {STRICT_REGISTRY_FORMAT!r}.", failures)

    sprites = registry.get("sprites")
    if not _check(isinstance(sprites, list), "Registry sprites must be an array.", failures):
        sprites = []
    assert isinstance(sprites, list)
    _check(len(sprites) >= len(ARCHETYPES), "Registry contains fewer sprites than archetypes.", failures)
    _check(registry.get("sprite_count") == len(sprites), "Registry sprite_count does not match the sprites array.", failures)

    legacy_model_hash = registry.get("model_hash")
    _check(isinstance(legacy_model_hash, str) and bool(MODEL_HASH_PATTERN.fullmatch(legacy_model_hash)), "Registry model_hash must be 16 lowercase hexadecimal characters.", failures)
    model = registry.get("model")
    canonical_hash: str | None = None
    if _strict_check(isinstance(model, dict), "Registry v2 model metadata is missing.", strict, failures, warnings):
        assert isinstance(model, dict)
        candidate = model.get("canonical_hash")
        _strict_check(isinstance(candidate, str) and bool(CANONICAL_HASH_PATTERN.fullmatch(candidate)), "Registry model.canonical_hash must be 64 lowercase hexadecimal characters.", strict, failures, warnings)
        if isinstance(candidate, str) and CANONICAL_HASH_PATTERN.fullmatch(candidate):
            canonical_hash = candidate
            _check(legacy_model_hash == canonical_hash[:16], "Registry model_hash is not the canonical hash prefix.", failures)
        _strict_check(isinstance(model.get("architecture"), dict), "Registry model architecture metadata is missing.", strict, failures, warnings)

    pipeline = registry.get("pipeline")
    if _strict_check(isinstance(pipeline, dict), "Registry v2 pipeline metadata is missing.", strict, failures, warnings):
        assert isinstance(pipeline, dict)
        for name in ("inference_source_hash", "postprocess_version", "rig_version"):
            _strict_check(bool(pipeline.get(name)), f"Registry pipeline.{name} is missing.", strict, failures, warnings)
    sampler = registry.get("sampler")
    if _strict_check(isinstance(sampler, dict), "Registry v2 sampler metadata is missing.", strict, failures, warnings):
        assert isinstance(sampler, dict)
        for name in ("name", "schedule", "temperature", "steps"):
            _strict_check(name in sampler, f"Registry sampler.{name} is missing.", strict, failures, warnings)

    sprite_reports: list[dict[str, Any]] = []
    atlas_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()
    silhouettes: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    seen_archetypes: set[str] = set()

    expected_atlas_size = (
        max(len(poses) for poses in ANIMATIONS.values()) * FRAME_SIZE,
        len(ANIMATIONS) * FRAME_SIZE,
    )
    for sprite_index, sprite_value in enumerate(sprites):
        if not _check(isinstance(sprite_value, dict), f"sprites/{sprite_index}: manifest must be an object.", failures):
            continue
        sprite: dict[str, Any] = sprite_value
        sprite_id = str(sprite.get("id", f"missing-{sprite_index}"))
        for error in _schema_errors(sprite, schema_path, strict=strict):
            failures.append(f"{sprite_id}/schema: {error}")
        _check(sprite_id not in seen_ids, f"{sprite_id}: duplicate sprite id.", failures)
        seen_ids.add(sprite_id)

        archetype = sprite.get("archetype")
        archetype_index = ARCHETYPES.index(archetype) if archetype in ARCHETYPES else None
        _check(archetype_index is not None, f"{sprite_id}: archetype is unsupported.", failures)
        if isinstance(archetype, str):
            seen_archetypes.add(archetype)
        _check(sprite.get("model_hash") == legacy_model_hash, f"{sprite_id}: model_hash disagrees with the registry.", failures)
        seed = sprite.get("seed")
        genome = sprite.get("genome")
        if isinstance(genome, dict):
            _check(genome.get("seed") == seed, f"{sprite_id}: genome seed disagrees with manifest seed.", failures)

        atlas_path = _resolve_artifact(root, sprite.get("atlas"), f"{sprite_id}/atlas", failures)
        emission_path = _resolve_artifact(root, sprite.get("emission_atlas"), f"{sprite_id}/emission_atlas", failures)
        atlas, atlas_mode = _load_rgba(atlas_path, f"{sprite_id}/atlas", failures)
        emission, emission_mode = _load_rgba(emission_path, f"{sprite_id}/emission_atlas", failures)
        if atlas is None or emission is None:
            continue

        _strict_check(atlas_mode == "RGBA", f"{sprite_id}: base atlas must be stored as RGBA, not {atlas_mode!r}.", strict, failures, warnings)
        _strict_check(emission_mode == "RGBA", f"{sprite_id}: emission atlas must be stored as RGBA, not {emission_mode!r}.", strict, failures, warnings)
        if not _check(atlas.shape == emission.shape, f"{sprite_id}: base and emission atlas sizes differ.", failures):
            continue
        _check((atlas.shape[1], atlas.shape[0]) == expected_atlas_size, f"{sprite_id}: atlas dimensions must be {expected_atlas_size[0]}x{expected_atlas_size[1]}.", failures)
        base_alpha_values = set(np.unique(atlas[..., 3]).tolist())
        emission_alpha_values = set(np.unique(emission[..., 3]).tolist())
        binary_alpha = base_alpha_values.issubset({0, 255})
        binary_emission_alpha = emission_alpha_values.issubset({0, 255})
        _check(binary_alpha, f"{sprite_id}: base alpha contains non-binary values.", failures)
        _check(binary_emission_alpha, f"{sprite_id}: emission alpha contains non-binary values.", failures)
        _check(not bool(((emission[..., 3] > 0) & (atlas[..., 3] == 0)).any()), f"{sprite_id}: emission pixels escape the base atlas silhouette.", failures)
        opaque_colors = np.unique(atlas[atlas[..., 3] > 0, :3], axis=0)
        emission_colors = np.unique(emission[emission[..., 3] > 0, :3], axis=0)
        _check(0 < len(opaque_colors) <= 12, f"{sprite_id}: opaque palette must contain between 1 and 12 colors.", failures)
        _check(len(emission_colors) <= 1, f"{sprite_id}: emission atlas must use one opaque RGB color.", failures)

        atlas_digest = hashlib.sha256(atlas.tobytes()).hexdigest()
        duplicate_id = atlas_hashes.get(atlas_digest)
        _check(duplicate_id is None, f"{sprite_id}: baked atlas duplicates {duplicate_id}.", failures)
        atlas_hashes[atlas_digest] = sprite_id

        animation_reports, atlas_silhouette, declared = _validate_animation_contract(
            sprite, atlas, emission, strict, failures, warnings
        )
        _strict_check(not bool((atlas[..., 3][~declared] > 0).any()), f"{sprite_id}: opaque pixels exist outside declared frame rectangles.", strict, failures, warnings)
        _strict_check(not bool((emission[..., 3][~declared] > 0).any()), f"{sprite_id}: emission pixels exist outside declared frame rectangles.", strict, failures, warnings)

        first_idle_alpha = None
        idle = sprite.get("animations", {}).get("idle") if isinstance(sprite.get("animations"), dict) else None
        if isinstance(idle, dict) and isinstance(idle.get("frames"), list) and idle["frames"]:
            rect = idle["frames"][0].get("rect") if isinstance(idle["frames"][0], dict) else None
            if isinstance(rect, list) and len(rect) == 4 and all(isinstance(value, int) for value in rect):
                x, y, width, height = rect
                if width == FRAME_SIZE and height == FRAME_SIZE and x >= 0 and y >= 0 and x + width <= atlas.shape[1] and y + height <= atlas.shape[0]:
                    first_idle_alpha = atlas[y : y + height, x : x + width, 3]
        _validate_sockets(sprite.get("sockets"), f"{sprite_id}/sockets", first_idle_alpha, strict, failures, warnings)
        pivot = _socket_coordinates(sprite.get("pivot"))
        _check(pivot is not None and 0 <= pivot[0] < FRAME_SIZE and 0 <= pivot[1] < FRAME_SIZE, f"{sprite_id}: pivot must be an in-bounds integer [x, y] pair.", failures)

        semantic_report, semantic_silhouette = _validate_semantic_source(
            sprite, root, archetype_index, strict, failures, warnings
        )
        silhouette = semantic_silhouette if semantic_silhouette is not None else atlas_silhouette
        if isinstance(archetype, str) and silhouette is not None:
            silhouettes[archetype].append((sprite_id, silhouette))

        if strict and isinstance(pipeline, dict):
            _check(sprite.get("rig_version") == pipeline.get("rig_version"), f"{sprite_id}: rig_version disagrees with registry pipeline.", failures)
            _check(sprite.get("postprocess_version") == pipeline.get("postprocess_version"), f"{sprite_id}: postprocess_version disagrees with registry pipeline.", failures)
            source_hash = sprite.get("inference_source_hash")
            pipeline_hash = pipeline.get("inference_source_hash")
            _check(isinstance(source_hash, str) and isinstance(pipeline_hash, str) and pipeline_hash.startswith(source_hash), f"{sprite_id}: inference source hash disagrees with registry pipeline.", failures)
            _check(sprite.get("faction") in {"player", "hostile"}, f"{sprite_id}: faction is missing or unsupported.", failures)
            _check(isinstance(sprite.get("generation"), dict), f"{sprite_id}: generation metadata is missing.", failures)

        sprite_reports.append(
            {
                "id": sprite_id,
                "archetype": archetype,
                "atlas_sha256": atlas_digest,
                "opaque_palette_colors": int(len(opaque_colors)),
                "binary_alpha": binary_alpha,
                "binary_emission_alpha": binary_emission_alpha,
                "semantic_source": semantic_report or {"present": False},
                "animations": animation_reports,
            }
        )

    _check(seen_archetypes == set(ARCHETYPES), f"Registry archetype coverage is incomplete: {sorted(set(ARCHETYPES).difference(seen_archetypes))}.", failures)
    acceptance = registry.get("acceptance")
    if _strict_check(isinstance(acceptance, dict), "Registry v2 acceptance metadata is missing.", strict, failures, warnings):
        assert isinstance(acceptance, dict)
        _check(acceptance.get("selected_count") == len(sprites), "Registry acceptance.selected_count disagrees with sprite_count.", failures)
        candidate_count = acceptance.get("candidate_count")
        selected_count = acceptance.get("selected_count")
        _check(isinstance(candidate_count, int) and isinstance(selected_count, int) and candidate_count >= selected_count >= 0, "Registry acceptance counts are invalid.", failures)

    diversity = _diversity_report(silhouettes, warnings)
    return {
        "passed": not failures,
        "strict": strict,
        "format": registry_format,
        "model_hash": legacy_model_hash,
        "canonical_model_hash": canonical_hash,
        "sprite_count": len(sprite_reports),
        "unique_atlases": len(atlas_hashes),
        "schema": str(schema_path),
        "failures": failures,
        "warnings": warnings,
        "diversity": diversity,
        "sprites": sprite_reports,
    }


def main() -> None:
    args = parse_args()
    report = validate_registry(args.registry, strict=args.strict)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
