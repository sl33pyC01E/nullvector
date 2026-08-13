from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from forge.config import ARCHETYPES, LAYER_NAMES
from forge.grammar import (
    genome_from_seed,
    genome_vector,
    layers_to_tokens,
    palette_for_seed,
    render_layers,
    tokens_to_layers,
)
from forge.rig import (
    POSTPROCESS_VERSION,
    RIG_VERSION,
    bake_animation_atlas,
    postprocess_layers,
    structure_score,
)
from forge.validate import validate_registry


CANONICAL_HASH = "a" * 64
SOURCE_HASH = "b" * 64
MODEL_HASH = CANONICAL_HASH[:16]
VALID_SEEDS = (104, 100, 101, 100)


def _nearest_opaque(alpha: np.ndarray, point: list[int]) -> list[int]:
    x, y = point
    if 0 <= x < 32 and 0 <= y < 32 and alpha[y, x] > 0:
        return point
    points = np.argwhere(alpha > 0)
    assert points.size
    distances = (points[:, 1] - x) ** 2 + (points[:, 0] - y) ** 2
    nearest_y, nearest_x = points[int(np.argmin(distances))]
    return [int(nearest_x), int(nearest_y)]


def _ensure_attached_sockets(manifest: dict, atlas_path: Path) -> None:
    atlas = np.asarray(Image.open(atlas_path).convert("RGBA"), dtype=np.uint8)
    idle = manifest["animations"]["idle"]["frames"][0]
    x, y, width, height = idle["rect"]
    alpha = atlas[y : y + height, x : x + width, 3]
    for name, point in manifest["sockets"].items():
        manifest["sockets"][name] = _nearest_opaque(alpha, point)
    for animation in manifest["animations"].values():
        for frame in animation["frames"]:
            x, y, width, height = frame["rect"]
            alpha = atlas[y : y + height, x : x + width, 3]
            for name, point in frame["sockets"].items():
                frame["sockets"][name] = _nearest_opaque(alpha, point)


def _write_sprite(
    root: Path,
    archetype: int,
    seed: int,
    *,
    sprite_id: str | None = None,
    source_layers: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    faction: str = "hostile",
) -> dict:
    genome = genome_from_seed(seed, archetype)
    if source_layers is None:
        raw_tokens = layers_to_tokens(render_layers(genome))
        raw_layers = tokens_to_layers(raw_tokens)
        layers = postprocess_layers(raw_layers)
    else:
        raw_tokens, raw_layers, layers = (value.copy() for value in source_layers)
    raw_score, raw_valid = structure_score(raw_layers, archetype)
    processed_score, processed_valid = structure_score(layers, archetype)
    assert raw_valid and processed_valid

    sprite_id = sprite_id or f"{ARCHETYPES[archetype]}_{seed:08x}"
    atlas_path = root / f"{sprite_id}.png"
    manifest = bake_animation_atlas(
        layers,
        seed,
        archetype,
        atlas_path,
        faction=faction,
    )
    _ensure_attached_sockets(manifest, atlas_path)
    processed_tokens = layers_to_tokens(layers)
    raw_name = f"{sprite_id}_raw_tokens.png"
    processed_name = f"{sprite_id}_processed_tokens.png"
    semantic_name = f"{sprite_id}_semantic_layers.npz"
    Image.fromarray(raw_tokens.astype(np.uint8)).save(root / raw_name)
    Image.fromarray(processed_tokens.astype(np.uint8)).save(root / processed_name)
    np.savez_compressed(
        root / semantic_name,
        layers=layers.astype(np.uint8),
        raw_layers=raw_layers.astype(np.uint8),
        raw_tokens=raw_tokens.astype(np.uint8),
        processed_tokens=processed_tokens.astype(np.uint8),
        layer_names=np.asarray(LAYER_NAMES),
    )
    manifest.update(
        {
            "id": sprite_id,
            "model_hash": MODEL_HASH,
            "faction": faction,
            "palette": {
                key: list(value)
                for key, value in palette_for_seed(seed, faction).items()
            },
            "genome": {
                "seed": seed,
                "genes": [float(value) for value in genome_vector(genome)],
                "recipe": genome.to_dict(),
            },
            "generation": {
                "sampler": "absorbing-categorical-confidence-v2",
                "schedule": "sine-squared",
                "temperature": 0.86,
                "noise_seed": seed + 1,
                "candidate_index": archetype,
            },
            "selection": {
                "version": "validity-diversity-v2",
                "raw_score": round(float(raw_score), 6),
                "processed_score": round(float(processed_score), 6),
                "raw_valid": bool(raw_valid),
                "processed_valid": bool(processed_valid),
            },
            "source": {
                "raw_tokens": raw_name,
                "processed_tokens": processed_name,
                "semantic_layers": semantic_name,
            },
            "inference_source_hash": SOURCE_HASH[:16],
        }
    )
    return manifest


def _write_v2_registry(root: Path, *, duplicate_dart: bool = False) -> Path:
    sprites = [
        _write_sprite(
            root,
            archetype,
            VALID_SEEDS[archetype],
            faction="player" if archetype == 0 else "hostile",
        )
        for archetype in range(len(ARCHETYPES))
    ]
    if duplicate_dart:
        semantic_path = root / sprites[0]["source"]["semantic_layers"]
        with np.load(semantic_path, allow_pickle=False) as payload:
            source_layers = (
                np.asarray(payload["raw_tokens"]),
                np.asarray(payload["raw_layers"]),
                np.asarray(payload["layers"]),
            )
        sprites.append(
            _write_sprite(
                root,
                0,
                9_999,
                sprite_id="dart_duplicate_silhouette",
                source_layers=source_layers,
                faction="hostile",
            )
        )
    registry = {
        "format": "neural-sprite-registry-v2",
        "model_hash": MODEL_HASH,
        "model": {
            "canonical_hash": CANONICAL_HASH,
            "architecture": {"steps": 12, "width": 64},
        },
        "sampler": {
            "name": "absorbing-categorical-confidence-v2",
            "schedule": "sine-squared",
            "temperature": 0.86,
            "steps": 12,
        },
        "pipeline": {
            "inference_source_hash": SOURCE_HASH,
            "postprocess_version": POSTPROCESS_VERSION,
            "rig_version": RIG_VERSION,
        },
        "acceptance": {
            "candidate_count": len(sprites) * 2,
            "selected_count": len(sprites),
        },
        "sprite_count": len(sprites),
        "sprites": sprites,
    }
    path = root / "sprite_registry.json"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return path


def test_strict_v2_validates_schema_semantics_frames_and_provenance(tmp_path) -> None:
    report = validate_registry(_write_v2_registry(tmp_path))
    assert report["passed"], report["failures"]
    assert report["strict"] is True
    assert report["canonical_model_hash"] == CANONICAL_HASH
    assert all(sprite["semantic_source"]["processed_valid"] for sprite in report["sprites"])
    assert all(
        animation["transparent_margin_frames"] == animation["frames"]
        and animation["attached_sockets"] == animation["checked_sockets"]
        for sprite in report["sprites"]
        for animation in sprite["animations"].values()
    )


def test_strict_v2_rejects_cross_field_pixel_and_semantic_tampering(tmp_path) -> None:
    registry_path = _write_v2_registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    sprite = registry["sprites"][0]
    sprite["selection"]["processed_valid"] = False
    sprite["animations"]["attack"]["frames"][2]["duration_ms"] = 46
    sprite["animations"]["idle"]["frames"][0]["sockets"]["core"] = [0, 0]
    registry["model_hash"] = "c" * 16
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    processed_path = tmp_path / sprite["source"]["processed_tokens"]
    processed = np.asarray(Image.open(processed_path)).copy()
    processed[0, 0] = (int(processed[0, 0]) + 1) % (len(LAYER_NAMES) + 1)
    Image.fromarray(processed).save(processed_path)

    atlas_path = tmp_path / sprite["atlas"]
    atlas = np.asarray(Image.open(atlas_path).convert("RGBA")).copy()
    atlas[0, 5] = [255, 255, 255, 255]
    Image.fromarray(atlas).save(atlas_path)

    report = validate_registry(registry_path)
    assert not report["passed"]
    failures = "\n".join(report["failures"])
    assert "canonical hash prefix" in failures
    assert "duration_ms must be 45" in failures
    assert "not attached to an opaque" in failures
    assert "safety margin" in failures
    assert "processed token PNG and semantic NPZ disagree" in failures
    assert "processed_valid: metadata disagrees" in failures


def test_v1_is_backward_tolerant_but_explicit_strict_mode_rejects_it(tmp_path) -> None:
    registry_path = _write_v2_registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["format"] = "neural-sprite-registry-v1"
    for name in ("model", "pipeline", "sampler", "acceptance"):
        registry.pop(name)
    for sprite in registry["sprites"]:
        sprite.pop("source")
        sprite.pop("selection")
        for animation in sprite["animations"].values():
            for frame in animation["frames"]:
                frame.pop("sockets")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    legacy = validate_registry(registry_path)
    assert legacy["passed"]
    assert legacy["strict"] is False
    assert legacy["warnings"]

    strict = validate_registry(registry_path, strict=True)
    assert not strict["passed"]
    assert any("Strict artifacts must use" in failure for failure in strict["failures"])


def test_diversity_report_warns_on_near_duplicate_silhouettes(tmp_path) -> None:
    report = validate_registry(_write_v2_registry(tmp_path, duplicate_dart=True))
    assert report["passed"], report["failures"]
    assert report["diversity"]["dart"]["maximum_pairwise_iou"] == 1.0
    assert report["diversity"]["dart"]["near_duplicate_pairs"]
    assert any("silhouette pair" in warning for warning in report["warnings"])
