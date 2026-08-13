from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np

from .constants import (
    APPENDAGE,
    BODY,
    CANVAS_SIZE,
    HEAD,
    JOINT_LAYER,
    LAYER_NAMES,
    LEFT_ARM,
    LEFT_LEG,
    MANIFEST_FORMAT,
    RIGHT_ARM,
    RIGHT_LEG,
    SAFETY_MARGIN,
    SEMANTIC_FORMAT,
    SOCKET_LAYER,
    STRUCTURAL_LAYERS,
    WEAPON,
)
from .render import MorphologySpecimen, layers_to_tokens


def component_count(mask: np.ndarray) -> int:
    active = mask.astype(bool)
    seen = np.zeros_like(active)
    components = 0
    height, width = active.shape
    for y in range(height):
        for x in range(width):
            if not active[y, x] or seen[y, x]:
                continue
            components += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                py, px = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = py + dy, px + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and active[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return components


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    return np.logical_or.reduce(
        [
            padded[y : y + mask.shape[0], x : x + mask.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
    )


def _point_errors(
    points: object,
    expected: dict[str, int],
    layers: np.ndarray,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(points, dict):
        return [f"{label} must be an object"]
    if set(points) != set(expected):
        errors.append(
            f"{label} keys must be {sorted(expected)}, got {sorted(points)}"
        )
    for name, layer_index in expected.items():
        value = points.get(name)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            errors.append(f"{label}.{name} must be an integer [x, y] pair")
            continue
        x, y = value
        if not 0 <= x < CANVAS_SIZE or not 0 <= y < CANVAS_SIZE:
            errors.append(f"{label}.{name} is outside the canvas")
            continue
        if not layers[layer_index, y, x]:
            errors.append(
                f"{label}.{name} does not land on {LAYER_NAMES[layer_index]}"
            )
    return errors


def _attached(part: np.ndarray, anchor: np.ndarray) -> bool:
    return bool((part.astype(bool) & _dilate(anchor)).any())


def validate_specimen(specimen: MorphologySpecimen) -> list[str]:
    errors: list[str] = []
    try:
        specimen.genome.validate()
    except ValueError as error:
        errors.append(f"genome: {error}")

    layers = specimen.layers
    tokens = specimen.tokens
    expected_layers = (len(LAYER_NAMES), CANVAS_SIZE, CANVAS_SIZE)
    if layers.shape != expected_layers:
        return [f"layers shape must be {expected_layers}, got {layers.shape}"]
    if layers.dtype != np.uint8:
        errors.append(f"layers dtype must be uint8, got {layers.dtype}")
    if not np.isin(layers, (0, 1)).all():
        errors.append("layers must be binary")
    if tokens.shape != (CANVAS_SIZE, CANVAS_SIZE):
        errors.append(f"tokens shape must be {(CANVAS_SIZE, CANVAS_SIZE)}")
    elif tokens.dtype != np.uint8:
        errors.append(f"tokens dtype must be uint8, got {tokens.dtype}")
    elif not np.isin(tokens, np.arange(len(LAYER_NAMES) + 1)).all():
        errors.append("token values are outside the semantic vocabulary")
    elif not np.array_equal(tokens, layers_to_tokens(layers)):
        errors.append("tokens do not encode the supplied semantic layers")

    empty_layers = [
        LAYER_NAMES[index]
        for index in range(len(LAYER_NAMES))
        if int(layers[index].sum()) == 0
    ]
    if empty_layers:
        errors.append(f"semantic layers are empty: {empty_layers}")

    visible = np.logical_or.reduce(layers > 0)
    unsafe = bool(
        visible[:SAFETY_MARGIN].any()
        or visible[-SAFETY_MARGIN:].any()
        or visible[:, :SAFETY_MARGIN].any()
        or visible[:, -SAFETY_MARGIN:].any()
    )
    if unsafe:
        errors.append("semantic pixels violate the safety margin")

    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    components = component_count(structural)
    if components != 1:
        errors.append(f"structural union has {components} connected components")

    attachments: tuple[tuple[int, Iterable[int]], ...] = (
        (HEAD, (BODY,)),
        (LEFT_ARM, (BODY,)),
        (RIGHT_ARM, (BODY,)),
        (LEFT_LEG, (BODY,)),
        (RIGHT_LEG, (BODY,)),
        (APPENDAGE, (BODY, HEAD)),
        (WEAPON, (BODY, HEAD, RIGHT_ARM)),
    )
    for part_index, anchor_indices in attachments:
        anchor = np.logical_or.reduce(layers[list(anchor_indices)] > 0)
        if not _attached(layers[part_index], anchor):
            errors.append(
                f"{LAYER_NAMES[part_index]} is not attached to its parent anatomy"
            )

    if bool((layers[-2:].astype(bool) & ~structural).any()):
        errors.append("detail/emission semantics escape the structural silhouette")

    errors.extend(_point_errors(specimen.joints, JOINT_LAYER, layers, "joints"))
    errors.extend(_point_errors(specimen.sockets, SOCKET_LAYER, layers, "sockets"))

    manifest = specimen.manifest
    if manifest.get("format") != MANIFEST_FORMAT:
        errors.append("manifest format is unsupported")
    if manifest.get("seed") != specimen.genome.seed:
        errors.append("manifest seed disagrees with genome")
    if manifest.get("family") != specimen.genome.family_name:
        errors.append("manifest family disagrees with genome")
    if manifest.get("genome") != specimen.genome.to_dict():
        errors.append("manifest genome is incomplete or inconsistent")
    semantic = manifest.get("semantic")
    if not isinstance(semantic, dict) or semantic.get("format") != SEMANTIC_FORMAT:
        errors.append("manifest semantic contract is missing")
    elif semantic.get("layer_names") != list(LAYER_NAMES):
        errors.append("manifest semantic layer order is incorrect")
    if manifest.get("joints") != specimen.joints:
        errors.append("manifest joints disagree with derived joints")
    if manifest.get("sockets") != specimen.sockets:
        errors.append("manifest sockets disagree with derived sockets")
    expected_palette = {name: list(value) for name, value in specimen.palette.items()}
    if manifest.get("palette") != expected_palette:
        errors.append("manifest palette disagrees with seeded palette")

    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        errors.append("manifest hashes are missing")
    else:
        genome_hash = hashlib.sha256(
            specimen.genome.canonical_json().encode("utf-8")
        ).hexdigest()
        semantic_hash = hashlib.sha256(
            layers.tobytes() + tokens.tobytes()
        ).hexdigest()
        training = specimen.training_fields()
        if hashes.get("genome_sha256") != genome_hash:
            errors.append("manifest genome hash is incorrect")
        if hashes.get("semantic_sha256") != semantic_hash:
            errors.append("manifest semantic hash is incorrect")
        if hashes.get("training_arrays_sha256") != training.arrays_hash():
            errors.append("manifest training arrays hash is incorrect")
        if manifest.get("training_contract") != training.metadata():
            errors.append("manifest training contract is incomplete or inconsistent")
    try:
        json.dumps(manifest, allow_nan=False)
    except (TypeError, ValueError) as error:
        errors.append(f"manifest is not strict JSON: {error}")
    return errors


def assert_valid_specimen(specimen: MorphologySpecimen) -> None:
    errors = validate_specimen(specimen)
    if errors:
        raise ValueError("; ".join(errors))
