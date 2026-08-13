from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ARCHETYPES, LAYER_NAMES
from .grammar import compose_rgba

POSTPROCESS_VERSION = "semantic-cleanup-v3"
RIG_VERSION = "layer-rig-v3"


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                py, px = stack.pop()
                component.append((py, px))
                for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) > len(best):
                best = component
    output = np.zeros_like(mask, dtype=np.uint8)
    for y, x in best:
        output[y, x] = 1
    return output


def _largest_components(mask: np.ndarray, count: int) -> np.ndarray:
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                py, px = stack.pop()
                component.append((py, px))
                for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(component)
    output = np.zeros_like(mask, dtype=np.uint8)
    for component in sorted(components, key=len, reverse=True)[:count]:
        for y, x in component:
            output[y, x] = 1
    return output


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    views = [
        padded[y : y + mask.shape[0], x : x + mask.shape[1]]
        for y in range(3)
        for x in range(3)
    ]
    return np.logical_or.reduce(views).astype(np.uint8)


def postprocess_layers(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    layers = (probabilities >= threshold).astype(np.uint8)
    layers[0] = _largest_component(layers[0])
    layers[1] = _largest_components(layers[1], 2)
    for index in (2, 3, 4, 5):
        if layers[index].sum() > 0:
            layers[index] = _largest_component(layers[index])
    body = np.maximum.reduce(layers[:6])
    layers[6] &= body
    layers[7] &= _dilate_mask(body)
    layers[7] |= layers[5]
    return layers


def structure_score(layers: np.ndarray, archetype: int) -> tuple[float, bool]:
    body = np.maximum.reduce(layers[:6])
    area = int(body.sum())
    target_area = (166, 223, 150, 331)[archetype]
    minimum_area = (85, 120, 70, 190)[archetype]
    maximum_area = (275, 360, 285, 470)[archetype]
    score = -abs(area - target_area) * 0.012
    score += min(int(layers[5].sum()), 12) * 0.12
    score += 1.0 if layers[2].sum() >= 2 else -1.5
    score += 1.0 if layers[3].sum() >= 2 else -1.5
    score += 0.7 if layers[4].sum() >= 2 else -0.8
    score += 0.4 if layers[6].sum() >= 1 else 0.0
    edge_pixels = int(body[0].sum() + body[-1].sum() + body[:, 0].sum() + body[:, -1].sum())
    score -= edge_pixels * 0.4
    largest_area = int(_largest_component(body).sum())
    connected_ratio = largest_area / max(area, 1)
    score += (connected_ratio - 0.9) * 4.0
    anchor = np.maximum.reduce(layers[[0, 1, 5]])
    attachment_zone = _dilate_mask(anchor)
    attached = [bool(np.any(layers[index] & attachment_zone)) for index in (2, 3, 4)]
    score += sum(0.35 if value else -1.0 for value in attached)
    points = np.argwhere(body > 0)
    if points.size:
        min_y, min_x = points.min(axis=0)
        max_y, max_x = points.max(axis=0)
        animation_safe = (max_x - min_x) <= 27 and (max_y - min_y) <= 27
    else:
        animation_safe = False
    score += 0.4 if animation_safe else -1.0
    valid = (
        minimum_area <= area <= maximum_area
        and edge_pixels == 0
        and connected_ratio >= 0.94
        and all(attached)
        and animation_safe
        and layers[5].sum() >= 2
        and layers[2].sum() >= 2
        and layers[3].sum() >= 2
        and layers[4].sum() >= 2
    )
    return score, valid


def _fit_margin(layers: np.ndarray, margin: int = 2) -> np.ndarray:
    visible = np.maximum.reduce(layers)
    points = np.argwhere(visible > 0)
    if points.size == 0:
        return layers
    min_y, min_x = points.min(axis=0)
    max_y, max_x = points.max(axis=0)
    lower_x = margin - int(min_x)
    upper_x = 31 - margin - int(max_x)
    lower_y = margin - int(min_y)
    upper_y = 31 - margin - int(max_y)
    dx = lower_x if lower_x > 0 else upper_x if upper_x < 0 else 0
    dy = lower_y if lower_y > 0 else upper_y if upper_y < 0 else 0
    fitted = layers
    if dx or dy:
        fitted = np.stack(
            [_transform_layer(layer, dx=dx, dy=dy) for layer in layers],
            axis=0,
        )
    # A one-pixel outline is added later, so keep two logical cells clear.
    fitted = fitted.copy()
    fitted[:, :margin, :] = 0
    fitted[:, -margin:, :] = 0
    fitted[:, :, :margin] = 0
    fitted[:, :, -margin:] = 0
    return fitted


def _mask_anchor(mask: np.ndarray, fallback: tuple[int, int]) -> list[int]:
    points = np.argwhere(mask > 0)
    if points.size == 0:
        return [fallback[0], fallback[1]]
    y = int(round(float(points[:, 0].mean())))
    x = int(round(float(points[:, 1].mean())))
    return [x, y]


def derive_sockets(layers: np.ndarray) -> dict[str, list[int]]:
    weapon_points = np.argwhere(layers[4] > 0)
    if weapon_points.size:
        muzzle_y = int(weapon_points[:, 0].min())
        muzzle_x = int(round(float(weapon_points[weapon_points[:, 0] == muzzle_y, 1].mean())))
        muzzle = [muzzle_x, muzzle_y]
    else:
        muzzle = [16, 3]
    return {
        "muzzle": muzzle,
        "core": _mask_anchor(layers[5], (16, 16)),
        "left": _mask_anchor(layers[2], (11, 17)),
        "right": _mask_anchor(layers[3], (21, 17)),
    }


def _transform_layer(
    layer: np.ndarray,
    *,
    angle: float = 0.0,
    dx: int = 0,
    dy: int = 0,
    center: tuple[int, int] = (16, 16),
) -> np.ndarray:
    image = Image.fromarray((layer * 255).astype(np.uint8))
    if angle:
        image = image.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            center=center,
            fillcolor=0,
        )
    if dx or dy:
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, -dx, 0, 1, -dy),
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class Pose:
    name: str
    duration_ms: int
    bob_y: int = 0
    lean_x: int = 0
    left_angle: float = 0.0
    right_angle: float = 0.0
    weapon_dy: int = 0
    flash: bool = False


ANIMATIONS: dict[str, tuple[Pose, ...]] = {
    "idle": (
        Pose("idle_0", 140, bob_y=0, left_angle=0, right_angle=0),
        Pose("idle_1", 140, bob_y=-1, left_angle=-5, right_angle=5),
        Pose("idle_2", 140, bob_y=0, left_angle=0, right_angle=0),
        Pose("idle_3", 140, bob_y=1, left_angle=5, right_angle=-5),
    ),
    "move": (
        Pose("move_0", 75, bob_y=0, lean_x=0, left_angle=-8, right_angle=8),
        Pose("move_1", 75, bob_y=-1, lean_x=1, left_angle=-13, right_angle=4),
        Pose("move_2", 75, bob_y=0, lean_x=1, left_angle=-5, right_angle=12),
        Pose("move_3", 75, bob_y=1, lean_x=0, left_angle=8, right_angle=-8),
        Pose("move_4", 75, bob_y=-1, lean_x=-1, left_angle=13, right_angle=-4),
        Pose("move_5", 75, bob_y=0, lean_x=-1, left_angle=5, right_angle=-12),
    ),
    "attack": (
        Pose("attack_0", 80, weapon_dy=1),
        Pose("attack_1", 55, weapon_dy=2),
        Pose("attack_2", 45, bob_y=-1, weapon_dy=-3, flash=True),
        Pose("attack_3", 60, weapon_dy=-1, flash=True),
        Pose("attack_4", 90, weapon_dy=0),
    ),
    "hit": (
        Pose("hit_0", 55, lean_x=-2, flash=True),
        Pose("hit_1", 65, lean_x=1, flash=True),
        Pose("hit_2", 95, lean_x=0),
    ),
}


def pose_layers(layers: np.ndarray, pose: Pose) -> np.ndarray:
    result = layers.copy()
    for index in (0, 1, 5, 6, 7):
        result[index] = _transform_layer(
            result[index], dx=pose.lean_x, dy=pose.bob_y
        )
    result[2] = _transform_layer(
        result[2],
        angle=pose.left_angle,
        dx=pose.lean_x,
        dy=pose.bob_y,
        center=(12, 17),
    )
    result[3] = _transform_layer(
        result[3],
        angle=pose.right_angle,
        dx=pose.lean_x,
        dy=pose.bob_y,
        center=(20, 17),
    )
    result[4] = _transform_layer(
        result[4],
        dx=pose.lean_x,
        dy=pose.bob_y + pose.weapon_dy,
        center=(16, 12),
    )
    return _fit_margin(result)


def emission_rgba(layers: np.ndarray) -> np.ndarray:
    mask = np.maximum.reduce(layers[[5, 6, 7]])
    weapon_points = np.argwhere(layers[4] > 0)
    if weapon_points.size:
        tip_y = int(weapon_points[:, 0].min())
        weapon_tip = layers[4].copy()
        weapon_tip[tip_y + 2 :, :] = 0
        mask = np.maximum(mask, weapon_tip)
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = mask.astype(np.uint8) * 255
    return rgba


def bake_animation_atlas(
    layers: np.ndarray,
    seed: int,
    archetype: int,
    destination: Path,
    *,
    faction: str = "hostile",
) -> dict[str, Any]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    layers = _fit_margin(layers)
    max_frames = max(len(poses) for poses in ANIMATIONS.values())
    atlas = Image.new("RGBA", (max_frames * 32, len(ANIMATIONS) * 32), (0, 0, 0, 0))
    emission_atlas = Image.new(
        "RGBA", (max_frames * 32, len(ANIMATIONS) * 32), (0, 0, 0, 0)
    )
    animations: dict[str, Any] = {}

    for row, (animation_name, poses) in enumerate(ANIMATIONS.items()):
        frame_entries = []
        for column, pose in enumerate(poses):
            posed = pose_layers(layers, pose)
            rgba = compose_rgba(posed, seed, faction=faction)
            if pose.flash:
                visible = rgba[..., 3] > 0
                rgba[visible, :3] = np.maximum(rgba[visible, :3], 232)
            frame = Image.fromarray(rgba)
            glow = Image.fromarray(emission_rgba(posed))
            atlas.paste(frame, (column * 32, row * 32), frame)
            emission_atlas.paste(glow, (column * 32, row * 32), glow)
            frame_entries.append(
                {
                    "name": pose.name,
                    "rect": [column * 32, row * 32, 32, 32],
                    "duration_ms": pose.duration_ms,
                    "event": "fire" if animation_name == "attack" and column == 2 else None,
                    "sockets": derive_sockets(posed),
                }
            )
        animations[animation_name] = {
            "loop": animation_name in {"idle", "move"},
            "frames": frame_entries,
        }

    atlas.save(destination)
    emission_path = destination.with_name(destination.stem + "_emission.png")
    emission_atlas.save(emission_path)
    return {
        "seed": int(seed),
        "archetype": ARCHETYPES[archetype],
        "atlas": destination.name,
        "emission_atlas": emission_path.name,
        "frame_size": [32, 32],
        "pivot": _mask_anchor(layers[0], (16, 16)),
        "sockets": derive_sockets(layers),
        "animations": animations,
        "layers": list(LAYER_NAMES),
        "rig_version": RIG_VERSION,
        "postprocess_version": POSTPROCESS_VERSION,
    }
