from __future__ import annotations

from dataclasses import dataclass
import colorsys
import hashlib
import json
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from .constants import (
    APPENDAGE,
    ARMOR,
    BODY,
    CANVAS_SIZE,
    CORE,
    DETAIL,
    EMISSION,
    FAMILIES,
    HEAD,
    JOINT_LAYER,
    LAYER_NAMES,
    LEFT_ARM,
    LEFT_LEG,
    MANIFEST_FORMAT,
    RENDERER_VERSION,
    RIGHT_ARM,
    RIGHT_LEG,
    SAFETY_MARGIN,
    SEMANTIC_FORMAT,
    SOCKET_LAYER,
    STRUCTURAL_LAYERS,
    WEAPON,
)
from .genome import MorphologyGenome, stream_value
from .fields import MorphologyTrainingFields, build_training_fields


Point = tuple[int, int]


@dataclass(frozen=True, slots=True)
class MorphologySpecimen:
    genome: MorphologyGenome
    layers: np.ndarray
    tokens: np.ndarray
    palette: dict[str, tuple[int, int, int, int]]
    joints: dict[str, list[int]]
    sockets: dict[str, list[int]]
    manifest: dict[str, Any]

    def training_fields(self) -> MorphologyTrainingFields:
        return build_training_fields(
            self.layers, self.genome, self.joints, self.sockets
        )


def _new_layers() -> list[Image.Image]:
    return [Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0) for _ in LAYER_NAMES]


def _draws(layers: list[Image.Image]) -> list[ImageDraw.ImageDraw]:
    return [ImageDraw.Draw(layer) for layer in layers]


def _point(x: float, y: float, pad: int = 2) -> Point:
    minimum = SAFETY_MARGIN + pad
    maximum = CANVAS_SIZE - SAFETY_MARGIN - pad - 1
    return (
        int(np.clip(round(x), minimum, maximum)),
        int(np.clip(round(y), minimum, maximum)),
    )


def _ellipse(draw: ImageDraw.ImageDraw, center: Point, rx: int, ry: int) -> None:
    x, y = center
    draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=255)


def _humanoid(genome: MorphologyGenome, draw: list[ImageDraw.ImageDraw]) -> None:
    cx = 24 + genome.x_offset
    cy = 24 + genome.y_offset + genome.posture // 2
    half_w = max(5, genome.body_width // 2)
    half_h = max(7, genome.body_height // 2 - 2)
    top, bottom = cy - half_h, cy + half_h
    shoulder_y = top + 4
    hip_y = bottom - 3
    head_y = top - genome.head_radius + 1
    head_x = cx + genome.dorsal_bias
    top_inset = 2 + genome.taper // 2

    draw[BODY].polygon(
        [
            _point(cx - half_w + top_inset, top),
            _point(cx + half_w - top_inset, top),
            _point(cx + half_w, cy + 1),
            _point(cx + half_w - 2, bottom),
            _point(cx - half_w + 2, bottom),
            _point(cx - half_w, cy + 1),
        ],
        fill=255,
    )
    _ellipse(draw[HEAD], _point(head_x, head_y), genome.head_radius, genome.head_radius)
    left_shoulder = _point(cx - half_w + 1, shoulder_y)
    right_shoulder = _point(cx + half_w - 1, shoulder_y)
    left_hand = _point(cx - half_w - genome.limb_length, shoulder_y + genome.limb_length // 2)
    right_hand = _point(
        cx + half_w + genome.limb_length,
        shoulder_y + genome.limb_length // 2 + genome.asymmetry,
    )
    draw[LEFT_ARM].line(
        [left_shoulder, _point(cx - half_w - genome.limb_length // 2, cy), left_hand],
        fill=255,
        width=genome.limb_thickness,
        joint="curve",
    )
    draw[RIGHT_ARM].line(
        [right_shoulder, _point(cx + half_w + genome.limb_length // 2, cy), right_hand],
        fill=255,
        width=genome.limb_thickness,
        joint="curve",
    )
    left_hip = _point(cx - max(2, half_w // 2), hip_y)
    right_hip = _point(cx + max(2, half_w // 2), hip_y)
    left_foot = _point(cx - genome.stance_width, bottom + genome.limb_length)
    right_foot = _point(cx + genome.stance_width, bottom + genome.limb_length)
    draw[LEFT_LEG].line([left_hip, _point(cx - 3, bottom + 3), left_foot], fill=255, width=genome.limb_thickness)
    draw[RIGHT_LEG].line([right_hip, _point(cx + 3, bottom + 3), right_foot], fill=255, width=genome.limb_thickness)
    draw[APPENDAGE].line(
        [_point(cx - half_w + 1, top + 2), _point(cx, cy + 2), _point(cx, bottom + genome.appendage_length // 2)],
        fill=255,
        width=max(2, genome.limb_thickness - 1),
    )
    draw[WEAPON].line(
        [right_hand, _point(right_hand[0] + genome.weapon_length // 3, right_hand[1] - genome.weapon_length // 2), _point(right_hand[0], right_hand[1] - genome.weapon_length)],
        fill=255,
        width=max(2, genome.limb_thickness - 1),
    )
    draw[ARMOR].polygon(
        [_point(cx - half_w + 2, top + 2), _point(cx + half_w - 2, top + 2), _point(cx, cy + 5)],
        fill=255,
    )
    _ellipse(draw[CORE], _point(cx, cy), genome.core_radius, genome.core_radius)


def _animalian(genome: MorphologyGenome, draw: list[ImageDraw.ImageDraw]) -> None:
    cx = 24 + genome.x_offset
    cy = 24 + genome.y_offset
    half_w = max(5, genome.body_width // 2)
    half_h = max(8, genome.body_height // 2)
    top, bottom = cy - half_h, cy + half_h
    draw[BODY].polygon(
        [
            _point(cx, top),
            _point(cx + half_w - genome.taper // 2, top + 5),
            _point(cx + half_w - 1, bottom - 4),
            _point(cx, bottom),
            _point(cx - half_w + 1, bottom - 4),
            _point(cx - half_w + genome.taper // 2, top + 5),
        ],
        fill=255,
    )
    head_y = top + 1
    head_x = cx + genome.dorsal_bias
    _ellipse(draw[HEAD], _point(head_x, head_y), genome.head_radius + 1, genome.head_radius)
    fore_y = top + 6
    rear_y = bottom - 5
    left_fore = _point(cx - half_w + 1, fore_y)
    right_fore = _point(cx + half_w - 1, fore_y)
    left_hind = _point(cx - half_w + 1, rear_y)
    right_hind = _point(cx + half_w - 1, rear_y)
    draw[LEFT_ARM].line([left_fore, _point(cx - half_w - genome.limb_length, fore_y - 1)], fill=255, width=genome.limb_thickness)
    draw[RIGHT_ARM].line([right_fore, _point(cx + half_w + genome.limb_length, fore_y - 1 + genome.asymmetry)], fill=255, width=genome.limb_thickness)
    draw[LEFT_LEG].line([left_hind, _point(cx - half_w - genome.stance_width, rear_y + genome.limb_length // 2)], fill=255, width=genome.limb_thickness)
    draw[RIGHT_LEG].line([right_hind, _point(cx + half_w + genome.stance_width, rear_y + genome.limb_length // 2)], fill=255, width=genome.limb_thickness)
    tail_bend = -1 if genome.silhouette_variant % 2 == 0 else 1
    draw[APPENDAGE].line(
        [_point(cx, bottom - 1), _point(cx + tail_bend * genome.appendage_length // 2, bottom + 4), _point(cx + tail_bend * genome.appendage_length, bottom + genome.appendage_length // 2)],
        fill=255,
        width=max(2, genome.limb_thickness - 1),
    )
    draw[WEAPON].polygon(
        [_point(head_x - 2, head_y - genome.head_radius + 1), _point(head_x, head_y - genome.head_radius - genome.weapon_length), _point(head_x + 2, head_y - genome.head_radius + 1)],
        fill=255,
    )
    draw[ARMOR].line(
        [_point(cx, top + 3), _point(cx, bottom - 3)],
        fill=255,
        width=genome.armor_depth + 1,
    )
    _ellipse(draw[CORE], _point(cx, cy - 1), genome.core_radius, genome.core_radius)


def _plantlike(genome: MorphologyGenome, draw: list[ImageDraw.ImageDraw]) -> None:
    cx = 24 + genome.x_offset
    cy = 25 + genome.y_offset
    trunk_half = max(3, genome.body_width // 4)
    trunk_top_half = max(2, trunk_half - genome.taper // 2)
    half_h = max(8, genome.body_height // 2)
    top, bottom = cy - half_h, cy + half_h
    draw[BODY].polygon(
        [_point(cx - trunk_top_half, top), _point(cx + trunk_top_half, top), _point(cx + trunk_half + 1, bottom), _point(cx - trunk_half - 1, bottom)],
        fill=255,
    )
    crown_radius = genome.head_radius + 2
    crown_x = cx + genome.dorsal_bias
    _ellipse(draw[HEAD], _point(crown_x, top + 1), crown_radius, crown_radius)
    branch_y = top + 5
    left_base = _point(cx - trunk_half + 1, branch_y)
    right_base = _point(cx + trunk_half - 1, branch_y)
    draw[LEFT_ARM].line(
        [left_base, _point(cx - genome.limb_length // 2, branch_y - 4), _point(cx - genome.limb_length - crown_radius // 2, branch_y - 1)],
        fill=255,
        width=genome.limb_thickness,
    )
    draw[RIGHT_ARM].line(
        [right_base, _point(cx + genome.limb_length // 2, branch_y - 4), _point(cx + genome.limb_length + crown_radius // 2, branch_y - 1 + genome.asymmetry)],
        fill=255,
        width=genome.limb_thickness,
    )
    root_y = bottom - 1
    draw[LEFT_LEG].line([_point(cx - 2, root_y), _point(cx - genome.stance_width, bottom + genome.limb_length // 2)], fill=255, width=genome.limb_thickness)
    draw[RIGHT_LEG].line([_point(cx + 2, root_y), _point(cx + genome.stance_width, bottom + genome.limb_length // 2)], fill=255, width=genome.limb_thickness)
    vine_side = -1 if genome.silhouette_variant < 2 else 1
    draw[APPENDAGE].line(
        [_point(cx + vine_side * trunk_half, cy), _point(cx + vine_side * (trunk_half + genome.appendage_length), cy + 2), _point(cx + vine_side * (trunk_half + genome.appendage_length // 2), bottom)],
        fill=255,
        width=2,
    )
    draw[WEAPON].line(
        [_point(crown_x, top), _point(crown_x + genome.asymmetry, top - genome.weapon_length)],
        fill=255,
        width=max(2, genome.limb_thickness - 1),
    )
    for offset in range(-half_h + 4, half_h - 2, max(3, genome.armor_depth + 2)):
        draw[ARMOR].line([_point(cx - trunk_half, cy + offset), _point(cx + trunk_half, cy + offset)], fill=255, width=genome.armor_depth)
    _ellipse(draw[CORE], _point(cx, cy), genome.core_radius, genome.core_radius + 1)


def _anomaly(genome: MorphologyGenome, draw: list[ImageDraw.ImageDraw]) -> None:
    cx = 24 + genome.x_offset
    cy = 24 + genome.y_offset
    half_w = max(6, genome.body_width // 2)
    half_h = max(6, genome.body_height // 2 - 2)
    wobble = genome.silhouette_variant + 1 + genome.taper // 2
    draw[BODY].polygon(
        [
            _point(cx - half_w, cy - 2),
            _point(cx - half_w + wobble, cy - half_h),
            _point(cx + 2, cy - half_h + 1),
            _point(cx + half_w, cy - 4),
            _point(cx + half_w - 1, cy + half_h),
            _point(cx - 2, cy + half_h - wobble),
        ],
        fill=255,
    )
    focus_x = cx + genome.dorsal_bias
    _ellipse(draw[HEAD], _point(focus_x, cy - 2), genome.head_radius + 1, genome.head_radius)
    left_base = _point(cx - half_w + 1, cy - 3)
    right_base = _point(cx + half_w - 1, cy - 3)
    draw[LEFT_ARM].line([left_base, _point(cx - half_w - genome.limb_length, cy - 7), _point(cx - half_w - genome.limb_length // 2, cy + 1)], fill=255, width=genome.limb_thickness)
    draw[RIGHT_ARM].line([right_base, _point(cx + half_w + genome.limb_length, cy - 6 + genome.asymmetry), _point(cx + half_w + genome.limb_length // 2, cy + 2)], fill=255, width=genome.limb_thickness)
    left_hip = _point(cx, cy + half_h - 3)
    right_hip = _point(cx + 1, cy + half_h - 3)
    draw[LEFT_LEG].line([left_hip, _point(cx - genome.stance_width, cy + half_h + genome.limb_length)], fill=255, width=genome.limb_thickness)
    draw[RIGHT_LEG].line([right_hip, _point(cx + genome.stance_width, cy + half_h + genome.limb_length)], fill=255, width=genome.limb_thickness)
    draw[APPENDAGE].line(
        [_point(cx, cy - half_h + 1), _point(cx - genome.appendage_length // 2, cy - half_h - 4), _point(cx + genome.appendage_length // 2, cy - half_h - genome.appendage_length)],
        fill=255,
        width=2,
    )
    draw[WEAPON].line([_point(focus_x, cy - 2), _point(focus_x, cy - half_h - genome.weapon_length)], fill=255, width=max(2, genome.limb_thickness - 1))
    draw[ARMOR].polygon([_point(cx - half_w + 2, cy + 1), _point(cx, cy - half_h + 2), _point(cx + half_w - 2, cy + 2), _point(cx, cy + half_h - 2)], fill=255)
    _ellipse(draw[CORE], _point(focus_x, cy - 2), genome.core_radius, genome.core_radius)


def _machine(genome: MorphologyGenome, draw: list[ImageDraw.ImageDraw]) -> None:
    cx = 24 + genome.x_offset
    cy = 24 + genome.y_offset
    half_w = max(6, genome.body_width // 2)
    half_h = max(7, genome.body_height // 2 - 1)
    top, bottom = cy - half_h, cy + half_h
    # Keep the top edge wide enough for the largest turret to remain attached.
    chamfer = min(
        3 + genome.silhouette_variant % 2 + genome.taper // 2,
        max(3, half_w - genome.head_radius),
    )
    draw[BODY].polygon(
        [
            _point(cx - half_w + chamfer, top),
            _point(cx + half_w - chamfer, top),
            _point(cx + half_w, top + chamfer),
            _point(cx + half_w, bottom - chamfer),
            _point(cx + half_w - chamfer, bottom),
            _point(cx - half_w + chamfer, bottom),
            _point(cx - half_w, bottom - chamfer),
            _point(cx - half_w, top + chamfer),
        ],
        fill=255,
    )
    head_half = genome.head_radius
    turret_x = cx + genome.dorsal_bias
    draw[HEAD].rounded_rectangle([turret_x - head_half, top - head_half, turret_x + head_half, top + 2], radius=2, fill=255)
    left_mount = _point(cx - half_w + 1, cy - 2)
    right_mount = _point(cx + half_w - 1, cy - 2)
    draw[LEFT_ARM].line(
        [left_mount, _point(cx - half_w - genome.limb_length, cy + 2)],
        fill=255,
        width=genome.limb_thickness + 2,
    )
    draw[RIGHT_ARM].line(
        [right_mount, _point(cx + half_w + genome.limb_length, cy + 2)],
        fill=255,
        width=genome.limb_thickness + 2,
    )
    tread_width = max(3, genome.limb_thickness + 1)
    left_tread_top = _point(cx - half_w + 1, bottom - 3)
    left_tread_bottom = _point(cx - half_w + 1, bottom + genome.limb_length)
    right_tread_top = _point(cx + half_w - 1, bottom - 3)
    right_tread_bottom = _point(cx + half_w - 1, bottom + genome.limb_length)
    draw[LEFT_LEG].line(
        [left_tread_top, left_tread_bottom],
        fill=255,
        width=tread_width + 1,
    )
    draw[RIGHT_LEG].line(
        [right_tread_top, right_tread_bottom],
        fill=255,
        width=tread_width + 1,
    )
    antenna_x = cx - 3 if genome.silhouette_variant < 2 else cx + 3
    draw[APPENDAGE].line([_point(antenna_x, top + 1), _point(antenna_x + genome.asymmetry, top - genome.appendage_length)], fill=255, width=2)
    draw[WEAPON].rectangle([turret_x - max(1, genome.limb_thickness // 2), top - genome.weapon_length, turret_x + max(1, genome.limb_thickness // 2), top + 1], fill=255)
    for offset in (-half_w // 2, half_w // 2):
        draw[ARMOR].rectangle([cx + offset - genome.armor_depth, top + 4, cx + offset + genome.armor_depth, bottom - 4], fill=255)
    _ellipse(draw[CORE], _point(cx, cy), genome.core_radius, genome.core_radius)


FAMILY_DRAWERS: tuple[Callable[[MorphologyGenome, list[ImageDraw.ImageDraw]], None], ...] = (
    _humanoid,
    _animalian,
    _plantlike,
    _anomaly,
    _machine,
)


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    return np.logical_or.reduce(
        [
            padded[y : y + mask.shape[0], x : x + mask.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
    )


def _nearest_pixel(mask: np.ndarray, target: Point) -> Point:
    points = np.argwhere(mask > 0)
    if not len(points):
        raise ValueError("Cannot derive an anchor from an empty semantic mask")
    tx, ty = target
    distances = (points[:, 1] - tx) ** 2 + (points[:, 0] - ty) ** 2
    index = int(np.argmin(distances))
    return int(points[index, 1]), int(points[index, 0])


def _centroid(mask: np.ndarray) -> Point:
    points = np.argwhere(mask > 0)
    if not len(points):
        raise ValueError("Cannot derive a centroid from an empty semantic mask")
    return int(round(float(points[:, 1].mean()))), int(round(float(points[:, 0].mean())))


def _attachment(part: np.ndarray, anchor: np.ndarray, target: Point) -> Point:
    candidates = part.astype(bool) & _dilate(anchor)
    return _nearest_pixel(candidates if candidates.any() else part, target)


def _farthest_pixel(mask: np.ndarray, origin: Point) -> Point:
    points = np.argwhere(mask > 0)
    if not len(points):
        raise ValueError("Cannot derive a socket from an empty semantic mask")
    ox, oy = origin
    distances = (points[:, 1] - ox) ** 2 + (points[:, 0] - oy) ** 2
    index = int(np.argmax(distances))
    return int(points[index, 1]), int(points[index, 0])


def _derive_anchors(layers: np.ndarray) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    body = layers[BODY] > 0
    head_anchor = body
    appendage_anchor = np.logical_or(body, layers[HEAD] > 0)
    weapon_anchor = np.logical_or.reduce((body, layers[HEAD] > 0, layers[RIGHT_ARM] > 0))
    body_center = _centroid(body)
    joints_points: dict[str, Point] = {
        "root": _nearest_pixel(body, body_center),
        "head": _attachment(layers[HEAD], head_anchor, (body_center[0], body_center[1] - 8)),
        "left_shoulder": _attachment(layers[LEFT_ARM], body, (body_center[0] - 6, body_center[1] - 5)),
        "right_shoulder": _attachment(layers[RIGHT_ARM], body, (body_center[0] + 6, body_center[1] - 5)),
        "left_hip": _attachment(layers[LEFT_LEG], body, (body_center[0] - 4, body_center[1] + 7)),
        "right_hip": _attachment(layers[RIGHT_LEG], body, (body_center[0] + 4, body_center[1] + 7)),
        "appendage_base": _attachment(layers[APPENDAGE], appendage_anchor, body_center),
        "weapon_mount": _attachment(layers[WEAPON], weapon_anchor, (body_center[0], body_center[1] - 8)),
    }
    sockets_points = {
        "focus": _nearest_pixel(layers[CORE], _centroid(layers[CORE])),
        "muzzle": _farthest_pixel(layers[WEAPON], joints_points["weapon_mount"]),
        "left_hand": _farthest_pixel(layers[LEFT_ARM], joints_points["left_shoulder"]),
        "right_hand": _farthest_pixel(layers[RIGHT_ARM], joints_points["right_shoulder"]),
        "left_foot": _farthest_pixel(layers[LEFT_LEG], joints_points["left_hip"]),
        "right_foot": _farthest_pixel(layers[RIGHT_LEG], joints_points["right_hip"]),
        "appendage_tip": _farthest_pixel(layers[APPENDAGE], joints_points["appendage_base"]),
    }
    joints = {name: [point[0], point[1]] for name, point in joints_points.items()}
    sockets = {name: [point[0], point[1]] for name, point in sockets_points.items()}
    return joints, sockets


def _add_surface_semantics(layers: np.ndarray, genome: MorphologyGenome) -> None:
    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    points = np.argwhere(structural)
    if not len(points):
        raise ValueError("Morphology renderer produced an empty structural mask")
    for index in range(genome.detail_count):
        point_index = (
            stream_value(genome.detail_seed, index)
            + index * genome.segmentation * 17
        ) % len(points)
        y, x = map(int, points[point_index])
        layers[DETAIL, y, x] = 1
        if index % 3 == 0:
            layers[EMISSION, y, x] = 1
    layers[DETAIL] |= layers[CORE]
    layers[EMISSION] |= layers[CORE]
    weapon_points = np.argwhere(layers[WEAPON] > 0)
    if len(weapon_points):
        reference = _centroid(layers[BODY])
        tip = _farthest_pixel(layers[WEAPON], reference)
        layers[EMISSION, tip[1], tip[0]] = 1
    layers[DETAIL] &= structural.astype(np.uint8)
    layers[EMISSION] &= structural.astype(np.uint8)


def _role_condition_semantics(
    layers: np.ndarray, genome: MorphologyGenome
) -> None:
    """Make combat-role conditioning observable without changing rig topology.

    Role is deliberately orthogonal to family: the same eight equipment/readout
    motifs apply to humanoids, beasts, plants, anomalies, and machines.  All
    writes are subsets of existing structural anatomy, so attachment, margins,
    and motion ownership remain invariant while target fields differ strongly
    enough for a conditional model to learn.
    """
    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    yy, xx = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
    points = np.argwhere(structural)
    if not len(points):
        raise ValueError("Cannot condition an empty morphology on combat role")
    center_y, center_x = points.mean(axis=0)
    x_band = np.abs(xx - center_x)
    y_band = np.abs(yy - center_y)
    parity_a = ((xx + 2 * yy + genome.topology_seed) % 5) == 0
    parity_b = ((2 * xx + yy + genome.detail_seed) % 7) == 0
    armor = layers[ARMOR] > 0
    weapon = layers[WEAPON] > 0
    core = layers[CORE] > 0
    head = layers[HEAD] > 0
    limbs = np.logical_or.reduce(
        layers[[LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG]] > 0
    )
    appendage = layers[APPENDAGE] > 0

    role = genome.role_id
    if role == 0:  # striker: hot weapon edge and forward limbs
        mark = (weapon | limbs) & ((yy <= center_y) | parity_a)
        glow = weapon | (limbs & parity_b)
    elif role == 1:  # defender: broad armored shell with restrained glow
        mark = armor | (structural & (x_band <= max(2, genome.armor_depth)))
        glow = core | (armor & parity_b)
    elif role == 2:  # scout: sparse extremity/sensor lights
        extremity = (x_band + y_band) >= np.percentile(
            (x_band + y_band)[structural], 64
        )
        mark = structural & extremity & (parity_a | head)
        glow = mark | (appendage & parity_b)
    elif role == 3:  # controller: cross-field circuitry
        cross = (x_band <= 1.5) | (y_band <= 1.5)
        mark = structural & (cross | parity_b)
        glow = core | (mark & parity_a)
    elif role == 4:  # support: paired side channels and bright core
        rails = (x_band >= max(2.0, genome.core_radius)) & (
            x_band <= max(4.0, genome.body_width / 2.0)
        )
        mark = structural & (rails | core)
        glow = core | (mark & parity_b)
    elif role == 5:  # artillery: weapon, dorsal axis, range pips
        dorsal = structural & (yy < center_y) & (x_band <= 2.0)
        mark = weapon | dorsal | (structural & parity_a)
        glow = weapon | dorsal
    elif role == 6:  # harvester: low-body/root extraction rake
        lower = structural & (yy >= center_y)
        mark = (lower & (parity_a | limbs | appendage)) | core
        glow = core | (lower & parity_b)
    else:  # disruptor: intentionally asymmetric interference bands
        diagonal = np.abs((xx - center_x) - (yy - center_y)) <= 1.5
        mark = structural & (diagonal | parity_a | appendage)
        glow = core | (mark & ~parity_b)

    layers[DETAIL] |= mark.astype(np.uint8)
    layers[EMISSION] |= glow.astype(np.uint8)
    layers[DETAIL] &= structural.astype(np.uint8)
    layers[EMISSION] &= structural.astype(np.uint8)


def _expand_appendages(layers: np.ndarray, genome: MorphologyGenome) -> None:
    """Use explicit topology genes to add connected secondary appendage branches."""
    image = Image.fromarray((layers[APPENDAGE] * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    points = np.argwhere(layers[APPENDAGE] > 0)
    directions = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
    for branch in range(1, genome.appendage_count):
        point_index = stream_value(genome.topology_seed, branch) % len(points)
        y, x = map(int, points[point_index])
        direction = directions[
            stream_value(genome.topology_seed, 32 + branch) % len(directions)
        ]
        length = 2 + (stream_value(genome.topology_seed, 64 + branch) % (3 + genome.taper))
        end_x = int(np.clip(x + direction[0] * length, SAFETY_MARGIN, CANVAS_SIZE - SAFETY_MARGIN - 1))
        end_y = int(np.clip(y + direction[1] * length, SAFETY_MARGIN, CANVAS_SIZE - SAFETY_MARGIN - 1))
        draw.line([(x, y), (end_x, end_y)], fill=255, width=1)
    layers[APPENDAGE] = (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def layers_to_tokens(layers: np.ndarray) -> np.ndarray:
    expected = (len(LAYER_NAMES), CANVAS_SIZE, CANVAS_SIZE)
    if layers.shape != expected:
        raise ValueError(f"Expected semantic layers {expected}, got {layers.shape}")
    tokens = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    for index in range(len(LAYER_NAMES)):
        tokens[layers[index] > 0] = index + 1
    return tokens


def palette_for_genome(genome: MorphologyGenome) -> dict[str, tuple[int, int, int, int]]:
    family_hues = (0.56, 0.08, 0.31, 0.82, 0.52)
    hue = (family_hues[genome.family] + genome.palette_id * 0.061 + genome.palette_shift * 0.013) % 1.0

    def rgba(offset: float, saturation: float, value: float) -> tuple[int, int, int, int]:
        red, green, blue = colorsys.hsv_to_rgb((hue + offset) % 1.0, saturation, value)
        return int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)), 255

    primary = rgba(0.0, 0.72, 0.92)
    return {
        "outline": (4, 6, 13, 255),
        "shadow": tuple(max(0, channel // 4) for channel in primary[:3]) + (255,),
        "primary": primary,
        "secondary": rgba(0.10 + genome.palette_shift * 0.006, 0.76, 1.0),
        "organic": rgba(-0.09, 0.58, 0.78),
        "hot": (243, 252, 255, 255),
    }


def _binary_dilate(mask: np.ndarray) -> np.ndarray:
    return _dilate(mask).astype(bool)


def compose_rgba(specimen_or_layers: MorphologySpecimen | np.ndarray, palette: dict[str, tuple[int, int, int, int]] | None = None) -> np.ndarray:
    if isinstance(specimen_or_layers, MorphologySpecimen):
        layers = specimen_or_layers.layers
        palette = specimen_or_layers.palette
    else:
        layers = specimen_or_layers
    if palette is None:
        raise ValueError("A palette is required when composing raw layers")
    major = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    outline = _binary_dilate(major) & ~major
    rgba = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)
    rgba[outline] = palette["outline"]
    rgba[layers[BODY] > 0] = palette["shadow"]
    rgba[(layers[LEFT_ARM] | layers[RIGHT_ARM] | layers[LEFT_LEG] | layers[RIGHT_LEG]) > 0] = palette["organic"]
    rgba[(layers[HEAD] | layers[APPENDAGE]) > 0] = palette["secondary"]
    rgba[(layers[ARMOR] | layers[WEAPON]) > 0] = palette["primary"]
    rgba[layers[CORE] > 0] = palette["hot"]
    rgba[layers[DETAIL] > 0] = palette["secondary"]
    rgba[layers[EMISSION] > 0] = palette["hot"]
    rgba[..., 3] = np.where(outline | major | (layers[DETAIL] > 0) | (layers[EMISSION] > 0), 255, 0).astype(np.uint8)
    return rgba


def _component_count(mask: np.ndarray) -> int:
    active = mask.astype(bool)
    seen = np.zeros_like(active)
    count = 0
    height, width = active.shape
    for y in range(height):
        for x in range(width):
            if not active[y, x] or seen[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                py, px = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < height and 0 <= nx < width and active[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return count


def _manifest(
    genome: MorphologyGenome,
    layers: np.ndarray,
    tokens: np.ndarray,
    palette: dict[str, tuple[int, int, int, int]],
    joints: dict[str, list[int]],
    sockets: dict[str, list[int]],
) -> dict[str, Any]:
    structural = np.logical_or.reduce(layers[list(STRUCTURAL_LAYERS)] > 0)
    points = np.argwhere(np.logical_or.reduce(layers > 0))
    min_y, min_x = points.min(axis=0)
    max_y, max_x = points.max(axis=0)
    semantic_hash = hashlib.sha256(layers.tobytes() + tokens.tobytes()).hexdigest()
    genome_hash = hashlib.sha256(genome.canonical_json().encode("utf-8")).hexdigest()
    training = build_training_fields(layers, genome, joints, sockets)
    return {
        "format": MANIFEST_FORMAT,
        "id": f"{genome.family_name}_{genome.seed:08x}",
        "seed": genome.seed,
        "family": genome.family_name,
        "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
        "safety_margin": SAFETY_MARGIN,
        "genome_version": genome.version,
        "renderer_version": RENDERER_VERSION,
        "semantic": {
            "format": SEMANTIC_FORMAT,
            "background_token": 0,
            "token_count": len(LAYER_NAMES) + 1,
            "layer_names": list(LAYER_NAMES),
            "layers_shape": list(layers.shape),
            "tokens_shape": list(tokens.shape),
        },
        "genome": genome.to_dict(),
        "condition_vector": [round(float(value), 7) for value in genome.condition_vector()],
        "training_contract": training.metadata(),
        "palette": {name: list(value) for name, value in palette.items()},
        "bounds": [int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)],
        "topology": {
            "structural_components": _component_count(structural),
            "structural_pixels": int(structural.sum()),
            "margin_clear": True,
        },
        "joints": joints,
        "sockets": sockets,
        "hashes": {
            "genome_sha256": genome_hash,
            "semantic_sha256": semantic_hash,
            "training_arrays_sha256": training.arrays_hash(),
        },
    }


def render_specimen(genome: MorphologyGenome) -> MorphologySpecimen:
    genome.validate()
    images = _new_layers()
    draw = _draws(images)
    FAMILY_DRAWERS[genome.family](genome, draw)
    layers = np.stack(
        [(np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8) for image in images],
        axis=0,
    )
    # Keep room for a one-pixel outline plus two genuinely clear logical cells.
    layers[:, :SAFETY_MARGIN, :] = 0
    layers[:, -SAFETY_MARGIN:, :] = 0
    layers[:, :, :SAFETY_MARGIN] = 0
    layers[:, :, -SAFETY_MARGIN:] = 0
    _expand_appendages(layers, genome)
    _add_surface_semantics(layers, genome)
    _role_condition_semantics(layers, genome)
    tokens = layers_to_tokens(layers)
    palette = palette_for_genome(genome)
    joints, sockets = _derive_anchors(layers)
    manifest = _manifest(genome, layers, tokens, palette, joints, sockets)
    return MorphologySpecimen(
        genome=genome,
        layers=layers,
        tokens=tokens,
        palette=palette,
        joints=joints,
        sockets=sockets,
        manifest=manifest,
    )


def canonical_manifest_json(specimen: MorphologySpecimen) -> str:
    return json.dumps(specimen.manifest, sort_keys=True, separators=(",", ":"))
