from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..maps.model import Point
from ..maps.render import HAZARD_PALETTE, TERRAIN_PALETTE
from .compiler import RawTopology


def render_categorical(
    terrain: np.ndarray,
    hazard: np.ndarray,
    elevation: np.ndarray,
    *,
    start: Point,
    exit: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
    scale: int = 4,
) -> Image.Image:
    if terrain.ndim != 2 or hazard.shape != terrain.shape or elevation.shape != terrain.shape:
        raise ValueError("Categorical render fields must share one two-dimensional shape.")
    if not 2 <= scale <= 16:
        raise ValueError("Categorical render scale must be in [2, 16].")
    height, width = terrain.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for terrain_id, color in TERRAIN_PALETTE.items():
        rgb[terrain == terrain_id] = color
    walkable = np.isin(terrain, (1, 4, 5, 6, 8))
    lift = np.clip(elevation.astype(np.int16) * 5, 0, 30)[..., None]
    rgb = np.where(
        walkable[..., None], np.clip(rgb.astype(np.int16) + lift, 0, 255), rgb
    ).astype(np.uint8)
    image = Image.fromarray(rgb).resize(
        (width * scale, height * scale), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(image)
    inset = max(1, scale // 3)
    for hazard_id, color in HAZARD_PALETTE.items():
        for y, x in np.argwhere(hazard == hazard_id):
            left, top = int(x) * scale + inset, int(y) * scale + inset
            draw.rectangle(
                (left, top, (int(x) + 1) * scale - inset, (int(y) + 1) * scale - inset),
                fill=color,
            )

    def marker(point: Point, color: tuple[int, int, int], shape: str) -> None:
        x, y = point
        left, top = x * scale, y * scale
        right, bottom = left + scale - 1, top + scale - 1
        center = (left + scale // 2, top + scale // 2)
        if shape == "diamond":
            draw.polygon(
                ((center[0], top), (right, center[1]), (center[0], bottom), (left, center[1])),
                fill=color,
                outline=(255, 255, 255),
            )
        elif shape == "cross":
            draw.line((left, center[1], right, center[1]), fill=color, width=max(1, scale // 2))
            draw.line((center[0], top, center[0], bottom), fill=color, width=max(1, scale // 2))
        else:
            draw.rectangle((left, top, right, bottom), fill=color, outline=(255, 255, 255))

    for x, y in spawns:
        draw.point((x * scale + scale // 2, y * scale + scale // 2), fill=(255, 70, 92))
    for point in objectives:
        marker(point, (255, 222, 54), "diamond")
    marker(start, (50, 255, 151), "cross")
    marker(exit, (255, 48, 213), "square")
    return image


def render_edit_overlay(
    raw: RawTopology,
    compiled: object,
    *,
    start: Point,
    exit: Point,
    objectives: tuple[Point, ...],
    spawns: tuple[Point, ...],
    scale: int = 4,
) -> Image.Image:
    terrain = getattr(compiled, "terrain")
    hazard = getattr(compiled, "hazard")
    elevation = getattr(compiled, "elevation")
    image = render_categorical(
        terrain,
        hazard,
        elevation,
        start=start,
        exit=exit,
        objectives=objectives,
        spawns=spawns,
        scale=scale,
    ).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    changed = {
        "terrain": raw.terrain != terrain,
        "hazard": raw.hazard != hazard,
        "elevation": raw.elevation != elevation,
    }
    colors = {
        "terrain": (255, 45, 210, 155),
        "hazard": (35, 225, 255, 180),
        "elevation": (255, 219, 55, 155),
    }
    for name in ("terrain", "hazard", "elevation"):
        for y, x in np.argwhere(changed[name]):
            left, top = int(x) * scale, int(y) * scale
            right, bottom = left + scale - 1, top + scale - 1
            draw.rectangle((left, top, right, bottom), outline=colors[name], width=max(1, scale // 2))
    return Image.alpha_composite(image, overlay).convert("RGB")


def contact_sheet_png_bytes(
    rows: Iterable[
        tuple[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray],
            RawTopology,
            object,
            Point,
            Point,
            tuple[Point, ...],
            tuple[Point, ...],
        ]
    ],
    *,
    scale: int = 4,
) -> bytes:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Contact sheet requires at least one theme row.")
    font = ImageFont.load_default()
    title_height = 31
    label_width = 82
    gutter = 6
    rendered: list[tuple[str, list[Image.Image]]] = []
    max_tile_width = 0
    max_tile_height = 0
    for theme, source, raw, compiled, start, exit_point, objectives, spawns in materialized:
        source_image = render_categorical(
            *source,
            start=start,
            exit=exit_point,
            objectives=objectives,
            spawns=spawns,
            scale=scale,
        )
        raw_image = render_categorical(
            raw.terrain,
            raw.hazard,
            raw.elevation,
            start=start,
            exit=exit_point,
            objectives=objectives,
            spawns=spawns,
            scale=scale,
        )
        compiled_image = render_categorical(
            getattr(compiled, "terrain"),
            getattr(compiled, "hazard"),
            getattr(compiled, "elevation"),
            start=start,
            exit=exit_point,
            objectives=objectives,
            spawns=spawns,
            scale=scale,
        )
        edit_image = render_edit_overlay(
            raw,
            compiled,
            start=start,
            exit=exit_point,
            objectives=objectives,
            spawns=spawns,
            scale=scale,
        )
        images = [source_image, raw_image, compiled_image, edit_image]
        max_tile_width = max(max_tile_width, *(image.width for image in images))
        max_tile_height = max(max_tile_height, *(image.height for image in images))
        rendered.append((theme, images))
    sheet_width = label_width + 4 * max_tile_width + 5 * gutter
    sheet_height = title_height + len(rendered) * (max_tile_height + gutter) + gutter
    sheet = Image.new("RGB", (sheet_width, sheet_height), (3, 5, 14))
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(("SOURCE", "CORRUPT", "COMPILED", "EDIT OVERLAY")):
        x = label_width + gutter + column * (max_tile_width + gutter)
        draw.text((x + 2, 3), label, fill=(183, 229, 255), font=font)
    legend_x = label_width + gutter + 3 * (max_tile_width + gutter) + 2
    for offset, (label, color) in enumerate(
        (("TERRAIN", (255, 45, 210)), ("HAZARD", (35, 225, 255)), ("ELEV", (255, 219, 55)))
    ):
        x = legend_x + offset * 48
        draw.rectangle((x, 17, x + 5, 22), fill=color)
        draw.text((x + 8, 15), label, fill=(170, 188, 207), font=font)
    for row_index, (theme, images) in enumerate(rendered):
        y = title_height + gutter + row_index * (max_tile_height + gutter)
        draw.text((6, y + 5), theme.upper(), fill=(255, 100, 225), font=font)
        for column, image in enumerate(images):
            x = label_width + gutter + column * (max_tile_width + gutter)
            sheet.paste(image, (x, y))
            draw.rectangle((x - 1, y - 1, x + image.width, y + image.height), outline=(28, 75, 105))
    output = BytesIO()
    sheet.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()
