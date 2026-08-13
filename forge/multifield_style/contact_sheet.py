from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .model import IMAGE_SIZE, RenderedLayers, StyleCondition
from .model import BackgroundCrop


LAYER_ORDER = (
    ("base", "BASE"),
    ("outline", "OUTLINE 1PX"),
    ("emission_core", "EMISSION CORE"),
    ("aura", "AURA"),
    ("bloom_r1", "BLOOM R1"),
    ("bloom_r2", "BLOOM R2"),
    ("composite", "COMPOSITE"),
)


def _checkerboard() -> Image.Image:
    yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE))
    check = ((xx // 4 + yy // 4) & 1).astype(np.uint8)
    rgb = np.where(check[..., None] == 0, np.array([9, 12, 24]), np.array([18, 23, 38]))
    return Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")


def build_layer_contact_sheet(
    conditions: Sequence[StyleCondition],
    rendered: Sequence[RenderedLayers],
    *,
    title: str,
    scale: int = 3,
) -> np.ndarray:
    if not conditions or len(conditions) != len(rendered):
        raise ValueError("Layer sheet needs matched non-empty conditions and renders")
    tile = IMAGE_SIZE * scale
    header = 34
    label = 34
    row_height = tile + label
    width = len(LAYER_ORDER) * tile
    height = header + len(rendered) * row_height
    sheet = Image.new("RGBA", (width, height), (3, 5, 13, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title, fill=(226, 246, 255, 255))
    draw.text((6, 18), "native layers 48px · preview scaling NEAREST", fill=(91, 202, 235, 255))
    checker = _checkerboard()
    for row, (condition, layers) in enumerate(zip(conditions, rendered)):
        row_y = header + row * row_height
        for column, (attribute, display_name) in enumerate(LAYER_ORDER):
            x = column * tile
            layer = Image.fromarray(getattr(layers, attribute))
            tile_image = checker.copy()
            tile_image.alpha_composite(layer)
            tile_image = tile_image.resize((tile, tile), Image.Resampling.NEAREST)
            sheet.alpha_composite(tile_image, (x, row_y))
            draw.rectangle((x, row_y + tile, x + tile - 1, row_y + row_height - 1), fill=(3, 5, 13, 255))
            draw.text((x + 3, row_y + tile + 2), display_name, fill=(226, 246, 255, 255))
            draw.text(
                (x + 3, row_y + tile + 16),
                f"{condition.morphology_name} · {condition.role_name}",
                fill=(102, 218, 242, 255),
            )
    return np.asarray(sheet, dtype=np.uint8).copy()


def build_hero_contact_sheet(
    conditions: Sequence[StyleCondition],
    rendered: Sequence[RenderedLayers],
    *,
    title: str,
    scale: int = 5,
) -> np.ndarray:
    if not conditions or len(conditions) != len(rendered):
        raise ValueError("Hero sheet needs matched non-empty conditions and renders")
    tile = IMAGE_SIZE * scale
    header = 38
    label = 38
    sheet = Image.new("RGBA", (len(rendered) * tile, header + tile + label), (3, 5, 13, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), title, fill=(232, 249, 255, 255))
    draw.text((8, 20), "categorical geometry unchanged · derived presentation only", fill=(94, 213, 238, 255))
    checker = _checkerboard()
    for index, (condition, layers) in enumerate(zip(conditions, rendered)):
        x = index * tile
        sprite = checker.copy()
        sprite.alpha_composite(Image.fromarray(layers.composite))
        sprite = sprite.resize((tile, tile), Image.Resampling.NEAREST)
        sheet.alpha_composite(sprite, (x, header))
        draw.rectangle((x, header + tile, x + tile - 1, header + tile + label - 1), fill=(3, 5, 13, 255))
        draw.text((x + 5, header + tile + 4), condition.morphology_name.upper(), fill=(232, 249, 255, 255))
        draw.text((x + 5, header + tile + 19), condition.role_name, fill=(102, 218, 242, 255))
    return np.asarray(sheet, dtype=np.uint8).copy()


def build_map_context_contact_sheet(
    conditions: Sequence[StyleCondition],
    rendered: Sequence[RenderedLayers],
    backgrounds: Sequence[BackgroundCrop],
    *,
    title: str,
    scale: int = 3,
) -> np.ndarray:
    if not conditions or len(conditions) != len(rendered):
        raise ValueError("Map-context sheet needs matched non-empty conditions and renders")
    selected: list[BackgroundCrop] = []
    for crop in backgrounds:
        if crop.theme not in {item.theme for item in selected}:
            selected.append(crop)
    if len(selected) != 6:
        raise ValueError("Map-context sheet requires all six canonical map themes")
    tile = IMAGE_SIZE * scale
    header = 36
    label = 22
    row_height = tile + label
    sheet = Image.new(
        "RGBA",
        (len(selected) * tile, header + len(rendered) * row_height),
        (3, 5, 13, 255),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title, fill=(232, 249, 255, 255))
    draw.text((6, 19), "deterministic representative map-art crops", fill=(94, 213, 238, 255))
    for row, (condition, layers) in enumerate(zip(conditions, rendered)):
        y = header + row * row_height
        for column, crop in enumerate(selected):
            x = column * tile
            context = Image.fromarray(crop.rgb).convert("RGBA")
            context.alpha_composite(Image.fromarray(layers.composite))
            context = context.resize((tile, tile), Image.Resampling.NEAREST)
            sheet.alpha_composite(context, (x, y))
            draw.rectangle((x, y + tile, x + tile - 1, y + row_height - 1), fill=(3, 5, 13, 255))
            draw.text(
                (x + 3, y + tile + 3),
                f"{crop.theme} · {condition.morphology_name}",
                fill=(212, 242, 250, 255),
            )
    return np.asarray(sheet, dtype=np.uint8).copy()
