from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from .model import Hazard, MapData, Terrain


TERRAIN_PALETTE: dict[int, tuple[int, int, int]] = {
    int(Terrain.VOID): (2, 2, 8),
    int(Terrain.FLOOR): (9, 15, 31),
    int(Terrain.WALL): (20, 12, 39),
    int(Terrain.WATER): (7, 18, 47),
    int(Terrain.BRIDGE): (91, 48, 29),
    int(Terrain.GROWTH): (14, 45, 35),
    int(Terrain.CRYSTAL): (41, 22, 63),
    int(Terrain.CHASM): (5, 1, 13),
    int(Terrain.SAND): (58, 49, 34),
}

HAZARD_PALETTE: dict[int, tuple[int, int, int]] = {
    int(Hazard.LASER): (255, 35, 164),
    int(Hazard.LAVA): (255, 87, 25),
    int(Hazard.SPORES): (95, 255, 110),
    int(Hazard.ARC): (53, 220, 255),
}


def render_preview(data: MapData, scale: int = 5) -> Image.Image:
    if scale < 2:
        raise ValueError("Preview scale must be at least 2 for semantic overlays.")
    height, width = data.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for terrain_id, color in TERRAIN_PALETTE.items():
        rgb[data.terrain == terrain_id] = color
    lift = np.clip(data.elevation.astype(np.int16) * 4, 0, 24)[..., None]
    rgb = np.where(data.walkability[..., None] > 0, np.clip(rgb.astype(np.int16) + lift, 0, 255), rgb).astype(np.uint8)

    # A subtle zone hatch exposes navigation partitioning without obscuring terrain.
    yy, xx = np.mgrid[0:height, 0:width]
    hatch = (data.walkability > 0) & (((xx + yy + np.maximum(data.zone, 0)) % 7) == 0)
    rgb[hatch] = np.clip(rgb[hatch].astype(np.int16) + np.array([4, 7, 12]), 0, 255).astype(np.uint8)

    base = Image.fromarray(rgb)
    image = base.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    # Cyan/magenta wall edges give the preview a crisp laser-cut silhouette.
    walk = data.walkability.astype(bool)
    for y in range(height):
        for x in range(width):
            if walk[y, x]:
                continue
            left, top = x * scale, y * scale
            right, bottom = left + scale - 1, top + scale - 1
            if y + 1 < height and walk[y + 1, x]:
                draw.line((left, bottom, right, bottom), fill=(36, 211, 255))
            if y > 0 and walk[y - 1, x]:
                draw.line((left, top, right, top), fill=(36, 211, 255))
            if x + 1 < width and walk[y, x + 1]:
                draw.line((right, top, right, bottom), fill=(204, 45, 255))
            if x > 0 and walk[y, x - 1]:
                draw.line((left, top, left, bottom), fill=(204, 45, 255))

    inset = max(1, scale // 3)
    for hazard_id, color in HAZARD_PALETTE.items():
        for y, x in np.argwhere(data.hazard == hazard_id):
            left, top = int(x) * scale + inset, int(y) * scale + inset
            draw.rectangle((left, top, (int(x) + 1) * scale - inset, (int(y) + 1) * scale - inset), fill=color)

    def marker(point: tuple[int, int], color: tuple[int, int, int], kind: str) -> None:
        x, y = point
        left, top = x * scale, y * scale
        right, bottom = left + scale - 1, top + scale - 1
        if kind == "diamond":
            center_x, center_y = left + scale // 2, top + scale // 2
            draw.polygon(
                ((center_x, top), (right, center_y), (center_x, bottom), (left, center_y)),
                fill=color,
                outline=(255, 255, 255),
            )
        elif kind == "cross":
            center_x, center_y = left + scale // 2, top + scale // 2
            draw.line((left, center_y, right, center_y), fill=color, width=max(1, scale // 2))
            draw.line((center_x, top, center_x, bottom), fill=color, width=max(1, scale // 2))
        else:
            draw.rectangle((left, top, right, bottom), fill=color, outline=(255, 255, 255))

    for spawn in data.spawns:
        x, y = spawn
        center_x, center_y = x * scale + scale // 2, y * scale + scale // 2
        draw.point((center_x, center_y), fill=(255, 69, 91))
    for objective in data.objectives:
        marker(objective, (255, 221, 53), "diamond")
    marker(data.start, (50, 255, 151), "cross")
    marker(data.exit, (255, 48, 213), "square")
    return image


def preview_png_bytes(data: MapData, scale: int = 5) -> bytes:
    buffer = BytesIO()
    render_preview(data, scale=scale).save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()
