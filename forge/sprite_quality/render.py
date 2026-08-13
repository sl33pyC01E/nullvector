from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

from PIL import Image, ImageDraw


FAMILIES = ("humanoid", "animalian", "plantlike", "anomaly", "machine")
MOTIONS = (
    "idle_breathe",
    "idle_wiggle",
    "locomote",
    "joy",
    "anger",
    "fear",
    "confused",
    "sleep",
    "taunt",
    "attack",
    "cast",
    "hit",
    "death",
)


def _energy_color(value: float) -> tuple[int, int, int]:
    level = max(0.0, min(1.0, float(value) / 0.75))
    if level < 0.5:
        local = level * 2.0
        return (
            int(round(12 + 14 * local)),
            int(round(30 + 172 * local)),
            int(round(54 + 142 * local)),
        )
    local = (level - 0.5) * 2.0
    return (
        int(round(26 + 225 * local)),
        int(round(202 - 96 * local)),
        int(round(196 + 47 * local)),
    )


def render_motion_energy_heatmap(report: Mapping[str, Any]) -> bytes:
    records = {
        (record["family"], record["motion"]): record
        for record in report["motion_quality"]["by_family_motion"]
    }
    width, height = 620, 478
    image = Image.new("RGB", (width, height), (5, 8, 19))
    draw = ImageDraw.Draw(image)
    draw.text((14, 10), "NEURAL MOTION ENERGY / MEAN COMPOSITE CHANGE", fill=(209, 239, 255))
    draw.text((14, 26), "5 actual neural representatives / 520 clips / effective loop frames", fill=(91, 153, 188))
    left, top, cell_w, cell_h = 116, 58, 96, 29
    for column, family in enumerate(FAMILIES):
        draw.text((left + column * cell_w + 3, top - 16), family[:10], fill=(153, 231, 255))
    for row, motion in enumerate(MOTIONS):
        y = top + row * cell_h
        draw.text((8, y + 8), motion, fill=(177, 194, 219))
        for column, family in enumerate(FAMILIES):
            x = left + column * cell_w
            value = float(records[(family, motion)]["composite_change"]["mean"])
            color = _energy_color(value)
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), fill=color, outline=(50, 84, 112))
            ink = (2, 8, 16) if sum(color) > 360 else (231, 249, 255)
            draw.text((x + 28, y + 7), f"{value:.3f}", fill=ink)
    draw.text((14, height - 26), "cyan = restrained motion   magenta = high motion   diagnostic, not an aesthetic score", fill=(91, 153, 188))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()
