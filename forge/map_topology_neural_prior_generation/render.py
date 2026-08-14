from __future__ import annotations

from io import BytesIO
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from ..map_topology_neural.render import render_categorical, render_edit_overlay


def case_preview_png_bytes(source: object, raw: object, compiled: object, *, scale: int = 4) -> bytes:
    points = {
        "start": getattr(source, "start"), "exit": getattr(source, "exit"),
        "objectives": getattr(source, "objectives"), "spawns": getattr(source, "spawns"),
    }
    images = [
        render_categorical(source.raw.terrain, source.raw.hazard, source.raw.elevation, **points, scale=scale),
        render_categorical(raw.terrain, raw.hazard, raw.elevation, **points, scale=scale),
        render_categorical(compiled.terrain, compiled.hazard, compiled.elevation, **points, scale=scale),
        render_edit_overlay(raw, compiled, **points, scale=scale),
    ]
    labels = ("CONDITION SOURCE", "RAW SAMPLE", "COMPILED", "EDIT OVERLAY")
    font = ImageFont.load_default()
    label_height, gutter = 24, 6
    output = Image.new("RGB", (sum(image.width for image in images) + gutter * 5, max(image.height for image in images) + label_height + gutter * 2), (3, 5, 14))
    draw = ImageDraw.Draw(output)
    x = gutter
    for label, image in zip(labels, images, strict=True):
        draw.text((x + 2, 5), label, fill=(183, 229, 255), font=font)
        output.paste(image.convert("RGB"), (x, label_height))
        draw.rectangle((x - 1, label_height - 1, x + image.width, label_height + image.height), outline=(28, 75, 105))
        x += image.width + gutter
    encoded = BytesIO(); output.save(encoded, format="PNG", optimize=False, compress_level=9)
    return encoded.getvalue()


def contact_sheet_png_bytes(rows: Iterable[tuple[str, bytes]]) -> bytes:
    materialized = [(label, Image.open(BytesIO(payload)).convert("RGB")) for label, payload in rows]
    if not materialized:
        raise ValueError("Generation contact sheet requires at least one row.")
    font = ImageFont.load_default(); label_width, gutter = 118, 6
    width = label_width + max(image.width for _, image in materialized) + gutter * 2
    height = gutter + sum(image.height + gutter for _, image in materialized)
    sheet = Image.new("RGB", (width, height), (2, 5, 12)); draw = ImageDraw.Draw(sheet)
    y = gutter
    for label, image in materialized:
        draw.text((6, y + 5), label.upper(), fill=(255, 100, 225), font=font)
        sheet.paste(image, (label_width, y)); y += image.height + gutter
    encoded = BytesIO(); sheet.save(encoded, format="PNG", optimize=False, compress_level=9)
    return encoded.getvalue()
