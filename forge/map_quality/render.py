from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..maps.io import load_map_pack
from ..maps.render import render_preview
from ..safety import require_disk_floor, write_json_atomic
from .audit import (
    _articulation_points,
    _distances,
    _square_erode,
    audit_packs,
    audit_source_hash,
)


SHOWCASE_FORMAT = "nullvector-map-quality-showcase-v1"


def showcase_source_hash() -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-map-quality-showcase-source-v1\0")
    digest.update(Path(__file__).read_bytes())
    digest.update(b"\0")
    digest.update(audit_source_hash().encode("ascii"))
    return digest.hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, planned_bytes=len(payload) + 256 * 1024)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _agent_chokepoints(data: Any) -> tuple[set[int], set[int]]:
    safe_center = _square_erode(
        data.walkability.astype(bool) & (data.hazard == 0)
    )
    component = _distances(safe_center, (data.start,)) >= 0
    required = (data.exit, *data.objectives)
    return _articulation_points(
        safe_center, component, data.start, required
    )


def render_quality_overlay(pack: Path, *, scale: int = 4) -> Image.Image:
    if not 3 <= scale <= 12:
        raise ValueError("quality overlay scale must be in [3, 12]")
    data = load_map_pack(Path(pack), verify_hashes=True)
    image = render_preview(data, scale=scale).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = data.config.width
    required = {data.start, data.exit, *data.objectives}

    # Generation-time route footprint: a quiet blue center pixel per cell.
    for y, x in np.argwhere(data.protected_backbone != 0):
        point = (int(x), int(y))
        if point not in required:
            draw.point(
                (int(x) * scale + scale // 2, int(y) * scale + scale // 2),
                fill=(50, 101, 255),
            )

    all_points, mission_points = _agent_chokepoints(data)
    for index in sorted(all_points):
        x, y = index % width, index // width
        if (x, y) in required:
            continue
        center_x = x * scale + scale // 2
        center_y = y * scale + scale // 2
        color = (255, 184, 39) if index not in mission_points else (255, 36, 79)
        draw.point((center_x, center_y), fill=color)
        if index in mission_points:
            draw.line(
                (x * scale, center_y, (x + 1) * scale - 1, center_y),
                fill=color,
            )
            draw.line(
                (center_x, y * scale, center_x, (y + 1) * scale - 1),
                fill=color,
            )
    return image


def render_quality_contact_sheet(
    packs: Sequence[Path], report: dict[str, Any], *, scale: int = 4
) -> Image.Image:
    if len(packs) != len(report["maps"]):
        raise ValueError("quality showcase pack/report count mismatch")
    records_by_manifest = {
        record["source_manifest_sha256"]: record for record in report["maps"]
    }
    cells: list[tuple[str, Image.Image, dict[str, Any]]] = []
    from ..maps.io import file_sha256

    for pack in packs:
        pack = Path(pack)
        manifest = pack if pack.name == "manifest.json" else pack / "manifest.json"
        manifest_hash = file_sha256(manifest)
        if manifest_hash not in records_by_manifest:
            raise ValueError("quality showcase source pack is absent from report")
        record = records_by_manifest[manifest_hash]
        cells.append((record["theme"], render_quality_overlay(pack, scale=scale), record))
    cells.sort(key=lambda item: item[0])
    cell_width = max(image.width for _, image, _ in cells) + 16
    cell_height = max(image.height for _, image, _ in cells) + 38
    columns = min(3, len(cells))
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 22), (4, 3, 14))
    draw = ImageDraw.Draw(sheet)
    for index, (theme, image, record) in enumerate(cells):
        column = index % columns
        row = index // columns
        left = column * cell_width + 8
        top = row * cell_height + 18
        sheet.paste(image, (left, top))
        metrics = record["metrics"]
        draw.text((left, 4 + row * cell_height), theme.upper(), fill=(78, 224, 255))
        draw.text(
            (left, top + image.height + 3),
            (
                f"agent cuts {metrics['agent_scale_mission_articulation_count']}  "
                f"safe detour {metrics['maximum_radius_one_safe_detour_ratio_vs_geometric']:.3f}x"
            ),
            fill=(221, 216, 238),
        )
    draw.text(
        (8, sheet.height - 15),
        "blue: protected route   amber: agent articulation   red: mission cut",
        fill=(167, 157, 194),
    )
    return sheet


def write_quality_showcase(
    packs: Sequence[Path], output_root: Path, *, scale: int = 4
) -> dict[str, Any]:
    packs = tuple(Path(path) for path in packs)
    report = audit_packs(packs)
    image = render_quality_contact_sheet(packs, report, scale=scale)
    png = _png_bytes(image)
    output_root = Path(output_root)
    image_path = output_root / "quality_contact_sheet.png"
    _atomic_bytes(image_path, png)
    manifest: dict[str, Any] = {
        "format": SHOWCASE_FORMAT,
        "quality_report_sha256": report["report_sha256"],
        "audit_source_sha256": report["audit_source_sha256"],
        "showcase_source_sha256": showcase_source_hash(),
        "map_count": report["map_count"],
        "scale": scale,
        "image": {
            "file": image_path.name,
            "sha256": hashlib.sha256(png).hexdigest(),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        },
    }
    unhashed = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(unhashed).hexdigest()
    write_json_atomic(output_root / "showcase_manifest.json", manifest)
    return manifest


def assert_exact_quality_showcase(
    manifest_path: Path, packs: Sequence[Path]
) -> None:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        "format",
        "quality_report_sha256",
        "audit_source_sha256",
        "showcase_source_sha256",
        "map_count",
        "scale",
        "image",
        "manifest_sha256",
    }
    if set(payload) != expected_keys or payload.get("format") != SHOWCASE_FORMAT:
        raise ValueError("map quality showcase manifest shape is unsupported")
    if payload.get("showcase_source_sha256") != showcase_source_hash():
        raise ValueError("map quality showcase source hash is stale")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    expected_manifest_hash = hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if payload.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("map quality showcase manifest hash mismatch")
    report = audit_packs(tuple(Path(path) for path in packs))
    if payload.get("quality_report_sha256") != report["report_sha256"]:
        raise ValueError("map quality showcase report provenance mismatch")
    if payload.get("audit_source_sha256") != report["audit_source_sha256"]:
        raise ValueError("map quality showcase audit source mismatch")
    if payload.get("map_count") != report["map_count"]:
        raise ValueError("map quality showcase map count mismatch")
    image_record = payload.get("image")
    if not isinstance(image_record, dict) or set(image_record) != {
        "file",
        "sha256",
        "width",
        "height",
        "mode",
    }:
        raise ValueError("map quality showcase image record is unsupported")
    filename = image_record.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("map quality showcase image path is unsafe")
    expected_image = render_quality_contact_sheet(
        tuple(Path(path) for path in packs), report, scale=int(payload["scale"])
    )
    expected_bytes = _png_bytes(expected_image)
    actual_path = manifest_path.parent / filename
    actual_bytes = actual_path.read_bytes()
    if actual_bytes != expected_bytes:
        raise ValueError("map quality showcase image is not an exact replay")
    expected_image_record = {
        "file": filename,
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "width": expected_image.width,
        "height": expected_image.height,
        "mode": expected_image.mode,
    }
    if image_record != expected_image_record:
        raise ValueError("map quality showcase image metadata mismatch")
