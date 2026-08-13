from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..config import OUTPUT_DIR, PROJECT_ROOT
from ..maps.generator import generate_map, splitmix64
from ..maps.io import array_digest, file_sha256, load_map_pack
from ..maps.model import (
    GENERATOR_NAME,
    GENERATOR_VERSION,
    MAP_SCHEMA_VERSION,
    TOPOLOGY_MASK_CAPTURE_POLICY,
    TOPOLOGY_MASK_CONTRACT_NAME,
    TOPOLOGY_MASK_CONTRACT_VERSION,
    TOPOLOGY_MASK_NAMES,
    MapConfig,
    MapData,
    THEMES,
)
from ..safety import require_disk_floor, write_json_atomic
from .atlas import compose_preview, frame_grid_metadata, pack_frame_grid
from .io import _atomic_bytes, _json_bytes, _png_bytes, write_art_pack
from .model import HAZARD_FRAME_COUNT, TILE_SIZE
from .renderer import render_hazard_tile, render_map_art
from .styles import style_for
from .validate import validate_art_pack, validate_layers


def _find_map_manifests(paths: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    candidates: list[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_file() and path.name == "manifest.json":
            candidates.append(path)
        elif path.is_dir() and (path / "manifest.json").is_file():
            candidates.append(path / "manifest.json")
        elif path.is_dir():
            candidates.extend(path.rglob("manifest.json"))
    for manifest in sorted(set(candidates)):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("generator", {}).get("name") == "nullvector-map-forge":
                found.append(manifest)
        except (OSError, json.JSONDecodeError):
            continue
    return found


def _find_art_manifests(paths: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_file() and path.name == "manifest.json":
            candidates = [path]
        elif path.is_dir() and (path / "manifest.json").is_file():
            candidates = [path / "manifest.json"]
        elif path.is_dir():
            candidates = list(path.rglob("manifest.json"))
        else:
            candidates = []
        for manifest in candidates:
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if payload.get("renderer", {}).get("name") == "nullvector-map-art-forge":
                    found.append(manifest)
            except (OSError, json.JSONDecodeError):
                continue
    return sorted(set(found))


def fuzz_art(
    count: int,
    *,
    base_seed: int = 0x504958454C4E454F,
    width: int = 40,
    height: int = 40,
) -> dict[str, object]:
    if count < 1:
        raise ValueError("Fuzz count must be positive.")
    require_disk_floor(PROJECT_ROOT, planned_bytes=0)
    started = time.perf_counter()
    failures: list[dict[str, object]] = []
    per_theme = {theme: 0 for theme in THEMES}
    hashes: set[str] = set()
    instance_total = 0
    for index in range(count):
        theme = THEMES[index % len(THEMES)]
        seed = splitmix64(base_seed ^ index ^ ((index % len(THEMES) + 1) << 45))
        cfg = MapConfig(
            width=width + (index % 3) * 4,
            height=height + ((index // 3) % 3) * 4,
            spawn_count=8,
        )
        try:
            data = generate_map(seed, theme, cfg)
            first = render_map_art(data)
            report = validate_layers(data, first)
            if not report["passed"]:
                failures.append({"index": index, "theme": theme, "seed": seed, "failures": report["failures"]})
                continue
            second = render_map_art(data)
            digest = hashlib.sha256()
            digest.update(memoryview(first.base_color).cast("B"))
            digest.update(memoryview(first.emissive).cast("B"))
            digest.update(memoryview(first.hazard_color_frames).cast("B"))
            digest.update(array_digest(first.semantic_arrays()).encode("ascii"))
            first_hash = digest.hexdigest()
            second_digest = hashlib.sha256()
            second_digest.update(memoryview(second.base_color).cast("B"))
            second_digest.update(memoryview(second.emissive).cast("B"))
            second_digest.update(memoryview(second.hazard_color_frames).cast("B"))
            second_digest.update(array_digest(second.semantic_arrays()).encode("ascii"))
            if first_hash != second_digest.hexdigest():
                failures.append({"index": index, "theme": theme, "seed": seed, "error": "rerender mismatch"})
                continue
            hashes.add(first_hash)
            instance_total += len(first.instances)
            per_theme[theme] += 1
        except Exception as error:
            failures.append({"index": index, "theme": theme, "seed": seed, "error": repr(error)})
    elapsed = time.perf_counter() - started
    passed_count = count - len(failures)
    return {
        "passed": not failures,
        "requested": count,
        "passed_count": passed_count,
        "failure_count": len(failures),
        "per_theme": per_theme,
        "unique_visual_maps": len(hashes),
        "instance_total": instance_total,
        "elapsed_seconds": round(elapsed, 3),
        "maps_per_second": round(passed_count / max(elapsed, 1e-9), 3),
        "failures": failures[:50],
    }


def _showcase_contact(previews: list[tuple[str, np.ndarray]]) -> np.ndarray:
    columns = 3
    rows = 2
    cell_width = max(image.shape[1] for _, image in previews)
    image_height = max(image.shape[0] for _, image in previews)
    label_height = 18
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + label_height)), (2, 2, 9))
    draw = ImageDraw.Draw(sheet)
    for index, (theme, array) in enumerate(previews):
        row, column = divmod(index, columns)
        left = column * cell_width
        top = row * (image_height + label_height)
        sheet.paste(Image.fromarray(array), (left, top))
        draw.rectangle((left, top + image_height, left + cell_width - 1, top + image_height + label_height - 1), fill=(5, 7, 18))
        draw.text((left + 6, top + image_height + 3), theme.upper(), fill=(111, 238, 255))
    return np.asarray(sheet)


def _showcase_hazard_frames(scale: int = 4) -> np.ndarray:
    frame_width = 4 * TILE_SIZE * scale
    frame_height = len(THEMES) * TILE_SIZE * scale
    frames = np.zeros((HAZARD_FRAME_COUNT, frame_height, frame_width, 3), dtype=np.uint8)
    for frame in range(HAZARD_FRAME_COUNT):
        canvas = np.zeros((len(THEMES) * TILE_SIZE, 4 * TILE_SIZE, 3), dtype=np.uint8)
        canvas[:] = (3, 4, 12)
        for row, theme in enumerate(THEMES):
            style = style_for(theme)
            for hazard_id in range(1, 5):
                tile, glow = render_hazard_tile(style, hazard_id, frame, same_mask=15)
                base = np.full((TILE_SIZE, TILE_SIZE, 3), style.terrain[1], dtype=np.uint16)
                alpha = tile[..., 3:4].astype(np.uint16)
                composed = (tile[..., :3].astype(np.uint16) * alpha + base * (255 - alpha) + 127) // 255
                composed = np.minimum(composed + glow.astype(np.uint16) // 2, 255).astype(np.uint8)
                left = (hazard_id - 1) * TILE_SIZE
                top = row * TILE_SIZE
                canvas[top : top + TILE_SIZE, left : left + TILE_SIZE] = composed
        frames[frame] = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)
    return frames


def _topology_source_record(data: MapData) -> dict[str, object]:
    masks = {name: data.arrays()[name] for name in TOPOLOGY_MASK_NAMES}
    return {
        "contract_name": TOPOLOGY_MASK_CONTRACT_NAME,
        "contract_version": TOPOLOGY_MASK_CONTRACT_VERSION,
        "capture_policy": TOPOLOGY_MASK_CAPTURE_POLICY,
        "combined_sha256": array_digest(masks),
        "members": {
            name: {
                "sha256": array_digest({name: masks[name]}),
                "cell_count": int(np.count_nonzero(masks[name])),
            }
            for name in TOPOLOGY_MASK_NAMES
        },
    }


def _persisted_showcase_sources(
    paths: Sequence[str | Path],
) -> list[tuple[str, MapData, dict[str, object]]]:
    manifests = _find_map_manifests([str(path) for path in paths])
    if not manifests:
        raise ValueError("No authoritative map manifests were found for the showcase.")
    by_theme: dict[str, tuple[MapData, dict[str, object]]] = {}
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        theme = str(payload.get("theme", ""))
        if theme not in THEMES:
            raise ValueError(f"Unsupported showcase map theme {theme!r} in {manifest}.")
        if theme in by_theme:
            raise ValueError(
                f"Showcase map sources must contain exactly one pack per theme; duplicate {theme!r}."
            )
        data = load_map_pack(manifest)
        topology = payload["semantics"]["topology_masks"]
        record: dict[str, object] = {
            "origin": "persisted_map_pack",
            "theme": theme,
            "map_id": data.map_id,
            "seed": data.seed,
            "dimensions": [data.config.width, data.config.height],
            "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "schema_version": payload["schema_version"],
            "generator": payload["generator"],
            "semantic_array_sha256": payload["semantic_array_sha256"],
            "topology_masks": {
                "contract_name": topology["contract_name"],
                "contract_version": topology["contract_version"],
                "capture_policy": topology["capture_policy"],
                "combined_sha256": topology["combined_sha256"],
                "members": {
                    name: {
                        "sha256": topology["members"][name]["sha256"],
                        "cell_count": topology["members"][name]["cell_count"],
                    }
                    for name in TOPOLOGY_MASK_NAMES
                },
            },
        }
        if record["topology_masks"] != _topology_source_record(data):
            raise ValueError(f"Loaded topology-mask provenance disagrees with {manifest}.")
        by_theme[theme] = (data, record)
    missing = [theme for theme in THEMES if theme not in by_theme]
    if missing:
        raise ValueError(
            "Showcase map sources must contain every theme; missing " + ", ".join(missing) + "."
        )
    return [(theme, *by_theme[theme]) for theme in THEMES]


def build_showcase(
    output_root: Path,
    *,
    seed: int = 0x4E454F4E4D4150,
    map_size: int = 48,
    skip_existing: bool = True,
    source_paths: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    output_root = Path(output_root).resolve()
    require_disk_floor(output_root, planned_bytes=96 * 1024 * 1024)
    output_root.mkdir(parents=True, exist_ok=True)
    previews: list[tuple[str, np.ndarray]] = []
    packs: list[str] = []
    pack_reports: list[dict[str, object]] = []
    if source_paths is None:
        sources: list[tuple[str, MapData, dict[str, object]]] = []
        for theme_index, theme in enumerate(THEMES):
            map_seed = splitmix64(seed ^ ((theme_index + 1) << 48))
            data = generate_map(map_seed, theme, MapConfig(width=map_size, height=map_size))
            sources.append(
                (
                    theme,
                    data,
                    {
                        "origin": "deterministic_generator",
                        "theme": theme,
                        "map_id": data.map_id,
                        "seed": data.seed,
                        "dimensions": [data.config.width, data.config.height],
                        "manifest": None,
                        "manifest_sha256": None,
                        "schema_version": MAP_SCHEMA_VERSION,
                        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
                        "semantic_array_sha256": array_digest(data.arrays()),
                        "topology_masks": _topology_source_record(data),
                    },
                )
            )
        source_mode = "deterministic_generator"
    else:
        sources = _persisted_showcase_sources(source_paths)
        source_mode = "persisted_map_packs"
    source_records: list[dict[str, object]] = []
    for theme_index, (theme, data, source_record) in enumerate(sources):
        layers = render_map_art(data)
        previews.append((theme, compose_preview(layers, frame=theme_index % HAZARD_FRAME_COUNT)))
        pack = write_art_pack(data, output_root / "packs", skip_existing=skip_existing)
        report = validate_art_pack(pack, source_data=data)
        packs.append(str(pack))
        pack_reports.append(report)
        source_records.append(source_record)
    contact = _showcase_contact(previews)
    hazard_frames = _showcase_hazard_frames()
    hazard_sheet = pack_frame_grid(hazard_frames)
    contact_payload = _png_bytes(contact)
    hazard_payload = _png_bytes(hazard_sheet)
    hazard_meta = frame_grid_metadata(hazard_frames.shape[2], hazard_frames.shape[1])
    contact_path = output_root / "map_art_contact_sheet.png"
    hazard_path = output_root / "animated_hazards.png"
    hazard_meta_path = output_root / "animated_hazards.meta.json"
    _atomic_bytes(contact_path, contact_payload)
    _atomic_bytes(hazard_path, hazard_payload)
    _atomic_bytes(hazard_meta_path, _json_bytes(hazard_meta))
    report: dict[str, object] = {
        "passed": all(bool(item["passed"]) for item in pack_reports),
        "themes": list(THEMES),
        "source_mode": source_mode,
        "source_maps": source_records,
        "pack_count": len(packs),
        "packs": packs,
        "artifacts": {
            "contact_sheet": {"file": str(contact_path), "sha256": hashlib.sha256(contact_payload).hexdigest()},
            "animated_hazards": {"file": str(hazard_path), "sha256": hashlib.sha256(hazard_payload).hexdigest()},
            "animation_metadata": {"file": str(hazard_meta_path), "sha256": hashlib.sha256(_json_bytes(hazard_meta)).hexdigest()},
        },
        "pack_reports": pack_reports,
    }
    write_json_atomic(output_root / "showcase_report.json", report)
    return report


def _render_command(args: argparse.Namespace) -> int:
    manifests = _find_map_manifests(args.paths)
    packs = []
    for manifest in manifests:
        data = load_map_pack(manifest)
        packs.append(str(write_art_pack(data, Path(args.output).resolve(), skip_existing=args.skip_existing)))
    payload = {"passed": bool(manifests), "source_count": len(manifests), "packs": packs}
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


def _validate_command(args: argparse.Namespace) -> int:
    manifests = _find_art_manifests(args.paths)
    reports = [validate_art_pack(manifest) for manifest in manifests]
    payload = {"passed": bool(reports) and all(report["passed"] for report in reports), "reports": reports}
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


def _fuzz_command(args: argparse.Namespace) -> int:
    report = fuzz_art(args.count, base_seed=args.seed, width=args.width, height=args.height)
    if args.report:
        write_json_atomic(Path(args.report).resolve(), report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def _showcase_command(args: argparse.Namespace) -> int:
    report = build_showcase(
        Path(args.output),
        seed=args.seed,
        map_size=args.map_size,
        skip_existing=args.skip_existing,
        source_paths=args.map_sources,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic pixel-neon map art forge.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="Render semantic map packs into art packs.")
    render.add_argument("paths", nargs="+", help="Map pack directories, manifests, or roots.")
    render.add_argument("--output", type=Path, default=OUTPUT_DIR / "map_art" / "packs")
    render.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    render.set_defaults(func=_render_command)
    validate = subparsers.add_parser("validate", help="Validate art packs recursively.")
    validate.add_argument("paths", nargs="+")
    validate.set_defaults(func=_validate_command)
    fuzz = subparsers.add_parser("fuzz", help="Property-fuzz semantic-to-visual rendering in memory.")
    fuzz.add_argument("--count", type=int, default=120)
    fuzz.add_argument("--seed", type=lambda value: int(value, 0), default=0x41525446555A5A)
    fuzz.add_argument("--width", type=int, default=40)
    fuzz.add_argument("--height", type=int, default=40)
    fuzz.add_argument("--report", type=Path)
    fuzz.set_defaults(func=_fuzz_command)
    showcase = subparsers.add_parser("showcase", help="Build all-theme packs, contact sheet, and animated hazard sheet.")
    showcase.add_argument("--output", type=Path, default=OUTPUT_DIR / "map_art")
    showcase.add_argument("--seed", type=lambda value: int(value, 0), default=0x4E454F4E4D4150)
    showcase.add_argument("--map-size", type=int, default=48)
    showcase.add_argument(
        "--map-sources",
        nargs="+",
        help=(
            "Authoritative v2 map pack directories, manifests, or a root containing "
            "exactly one pack for every theme. When supplied, --seed and --map-size are unused."
        ),
    )
    showcase.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    showcase.set_defaults(func=_showcase_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
