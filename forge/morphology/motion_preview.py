from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .constants import CANVAS_SIZE, FAMILIES, JOINT_LAYER, LAYER_NAMES, SOCKET_LAYER
from .contract import assert_valid_specimen
from .disk_guard import guard_corpus_destination
from .genome import genome_from_seed
from .motion import (
    DEFAULT_FRAME_COUNTS,
    MOTION_NAMES,
    MotionClip,
    assert_valid_motion_clip,
    generate_motion_clip,
)
from .render import MorphologySpecimen, render_specimen


MOTION_PREVIEW_SEED = 0x4D4F544E
HARD_DISK_FLOOR_BYTES = 100 * 1024**3
SHOWCASE_MOTIONS = (
    "idle_breathe",
    "idle_wiggle",
    "locomote",
    "joy",
    "anger",
    "fear",
    "attack",
    "cast",
    "death",
)


def preview_specimens() -> tuple[MorphologySpecimen, ...]:
    specimens: list[MorphologySpecimen] = []
    for family_index, family in enumerate(FAMILIES):
        seed = (MOTION_PREVIEW_SEED + family_index * 0x10203041) & 0xFFFFFFFF
        specimen = render_specimen(genome_from_seed(seed, family))
        assert_valid_specimen(specimen)
        specimens.append(specimen)
    return tuple(specimens)


def build_motion_bank(
    specimens: Sequence[MorphologySpecimen] | None = None,
    *,
    motions: Iterable[str] = MOTION_NAMES,
    facing: str = "north",
) -> tuple[tuple[MorphologySpecimen, ...], tuple[MotionClip, ...]]:
    resolved_specimens = tuple(preview_specimens() if specimens is None else specimens)
    resolved_motions = tuple(motions)
    if not resolved_specimens:
        raise ValueError("At least one specimen is required")
    if not resolved_motions:
        raise ValueError("At least one motion is required")
    unknown = set(resolved_motions) - set(MOTION_NAMES)
    if unknown:
        raise ValueError(f"Unknown motions: {sorted(unknown)}")
    clips: list[MotionClip] = []
    for specimen in resolved_specimens:
        for motion in resolved_motions:
            clip = generate_motion_clip(specimen, motion, facing=facing)
            assert_valid_motion_clip(clip)
            clips.append(clip)
    return resolved_specimens, tuple(clips)


def _cell_backdrop(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (3, 5, 13, 255))
    draw = ImageDraw.Draw(image)
    step = max(8, size // 6)
    for coordinate in range(0, size, step):
        draw.line((coordinate, 0, coordinate, size), fill=(9, 20, 35, 255))
        draw.line((0, coordinate, size, coordinate), fill=(9, 20, 35, 255))
    draw.rectangle((0, 0, size - 1, size - 1), outline=(31, 72, 98, 255))
    return image


def _clip_lookup(clips: Sequence[MotionClip]) -> dict[tuple[str, str], MotionClip]:
    return {
        (clip.specimen.genome.family_name, clip.motion): clip for clip in clips
    }


def build_motion_contact_sheet(
    specimens: Sequence[MorphologySpecimen],
    clips: Sequence[MotionClip],
    *,
    motions: Sequence[str] = MOTION_NAMES,
    scale: int = 2,
) -> Image.Image:
    if scale < 1:
        raise ValueError("scale must be positive")
    cell = CANVAS_SIZE * scale
    label_width = 74
    header_height = 42
    sheet = Image.new(
        "RGBA",
        (label_width + len(motions) * cell, header_height + len(specimens) * cell),
        (2, 4, 10, 255),
    )
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, header_height - 1), fill=(7, 13, 26, 255))
    draw.text((7, 6), "GRAPH-RIG MOTION BANK // 48x48", fill=(218, 247, 255, 255))
    for column, motion in enumerate(motions):
        x = label_width + column * cell
        draw.text((x + 4, 26), motion.replace("idle_", "i_")[:14], fill=(91, 226, 255, 255))
    lookup = _clip_lookup(clips)
    backdrop = _cell_backdrop(cell)
    for row, specimen in enumerate(specimens):
        family = specimen.genome.family_name
        y = header_height + row * cell
        draw.rectangle((0, y, label_width - 1, y + cell - 1), fill=(6, 12, 24, 255))
        draw.text((6, y + 8), family.upper(), fill=(231, 244, 255, 255))
        draw.text((6, y + 24), f"{specimen.genome.seed:08X}", fill=(83, 181, 219, 255))
        for column, motion in enumerate(motions):
            clip = lookup[(family, motion)]
            if motion == "death":
                frame = clip.frames[-1]
            elif motion in {"attack", "cast", "hit"}:
                frame = clip.frames[len(clip.frames) // 2]
            else:
                frame = clip.frames[max(1, len(clip.frames) // 4)]
            x = label_width + column * cell
            sheet.alpha_composite(backdrop, (x, y))
            sprite = Image.fromarray(frame.rgba).resize(
                (cell, cell), Image.Resampling.NEAREST
            )
            sheet.alpha_composite(sprite, (x, y))
    return sheet


def _timeline_index(clip: MotionClip, timeline_index: int, timeline_count: int) -> int:
    if clip.loop:
        phase = timeline_index / float(timeline_count)
    else:
        phase = timeline_index / float(max(1, timeline_count - 1))
    return min(len(clip.frames) - 1, int(round(phase * (len(clip.frames) - 1))))


def build_showcase_frames(
    specimens: Sequence[MorphologySpecimen],
    clips: Sequence[MotionClip],
    *,
    motions: Sequence[str] = SHOWCASE_MOTIONS,
    frame_count: int = 32,
    scale: int = 2,
) -> tuple[Image.Image, ...]:
    if frame_count < 3 or scale < 1:
        raise ValueError("frame_count must be at least 3 and scale must be positive")
    cell = CANVAS_SIZE * scale
    label_width = 72
    header_height = 36
    width = label_width + len(motions) * cell
    height = header_height + len(specimens) * cell
    lookup = _clip_lookup(clips)
    backdrop = _cell_backdrop(cell)
    frames: list[Image.Image] = []
    for timeline_index in range(frame_count):
        image = Image.new("RGBA", (width, height), (2, 4, 10, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, header_height - 1), fill=(7, 13, 26, 255))
        draw.text((7, 6), "NEURAL MORPHOLOGY // MOTION SHOWCASE", fill=(218, 247, 255, 255))
        for column, motion in enumerate(motions):
            x = label_width + column * cell
            draw.text((x + 4, 22), motion.replace("idle_", "i_")[:14], fill=(92, 228, 255, 255))
        for row, specimen in enumerate(specimens):
            family = specimen.genome.family_name
            y = header_height + row * cell
            draw.rectangle((0, y, label_width - 1, y + cell - 1), fill=(6, 12, 24, 255))
            draw.text((6, y + 8), family.upper(), fill=(231, 244, 255, 255))
            for column, motion in enumerate(motions):
                x = label_width + column * cell
                clip = lookup[(family, motion)]
                frame = clip.frames[_timeline_index(clip, timeline_index, frame_count)]
                image.alpha_composite(backdrop, (x, y))
                sprite = Image.fromarray(frame.rgba).resize(
                    (cell, cell), Image.Resampling.NEAREST
                )
                image.alpha_composite(sprite, (x, y))
        frames.append(image)
    return tuple(frames)


def build_vertical_sprite_sheet(frames: Sequence[Image.Image]) -> Image.Image:
    if not frames:
        raise ValueError("At least one frame is required")
    width, height = frames[0].size
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("All sprite-sheet frames must have the same size")
    sheet = Image.new("RGBA", (width, height * len(frames)), (0, 0, 0, 0))
    for index, frame in enumerate(frames):
        sheet.alpha_composite(frame, (0, index * height))
    return sheet


def _atomic_png(image: Image.Image, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    image.save(temporary, format="PNG", compress_level=9)
    os.replace(temporary, destination)


def _atomic_json(payload: object, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, destination)


def _write_gif_with_ffmpeg(
    frames: Sequence[Image.Image],
    destination: Path,
    *,
    fps: int,
) -> None:
    """Encode with ffmpeg, avoiding Pillow's unstable Windows GIF encoder."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required for animated GIF previews")
    with tempfile.TemporaryDirectory(prefix="motion-gif-", dir=destination.parent) as directory:
        temporary_directory = Path(directory)
        for index, frame in enumerate(frames):
            frame.save(
                temporary_directory / f"frame_{index:04d}.png",
                format="PNG",
                compress_level=2,
            )
        pattern = str(temporary_directory / "frame_%04d.png")
        palette = temporary_directory / "palette.png"
        encoded = temporary_directory / "showcase.gif"
        commands = (
            [
                executable,
                "-v",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-vf",
                "palettegen=stats_mode=diff:max_colors=192",
                str(palette),
            ],
            [
                executable,
                "-v",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-i",
                str(palette),
                "-lavfi",
                "paletteuse=dither=bayer:bayer_scale=3",
                "-loop",
                "0",
                str(encoded),
            ],
        )
        for command in commands:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "ffmpeg GIF encoding failed: " + completed.stderr.strip()
                )
        os.replace(encoded, destination)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_semantic_archive(clips: Sequence[MotionClip], destination: Path) -> dict[str, object]:
    frames = [frame for clip in clips for frame in clip.frames]
    offsets = [0]
    for clip in clips:
        offsets.append(offsets[-1] + len(clip.frames))
    joint_names = tuple(JOINT_LAYER)
    socket_names = tuple(SOCKET_LAYER)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            layers=np.stack([frame.layers for frame in frames]),
            tokens=np.stack([frame.tokens for frame in frames]),
            joints=np.asarray(
                [[frame.joints[name] for name in joint_names] for frame in frames],
                dtype=np.uint8,
            ),
            sockets=np.asarray(
                [[frame.sockets[name] for name in socket_names] for frame in frames],
                dtype=np.uint8,
            ),
            phases=np.asarray([frame.phase for frame in frames], dtype=np.float32),
            frame_sha256=np.asarray([frame.sha256 for frame in frames]),
            clip_offsets=np.asarray(offsets, dtype=np.uint32),
            clip_ids=np.asarray([clip.manifest["id"] for clip in clips]),
            layer_names=np.asarray(LAYER_NAMES),
            joint_names=np.asarray(joint_names),
            socket_names=np.asarray(socket_names),
        )
    os.replace(temporary, destination)
    return {
        "frame_count": len(frames),
        "clip_count": len(clips),
        "layers_shape": [len(frames), len(LAYER_NAMES), CANVAS_SIZE, CANVAS_SIZE],
        "tokens_shape": [len(frames), CANVAS_SIZE, CANVAS_SIZE],
        "clip_offsets_shape": [len(offsets)],
    }


def write_motion_outputs(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve()
    total_frames = len(FAMILIES) * sum(DEFAULT_FRAME_COUNTS.values())
    budget = guard_corpus_destination(
        destination,
        total_frames,
        reserve_bytes=HARD_DISK_FLOOR_BYTES,
    )
    destination.mkdir(parents=True, exist_ok=True)
    specimens, clips = build_motion_bank()

    contact_path = destination / "morphology_motion_contact_sheet.png"
    sprite_sheet_path = destination / "morphology_motion_showcase_frames.png"
    sprite_meta_path = destination / "morphology_motion_showcase_frames.meta.json"
    gif_path = destination / "morphology_motion_showcase.gif"
    archive_path = destination / "morphology_motion_semantics.npz"
    manifest_path = destination / "morphology_motion_manifest.json"

    _atomic_png(build_motion_contact_sheet(specimens, clips), contact_path)
    showcase_frames = build_showcase_frames(specimens, clips)
    showcase_sheet = build_vertical_sprite_sheet(showcase_frames)
    _atomic_png(showcase_sheet, sprite_sheet_path)
    _atomic_json(
        {
            "format": "vertical-animation-strip-v1",
            "n_frames": len(showcase_frames),
            "frame_w": showcase_frames[0].width,
            "frame_h": showcase_frames[0].height,
            "duration_ms": 100,
            "layout": "top-to-bottom",
        },
        sprite_meta_path,
    )
    _write_gif_with_ffmpeg(showcase_frames, gif_path, fps=10)
    archive = _write_semantic_archive(clips, archive_path)

    manifest_payload = {
        "format": "neural-morphology-motion-bank-v1",
        "families": list(FAMILIES),
        "motion_names": list(MOTION_NAMES),
        "facing": "north",
        "source_count": len(specimens),
        "sources": [specimen.manifest for specimen in specimens],
        "clip_count": len(clips),
        "frame_count": archive["frame_count"],
        "disk_budget": budget.to_dict(),
        "archive": {
            "file": archive_path.name,
            "sha256": _sha256_file(archive_path),
            **archive,
        },
        "previews": {
            "contact_sheet": contact_path.name,
            "contact_sheet_sha256": _sha256_file(contact_path),
            "sprite_sheet": sprite_sheet_path.name,
            "sprite_sheet_sha256": _sha256_file(sprite_sheet_path),
            "sprite_sheet_meta": sprite_meta_path.name,
            "animated_gif": gif_path.name,
            "animated_gif_sha256": _sha256_file(gif_path),
        },
        "clips": [clip.manifest for clip in clips],
    }
    _atomic_json(manifest_payload, manifest_path)
    return {
        "sources": len(specimens),
        "clips": len(clips),
        "frames": archive["frame_count"],
        "contact_sheet": str(contact_path),
        "sprite_sheet": str(sprite_sheet_path),
        "animated_gif": str(gif_path),
        "semantic_archive": str(archive_path),
        "manifest": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate the deterministic graph-rig morphology motion bank."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=project_root / "outputs" / "morphology_motion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(write_motion_outputs(args.destination), indent=2))


if __name__ == "__main__":
    main()
