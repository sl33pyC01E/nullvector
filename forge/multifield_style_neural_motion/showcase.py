from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..morphology import FAMILIES
from ..multifield_style_motion.hashing import array_sha256, png_bytes
from ..multifield_style_motion.io import verify_artifact
from ..multifield_style_motion.model import ATLAS_COLUMNS, IMAGE_SIZE, LAYER_NAMES
from ..multifield_style_motion.showcase import (
    encode_showcase_mp4,
    ffmpeg_provenance,
)


CONTACT_MOTIONS = ("idle_wiggle", "locomote", "joy", "attack", "cast", "death")
CONTACT_FACING = "southeast"
SHOWCASE_MOTION = "locomote"
SHOWCASE_FACING = "southeast"
SHOWCASE_FPS = 12
SHOWCASE_FRAMES = 36


@dataclass(frozen=True, slots=True)
class NeuralShowcasePayload:
    contact_png: bytes
    poster_png: bytes
    video_mp4: bytes
    contact_selections: tuple[Mapping[str, Any], ...]
    frame_sha256: tuple[str, ...]
    width: int
    height: int
    ffmpeg: Mapping[str, Any]
    encoding: Mapping[str, Any]


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _cell(atlas: np.ndarray, index: int) -> np.ndarray:
    row, column = divmod(int(index), ATLAS_COLUMNS)
    y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
    values = atlas[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE]
    if values.shape != (48, 48, 4):
        raise ValueError("Neural motion atlas cell is outside the 48px grid")
    return values


def _lookup(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {(clip["motion"], clip["facing"]): clip for clip in manifest["clips"]}
    if len(result) != manifest["clip_count"]:
        raise ValueError("Neural identity clip keys are not unique")
    return result


def load_atlases(
    root: Path,
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        manifest = manifests[family]
        expected = (
            manifest["layout"]["columns"] * IMAGE_SIZE,
            manifest["layout"]["rows"] * IMAGE_SIZE,
        )
        layers: dict[str, np.ndarray] = {}
        for name in LAYER_NAMES:
            path = verify_artifact(root, manifest["artifacts"]["layers"][name])
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGBA" or image.size != expected:
                    raise ValueError("Neural motion atlas image contract mismatch")
                layers[name] = np.asarray(image, dtype=np.uint8).copy()
        result[family] = layers
    return result


def build_contact_sheet(
    manifests: Mapping[str, Mapping[str, Any]],
    atlases: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
    scale = 2
    tile_width = IMAGE_SIZE * scale + 14
    tile_height = IMAGE_SIZE * scale + 28
    label_width = 132
    width = label_width + len(CONTACT_MOTIONS) * tile_width + 10
    height = 40 + len(FAMILIES) * tile_height + 10
    image = Image.new("RGBA", (width, height), (3, 5, 13, 255))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "ACTUAL NEURAL MOTION PRESENTATION / SE FACING", fill=(210, 240, 255, 255), font=_font())
    for column, motion in enumerate(CONTACT_MOTIONS):
        draw.text((label_width + column * tile_width, 27), motion.upper(), fill=(188, 226, 250, 255), font=_font())
    selections: list[Mapping[str, Any]] = []
    for row, family in enumerate(FAMILIES):
        manifest = manifests[family]
        lookup = _lookup(manifest)
        y = 40 + row * tile_height
        draw.rectangle((0, y, width - 1, y + tile_height - 2), fill=(8 + row * 2, 13 + row * 2, 27 + row * 2, 255))
        draw.text((8, y + 38), family.upper(), fill=(210, 240, 255, 255), font=_font())
        draw.text((8, y + 52), manifest["sample_id"], fill=(130, 179, 213, 255), font=_font())
        for column, motion in enumerate(CONTACT_MOTIONS):
            clip = lookup[(motion, CONTACT_FACING)]
            if motion in {"attack", "cast", "death"}:
                frame = int(clip["events"][-1]["frame"])
            else:
                frame = (int(clip["frame_count"]) - 1) // 2
            cell = int(clip["start_cell"]) + frame
            sprite = Image.fromarray(_cell(atlases[family]["composite"], cell), mode="RGBA").resize(
                (IMAGE_SIZE * scale, IMAGE_SIZE * scale), Image.Resampling.NEAREST
            )
            image.alpha_composite(sprite, (label_width + column * tile_width, y + 8))
            selections.append({
                "family": family, "sample_id": manifest["sample_id"], "motion": motion,
                "facing": CONTACT_FACING, "clip_id": clip["id"], "source_frame": frame,
                "atlas_cell": cell,
            })
    return np.asarray(image, dtype=np.uint8).copy(), tuple(selections)


def build_showcase_frames(
    manifests: Mapping[str, Mapping[str, Any]],
    atlases: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, ...]:
    scale = 3
    tile = IMAGE_SIZE * scale + 22
    width = max(640, len(FAMILIES) * tile + 10)
    height = 220
    if width % 2:
        width += 1
    frames: list[np.ndarray] = []
    lookups = {family: _lookup(manifests[family]) for family in FAMILIES}
    for output_frame in range(SHOWCASE_FRAMES):
        image = Image.new("RGBA", (width, height), (2, 4, 11, 255))
        draw = ImageDraw.Draw(image)
        draw.text((10, 9), "ACTUAL NEURAL LOCOMOTION / DERIVED GRAPH-RIG MOTION / 12 FPS", fill=(210, 240, 255, 255), font=_font())
        for column, family in enumerate(FAMILIES):
            manifest = manifests[family]
            clip = lookups[family][(SHOWCASE_MOTION, SHOWCASE_FACING)]
            cycle = int(clip["frame_count"]) - 1
            frame = (output_frame * int(clip["fps"]) // SHOWCASE_FPS) % cycle
            cell = int(clip["start_cell"]) + frame
            sprite = Image.fromarray(_cell(atlases[family]["composite"], cell), mode="RGBA").resize(
                (IMAGE_SIZE * scale, IMAGE_SIZE * scale), Image.Resampling.NEAREST
            )
            x = 10 + column * tile
            image.alpha_composite(sprite, (x, 36))
            draw.text((x, 184), family.upper(), fill=(210, 240, 255, 255), font=_font())
            draw.text((x, 198), manifest["sample_id"], fill=(130, 179, 213, 255), font=_font())
        frames.append(np.asarray(image, dtype=np.uint8).copy())
    return tuple(frames)


def compile_showcase(
    root: Path,
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    ffmpeg_executable: Path | None = None,
) -> NeuralShowcasePayload:
    atlases = load_atlases(root, manifests)
    contact, selections = build_contact_sheet(manifests, atlases)
    frames = build_showcase_frames(manifests, atlases)
    ffmpeg = ffmpeg_provenance(ffmpeg_executable)
    video, encoding = encode_showcase_mp4(frames, ffmpeg, fps=SHOWCASE_FPS)
    return NeuralShowcasePayload(
        contact_png=png_bytes(contact), poster_png=png_bytes(frames[0]), video_mp4=video,
        contact_selections=selections,
        frame_sha256=tuple(array_sha256(f"neural_showcase_frame_{i:03d}", frame) for i, frame in enumerate(frames)),
        width=int(frames[0].shape[1]), height=int(frames[0].shape[0]),
        ffmpeg=ffmpeg, encoding=encoding,
    )
