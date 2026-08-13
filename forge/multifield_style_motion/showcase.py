from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..morphology import FAMILIES
from .hashing import artifact_record_from_bytes, array_sha256, png_bytes, sha256_file
from .io import verify_artifact
from .model import ATLAS_COLUMNS, IMAGE_SIZE, LAYER_NAMES


CONTACT_MOTIONS = (
    "idle_wiggle",
    "locomote",
    "joy",
    "attack",
    "cast",
    "death",
)
SHOWCASE_MOTIONS = (
    "idle_wiggle",
    "locomote",
    "joy",
    "confused",
    "taunt",
)
CONTACT_FACING = "southeast"
SHOWCASE_FACING = "southeast"
SHOWCASE_FPS = 12
SHOWCASE_FRAME_COUNT = 36


@dataclass(frozen=True, slots=True)
class ShowcasePayload:
    contact_png: bytes
    poster_png: bytes
    video_mp4: bytes
    contact_selections: tuple[Mapping[str, Any], ...]
    showcase_selections: tuple[Mapping[str, Any], ...]
    frame_sha256: tuple[str, ...]
    width: int
    height: int
    fps: int
    frame_count: int
    ffmpeg: Mapping[str, Any]
    encoding: Mapping[str, Any]


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(206, 238, 255, 255), font=_font())


def _open_rgba(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGBA" or image.size != expected_size:
            raise ValueError(f"Style-motion atlas image contract mismatch: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


def load_family_atlases(
    output_root: Path,
    family_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    root = Path(output_root).resolve()
    result: dict[str, dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        manifest = family_manifests[family]
        layout = manifest["layout"]
        expected_size = (
            int(layout["columns"]) * int(layout["cell_size"]),
            int(layout["rows"]) * int(layout["cell_size"]),
        )
        layers: dict[str, np.ndarray] = {}
        for layer_name in LAYER_NAMES:
            path = verify_artifact(root, manifest["artifacts"]["layers"][layer_name])
            layers[layer_name] = _open_rgba(path, expected_size)
        result[family] = layers
    return result


def _extract_cell(atlas: np.ndarray, cell: int) -> np.ndarray:
    row, column = divmod(int(cell), ATLAS_COLUMNS)
    y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
    values = atlas[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE]
    if values.shape != (IMAGE_SIZE, IMAGE_SIZE, 4):
        raise ValueError(f"Atlas cell is outside the native 48px grid: {cell}")
    return values


def _clip_lookup(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    clips = {
        (str(clip["motion"]), str(clip["facing"])): clip
        for clip in manifest["clips"]
    }
    if len(clips) != int(manifest["clip_count"]):
        raise ValueError("Family showcase clip keys are not unique")
    return clips


def _representative_frame(clip: Mapping[str, Any]) -> int:
    motion = str(clip["motion"])
    if motion in {"attack", "cast", "hit", "death"}:
        events = list(clip["events"])
        preferred = next(
            (
                event
                for event in events
                if event["name"] not in {"recover", "settle", "death_settle"}
            ),
            events[-1],
        )
        return min(int(preferred["frame"]), int(clip["frame_count"]) - 1)
    return max(0, (int(clip["frame_count"]) - 1) // 2)


def build_contact_sheet(
    family_manifests: Mapping[str, Mapping[str, Any]],
    atlases: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
    tile_width = 6 * IMAGE_SIZE + 10
    tile_height = 2 * IMAGE_SIZE + 70
    margin_x = 12
    header = 38
    label_column = 86
    width = label_column + len(CONTACT_MOTIONS) * tile_width + margin_x
    height = header + len(FAMILIES) * tile_height + 12
    sheet = Image.new("RGBA", (width, height), (4, 7, 16, 255))
    draw = ImageDraw.Draw(sheet)
    _label(draw, (12, 10), "MOTION STYLE: NATIVE LAYERS + COMPOSITE / SE FACING")
    selections: list[Mapping[str, Any]] = []
    layer_preview_names = tuple(name for name in LAYER_NAMES if name != "composite")
    for column, motion in enumerate(CONTACT_MOTIONS):
        x = label_column + column * tile_width
        _label(draw, (x + 4, 25), motion.upper())
    for row, family in enumerate(FAMILIES):
        y = header + row * tile_height
        draw.rectangle(
            (0, y, width - 1, y + tile_height - 2),
            fill=(7 + row * 2, 12 + row * 2, 26 + row * 3, 255),
        )
        _label(draw, (10, y + 68), family.upper())
        lookup = _clip_lookup(family_manifests[family])
        for column, motion in enumerate(CONTACT_MOTIONS):
            clip = lookup[(motion, CONTACT_FACING)]
            frame = _representative_frame(clip)
            cell = int(clip["start_cell"]) + frame
            x = label_column + column * tile_width
            for layer_column, layer_name in enumerate(layer_preview_names):
                pixels = _extract_cell(atlases[family][layer_name], cell)
                sheet.alpha_composite(
                    Image.fromarray(pixels, mode="RGBA"),
                    (x + 4 + layer_column * IMAGE_SIZE, y + 5),
                )
            composite = Image.fromarray(
                _extract_cell(atlases[family]["composite"], cell),
                mode="RGBA",
            ).resize((2 * IMAGE_SIZE, 2 * IMAGE_SIZE), resample=Image.Resampling.NEAREST)
            sheet.alpha_composite(composite, (x + (tile_width - 2 * IMAGE_SIZE) // 2, y + 58))
            selections.append(
                {
                    "family": family,
                    "motion": motion,
                    "facing": CONTACT_FACING,
                    "clip_id": clip["id"],
                    "source_frame": frame,
                    "atlas_cell": cell,
                }
            )
    return np.asarray(sheet, dtype=np.uint8).copy(), tuple(selections)


def _showcase_geometry() -> tuple[int, int, int, int, int]:
    scale = 2
    tile = IMAGE_SIZE * scale + 10
    label_column = 82
    header = 34
    width = max(640, label_column + len(SHOWCASE_MOTIONS) * tile + 8)
    height = max(576, header + len(FAMILIES) * tile + 8)
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    return width, height, scale, tile, label_column


def _source_frame_at_time(clip: Mapping[str, Any], output_frame: int) -> int:
    count = int(clip["frame_count"])
    cycle = count - 1 if bool(clip["loop"]) else count
    if cycle <= 0:
        raise ValueError("Showcase clip has no drawable frames")
    return int((output_frame * int(clip["fps"]) // SHOWCASE_FPS) % cycle)


def build_showcase_frames(
    family_manifests: Mapping[str, Mapping[str, Any]],
    atlases: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[tuple[np.ndarray, ...], tuple[Mapping[str, Any], ...]]:
    width, height, scale, tile, label_column = _showcase_geometry()
    lookups = {family: _clip_lookup(family_manifests[family]) for family in FAMILIES}
    selections = tuple(
        {
            "family": family,
            "motion": motion,
            "facing": SHOWCASE_FACING,
            "clip_id": lookups[family][(motion, SHOWCASE_FACING)]["id"],
            "fps": int(lookups[family][(motion, SHOWCASE_FACING)]["fps"]),
            "source_frame_count": int(
                lookups[family][(motion, SHOWCASE_FACING)]["frame_count"]
            ),
        }
        for family in FAMILIES
        for motion in SHOWCASE_MOTIONS
    )
    frames: list[np.ndarray] = []
    for output_frame in range(SHOWCASE_FRAME_COUNT):
        image = Image.new("RGBA", (width, height), (3, 5, 13, 255))
        draw = ImageDraw.Draw(image)
        _label(draw, (10, 9), "MOTION-COHERENT PRESENTATION / 12 FPS / SE FACING")
        for column, motion in enumerate(SHOWCASE_MOTIONS):
            _label(draw, (label_column + column * tile + 6, 22), motion.upper())
        for row, family in enumerate(FAMILIES):
            y = 34 + row * tile
            _label(draw, (8, y + 66), family.upper())
            for column, motion in enumerate(SHOWCASE_MOTIONS):
                clip = lookups[family][(motion, SHOWCASE_FACING)]
                source_frame = _source_frame_at_time(clip, output_frame)
                cell = int(clip["start_cell"]) + source_frame
                sprite = Image.fromarray(
                    _extract_cell(atlases[family]["composite"], cell),
                    mode="RGBA",
                ).resize(
                    (IMAGE_SIZE * scale, IMAGE_SIZE * scale),
                    resample=Image.Resampling.NEAREST,
                )
                x = label_column + column * tile + 5
                image.alpha_composite(sprite, (x, y + 4))
        frame = np.asarray(image, dtype=np.uint8).copy()
        if frame.shape != (height, width, 4):
            raise RuntimeError("Showcase frame geometry drifted")
        frames.append(frame)
    return tuple(frames), selections


def ffmpeg_provenance(executable: Path | None = None) -> dict[str, Any]:
    candidate = str(executable) if executable is not None else shutil.which("ffmpeg")
    if not candidate:
        raise FileNotFoundError("ffmpeg is required for the animated showcase")
    path = Path(candidate).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ffmpeg executable is missing: {path}")
    completed = subprocess.run(
        [str(path), "-version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    first_line = completed.stdout.splitlines()[0].strip()
    if not first_line.startswith("ffmpeg version "):
        raise ValueError("Unexpected ffmpeg version output")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "version_line": first_line,
    }


def _encoding_args(width: int, height: int, fps: int) -> tuple[str, ...]:
    return (
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        "-y",
    )


def encode_showcase_mp4(
    frames: Sequence[np.ndarray],
    ffmpeg: Mapping[str, Any],
    *,
    fps: int = SHOWCASE_FPS,
) -> tuple[bytes, Mapping[str, Any]]:
    if not frames:
        raise ValueError("Animated showcase needs at least one frame")
    first = np.asarray(frames[0])
    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 4:
        raise ValueError("Animated showcase frames must be uint8 RGBA")
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise ValueError("YUV420 showcase dimensions must be even")
    args = _encoding_args(width, height, fps)
    with tempfile.TemporaryDirectory(prefix="nullvector-style-motion-") as temporary:
        output = Path(temporary) / "showcase.mp4"
        command = [str(ffmpeg["path"]), *args, str(output)]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        try:
            for ordinal, frame in enumerate(frames):
                values = np.asarray(frame)
                if values.dtype != np.uint8 or values.shape != first.shape:
                    raise ValueError(f"Animated showcase frame {ordinal} contract mismatch")
                process.stdin.write(np.ascontiguousarray(values).tobytes())
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait(timeout=45)
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            raise
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg showcase encoding failed ({return_code}): "
                + stderr.decode("utf-8", errors="replace")[-2000:]
            )
        payload = output.read_bytes()
    if len(payload) < 32 or b"ftyp" not in payload[:32]:
        raise ValueError("ffmpeg did not produce a valid MP4 container")
    return payload, {
        "codec": "libx264",
        "preset": "slow",
        "crf": 12,
        "pixel_format": "yuv420p",
        "threads": 1,
        "metadata": "stripped",
        "arguments": list(args),
    }


def compile_showcase(
    output_root: Path,
    family_manifests: Mapping[str, Mapping[str, Any]],
    *,
    ffmpeg_executable: Path | None = None,
) -> ShowcasePayload:
    atlases = load_family_atlases(output_root, family_manifests)
    contact, contact_selections = build_contact_sheet(family_manifests, atlases)
    frames, showcase_selections = build_showcase_frames(family_manifests, atlases)
    provenance = ffmpeg_provenance(ffmpeg_executable)
    video, encoding = encode_showcase_mp4(frames, provenance)
    frame_hashes = tuple(
        array_sha256(f"showcase_frame_{ordinal:03d}", frame)
        for ordinal, frame in enumerate(frames)
    )
    return ShowcasePayload(
        contact_png=png_bytes(contact),
        poster_png=png_bytes(frames[0]),
        video_mp4=video,
        contact_selections=contact_selections,
        showcase_selections=showcase_selections,
        frame_sha256=frame_hashes,
        width=int(frames[0].shape[1]),
        height=int(frames[0].shape[0]),
        fps=SHOWCASE_FPS,
        frame_count=len(frames),
        ffmpeg=provenance,
        encoding=encoding,
    )


def showcase_artifacts(
    payload: ShowcasePayload,
) -> dict[str, Mapping[str, Any]]:
    return {
        "contact_sheet": artifact_record_from_bytes(
            "motion_contact_sheet.png", payload.contact_png
        ),
        "poster": artifact_record_from_bytes("motion_showcase_poster.png", payload.poster_png),
        "video": artifact_record_from_bytes("motion_showcase.mp4", payload.video_mp4),
    }
