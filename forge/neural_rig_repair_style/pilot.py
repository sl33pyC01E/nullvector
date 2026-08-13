from __future__ import annotations

import io
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..morphology import FAMILIES
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    png_bytes,
)
from ..multifield_style_motion.io import require_disk_floor, write_exact
from ..multifield_style_motion.model import IMAGE_SIZE, LAYER_NAMES
from ..multifield_style_neural_motion.rendering import render_neural_motion_frame
from ..neural_rig_repair.binding import bind_repair_plan
from ..neural_rig_repair.constants import MAX_PLAN_BYTES, PROJECT_ROOT
from ..neural_rig_repair.hashing import sha256_bytes
from ..neural_rig_repair.planner import load_repair_plan
from ..neural_rig_repair.schema import resolve_artifact_record
from .authority import DEFAULT_REPAIR_BANK, RepairStyleAuthority, load_repair_style_authority
from .hashing import source_hash
from .projection import reconstruct_clip


PILOT_FORMAT = "nullvector-neural-rig-repair-style-pilot-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_rig_repair_style_pilot_v2"
PILOT_ORDINALS = (0, 16, 32, 48, 64)
PILOT_CLIPS = (
    ("idle_breathe", "north"),
    ("idle_wiggle", "north"),
    ("locomote", "southeast"),
    ("joy", "north"),
    ("fear", "northeast"),
    ("attack", "east"),
    ("cast", "north"),
    ("death", "south"),
)
ATLAS_COLUMNS = 16


def _artifact(path: str, payload: bytes) -> dict[str, Any]:
    return artifact_record_from_bytes(path, payload)


def _binding(authority: RepairStyleAuthority, ordinal: int):
    record = authority.bank["plans"][ordinal]
    plan_path = resolve_artifact_record(
        authority.bank_path.parent,
        record["artifact"],
        label=f"repair style pilot plan {ordinal}",
        maximum_bytes=MAX_PLAN_BYTES,
    )
    plan = load_repair_plan(plan_path)
    return bind_repair_plan(
        authority.repair_source,
        authority.repair_source.samples[ordinal],
        plan,
        verify_exact_plan=True,
    )


def _clip_audit(authority: RepairStyleAuthority, ordinal: int, motion: str, facing: str):
    matches = [
        clip
        for clip in authority.motion_audits[ordinal]["clips"]
        if clip["motion"] == motion and clip["facing"] == facing
    ]
    if len(matches) != 1:
        raise ValueError("repair style pilot clip audit registry is not exact")
    return matches[0]


def _contact_sheet(
    keyframes: Mapping[tuple[int, str, str], np.ndarray],
) -> bytes:
    scale = 4
    tile = IMAGE_SIZE * scale
    label_h = 28
    left = 126
    top = 60
    width = left + len(PILOT_CLIPS) * tile
    height = top + len(PILOT_ORDINALS) * (tile + label_h)
    sheet = Image.new("RGB", (width, height), (3, 9, 17))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 10), "ALL-80 REPAIRED RIG", fill=(102, 244, 255), font=font)
    draw.text((12, 25), "FIVE-FAMILY VISUAL PILOT", fill=(91, 124, 142), font=font)
    for column, (motion, facing) in enumerate(PILOT_CLIPS):
        x = left + column * tile
        draw.text((x + 5, 10), motion.upper(), fill=(196, 218, 229), font=font)
        draw.text((x + 5, 25), facing.upper(), fill=(91, 124, 142), font=font)
    for row, (ordinal, family) in enumerate(zip(PILOT_ORDINALS, FAMILIES, strict=True)):
        y = top + row * (tile + label_h)
        draw.text((12, y + 8), family.upper(), fill=(191, 255, 72), font=font)
        draw.text((12, y + 22), f"ORD {ordinal:02d}", fill=(91, 124, 142), font=font)
        for column, (motion, facing) in enumerate(PILOT_CLIPS):
            values = keyframes[(ordinal, motion, facing)]
            image = Image.fromarray(values, mode="RGBA").resize((tile, tile), Image.Resampling.NEAREST)
            background = Image.new("RGBA", (tile, tile), (5, 13, 23, 255))
            background.alpha_composite(image)
            x = left + column * tile
            sheet.paste(background.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(27, 62, 78), width=1)
    output = io.BytesIO()
    sheet.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _preview_frame(
    frames: Mapping[tuple[int, str, str], tuple[np.ndarray, ...]],
    motion: str,
    facing: str,
    index: int,
) -> np.ndarray:
    scale = 4
    tile = IMAGE_SIZE * scale
    top = 54
    width = tile * len(PILOT_ORDINALS)
    height = top + tile
    image = Image.new("RGB", (width, height), (3, 9, 17))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 9), "ALL-80 REPAIRED NEURAL RIG // FIVE REPRESENTATIVES", fill=(102, 244, 255), font=font)
    draw.text((12, 25), f"{motion.upper()} // {facing.upper()}", fill=(191, 255, 72), font=font)
    for column, (ordinal, family) in enumerate(zip(PILOT_ORDINALS, FAMILIES, strict=True)):
        sequence = frames[(ordinal, motion, facing)]
        selected = sequence[index % len(sequence)]
        sprite = Image.fromarray(selected, mode="RGBA").resize((tile, tile), Image.Resampling.NEAREST)
        cell = Image.new("RGBA", (tile, tile), (5, 13, 23, 255))
        cell.alpha_composite(sprite)
        x = column * tile
        image.paste(cell.convert("RGB"), (x, top))
        draw.rectangle((x, top, x + tile - 1, top + tile - 1), outline=(27, 62, 78), width=1)
        draw.text((x + 7, top + 7), family.upper(), fill=(196, 218, 229), font=font)
    return np.asarray(image, dtype=np.uint8)


def _animated_preview(
    frames: Mapping[tuple[int, str, str], tuple[np.ndarray, ...]],
    destination: Path,
) -> tuple[bytes, dict[str, Any]]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the repair style pilot preview")
    fps = 24
    width = IMAGE_SIZE * 4 * len(PILOT_ORDINALS)
    height = 54 + IMAGE_SIZE * 4
    temporary = destination / ".repair_style_pilot_preview.mp4.tmp"
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        "creation_time=1970-01-01T00:00:00Z",
        "-f",
        "mp4",
        "-y",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    frame_total = 0
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg raw-video input pipe is unavailable")
        for motion, facing in PILOT_CLIPS:
            source_count = len(frames[(PILOT_ORDINALS[0], motion, facing)])
            playback_count = source_count - 1 if motion not in {"attack", "cast", "hit", "death"} else source_count
            source_fps = int(
                8 if motion == "idle_breathe" else
                10 if motion in {"idle_wiggle", "joy", "death"} else
                14 if motion in {"fear", "attack"} else 12
            )
            output_count = max(1, int(round(playback_count * fps / source_fps)))
            for output_index in range(output_count):
                source_index = min(playback_count - 1, int(output_index * source_fps / fps))
                process.stdin.write(_preview_frame(frames, motion, facing, source_index).tobytes())
                frame_total += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait(timeout=120)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg pilot preview failed ({return_code}): {stderr[-1000:]}")
        payload = temporary.read_bytes()
    finally:
        if process.poll() is None:
            process.kill()
        temporary.unlink(missing_ok=True)
    version = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, text=True, timeout=10, check=True
    ).stdout.splitlines()[0]
    return payload, {
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_total,
        "ffmpeg_version": version,
        "loop_terminal_proof_frame_omitted": True,
    }


def compile_pilot(
    destination: Path = DEFAULT_OUTPUT,
    *,
    bank_path: Path = DEFAULT_REPAIR_BANK,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    require_disk_floor(destination, planned_bytes=512 * 1024 * 1024)
    authority = load_repair_style_authority(bank_path)
    if destination.exists() and (destination / "pilot_manifest.json").is_file():
        raise FileExistsError("repair style pilot destination is already sealed")
    destination.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Mapping[str, Any]] = {}
    identity_records: list[dict[str, Any]] = []
    keyframes: dict[tuple[int, str, str], np.ndarray] = {}
    animation_frames: dict[tuple[int, str, str], tuple[np.ndarray, ...]] = {}
    total_frames = 0
    for ordinal, family in zip(PILOT_ORDINALS, FAMILIES, strict=True):
        binding = _binding(authority, ordinal)
        sample = authority.neural_source.bank.samples[ordinal]
        if binding.family != family or sample.condition.sample_id != binding.sample_id:
            raise ValueError("repair style pilot family identity registry drifted")
        palette = authority.style_parent.palettes[binding.sample_id]
        palette_artifact = authority.style_parent.palette_artifacts[binding.sample_id]
        expected_frames = sum(
            int(_clip_audit(authority, ordinal, motion, facing)["frame_count"])
            for motion, facing in PILOT_CLIPS
        )
        rows = math.ceil(expected_frames / ATLAS_COLUMNS)
        atlases = {
            name: np.zeros((rows * IMAGE_SIZE, ATLAS_COLUMNS * IMAGE_SIZE, 4), dtype=np.uint8)
            for name in LAYER_NAMES
        }
        clip_records: list[dict[str, Any]] = []
        cursor = 0
        for motion, facing in PILOT_CLIPS:
            audit = _clip_audit(authority, ordinal, motion, facing)
            clip = reconstruct_clip(binding, audit)
            start_cell = cursor
            frame_hashes: list[str] = []
            presentation_hashes: list[list[str]] = []
            rendered_frames: list[Mapping[str, np.ndarray]] = []
            for frame in clip.frames:
                rendered = render_neural_motion_frame(
                    frame,
                    sample.condition,
                    sample.fields.aligned_sha256,
                    palette,
                    palette_artifact["sha256"],
                )
                row, column = divmod(cursor, ATLAS_COLUMNS)
                y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
                for layer in LAYER_NAMES:
                    atlases[layer][y : y + IMAGE_SIZE, x : x + IMAGE_SIZE] = rendered.layers[layer]
                frame_hashes.append(frame.sha256)
                presentation_hashes.append(list(rendered.presentation_sha256))
                rendered_frames.append(rendered.layers)
                cursor += 1
            key_index = min(len(clip.frames) - 1, max(0, len(clip.frames) // 2))
            keyframes[(ordinal, motion, facing)] = rendered_frames[key_index]["composite"].copy()
            animation_frames[(ordinal, motion, facing)] = tuple(
                layers["composite"].copy() for layers in rendered_frames
            )
            if clip.loop and any(
                not np.array_equal(rendered_frames[0][name], rendered_frames[-1][name])
                for name in LAYER_NAMES
            ):
                raise ValueError("repair style pilot rendered loop endpoint differs")
            clip_records.append(
                {
                    "id": clip.manifest["id"],
                    "motion": motion,
                    "facing": facing,
                    "fps": clip.fps,
                    "loop": clip.loop,
                    "start_cell": start_cell,
                    "frame_count": len(clip.frames),
                    "repair_audit_sha256": audit["clip_sha256"],
                    "repair_style_clip_sha256": clip.sha256,
                    "frame_sequence_sha256": sha256_bytes(canonical_json_bytes(frame_hashes)),
                    "presentation_sequence_sha256": sha256_bytes(canonical_json_bytes(presentation_hashes)),
                }
            )
        if cursor != expected_frames:
            raise ValueError("repair style pilot frame accounting mismatch")
        layer_records: dict[str, Any] = {}
        for layer, values in atlases.items():
            relative = f"identities/{family}/{binding.sample_id}/{layer}.png"
            payload = png_bytes(values)
            write_exact(destination / relative, payload)
            record = _artifact(relative, payload)
            artifacts[relative] = record
            layer_records[layer] = record
        identity_records.append(
            {
                "ordinal": ordinal,
                "sample_id": binding.sample_id,
                "family": family,
                "condition": sample.condition.as_dict(),
                "binding_sha256": binding.sha256,
                "static_palette_sha256": palette_artifact["sha256"],
                "layout": {
                    "cell_size": IMAGE_SIZE,
                    "columns": ATLAS_COLUMNS,
                    "rows": rows,
                    "frame_count": cursor,
                },
                "layers": layer_records,
                "clips": clip_records,
                "gates": {
                    "sealed_repair_audits_exact": True,
                    "all_bound_frames_exact": True,
                    "all_seven_style_layers_valid": True,
                    "palette_identity_invariant": True,
                    "loop_endpoints_exact": True,
                },
            }
        )
        total_frames += cursor

    contact_payload = _contact_sheet(keyframes)
    contact_relative = "repair_style_pilot_contact_sheet.png"
    write_exact(destination / contact_relative, contact_payload)
    artifacts[contact_relative] = _artifact(contact_relative, contact_payload)
    preview_payload, preview_encoding = _animated_preview(animation_frames, destination)
    preview_relative = "repair_style_pilot_preview.mp4"
    write_exact(destination / preview_relative, preview_payload)
    artifacts[preview_relative] = _artifact(preview_relative, preview_payload)
    manifest: dict[str, Any] = {
        "format": PILOT_FORMAT,
        "status": "ready",
        "neural_output": True,
        "scope": "five-family-representative-visual-pilot",
        "compiler": {
            "source_sha256": source_hash(),
            "repair_source_sha256": authority.bank["source"]["repair_source_sha256"],
        },
        "authority": {
            "repair_bank_sha256": authority.bank["bank_sha256"],
            "generation_manifest_sha256": authority.repair_source.generation_manifest_sha256,
            "style_manifest_sha256": authority.repair_source.style_manifest_sha256,
            "motion_stress_sha256": authority.bank["motion_result"]["stress_sha256"],
            "motion_replay_sha256": authority.bank["motion_result"]["replay_stress_sha256"],
        },
        "counts": {
            "identity_count": len(identity_records),
            "family_count": len(FAMILIES),
            "clip_count": len(identity_records) * len(PILOT_CLIPS),
            "frame_count": total_frames,
            "layer_atlas_count": len(identity_records) * len(LAYER_NAMES),
        },
        "identities": identity_records,
        "artifacts": dict(sorted(artifacts.items())),
        "preview_encoding": preview_encoding,
        "gates": {
            "sealed_all_80_bank_loaded": True,
            "two_process_sharded_motion_authorities_exact": True,
            "five_families_represented": True,
            "idles_locomotion_emotes_actions_represented": True,
            "every_consumed_motion_audit_exactly_reconstructed": True,
            "all_categorical_fields_unchanged": True,
            "all_style_layer_gates_passed": True,
            "native_48px_nearest_neighbor": True,
            "animated_preview_omits_loop_proof_frames": True,
            "disk_floor_preserved": True,
        },
    }
    manifest["pilot_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    write_exact(destination / "pilot_manifest.json", canonical_json_bytes(manifest))
    return manifest
