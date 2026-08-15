from __future__ import annotations

from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_developmental.contract import FAMILIES, TISSUES, source_sha256 as developmental_source_sha256
from ..creature_stage_developmental.development import DevelopedOrganism, develop
from ..creature_stage_developmental.genomes import review_genomes
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    sha256_bytes,
    sha256_file,
)
from .contract import FORMAT, LOCOMOTOR_MODES, SCHEMA_PATH, GroundedLocomotionConfig, source_sha256
from .physics import GroundedCycle, simulate_grounded_cycle


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/creature_stage_grounded_locomotion/review_v1"
MANIFEST_NAME = "grounded_locomotion_manifest.json"
MAX_ARRAY_BYTES = 128 * 1024**2
TISSUE_COLORS = {
    "skin": (244, 111, 126), "bone": (238, 237, 204), "muscle": (255, 69, 113),
    "tendon": (248, 188, 144), "armor": (113, 158, 177), "neural": (72, 225, 246),
    "vascular": (234, 52, 101), "respiratory": (123, 216, 244), "digestive": (241, 169, 63),
    "sensor": (253, 244, 105), "storage": (164, 102, 230), "root": (115, 229, 92),
    "phase": (185, 78, 255), "machine": (177, 195, 207), "weapon": (255, 131, 61),
}
MODE_COLORS = {
    "passive": (99, 120, 132), "step": (72, 229, 255), "drag": (151, 245, 82),
    "float": (199, 93, 255), "wheel": (255, 188, 70),
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _cycles(config: GroundedLocomotionConfig) -> tuple[tuple[DevelopedOrganism, ...], tuple[GroundedCycle, ...]]:
    organisms = tuple(develop(genome) for genome in review_genomes())
    cycles = tuple(simulate_grounded_cycle(organism, config) for organism in organisms)
    return organisms, cycles


def _array_payload(organisms: tuple[DevelopedOrganism, ...], cycles: tuple[GroundedCycle, ...]) -> dict[str, np.ndarray]:
    count = len(organisms)
    frames = len(cycles[0].frames)
    max_nodes = max(len(organism.skeleton_nodes) for organism in organisms)
    max_cells = max(organism.cell_count for organism in organisms)
    max_appendages = max(len(organism.genome.appendages) for organism in organisms)
    max_muscles = max(len(organism.muscles) for organism in organisms)
    max_components = max(len(organism.genome.components) for organism in organisms)
    arrays = {
        "family": np.zeros(count, dtype=np.uint8),
        "grafted": np.zeros(count, dtype=np.uint8),
        "ground_y": np.zeros(count, dtype=np.float32),
        "body_world_x": np.zeros((count, frames), dtype=np.float32),
        "body_velocity_x": np.zeros((count, frames), dtype=np.float32),
        "nodes_local": np.zeros((count, frames, max_nodes, 2), dtype=np.float32),
        "node_velocity": np.zeros((count, frames, max_nodes, 2), dtype=np.float32),
        "node_mask": np.zeros((count, max_nodes), dtype=np.uint8),
        "cells_local": np.zeros((count, frames, max_cells, 2), dtype=np.float32),
        "cell_mask": np.zeros((count, max_cells), dtype=np.uint8),
        "rest_cells": np.zeros((count, max_cells, 2), dtype=np.float32),
        "tissue": np.full((count, max_cells), 255, dtype=np.uint8),
        "appendage_owner": np.full((count, max_cells), -1, dtype=np.int16),
        "trait_fields": np.zeros((count, max_cells, 15), dtype=np.float32),
        "component_weights": np.zeros((count, max_cells, max_components), dtype=np.float32),
        "component_mask": np.zeros((count, max_components), dtype=np.uint8),
        "contact_active": np.zeros((count, frames, max_appendages), dtype=np.uint8),
        "contact_anchor_local": np.zeros((count, frames, max_appendages, 2), dtype=np.float32),
        "contact_force": np.zeros((count, frames, max_appendages, 2), dtype=np.float32),
        "appendage_mask": np.zeros((count, max_appendages), dtype=np.uint8),
        "locomotor_mode": np.zeros((count, max_appendages), dtype=np.uint8),
        "muscle_activation": np.zeros((count, frames, max_muscles), dtype=np.float32),
        "muscle_mask": np.zeros((count, max_muscles), dtype=np.uint8),
    }
    for index, (organism, cycle) in enumerate(zip(organisms, cycles, strict=True)):
        node_count, cell_count = len(organism.skeleton_nodes), organism.cell_count
        appendage_count, muscle_count = len(cycle.modes), len(organism.muscles)
        component_count = len(organism.genome.components)
        arrays["family"][index] = int(np.argmax(organism.genome.family_mix))
        arrays["grafted"][index] = int(bool(organism.genome.parent_ids))
        arrays["ground_y"][index] = cycle.ground_y
        arrays["node_mask"][index, :node_count] = 1
        arrays["cell_mask"][index, :cell_count] = 1
        arrays["appendage_mask"][index, :appendage_count] = 1
        arrays["muscle_mask"][index, :muscle_count] = 1
        arrays["component_mask"][index, :component_count] = 1
        arrays["rest_cells"][index, :cell_count] = organism.cell_xy
        arrays["tissue"][index, :cell_count] = organism.tissue
        arrays["appendage_owner"][index, :cell_count] = organism.appendage_index
        arrays["trait_fields"][index, :cell_count] = organism.trait_fields
        arrays["component_weights"][index, :cell_count, :component_count] = organism.component_weights
        arrays["locomotor_mode"][index, :appendage_count] = [LOCOMOTOR_MODES.index(mode) for mode in cycle.modes]
        for frame_index, frame in enumerate(cycle.frames):
            arrays["body_world_x"][index, frame_index] = frame.body_world_x
            arrays["body_velocity_x"][index, frame_index] = frame.body_velocity_x
            arrays["nodes_local"][index, frame_index, :node_count] = frame.nodes_local
            arrays["node_velocity"][index, frame_index, :node_count] = frame.node_velocity
            arrays["cells_local"][index, frame_index, :cell_count] = frame.cells_local
            arrays["contact_active"][index, frame_index, :appendage_count] = frame.contact_active
            anchor = frame.contact_anchor_world.copy()
            anchor[:, 0] -= frame.body_world_x
            anchor[~frame.contact_active] = 0.0
            arrays["contact_anchor_local"][index, frame_index, :appendage_count] = anchor
            arrays["contact_force"][index, frame_index, :appendage_count] = frame.contact_force
            arrays["muscle_activation"][index, frame_index, :muscle_count] = frame.muscle_activation
    return arrays


def _record(organism: DevelopedOrganism, cycle: GroundedCycle) -> dict[str, Any]:
    return {
        "genome_id": organism.genome.genome_id,
        "family_id": int(np.argmax(organism.genome.family_mix)),
        "family": FAMILIES[int(np.argmax(organism.genome.family_mix))],
        "grafted": bool(organism.genome.parent_ids),
        "parent_ids": list(organism.genome.parent_ids),
        "organism_identity_sha256": organism.identity_sha256,
        "cycle_identity_sha256": cycle.identity_sha256,
        "primary_mode": cycle.primary_mode,
        "locomotor_modes": list(cycle.modes),
        "cell_count": organism.cell_count,
        "node_count": len(organism.skeleton_nodes),
        "appendage_count": len(organism.genome.appendages),
        "diagnostics": {
            "distance_px": round(cycle.distance_px, 9),
            "average_speed_px_per_frame": round(cycle.average_speed_px_per_frame, 9),
            "loop_seam_max_abs": round(cycle.loop_seam_max_abs, 9),
            "maximum_edge_strain": round(cycle.maximum_edge_strain, 9),
            "maximum_contact_slip_px": round(cycle.maximum_contact_slip_px, 9),
            "traction_work": round(cycle.traction_work, 9),
            "vertical_axis_max_degrees": round(cycle.vertical_axis_max_degrees, 9),
        },
    }


def _gates(organisms: tuple[DevelopedOrganism, ...], cycles: tuple[GroundedCycle, ...]) -> dict[str, bool]:
    base = cycles[::2]
    return {
        "all_five_families": [int(np.argmax(item.genome.family_mix)) for item in organisms[::2]] == list(range(5)),
        "base_mode_authority": [item.primary_mode for item in base] == ["step", "step", "drag", "float", "wheel"],
        "all_entities_advance": all(item.distance_px > .25 for item in cycles),
        "ground_reaction_produces_motion": all(item.traction_work > .1 for item in cycles if item.primary_mode != "float"),
        "floating_anomaly_has_no_ground_contact": not any(bool(frame.contact_active.any()) for frame in base[3].frames) and base[3].traction_work == 0.0,
        "contacts_hold_without_sliding": max(item.maximum_contact_slip_px for item in cycles) < .05,
        "limb_tethers_preserve_length": max(item.maximum_edge_strain for item in cycles) < .12,
        "local_cycles_close": max(item.loop_seam_max_abs for item in cycles) < .002,
        "vertical_2_5d_axis_preserved": max(item.vertical_axis_max_degrees for item in cycles) < 5.0,
        "rare_cross_family_locomotors_survive": {"step", "drag"}.issubset(set(cycles[1].modes)) and "step" in cycles[7].modes,
        "cell_correspondence_preserved": all(len(frame.cells_local) == organism.cell_count for organism, cycle in zip(organisms, cycles, strict=True) for frame in cycle.frames),
    }


def _render_frame(organisms: tuple[DevelopedOrganism, ...], cycles: tuple[GroundedCycle, ...], frame_index: int) -> Image.Image:
    panel_w, panel_h, header = 300, 270, 64
    image = Image.new("RGB", (panel_w * 5, header + panel_h * 2), (3, 8, 14))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, header), fill=(6, 17, 25, 255))
    draw.text((20, 12), "GROUNDED CELLULAR LOCOMOTION // CONTACT + TETHER AUTHORITY", font=_font(22), fill=(220, 243, 248, 255))
    draw.text((image.width - 190, 18), f"PHASE {frame_index:02d}/71", font=_font(15), fill=(77, 226, 247, 255))
    for index, (organism, cycle) in enumerate(zip(organisms, cycles, strict=True)):
        column, row = index // 2, index % 2
        x0, y0 = column * panel_w, header + row * panel_h
        frame = cycle.frames[frame_index]
        accent = MODE_COLORS[cycle.primary_mode]
        current_distance = frame.body_world_x - cycle.frames[0].body_world_x
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline=(26, 65, 78, 255), width=1)
        draw.text((x0 + 10, y0 + 8), f"{organism.genome.genome_id.upper()}", font=_font(12), fill=(220, 239, 244, 255))
        draw.text((x0 + 10, y0 + 27), f"{cycle.primary_mode.upper()}  X={current_distance:05.2f}/{cycle.distance_px:05.2f}px  V={frame.body_velocity_x:+.3f}", font=_font(9), fill=(*accent, 255))
        cx, cy, scale = x0 + panel_w * .5, y0 + 132, 4.4
        ground = cy + cycle.ground_y * scale
        draw.ellipse((cx - 72, ground - 7, cx + 72, ground + 8), fill=(*accent, 20), outline=(*accent, 80), width=1)
        first_tick = math.floor((frame.body_world_x - 34.0) / 8.0) * 8.0
        for world_tick in np.arange(first_tick, frame.body_world_x + 36.0, 8.0):
            tick_x = cx + (float(world_tick) - frame.body_world_x) * scale
            draw.line((tick_x, ground + 4, tick_x, ground + 9), fill=(*accent, 70), width=1)
        if cycle.primary_mode == "float":
            for radius in (35, 52, 70):
                draw.ellipse((cx-radius, cy-radius*.42, cx+radius, cy+radius*.42), outline=(*accent, 40), width=1)
        for left, right in organism.skeleton_edges:
            a, b = frame.nodes_local[int(left)], frame.nodes_local[int(right)]
            draw.line((cx+a[0]*scale, cy+a[1]*scale, cx+b[0]*scale, cy+b[1]*scale), fill=(70, 216, 242, 115), width=2)
        for cell_index, (x, y) in enumerate(frame.cells_local):
            color = TISSUE_COLORS[TISSUES[int(organism.tissue[cell_index])]]
            px, py = cx + float(x)*scale, cy + float(y)*scale
            draw.rectangle((px-1.55, py-1.55, px+1.55, py+1.55), fill=(*color, 225))
        for appendage_index in np.flatnonzero(frame.contact_active):
            terminal_edges = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
            terminal = int(organism.skeleton_edges[int(terminal_edges[-1]), 1])
            tip = frame.nodes_local[terminal]
            anchor = frame.contact_anchor_world[appendage_index].copy()
            anchor[0] -= frame.body_world_x
            tx, ty = cx+tip[0]*scale, cy+tip[1]*scale
            ax, ay = cx+anchor[0]*scale, cy+anchor[1]*scale
            draw.line((tx, ty, ax, ay), fill=(*accent, 180), width=2)
            draw.ellipse((ax-4, ay-2, ax+4, ay+2), fill=(*accent, 230))
            force = frame.contact_force[appendage_index]
            draw.line((ax, ay, ax+float(force[0])*45, ay-float(force[1])*45), fill=(255, 246, 164, 220), width=2)
        progress = np.clip(current_distance / max(cycle.distance_px, 1e-6), 0.0, 1.0)
        draw.rectangle((x0+10, y0+panel_h-18, x0+panel_w-10, y0+panel_h-12), fill=(16, 38, 46, 255))
        draw.rectangle((x0+10, y0+panel_h-18, x0+10+(panel_w-20)*progress, y0+panel_h-12), fill=(*accent, 210))
    return image


def _contact_sheet(organisms: tuple[DevelopedOrganism, ...], cycles: tuple[GroundedCycle, ...]) -> bytes:
    snapshots = []
    for frame_index in (0, 12, 24, 36, 48, 60):
        frame = _render_frame(organisms, cycles, frame_index)
        snapshots.append(frame.resize((750, 302), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (2250, 604), (3, 8, 14))
    for index, frame in enumerate(snapshots):
        sheet.paste(frame, ((index % 3) * 750, (index // 3) * 302))
    stream = BytesIO(); sheet.save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def build_review(
    output: Path = DEFAULT_OUTPUT,
    *,
    config: GroundedLocomotionConfig | None = None,
    visually_inspected: bool = False,
) -> dict[str, Any]:
    output = Path(output).resolve()
    config = config or GroundedLocomotionConfig()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        organisms, cycles = _cycles(config)
        arrays_payload = deterministic_npz_bytes(_array_payload(organisms, cycles))
        contact_payload = _contact_sheet(organisms, cycles)
        (stage / "grounded_cycles.npz").write_bytes(arrays_payload)
        (stage / "grounded_contact_sheet.png").write_bytes(contact_payload)
        frames_dir = stage / "frames"; frames_dir.mkdir()
        for frame_index in range(config.frame_count):
            _render_frame(organisms, cycles, frame_index).save(frames_dir / f"frame_{frame_index:03d}.png", compress_level=9)
        gif_path = stage / "grounded_locomotion.gif"
        mp4_path = stage / "grounded_locomotion.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "18", "-i", str(frames_dir / "frame_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "18", "-i", str(frames_dir / "frame_%03d.png"), "-vf", "fps=18,scale=1100:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", str(gif_path)], check=True)
        shutil.rmtree(frames_dir)
        gates = _gates(organisms, cycles)
        artifacts = {
            "arrays": artifact_record_from_bytes("grounded_cycles.npz", arrays_payload),
            "contact_sheet": artifact_record_from_bytes("grounded_contact_sheet.png", contact_payload),
            "gif": artifact_record_from_bytes("grounded_locomotion.gif", gif_path.read_bytes()),
            "mp4": artifact_record_from_bytes("grounded_locomotion.mp4", mp4_path.read_bytes()),
        }
        report: dict[str, Any] = {
            "format": FORMAT,
            "status": "passed" if all(gates.values()) else "failed-quality",
            "source_sha256": source_sha256(),
            "developmental_source_sha256": developmental_source_sha256(),
            "config": config.to_dict(),
            "scope": {"organisms": 10, "families": 5, "base_organisms": 5, "grafted_organisms": 5, "frames_per_cycle": config.frame_count},
            "cycles": [_record(organism, cycle) for organism, cycle in zip(organisms, cycles, strict=True)],
            "artifacts": artifacts,
            "gates": gates,
            "visually_inspected": bool(visually_inspected),
            "promotion_ready": all(gates.values()) and bool(visually_inspected),
        }
        report["semantic_sha256"] = sha256_bytes(canonical_json_bytes(report))
        (stage / MANIFEST_NAME).write_bytes(canonical_json_bytes(report))
        os.replace(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return validate_review(output, replay=False)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_ARRAY_BYTES:
        raise ValueError("grounded locomotion array artifact is missing, linked, or oversized")
    with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def validate_review(output: Path, *, replay: bool = True) -> dict[str, Any]:
    output = Path(output).resolve(); manifest_path = output / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= 2 * 1024**2:
        raise ValueError("grounded locomotion manifest is missing, linked, or oversized")
    raw = manifest_path.read_bytes(); report = json.loads(raw)
    errors = sorted(Draft202012Validator(json.loads(SCHEMA_PATH.read_bytes())).iter_errors(report), key=lambda error: list(error.path))
    if raw != canonical_json_bytes(report) or errors:
        detail = errors[0].message if errors else "canonical bytes"
        raise ValueError(f"grounded locomotion manifest drifted: {detail}")
    if (
        report["source_sha256"] != source_sha256()
        or report["developmental_source_sha256"] != developmental_source_sha256()
        or report["config"] != GroundedLocomotionConfig().to_dict()
        or report["semantic_sha256"] != sha256_bytes(canonical_json_bytes({key: value for key, value in report.items() if key != "semantic_sha256"}))
        or (report["status"] == "passed") is not all(report["gates"].values())
        or report["promotion_ready"] is not (all(report["gates"].values()) and report["visually_inspected"])
    ):
        raise ValueError("grounded locomotion authority drifted")
    for artifact in report["artifacts"].values():
        path = output / artifact["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("grounded locomotion artifact drifted")
    if replay:
        config = GroundedLocomotionConfig(**report["config"])
        organisms, cycles = _cycles(config)
        expected_records = [_record(organism, cycle) for organism, cycle in zip(organisms, cycles, strict=True)]
        if expected_records != report["cycles"] or _gates(organisms, cycles) != report["gates"]:
            raise ValueError("grounded locomotion semantic replay drifted")
        expected_arrays = _array_payload(organisms, cycles)
        stored_arrays = _load_arrays(output / report["artifacts"]["arrays"]["path"])
        if stored_arrays.keys() != expected_arrays.keys() or any(not np.array_equal(stored_arrays[name], expected_arrays[name]) for name in expected_arrays):
            raise ValueError("grounded locomotion array replay drifted")
        if deterministic_npz_bytes(expected_arrays) != (output / report["artifacts"]["arrays"]["path"]).read_bytes():
            raise ValueError("grounded locomotion archive byte replay drifted")
        if _contact_sheet(organisms, cycles) != (output / report["artifacts"]["contact_sheet"]["path"]).read_bytes():
            raise ValueError("grounded locomotion contact sheet replay drifted")
    return {
        "passed": report["status"] == "passed",
        "promotion_ready": report["promotion_ready"],
        "organisms": report["scope"]["organisms"],
        "frames": report["scope"]["frames_per_cycle"],
        "semantic_sha256": report["semantic_sha256"],
        "gates": report["gates"],
    }
