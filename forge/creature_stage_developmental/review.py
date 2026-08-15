from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from jsonschema import Draft202012Validator

from ..safety import require_disk_floor
from .contract import FAMILIES, SCHEMA_PATH, TISSUES, TRAITS, source_sha256
from .development import DevelopedOrganism, develop
from .genomes import review_genomes
from .motion import MotionPose, pose


FORMAT = "nullvector-creature-stage-developmental-review-v1"
TISSUE_COLORS = {
    "skin": (80, 207, 236), "bone": (235, 229, 194), "muscle": (239, 79, 102),
    "tendon": (244, 159, 78), "armor": (139, 158, 178), "neural": (242, 78, 188),
    "vascular": (208, 57, 74), "respiratory": (83, 225, 215), "digestive": (222, 186, 64),
    "sensor": (250, 239, 133), "storage": (177, 123, 232), "root": (119, 220, 89),
    "phase": (184, 88, 249), "machine": (239, 103, 76), "weapon": (255, 91, 91),
}


def _font(size: int):
    for path in (Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_schema(payload: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    # Dataclass serialization retains tuples in memory even though canonical
    # JSON represents them as arrays.  Validate the actual JSON data model.
    document = json.loads(json.dumps(payload, ensure_ascii=True, allow_nan=False))
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        location = "/".join(str(value) for value in first.absolute_path) or "<root>"
        raise ValueError(f"developmental review schema failed at {location}: {first.message}")


def _family_label(mix: tuple[float, ...]) -> str:
    abbreviations = ("HUM", "ANI", "PLT", "ANO", "MCH")
    active = [(abbreviations[index], int(round(value * 100))) for index, value in enumerate(mix) if value >= .01]
    return " / ".join(f"{name}{value}" for name, value in active)


def _bounds(organism: DevelopedOrganism) -> tuple[np.ndarray, np.ndarray]:
    minimum = organism.cell_xy.min(axis=0).astype(np.float32)
    maximum = organism.cell_xy.max(axis=0).astype(np.float32)
    return minimum, maximum


def _map_point(point: np.ndarray, center: tuple[float, float], scale: float, midpoint: np.ndarray) -> tuple[float, float]:
    return center[0] + float(point[0] - midpoint[0]) * scale, center[1] + float(point[1] - midpoint[1]) * scale


def _render_cells(
    draw: ImageDraw.ImageDraw,
    organism: DevelopedOrganism,
    center: tuple[float, float],
    scale: float,
    *,
    points: np.ndarray | None = None,
    midpoint: np.ndarray | None = None,
    fade_support: bool = False,
) -> None:
    minimum, maximum = _bounds(organism)
    midpoint = (minimum + maximum) * .5 if midpoint is None else midpoint
    coordinates = organism.cell_xy if points is None else points
    order = np.argsort(coordinates[:, 1])
    radius = max(1.1, scale * .36)
    for index in order:
        tissue = TISSUES[int(organism.tissue[index])]
        color = TISSUE_COLORS[tissue]
        alpha = 72 if fade_support and tissue in {"skin", "root", "phase", "machine", "armor"} else 238
        x, y = _map_point(coordinates[index], center, scale, midpoint)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(*color, alpha))
        if tissue in {"neural", "sensor"}:
            draw.point((round(x), round(y)), fill=(255, 255, 224, 255))


def _render_structure(
    draw: ImageDraw.ImageDraw,
    organism: DevelopedOrganism,
    center: tuple[float, float],
    scale: float,
    *,
    nodes: np.ndarray | None = None,
    activations: np.ndarray | None = None,
    planted_contacts: np.ndarray | None = None,
    midpoint: np.ndarray | None = None,
) -> None:
    minimum, maximum = _bounds(organism)
    midpoint = (minimum + maximum) * .5 if midpoint is None else midpoint
    node_positions = organism.skeleton_nodes if nodes is None else nodes
    for edge in organism.skeleton_edges:
        a = _map_point(node_positions[int(edge[0]), :2], center, scale, midpoint)
        b = _map_point(node_positions[int(edge[1]), :2], center, scale, midpoint)
        draw.line((*a, *b), fill=(235, 229, 194, 235), width=max(2, round(scale * .55)))
    # Flexor and extensor actuators are paired around the same load path.
    for index, muscle in enumerate(organism.muscles):
        a = node_positions[int(muscle[0]), :2]
        b = node_positions[int(muscle[1]), :2]
        direction = b - a
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        normal /= max(float(np.linalg.norm(normal)), 1e-6)
        normal *= .35 * float(muscle[3])
        pa = _map_point(a + normal, center, scale, midpoint)
        pb = _map_point(b + normal, center, scale, midpoint)
        activation = float(muscle[4]) if activations is None else float(activations[index])
        glow = int(round(95 + 160 * np.clip(activation, 0.0, 1.0)))
        color = (255, 74, 104, glow) if index % 2 == 0 else (72, 218, 255, glow)
        width = max(1, round(scale * (.22 + activation * .24)))
        draw.line((*pa, *pb), fill=color, width=width)
    for node in node_positions:
        x, y = _map_point(node[:2], center, scale, midpoint)
        r = max(1.8, scale * .48)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(244, 235, 195, 255))
    if planted_contacts is not None:
        for appendage_index in np.flatnonzero(planted_contacts):
            edge_ids = np.flatnonzero(organism.skeleton_edge_appendage == appendage_index)
            terminal = int(organism.skeleton_edges[int(edge_ids[-1]), 1])
            x, y = _map_point(node_positions[terminal, :2], center, scale, midpoint)
            r = max(2.4, scale * .72)
            draw.ellipse((x-r, y-r, x+r, y+r), outline=(133, 255, 126, 255), width=max(1, round(scale * .22)))


def _render_sheet(organisms: list[DevelopedOrganism]) -> Image.Image:
    width, height = 1800, 940
    image = Image.new("RGBA", (width, height), (3, 8, 14, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width, 62), fill=(6, 15, 24, 255))
    draw.text((24, 16), "DEVELOPMENTAL ORGANISM LAB // COMPONENT + TRAIT DIFFUSION", font=_font(25), fill=(219, 242, 249, 255))
    draw.text((width-430, 19), "CELL FIELD  /  SKELETON + MUSCLE CUTAWAY", font=_font(15), fill=(75, 229, 247, 255))
    col_width = width / 5
    row_height = (height - 62) / 2
    for index, organism in enumerate(organisms):
        family = index // 2
        row = index % 2
        left = family * col_width
        top = 62 + row * row_height
        if family:
            draw.line((left, 62, left, height), fill=(29, 55, 68, 180), width=1)
        if row:
            draw.line((left, top, left + col_width, top), fill=(25, 49, 61, 170), width=1)
        kind = "BASE PRIOR" if row == 0 else "COMPONENT GRAFT"
        draw.text((left+14, top+14), f"{FAMILIES[family].upper()} // {kind}", font=_font(14), fill=(204, 225, 232, 255))
        draw.text((left+14, top+36), _family_label(organism.genome.family_mix), font=_font(11), fill=(121, 151, 164, 255))
        minimum, maximum = _bounds(organism)
        span = float(max(maximum - minimum))
        scale = min(4.4, 106.0 / max(span, 1.0))
        center_y = top + row_height * .55
        _render_cells(draw, organism, (left + col_width * .27, center_y), scale)
        _render_cells(draw, organism, (left + col_width * .73, center_y), scale, fade_support=True)
        _render_structure(draw, organism, (left + col_width * .73, center_y), scale)
        draw.text((left+18, top+row_height-52), f"{organism.cell_count} CELLS  {len(organism.skeleton_nodes)} NODES  {len(organism.muscles)} MUSCLES", font=_font(10), fill=(121, 151, 164, 255))
        draw.text((left+18, top+row_height-34), organism.genome.genome_id.upper(), font=_font(10), fill=(75, 229, 247, 255))
    draw.text((24, height-20), "Traits are interpolated from overlapping component fields. Bone carries load; paired muscles actuate the same graph; organs occupy protected interior cells.", font=_font(12), fill=(123, 154, 166, 255))
    return image.convert("RGB")


def _render_motion_frame(organisms: list[DevelopedOrganism], poses: list[MotionPose], phase: float) -> Image.Image:
    width, height = 1800, 940
    image = Image.new("RGBA", (width, height), (3, 8, 14, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width, 62), fill=(6, 15, 24, 255))
    draw.text((24, 16), "SKELETON-DRIVEN LOCOMOTION // PAIRED MUSCLE ACTUATION", font=_font(25), fill=(219, 242, 249, 255))
    draw.text((width-330, 19), f"CYCLE {phase:0.3f}  /  12 FPS", font=_font(15), fill=(75, 229, 247, 255))
    col_width = width / 5
    row_height = (height - 62) / 2
    for index, (organism, motion) in enumerate(zip(organisms, poses, strict=True)):
        family = index // 2
        row = index % 2
        left = family * col_width
        top = 62 + row * row_height
        if family:
            draw.line((left, 62, left, height), fill=(29, 55, 68, 180), width=1)
        if row:
            draw.line((left, top, left + col_width, top), fill=(25, 49, 61, 170), width=1)
        kind = "BASE PRIOR" if row == 0 else "COMPONENT GRAFT"
        draw.text((left+14, top+14), f"{FAMILIES[family].upper()} // {kind}", font=_font(14), fill=(204, 225, 232, 255))
        minimum, maximum = _bounds(organism)
        midpoint = (minimum + maximum) * .5
        span = float(max(maximum - minimum))
        scale = min(4.4, 106.0 / max(span, 1.0))
        center_y = top + row_height * .55
        floor_y = center_y + float(maximum[1] - midpoint[1]) * scale + 8
        for center_x in (left + col_width * .27, left + col_width * .73):
            draw.ellipse((center_x-52, floor_y-7, center_x+52, floor_y+8), fill=(31, 92, 105, 55), outline=(55, 170, 183, 80))
        _render_cells(draw, organism, (left + col_width * .27, center_y), scale, points=motion.cells, midpoint=midpoint)
        _render_cells(draw, organism, (left + col_width * .73, center_y), scale, points=motion.cells, midpoint=midpoint, fade_support=True)
        _render_structure(
            draw,
            organism,
            (left + col_width * .73, center_y),
            scale,
            nodes=motion.nodes,
            activations=motion.muscle_activation,
            planted_contacts=motion.planted_contacts,
            midpoint=midpoint,
        )
        active = int(np.count_nonzero(motion.muscle_activation >= .30))
        planted = int(np.count_nonzero(motion.planted_contacts))
        draw.text((left+18, top+row_height-52), f"{active:02d} ACTIVE MUSCLES  {planted:02d} PLANTED CONTACTS", font=_font(10), fill=(121, 151, 164, 255))
        draw.text((left+18, top+row_height-34), organism.genome.genome_id.upper(), font=_font(10), fill=(75, 229, 247, 255))
    draw.text((24, height-20), "Left: living cell field. Right: identical cells with load graph, antagonistic actuators, and planted contacts. No whole-body rotation or sprite mirroring.", font=_font(12), fill=(123, 154, 166, 255))
    return image.convert("RGB")


def _motion_semantic_sha256(organisms: list[DevelopedOrganism], frame_count: int) -> str:
    digest = hashlib.sha256(b"nullvector-developmental-motion-cycle-v1\0")
    for frame in range(frame_count):
        phase = frame / frame_count
        digest.update(np.asarray([phase], dtype="<f4").tobytes())
        for organism in organisms:
            motion = pose(organism, phase)
            digest.update(organism.identity_sha256.encode("ascii") + b"\0")
            digest.update(motion.nodes.astype("<f4", copy=False).tobytes())
            digest.update(motion.cells.astype("<f4", copy=False).tobytes())
            digest.update(motion.muscle_activation.astype("<f4", copy=False).tobytes())
            digest.update(motion.planted_contacts.astype(np.uint8, copy=False).tobytes())
    return digest.hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=7)
    return buffer.getvalue()


def _render_motion_frames(organisms: list[DevelopedOrganism], frame_count: int, destination: Path | None = None) -> str:
    digest = hashlib.sha256(b"nullvector-developmental-motion-frame-stream-v1\0")
    for frame in range(frame_count):
        phase = frame / frame_count
        motions = [pose(organism, phase) for organism in organisms]
        encoded = _png_bytes(_render_motion_frame(organisms, motions, phase))
        digest.update(len(encoded).to_bytes(8, "little") + encoded)
        if destination is not None:
            (destination / f"frame_{frame:03d}.png").write_bytes(encoded)
    return digest.hexdigest()


def _encode_motion(frames: Path, staging: Path, fps: int) -> tuple[Path, Path, str]:
    ffmpeg = Path("C:/ffmpeg.exe")
    if not ffmpeg.is_file():
        resolved = shutil.which("ffmpeg")
        if resolved is None:
            raise RuntimeError("ffmpeg is required for the developmental motion review")
        ffmpeg = Path(resolved)
    pattern = str(frames / "frame_%03d.png")
    mp4 = staging / "developmental_locomotion.mp4"
    gif = staging / "developmental_locomotion.gif"
    common = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern]
    subprocess.run(common + ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    palette = "fps=%d,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" % fps
    subprocess.run(common + ["-filter_complex", palette, "-loop", "0", str(gif)], check=True)
    version = subprocess.run([str(ffmpeg), "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    return mp4, gif, version


def publish_review(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    require_disk_floor(output.parent, planned_bytes=512 * 1024**2)
    if output.exists():
        raise FileExistsError("developmental review destination already exists")
    staging = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    specimens = staging / "specimens"
    specimens.mkdir(parents=True)
    organisms = [develop(genome) for genome in review_genomes()]
    records: list[dict[str, Any]] = []
    for organism in organisms:
        path = specimens / f"{organism.genome.genome_id}.npz"
        np.savez_compressed(
            path,
            cell_xy=organism.cell_xy,
            tissue=organism.tissue,
            component_weights=organism.component_weights,
            trait_fields=organism.trait_fields,
            skeleton_nodes=organism.skeleton_nodes,
            skeleton_edges=organism.skeleton_edges,
            skeleton_edge_appendage=organism.skeleton_edge_appendage,
            skeleton_edge_side=organism.skeleton_edge_side,
            muscles=organism.muscles,
            appendage_index=organism.appendage_index,
            side=organism.side,
        )
        records.append({
            "genome": asdict(organism.genome),
            "cell_count": organism.cell_count,
            "skeleton_node_count": len(organism.skeleton_nodes),
            "skeleton_edge_count": len(organism.skeleton_edges),
            "muscle_count": len(organism.muscles),
            "identity_sha256": organism.identity_sha256,
            "artifact": {"path": path.relative_to(staging).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)},
        })
    sheet = staging / "developmental_contact_sheet.png"
    _render_sheet(organisms).save(sheet, compress_level=7)
    frame_count = 72
    fps = 12
    frames = staging / "motion_frames"
    frames.mkdir()
    frame_stream_sha256 = _render_motion_frames(organisms, frame_count, frames)
    motion_semantic_sha256 = _motion_semantic_sha256(organisms, frame_count)
    mp4, gif, ffmpeg_version = _encode_motion(frames, staging, fps)
    shutil.rmtree(frames)
    seam_drift = 0.0
    motion_metrics: list[dict[str, Any]] = []
    for organism in organisms:
        rest = pose(organism, 0.0)
        near_loop = pose(organism, 1.0 - 1e-6)
        local_seam = float(np.max(np.abs(rest.cells - near_loop.cells)))
        seam_drift = max(seam_drift, local_seam)
        quarter = pose(organism, .25)
        displacement = np.linalg.norm(quarter.cells - organism.cell_xy.astype(np.float32), axis=1)
        motion_metrics.append({
            "genome_id": organism.genome.genome_id,
            "loop_seam_max_abs": local_seam,
            "quarter_cycle_mean_cell_displacement": float(displacement.mean()),
            "quarter_cycle_max_cell_displacement": float(displacement.max()),
        })
    if seam_drift > .005:
        raise ValueError("developmental locomotion is not loop closed")
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "passed": True,
        "source_sha256": source_sha256(),
        "contracts": {
            "family": "five-prior-simplex-v1",
            "components": "graftable-developmental-sources-v1",
            "traits": "overlap-normalized-diffusion-v1",
            "skeleton": "breakable-node-edge-load-graph-v1",
            "muscle": "paired-flexor-extensor-actuator-v1",
            "orientation": "vertical-locked-2.5d-v1",
        },
        "trait_order": list(TRAITS),
        "tissue_order": list(TISSUES),
        "specimen_count": len(records),
        "specimens": records,
        "contact_sheet": {"path": sheet.name, "bytes": sheet.stat().st_size, "sha256": _sha256(sheet)},
        "motion": {
            "contract": "vertical-locked-skeleton-muscle-locomotion-v1",
            "frame_count": frame_count,
            "fps": fps,
            "loop": True,
            "semantic_sha256": motion_semantic_sha256,
            "frame_stream_sha256": frame_stream_sha256,
            "loop_seam_max_abs": seam_drift,
            "metrics": motion_metrics,
            "ffmpeg": ffmpeg_version,
            "artifacts": {
                "mp4": {"path": mp4.name, "bytes": mp4.stat().st_size, "sha256": _sha256(mp4)},
                "gif": {"path": gif.name, "bytes": gif.stat().st_size, "sha256": _sha256(gif)},
            },
        },
    }
    _validate_schema(manifest)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
    manifest_path = staging / "review_manifest.json"
    manifest_path.write_bytes(encoded)
    os.replace(staging, output)
    return {
        "passed": True,
        "output": str(output),
        "manifest_sha256": _sha256(output / "review_manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "specimen_count": len(records),
        "contact_sheet_sha256": manifest["contact_sheet"]["sha256"],
        "motion_semantic_sha256": motion_semantic_sha256,
        "motion_frame_stream_sha256": frame_stream_sha256,
    }


def validate_review(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / "review_manifest.json"
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    _validate_schema(payload)
    if raw != json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n":
        raise ValueError("developmental review manifest is not canonical")
    if payload.get("format") != FORMAT or payload.get("passed") is not True or payload.get("source_sha256") != source_sha256():
        raise ValueError("developmental review authority drifted")
    expected = [develop(genome) for genome in review_genomes()]
    if len(payload.get("specimens", [])) != len(expected):
        raise ValueError("developmental review specimen count drifted")
    for record, organism in zip(payload["specimens"], expected, strict=True):
        if record["identity_sha256"] != organism.identity_sha256 or record["cell_count"] != organism.cell_count:
            raise ValueError("developmental review semantic replay drifted")
        path = output / record["artifact"]["path"]
        if path.stat().st_size != record["artifact"]["bytes"] or _sha256(path) != record["artifact"]["sha256"]:
            raise ValueError("developmental review artifact drifted")
        with np.load(path, allow_pickle=False) as archive:
            expected_arrays = {
                "cell_xy": organism.cell_xy, "tissue": organism.tissue,
                "component_weights": organism.component_weights, "trait_fields": organism.trait_fields,
                "skeleton_nodes": organism.skeleton_nodes, "skeleton_edges": organism.skeleton_edges,
                "skeleton_edge_appendage": organism.skeleton_edge_appendage,
                "skeleton_edge_side": organism.skeleton_edge_side,
                "muscles": organism.muscles, "appendage_index": organism.appendage_index, "side": organism.side,
            }
            if set(archive.files) != set(expected_arrays) or any(not np.array_equal(archive[name], value) for name, value in expected_arrays.items()):
                raise ValueError("developmental review array replay drifted")
    sheet = output / payload["contact_sheet"]["path"]
    if sheet.stat().st_size != payload["contact_sheet"]["bytes"] or _sha256(sheet) != payload["contact_sheet"]["sha256"]:
        raise ValueError("developmental contact sheet drifted")
    motion = payload.get("motion")
    if not isinstance(motion, dict) or motion.get("loop") is not True:
        raise ValueError("developmental motion manifest drifted")
    frame_count = motion.get("frame_count")
    if type(frame_count) is not int or not 8 <= frame_count <= 240:
        raise ValueError("developmental motion frame count drifted")
    if motion.get("semantic_sha256") != _motion_semantic_sha256(expected, frame_count):
        raise ValueError("developmental motion semantic replay drifted")
    if motion.get("frame_stream_sha256") != _render_motion_frames(expected, frame_count):
        raise ValueError("developmental motion frame replay drifted")
    if float(motion.get("loop_seam_max_abs", 1.0)) > .005:
        raise ValueError("developmental motion loop gate failed")
    artifacts = motion.get("artifacts", {})
    for name in ("mp4", "gif"):
        artifact = artifacts.get(name, {})
        path = output / artifact.get("path", "")
        if not path.is_file() or path.stat().st_size != artifact.get("bytes") or _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"developmental motion {name} artifact drifted")
    return {
        "passed": True,
        "specimen_count": len(expected),
        "source_sha256": payload["source_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "contact_sheet_sha256": payload["contact_sheet"]["sha256"],
        "motion_semantic_sha256": motion["semantic_sha256"],
        "motion_frame_stream_sha256": motion["frame_stream_sha256"],
    }
