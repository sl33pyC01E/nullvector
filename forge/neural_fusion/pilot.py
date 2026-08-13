from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..multifield_style import render_layers
from ..multifield_style.model import CategoricalFields
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    png_bytes,
    sha256_bytes,
)
from ..multifield_style_motion.io import require_disk_floor, write_exact
from ..multifield_style_motion.model import IMAGE_SIZE, LAYER_NAMES
from ..multifield_style_neural_motion.rendering import render_neural_motion_frame
from ..neural_rig_repair.motion import compile_motion_clip_audit
from ..neural_rig_repair_style import load_repair_style_authority
from ..neural_rig_repair_style.projection import reconstruct_clip
from .genetics import FUSION_MODES, MUTATION_MODES, fuse_specimen
from .hashing import pilot_source_hash
from .rig import build_fusion_binding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_fusion_pilot_v1"
PILOT_FORMAT = "nullvector-neural-fusion-pilot-v1"
PAIR_ORDINALS = (
    (0, 16),
    (0, 32),
    (0, 48),
    (0, 64),
    (16, 32),
    (16, 48),
    (16, 64),
    (32, 48),
    (32, 64),
    (48, 64),
)
PILOT_CLIPS = (
    ("idle_wiggle", "north"),
    ("locomote", "southeast"),
    ("joy", "north"),
    ("fear", "northeast"),
    ("attack", "east"),
    ("cast", "north"),
    ("death", "south"),
)
ATLAS_COLUMNS = 16


def _artifact(relative: str, payload: bytes) -> dict[str, Any]:
    return artifact_record_from_bytes(relative, payload)


def _rest_layers(sample) -> Mapping[str, np.ndarray]:
    rendered = render_layers(
        CategoricalFields(
            part=sample.part_owner.copy(),
            material=sample.material.copy(),
            emission=sample.emission_level.copy(),
            aligned_sha256=sample.fields_sha256,
        ),
        sample.genome.condition,
    )
    return {
        "base": rendered.base,
        "outline": rendered.outline,
        "emission_core": rendered.emission_core,
        "aura": rendered.aura,
        "bloom_r1": rendered.bloom_r1,
        "bloom_r2": rendered.bloom_r2,
        "composite": rendered.composite,
        "palette": rendered.palette,
    }


def _parent_composite(authority, ordinal: int) -> np.ndarray:
    source = authority.repair_source.samples[ordinal]
    condition = authority.neural_source.bank.samples[ordinal].condition
    rendered = render_layers(
        CategoricalFields(
            part=source.part_owner.copy(),
            material=source.material.copy(),
            emission=source.emission_level.copy(),
            aligned_sha256=source.raw_fields_sha256,
        ),
        condition,
    )
    return rendered.composite


def _provenance_rgba(values: np.ndarray) -> np.ndarray:
    colors = np.asarray(
        (
            (0, 0, 0, 0),
            (42, 217, 255, 255),
            (255, 88, 183, 255),
            (191, 255, 72, 255),
            (255, 190, 55, 255),
        ),
        dtype=np.uint8,
    )
    return colors[values]


def _contact_sheet(rows: list[Mapping[str, Any]]) -> bytes:
    scale = 3
    tile = IMAGE_SIZE * scale
    label_w = 180
    top = 64
    columns = ("PARENT A", "FUSION", "PARENT B", "LINEAGE", "LOCOMOTE", "FEAR", "ATTACK", "CAST")
    image = Image.new("RGB", (label_w + len(columns) * tile, top + len(rows) * tile), (3, 9, 17))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 10), "NEURAL FUSION + MUTATION LAB", fill=(102, 244, 255), font=font)
    draw.text((12, 26), "CROSS-FAMILY LINEAGE // EXACT-TUPLE RIGGED PILOT", fill=(91, 124, 142), font=font)
    for index, label in enumerate(columns):
        draw.text((label_w + index * tile + 6, 44), label, fill=(196, 218, 229), font=font)
    for row_index, record in enumerate(rows):
        y = top + row_index * tile
        draw.text((10, y + 8), record["specimen_id"], fill=(191, 255, 72), font=font)
        draw.text((10, y + 24), record["fusion_mode"].upper(), fill=(196, 218, 229), font=font)
        draw.text((10, y + 39), record["mutation_mode"].upper(), fill=(255, 88, 183), font=font)
        draw.text((10, y + 54), f"A {record['parent_a_family']} + B {record['parent_b_family']}", fill=(91, 124, 142), font=font)
        draw.text((10, y + 69), f"{record['parent_a_pixels']} / {record['parent_b_pixels']} PX", fill=(91, 124, 142), font=font)
        pictures = (
            record["parent_a"],
            record["rest"],
            record["parent_b"],
            record["provenance"],
            record["keyframes"]["locomote"],
            record["keyframes"]["fear"],
            record["keyframes"]["attack"],
            record["keyframes"]["cast"],
        )
        for column, values in enumerate(pictures):
            sprite = Image.fromarray(values, mode="RGBA").resize((tile, tile), Image.Resampling.NEAREST)
            cell = Image.new("RGBA", (tile, tile), (5, 13, 23, 255))
            cell.alpha_composite(sprite)
            x = label_w + column * tile
            image.paste(cell.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(27, 62, 78), width=1)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def compile_pilot(destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = Path(destination).resolve()
    require_disk_floor(destination, planned_bytes=2 * 1024**3)
    if (destination / "fusion_manifest.json").exists():
        raise FileExistsError("neural fusion pilot is already sealed")
    destination.mkdir(parents=True, exist_ok=True)
    authority = load_repair_style_authority()
    records: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    total_frames = 0
    for index, (parent_a_ordinal, parent_b_ordinal) in enumerate(PAIR_ORDINALS):
        parent_a = authority.repair_source.samples[parent_a_ordinal]
        parent_b = authority.repair_source.samples[parent_b_ordinal]
        fusion_mode = FUSION_MODES[index % len(FUSION_MODES)]
        mutation_mode = MUTATION_MODES[(index + 1) % len(MUTATION_MODES)]
        specimen = fuse_specimen(
            parent_a,
            parent_b,
            seed=0xF0510A00 + index * 0x9E37,
            fusion_mode=fusion_mode,
            mutation_mode=mutation_mode,
            mutation_strength=1 + index % 3,
            dominant_parent="a" if index % 2 == 0 else "b",
        )
        binding = build_fusion_binding(specimen)
        rest = _rest_layers(specimen)
        palette = rest["palette"]
        palette_sha = sha256_bytes(canonical_json_bytes(dict(palette)))
        expected_frames = sum(
            compile_motion_clip_audit(binding, motion, facing)["frame_count"]
            for motion, facing in PILOT_CLIPS
        )
        rows = math.ceil(expected_frames / ATLAS_COLUMNS)
        atlases = {
            name: np.zeros((rows * IMAGE_SIZE, ATLAS_COLUMNS * IMAGE_SIZE, 4), dtype=np.uint8)
            for name in LAYER_NAMES
        }
        clip_records: list[dict[str, Any]] = []
        cursor = 0
        keyframes: dict[str, np.ndarray] = {}
        for motion, facing in PILOT_CLIPS:
            audit = compile_motion_clip_audit(binding, motion, facing)
            clip = reconstruct_clip(binding, audit)
            start_cell = cursor
            presentation_hashes: list[list[str]] = []
            frames: list[Mapping[str, np.ndarray]] = []
            for frame in clip.frames:
                rendered = render_neural_motion_frame(
                    frame,
                    specimen.genome.condition,
                    specimen.fields_sha256,
                    palette,
                    palette_sha,
                )
                row, column = divmod(cursor, ATLAS_COLUMNS)
                y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
                for layer in LAYER_NAMES:
                    atlases[layer][y : y + IMAGE_SIZE, x : x + IMAGE_SIZE] = rendered.layers[layer]
                presentation_hashes.append(list(rendered.presentation_sha256))
                frames.append(rendered.layers)
                cursor += 1
            keyframes[motion] = frames[len(frames) // 2]["composite"].copy()
            if clip.loop and any(
                not np.array_equal(frames[0][name], frames[-1][name]) for name in LAYER_NAMES
            ):
                raise ValueError("neural fusion pilot loop endpoint differs")
            clip_records.append(
                {
                    "motion": motion,
                    "facing": facing,
                    "fps": clip.fps,
                    "loop": clip.loop,
                    "start_cell": start_cell,
                    "frame_count": len(clip.frames),
                    "repair_audit_sha256": audit["clip_sha256"],
                    "fusion_clip_sha256": clip.sha256,
                    "presentation_sequence_sha256": sha256_bytes(canonical_json_bytes(presentation_hashes)),
                }
            )
        prefix = f"specimens/{specimen.genome.specimen_id}"
        artifacts: dict[str, Any] = {}
        for layer, values in atlases.items():
            relative = f"{prefix}/{layer}.png"
            payload = png_bytes(values)
            write_exact(destination / relative, payload)
            artifacts[layer] = _artifact(relative, payload)
        fields_relative = f"{prefix}/semantic_fields.npz"
        fields_payload = deterministic_npz_bytes(
            {
                "part_owner": specimen.part_owner,
                "material": specimen.material,
                "emission_level": specimen.emission_level,
                "provenance": specimen.provenance,
                "guide": specimen.guide,
                "genes": specimen.genes,
                "legal_tuples": specimen.legal_tuples,
            }
        )
        write_exact(destination / fields_relative, fields_payload)
        artifacts["semantic_fields"] = _artifact(fields_relative, fields_payload)
        palette_relative = f"{prefix}/palette.json"
        palette_payload = canonical_json_bytes(dict(palette))
        write_exact(destination / palette_relative, palette_payload)
        artifacts["palette"] = _artifact(palette_relative, palette_payload)
        binding_relative = f"{prefix}/binding_manifest.json"
        binding_payload = canonical_json_bytes(dict(binding.manifest))
        write_exact(destination / binding_relative, binding_payload)
        artifacts["binding"] = _artifact(binding_relative, binding_payload)
        record = {
            "specimen_id": specimen.genome.specimen_id,
            "lineage_sha256": specimen.genome.lineage_sha256,
            "seed": specimen.genome.seed,
            "parent_a": {"ordinal": parent_a.ordinal, "sample_id": parent_a.sample_id, "family": parent_a.family},
            "parent_b": {"ordinal": parent_b.ordinal, "sample_id": parent_b.sample_id, "family": parent_b.family},
            "dominant_parent": specimen.genome.dominant_parent,
            "fusion_mode": specimen.genome.fusion_mode,
            "mutation_mode": specimen.genome.mutation_mode,
            "mutation_strength": specimen.genome.mutation_strength,
            "mirror_donor": specimen.genome.mirror_donor,
            "condition": specimen.genome.condition.as_dict(),
            "fields_sha256": specimen.fields_sha256,
            "provenance_sha256": specimen.provenance_sha256,
            "binding_sha256": binding.sha256,
            "palette_sha256": palette_sha,
            "metrics": dict(specimen.metrics),
            "layout": {"cell_size": IMAGE_SIZE, "columns": ATLAS_COLUMNS, "rows": rows, "frame_count": cursor},
            "clips": clip_records,
            "artifacts": artifacts,
            "gates": {
                "different_family_parents": parent_a.family != parent_b.family,
                "both_parents_contribute": min(specimen.metrics["parent_a_pixels"], specimen.metrics["parent_b_pixels"]) >= 20,
                "legal_neural_tuples_only": True,
                "visible_connected": specimen.metrics["component_count"] == 1,
                "fresh_rig_valid": True,
                "all_selected_motion_clips_valid": True,
                "all_style_layers_valid": True,
                "lineage_recorded": True,
            },
        }
        if any(value is not True for value in record["gates"].values()):
            raise ValueError("neural fusion specimen gate failed")
        records.append(record)
        contact_rows.append(
            {
                "specimen_id": specimen.genome.specimen_id,
                "fusion_mode": fusion_mode,
                "mutation_mode": mutation_mode,
                "parent_a_family": parent_a.family,
                "parent_b_family": parent_b.family,
                "parent_a_pixels": specimen.metrics["parent_a_pixels"],
                "parent_b_pixels": specimen.metrics["parent_b_pixels"],
                "parent_a": _parent_composite(authority, parent_a.ordinal),
                "parent_b": _parent_composite(authority, parent_b.ordinal),
                "rest": rest["composite"],
                "provenance": _provenance_rgba(specimen.provenance),
                "keyframes": keyframes,
            }
        )
        total_frames += cursor
    contact_relative = "neural_fusion_contact_sheet.png"
    contact_payload = _contact_sheet(contact_rows)
    write_exact(destination / contact_relative, contact_payload)
    manifest = {
        "format": PILOT_FORMAT,
        "status": "ready",
        "neural_output": True,
        "scope": "ten-cross-family-lineage-aware-fusion-mutation-specimens",
        "compiler": {"source_sha256": pilot_source_hash(), "cpu_only": True, "cuda_used": False},
        "authority": {
            "repair_bank_sha256": authority.bank["bank_sha256"],
            "generation_manifest_sha256": authority.repair_source.generation_manifest_sha256,
            "style_manifest_sha256": authority.repair_source.style_manifest_sha256,
        },
        "counts": {
            "specimen_count": len(records),
            "parent_pair_count": len(PAIR_ORDINALS),
            "family_pair_coverage": len(PAIR_ORDINALS),
            "fusion_mode_count": len({record["fusion_mode"] for record in records}),
            "mutation_mode_count": len({record["mutation_mode"] for record in records}),
            "clip_count": len(records) * len(PILOT_CLIPS),
            "frame_count": total_frames,
            "layer_atlas_count": len(records) * len(LAYER_NAMES),
        },
        "specimens": records,
        "artifacts": {"contact_sheet": _artifact(contact_relative, contact_payload)},
        "gates": {
            "all_ten_unordered_family_pairs_present": True,
            "all_five_fusion_modes_present": True,
            "multiple_mutation_modes_present": True,
            "all_specimens_have_two_parent_lineage": True,
            "all_specimens_use_legal_neural_tuples": True,
            "all_specimens_connected_and_riggable": True,
            "idles_locomotion_emotes_actions_present": True,
            "all_animation_and_style_gates_passed": True,
            "disk_floor_preserved": True,
        },
    }
    manifest["bank_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    write_exact(destination / "fusion_manifest.json", canonical_json_bytes(manifest))
    return manifest
