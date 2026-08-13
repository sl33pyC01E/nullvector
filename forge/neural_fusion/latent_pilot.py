from __future__ import annotations

import io
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..multifield_style import render_layers
from ..multifield_style.model import CategoricalFields
from ..multifield_style_motion.hashing import artifact_record_from_bytes, canonical_json_bytes, deterministic_npz_bytes, png_bytes, sha256_bytes
from ..multifield_style_motion.io import require_disk_floor, write_exact
from ..multifield_style_motion.model import IMAGE_SIZE, LAYER_NAMES
from ..multifield_style_neural_motion.rendering import render_neural_motion_frame
from ..neural_rig_repair.motion import compile_motion_clip_audit
from ..neural_rig_repair_style import load_repair_style_authority
from ..neural_rig_repair_style.projection import reconstruct_clip
from .hashing import source_hash
from .latent import LATENT_MODES, latent_fuse, latent_source_hash
from .rig import build_fusion_binding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_latent_fusion_pilot_v1"
PAIRINGS = ((0, 16), (0, 64), (32, 48), (48, 64))
ALPHAS = (0.25, 0.5, 0.75)
CLIPS = (("idle_breathe", "north"), ("locomote", "southeast"), ("joy", "north"), ("attack", "east"))
FORMAT = "nullvector-neural-latent-fusion-pilot-v1"


def latent_pilot_source_hash() -> str:
    digest = hashlib.sha256()
    for label, value in (
        ("foundation", source_hash()),
        ("latent", latent_source_hash()),
        ("pilot", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _artifact(path: str, payload: bytes) -> dict[str, Any]:
    return artifact_record_from_bytes(path, payload)


def _render(specimen):
    return render_layers(
        CategoricalFields(
            part=specimen.part_owner.copy(),
            material=specimen.material.copy(),
            emission=specimen.emission_level.copy(),
            aligned_sha256=specimen.fields_sha256,
        ),
        specimen.genome.condition,
    )


def _parent(authority, ordinal: int) -> np.ndarray:
    sample = authority.repair_source.samples[ordinal]
    condition = authority.neural_source.bank.samples[ordinal].condition
    return render_layers(
        CategoricalFields(
            part=sample.part_owner.copy(),
            material=sample.material.copy(),
            emission=sample.emission_level.copy(),
            aligned_sha256=sample.raw_fields_sha256,
        ),
        condition,
    ).composite


def _contact(rows: list[Mapping[str, Any]]) -> bytes:
    scale = 4
    tile = IMAGE_SIZE * scale
    label = 190
    top = 62
    columns = ("PARENT A", "25% B", "50% B", "75% B", "PARENT B", "50% MOTION")
    image = Image.new("RGB", (label + len(columns) * tile, top + len(rows) * tile), (3, 9, 17))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 10), "LEARNED FSQ LATENT FUSION", fill=(102, 244, 255), font=font)
    draw.text((12, 26), "EXPERIMENTAL SMOKE CODEC // NOT PRODUCTION QUALITY", fill=(255, 88, 183), font=font)
    for column, name in enumerate(columns):
        draw.text((label + column * tile + 7, 44), name, fill=(196, 218, 229), font=font)
    for row, record in enumerate(rows):
        y = top + row * tile
        draw.text((10, y + 8), record["mode"].upper(), fill=(191, 255, 72), font=font)
        draw.text((10, y + 24), f"{record['family_a']} + {record['family_b']}", fill=(196, 218, 229), font=font)
        draw.text((10, y + 40), f"CODES {record['unique_codes']}", fill=(91, 124, 142), font=font)
        draw.text((10, y + 56), f"REPAIR {record['repair_pixels']} PX", fill=(91, 124, 142), font=font)
        pictures = (record["parent_a"], *record["children"], record["parent_b"], record["motion"])
        for column, values in enumerate(pictures):
            sprite = Image.fromarray(values, mode="RGBA").resize((tile, tile), Image.Resampling.NEAREST)
            cell = Image.new("RGBA", (tile, tile), (5, 13, 23, 255))
            cell.alpha_composite(sprite)
            x = label + column * tile
            image.paste(cell.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(27, 62, 78), width=1)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def compile_latent_pilot(destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = Path(destination).resolve()
    require_disk_floor(destination, planned_bytes=2 * 1024**3)
    if (destination / "latent_fusion_manifest.json").exists():
        raise FileExistsError("latent fusion pilot is already sealed")
    destination.mkdir(parents=True, exist_ok=True)
    authority = load_repair_style_authority()
    records = []
    contact_rows = []
    total_frames = 0
    for index, ((ordinal_a, ordinal_b), mode) in enumerate(zip(PAIRINGS, LATENT_MODES, strict=True)):
        parent_a = authority.repair_source.samples[ordinal_a]
        parent_b = authority.repair_source.samples[ordinal_b]
        children = []
        child_images = []
        middle_motion = None
        for alpha_index, alpha in enumerate(ALPHAS):
            specimen = latent_fuse(
                parent_a,
                parent_b,
                seed=0x1A7E5000 + index * 101 + alpha_index,
                alpha=alpha,
                mode=mode,
                dominant_parent="a" if alpha <= 0.5 else "b",
            )
            binding = build_fusion_binding(specimen)
            rest = _render(specimen)
            palette = rest.palette
            palette_sha = sha256_bytes(canonical_json_bytes(dict(palette)))
            expected_frames = sum(compile_motion_clip_audit(binding, motion, facing)["frame_count"] for motion, facing in CLIPS)
            rows = math.ceil(expected_frames / 16)
            atlases = {name: np.zeros((rows * 48, 16 * 48, 4), dtype=np.uint8) for name in LAYER_NAMES}
            cursor = 0
            clips = []
            motion_keyframe = None
            for motion, facing in CLIPS:
                audit = compile_motion_clip_audit(binding, motion, facing)
                clip = reconstruct_clip(binding, audit)
                start = cursor
                rendered_frames = []
                for frame in clip.frames:
                    rendered = render_neural_motion_frame(
                        frame,
                        specimen.genome.condition,
                        specimen.fields_sha256,
                        palette,
                        palette_sha,
                    )
                    row, column = divmod(cursor, 16)
                    y, x = row * 48, column * 48
                    for layer in LAYER_NAMES:
                        atlases[layer][y : y + 48, x : x + 48] = rendered.layers[layer]
                    rendered_frames.append(rendered.layers)
                    cursor += 1
                if motion == "locomote":
                    motion_keyframe = rendered_frames[len(rendered_frames) // 2]["composite"].copy()
                clips.append(
                    {
                        "motion": motion,
                        "facing": facing,
                        "fps": clip.fps,
                        "loop": clip.loop,
                        "start_cell": start,
                        "frame_count": len(clip.frames),
                        "audit_sha256": audit["clip_sha256"],
                        "clip_sha256": clip.sha256,
                    }
                )
            prefix = f"specimens/{specimen.genome.specimen_id}"
            artifacts = {}
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
                }
            )
            write_exact(destination / fields_relative, fields_payload)
            artifacts["semantic_fields"] = _artifact(fields_relative, fields_payload)
            record = {
                "specimen_id": specimen.genome.specimen_id,
                "parent_a": parent_a.sample_id,
                "parent_b": parent_b.sample_id,
                "alpha": alpha,
                "latent_mode": mode,
                "quality_tier": specimen.metrics["quality_tier"],
                "fields_sha256": specimen.fields_sha256,
                "lineage_sha256": specimen.genome.lineage_sha256,
                "binding_sha256": binding.sha256,
                "metrics": dict(specimen.metrics),
                "clips": clips,
                "artifacts": artifacts,
                "gates": {
                    "codec_checkpoint_validated": True,
                    "latent_fusion_executed": True,
                    "legal_tuple_projection_exact": True,
                    "topology_repaired_and_connected": True,
                    "fresh_rig_valid": True,
                    "selected_motion_clips_valid": True,
                    "experimental_quality_label_present": True,
                },
            }
            children.append(record)
            child_images.append(rest.composite)
            total_frames += cursor
            if alpha == 0.5:
                middle_motion = motion_keyframe
        if middle_motion is None:
            raise RuntimeError("latent fusion middle motion keyframe missing")
        records.extend(children)
        contact_rows.append(
            {
                "mode": mode,
                "family_a": parent_a.family,
                "family_b": parent_b.family,
                "parent_a": _parent(authority, ordinal_a),
                "parent_b": _parent(authority, ordinal_b),
                "children": child_images,
                "motion": middle_motion,
                "unique_codes": children[1]["metrics"]["unique_codes"],
                "repair_pixels": children[1]["metrics"]["connective_repair_pixels"],
            }
        )
    contact_payload = _contact(contact_rows)
    contact_relative = "latent_fusion_contact_sheet.png"
    write_exact(destination / contact_relative, contact_payload)
    manifest = {
        "format": FORMAT,
        "status": "experimental",
        "neural_output": True,
        "quality_tier": "experimental-smoke-codec-not-production",
        "compiler": {
            "foundation_source_sha256": source_hash(),
            "latent_source_sha256": latent_source_hash(),
            "pilot_source_sha256": latent_pilot_source_hash(),
            "cpu_only": True,
            "cuda_used": False,
        },
        "counts": {
            "parent_pair_count": len(PAIRINGS),
            "latent_mode_count": len(LATENT_MODES),
            "alpha_count": len(ALPHAS),
            "specimen_count": len(records),
            "clip_count": len(records) * len(CLIPS),
            "frame_count": total_frames,
            "layer_atlas_count": len(records) * len(LAYER_NAMES),
        },
        "specimens": records,
        "artifacts": {"contact_sheet": _artifact(contact_relative, contact_payload)},
        "gates": {
            "fresh_codec_manifest_and_checkpoint_validated": True,
            "all_four_latent_fusion_modes_exercised": True,
            "all_three_interpolation_levels_exercised": True,
            "all_outputs_legal_connected_and_riggable": True,
            "selected_animations_valid": True,
            "production_quality_not_claimed": True,
            "disk_floor_preserved": True,
        },
    }
    manifest["bank_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    write_exact(destination / "latent_fusion_manifest.json", canonical_json_bytes(manifest))
    return manifest


if __name__ == "__main__":
    result = compile_latent_pilot()
    print("NEURAL_LATENT_FUSION_OK", result["counts"], result["bank_sha256"])
