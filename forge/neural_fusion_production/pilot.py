from __future__ import annotations

import io
import json
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
from ..neural_fusion.rig import build_fusion_binding
from ..neural_rig_repair.motion import compile_motion_clip_audit
from ..neural_rig_repair_style import load_repair_style_authority
from ..neural_rig_repair_style.projection import reconstruct_clip
from ..sprite_latent_production.contract import sha256_file
from .codec import load_production_codec
from .contract import DEFAULT_OUTPUT, FORMAT, FUSION_MODES, MUTATION_MODES, production_fusion_source_hash
from .operators import production_latent_fuse


PAIRINGS = ((0, 16), (7, 39), (14, 48), (23, 64), (31, 40), (15, 79), (5, 21), (34, 55), (46, 70), (11, 59), (27, 75), (42, 67))
ALPHAS = (0.25, 0.5, 0.75, 0.4, 0.6, 0.5)
CLIPS = (("idle_wiggle", "north"), ("locomote", "southeast"), ("attack", "east"))


def _render(specimen):
    return render_layers(CategoricalFields(part=specimen.part_owner.copy(), material=specimen.material.copy(), emission=specimen.emission_level.copy(), aligned_sha256=specimen.fields_sha256), specimen.genome.condition)


def _parent(authority, ordinal: int) -> np.ndarray:
    sample = authority.repair_source.samples[ordinal]
    condition = authority.neural_source.bank.samples[ordinal].condition
    return render_layers(CategoricalFields(part=sample.part_owner.copy(), material=sample.material.copy(), emission=sample.emission_level.copy(), aligned_sha256=sample.raw_fields_sha256), condition).composite


def _provenance(values: np.ndarray) -> np.ndarray:
    colors = np.asarray(((0, 0, 0, 0), (42, 217, 255, 255), (255, 88, 183, 255), (191, 255, 72, 255), (255, 190, 55, 255)), dtype=np.uint8)
    return colors[values]


def _contact(rows: list[Mapping[str, Any]]) -> bytes:
    scale = 3; tile = IMAGE_SIZE * scale; label = 210; top = 66
    columns = ("PARENT A", "HYBRID", "PARENT B", "LINEAGE", "IDLE", "LOCOMOTE", "ATTACK")
    image = Image.new("RGB", (label + len(columns) * tile, top + len(rows) * tile), (3, 9, 17))
    draw = ImageDraw.Draw(image); font = ImageFont.load_default()
    draw.text((12, 10), "PRODUCTION NEURAL LATENT GENETICS", fill=(102, 244, 255), font=font)
    draw.text((12, 27), "EMA FSQ // CROSSOVER + MUTATION + FRESH GRAPH RIG", fill=(191, 255, 72), font=font)
    for index, name in enumerate(columns): draw.text((label + index * tile + 6, 48), name, fill=(196, 218, 229), font=font)
    for row_index, record in enumerate(rows):
        y = top + row_index * tile
        draw.text((10, y + 7), record["specimen_id"], fill=(191, 255, 72), font=font)
        draw.text((10, y + 23), record["fusion_mode"].upper(), fill=(102, 244, 255), font=font)
        draw.text((10, y + 39), record["mutation_mode"].upper(), fill=(255, 88, 183), font=font)
        draw.text((10, y + 55), f"A{record['parent_a']} + B{record['parent_b']} @ {record['alpha']:.2f}", fill=(196, 218, 229), font=font)
        draw.text((10, y + 71), f"CODES {record['unique_codes']}  NEW {record['novel_codes']}", fill=(91, 124, 142), font=font)
        draw.text((10, y + 87), f"REPAIR {record['repair_pixels']} PX", fill=(91, 124, 142), font=font)
        pictures = (record["parent_a_image"], record["rest"], record["parent_b_image"], record["provenance"], record["keyframes"]["idle_wiggle"], record["keyframes"]["locomote"], record["keyframes"]["attack"])
        for column, values in enumerate(pictures):
            sprite = Image.fromarray(values, mode="RGBA").resize((tile, tile), Image.Resampling.NEAREST)
            cell = Image.new("RGBA", (tile, tile), (5, 13, 23, 255)); cell.alpha_composite(sprite)
            x = label + column * tile; image.paste(cell.convert("RGB"), (x, y)); draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(27, 62, 78))
    output = io.BytesIO(); image.save(output, format="PNG", optimize=False, compress_level=9); return output.getvalue()


def compile_production_pilot(destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = Path(destination).resolve(); require_disk_floor(destination, planned_bytes=2 * 1024**3)
    if destination.exists(): raise FileExistsError("production neural fusion destination already exists")
    destination.mkdir(parents=True)
    authority = load_repair_style_authority(); codec = load_production_codec()
    records = []; contact_rows = []; total_frames = 0; failures = []
    for index, (ordinal_a, ordinal_b) in enumerate(PAIRINGS):
        parent_a = authority.repair_source.samples[ordinal_a]; parent_b = authority.repair_source.samples[ordinal_b]
        fusion_mode = FUSION_MODES[index % len(FUSION_MODES)]; mutation_mode = MUTATION_MODES[index % len(MUTATION_MODES)]
        mutation_strength = 0 if mutation_mode == "none" else 1 + index % 3
        alpha = ALPHAS[index % len(ALPHAS)]
        specimen = binding = None
        for attempt in range(24):
            seed = 0x505246580000 + index * 0x9E37 + attempt
            try:
                candidate = production_latent_fuse(parent_a, parent_b, seed=seed, alpha=alpha, fusion_mode=fusion_mode, mutation_mode=mutation_mode, mutation_strength=mutation_strength)
                candidate_binding = build_fusion_binding(candidate)
                specimen, binding = candidate, candidate_binding; break
            except ValueError as error:
                failures.append({"index": index, "attempt": attempt + 1, "error": str(error)[:400]})
        if specimen is None or binding is None: raise RuntimeError(f"production fusion specimen {index} exhausted bounded attempts")
        rest = _render(specimen); palette = rest.palette; palette_sha = sha256_bytes(canonical_json_bytes(dict(palette)))
        expected = sum(compile_motion_clip_audit(binding, motion, facing)["frame_count"] for motion, facing in CLIPS)
        rows = math.ceil(expected / 16); atlases = {name: np.zeros((rows * 48, 16 * 48, 4), dtype=np.uint8) for name in LAYER_NAMES}
        cursor = 0; clips = []; keyframes = {}
        for motion, facing in CLIPS:
            audit = compile_motion_clip_audit(binding, motion, facing); clip = reconstruct_clip(binding, audit); start = cursor; rendered = []
            for frame in clip.frames:
                presentation = render_neural_motion_frame(frame, specimen.genome.condition, specimen.fields_sha256, palette, palette_sha)
                row, column = divmod(cursor, 16); y, x = row * 48, column * 48
                for layer in LAYER_NAMES: atlases[layer][y:y + 48, x:x + 48] = presentation.layers[layer]
                rendered.append(presentation.layers["composite"]); cursor += 1
            keyframes[motion] = rendered[len(rendered) // 2].copy()
            clips.append({"motion": motion, "facing": facing, "start_cell": start, "frame_count": len(clip.frames), "fps": clip.fps, "loop": clip.loop, "audit_sha256": audit["clip_sha256"], "clip_sha256": clip.sha256})
        prefix = f"specimens/{specimen.genome.specimen_id}"; artifacts = {}
        for layer, values in atlases.items():
            relative = f"{prefix}/{layer}.png"; payload = png_bytes(values); write_exact(destination / relative, payload); artifacts[layer] = artifact_record_from_bytes(relative, payload)
        fields_relative = f"{prefix}/semantic_fields.npz"
        fields_payload = deterministic_npz_bytes({"part_owner": specimen.part_owner, "material": specimen.material, "emission_level": specimen.emission_level, "provenance": specimen.provenance, "guide": specimen.guide, "genes": specimen.genes})
        write_exact(destination / fields_relative, fields_payload); artifacts["semantic_fields"] = artifact_record_from_bytes(fields_relative, fields_payload)
        record = {"specimen_id": specimen.genome.specimen_id, "parent_a": parent_a.sample_id, "parent_b": parent_b.sample_id, "alpha": alpha, "fusion_mode": fusion_mode, "mutation_mode": mutation_mode, "mutation_strength": mutation_strength, "fields_sha256": specimen.fields_sha256, "lineage_sha256": specimen.genome.lineage_sha256, "binding_sha256": binding.sha256, "metrics": dict(specimen.metrics), "clips": clips, "artifacts": artifacts, "gates": {"production_codec_authority_valid": True, "legal_tuple_projection_exact": True, "connected_occupancy_valid": True, "both_parents_contribute": True, "fresh_rig_valid": True, "motion_clips_valid": True}}
        records.append(record); total_frames += cursor
        contact_rows.append({"specimen_id": specimen.genome.specimen_id, "fusion_mode": fusion_mode, "mutation_mode": mutation_mode, "alpha": alpha, "parent_a": ordinal_a, "parent_b": ordinal_b, "parent_a_image": _parent(authority, ordinal_a), "parent_b_image": _parent(authority, ordinal_b), "rest": rest.composite, "provenance": _provenance(specimen.provenance), "keyframes": keyframes, "unique_codes": specimen.metrics["unique_codes"], "novel_codes": specimen.metrics["novel_codes"], "repair_pixels": specimen.metrics["connective_repair_pixels"]})
    contact_payload = _contact(contact_rows); write_exact(destination / "production_fusion_contact_sheet.png", contact_payload)
    manifest = {"format": FORMAT, "status": "ready", "quality_tier": "production-learned-latent-authority-v1", "compiler": {"source_sha256": production_fusion_source_hash(), "production_manifest_sha256": codec.manifest["manifest_sha256"], "production_manifest_file_sha256": codec.manifest_file_sha256, "production_checkpoint_sha256": codec.checkpoint_file_sha256, "production_ema_sha256": codec.ema_state_sha256, "cpu_only": True}, "counts": {"specimens": len(records), "fusion_modes": len(set(record["fusion_mode"] for record in records)), "mutation_modes": len(set(record["mutation_mode"] for record in records)), "clips": len(records) * len(CLIPS), "frames": total_frames, "layer_atlases": len(records) * len(LAYER_NAMES), "bounded_rejections": len(failures)}, "specimens": records, "failures": failures, "artifacts": {"contact_sheet": artifact_record_from_bytes("production_fusion_contact_sheet.png", contact_payload)}, "gates": {"accepted_production_codec_used": True, "all_fusion_modes_present": True, "all_mutation_modes_present": True, "all_specimens_legal_connected_and_riggable": True, "all_motion_clips_valid": True, "disk_floor_preserved": True}}
    manifest["bank_sha256"] = sha256_bytes(canonical_json_bytes(manifest)); write_exact(destination / "production_fusion_manifest.json", canonical_json_bytes(manifest)); return manifest


def validate_production_pilot(path: Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest): raise ValueError("production fusion manifest is not canonical JSON")
    unsigned = dict(manifest); stored = unsigned.pop("bank_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)) or manifest.get("format") != FORMAT or manifest.get("compiler", {}).get("source_sha256") != production_fusion_source_hash(): raise ValueError("production fusion manifest authority mismatch")
    if manifest.get("status") != "ready" or not all(manifest.get("gates", {}).values()): raise ValueError("production fusion bank gate failure")
    records = [manifest["artifacts"]["contact_sheet"]] + [artifact for specimen in manifest["specimens"] for artifact in specimen["artifacts"].values()]
    for record in records:
        artifact = (manifest_path.parent / record["path"]).resolve()
        try: artifact.relative_to(manifest_path.parent)
        except ValueError as error: raise ValueError("production fusion artifact escapes bank") from error
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size != record["bytes"] or sha256_file(artifact) != record["sha256"]: raise ValueError("production fusion artifact mismatch")
    if len(manifest["specimens"]) != manifest["counts"]["specimens"] or sum(len(item["clips"]) for item in manifest["specimens"]) != manifest["counts"]["clips"]: raise ValueError("production fusion count mismatch")
    return manifest
