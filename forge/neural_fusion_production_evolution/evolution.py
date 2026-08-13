from __future__ import annotations

import io
import json
import math
from pathlib import Path, PurePosixPath
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
from ..multifield_style_motion.model import LAYER_NAMES
from ..multifield_style_neural_motion.rendering import render_neural_motion_frame
from ..neural_fusion.model import FusionSpecimen
from ..neural_fusion.rig import build_fusion_binding
from ..neural_fusion_production.contract import FUSION_MODES, MUTATION_MODES
from ..neural_fusion_production.operators import production_latent_fuse
from ..neural_fusion_production.pilot import validate_production_pilot
from ..neural_rig_bridge.hashing import aligned_fields_hash
from ..neural_rig_repair.model import RepairSourceSample
from ..neural_rig_repair.motion import compile_motion_clip_audit
from ..neural_rig_repair_style import load_repair_style_authority
from ..neural_rig_repair_style.projection import reconstruct_clip
from ..sprite_latent_production.contract import sha256_file
from .contract import DEFAULT_FOUNDERS, DEFAULT_OUTPUT, FORMAT, evolution_source_hash


GENERATION_SIZES = (30, 36, 42)
SURVIVORS = 12
CLIPS = (("idle_wiggle", "north"), ("locomote", "southeast"), ("attack", "east"))
MIN_FREE_BYTES = 100 * 1024**3


def _safe_artifact(root: Path, record: Mapping[str, Any]) -> Path:
    text = str(record.get("path", ""))
    relative = PurePosixPath(text)
    if not text or "\\" in text or relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("unsafe production evolution artifact path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("production evolution artifact escapes bank") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError("production evolution artifact is missing or a symlink")
    if path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
        raise ValueError("production evolution artifact integrity mismatch")
    return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("production evolution semantic archive exceeds bound")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"part_owner", "material", "emission_level", "provenance", "guide", "genes"}:
            raise ValueError("production evolution semantic archive member contract")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _sample(
    *, specimen_id: str, ordinal: int, condition_source: RepairSourceSample,
    part: np.ndarray, material: np.ndarray, emission: np.ndarray,
    guide: np.ndarray, genes: np.ndarray, fields_sha256: str, provenance_path: Path,
) -> RepairSourceSample:
    if part.shape != (48, 48) or material.shape != (48, 48) or emission.shape != (48, 48):
        raise ValueError("production evolution parent field shape")
    if guide.shape != (8, 48, 48) or genes.shape != (24,):
        raise ValueError("production evolution parent conditioning shape")
    if any(values.dtype != np.uint8 for values in (part, material, emission)) or guide.dtype != np.float32 or genes.dtype != np.float32:
        raise ValueError("production evolution parent dtype")
    if aligned_fields_hash(part, material, emission) != fields_sha256:
        raise ValueError("production evolution parent field hash")
    sample_seed = int.from_bytes(bytes.fromhex(sha256_bytes(specimen_id.encode("utf-8")))[:4], "big") & 0x7FFFFFFF
    return RepairSourceSample(
        sample_id=specimen_id, ordinal=ordinal, family=condition_source.family,
        family_id=condition_source.family_id, subtype_id=condition_source.subtype_id,
        role_id=condition_source.role_id, corpus_seed=0, sample_seed=sample_seed,
        part_owner=part.copy(), material=material.copy(), emission_level=emission.copy(),
        guide=guide.copy(), genes=genes.copy(), legal_tuples=condition_source.legal_tuples.copy(),
        raw_manifest_path=provenance_path, raw_manifest_bytes=provenance_path.stat().st_size,
        raw_manifest_sha256=sha256_file(provenance_path), raw_archive_path=provenance_path,
        raw_archive_bytes=provenance_path.stat().st_size, raw_archive_sha256=sha256_file(provenance_path),
        raw_fields_sha256=fields_sha256, compiled_fields_sha256=fields_sha256,
        static_palette_sha256="0" * 64,
    )


def _load_founders(manifest_path: Path) -> tuple[dict[str, Any], list[RepairSourceSample], Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = validate_production_pilot(manifest_path)
    authority = load_repair_style_authority()
    by_id = {sample.sample_id: sample for sample in authority.repair_source.samples}
    founders = []
    for index, record in enumerate(manifest["specimens"]):
        parent_a = by_id[str(record["parent_a"])]
        parent_b = by_id[str(record["parent_b"])]
        dominant = parent_a if float(record["alpha"]) < 0.5 else parent_b
        semantic_path = _safe_artifact(manifest_path.parent, record["artifacts"]["semantic_fields"])
        arrays = _load_npz(semantic_path)
        founders.append(_sample(
            specimen_id=str(record["specimen_id"]), ordinal=10_000 + index,
            condition_source=dominant, part=arrays["part_owner"], material=arrays["material"],
            emission=arrays["emission_level"], guide=arrays["guide"], genes=arrays["genes"],
            fields_sha256=str(record["fields_sha256"]), provenance_path=manifest_path,
        ))
    if len(founders) != 12 or len({sample.raw_fields_sha256 for sample in founders}) != 12:
        raise ValueError("production evolution founder census/uniqueness")
    return manifest, founders, authority


def _as_parent(specimen: FusionSpecimen, ordinal: int, provenance_path: Path) -> RepairSourceSample:
    condition = specimen.genome.condition
    basis = RepairSourceSample(
        sample_id=specimen.genome.specimen_id, ordinal=ordinal, family=condition.morphology_name,
        family_id=condition.morphology_id, subtype_id=condition.subtype_id, role_id=condition.role_id,
        corpus_seed=0, sample_seed=condition.sample_seed, part_owner=specimen.part_owner.copy(),
        material=specimen.material.copy(), emission_level=specimen.emission_level.copy(),
        guide=specimen.guide.copy(), genes=specimen.genes.copy(), legal_tuples=specimen.legal_tuples.copy(),
        raw_manifest_path=provenance_path, raw_manifest_bytes=0, raw_manifest_sha256="0" * 64,
        raw_archive_path=provenance_path, raw_archive_bytes=0, raw_archive_sha256="0" * 64,
        raw_fields_sha256=specimen.fields_sha256, compiled_fields_sha256=specimen.fields_sha256,
        static_palette_sha256="0" * 64,
    )
    return basis


def _codes(value: RepairSourceSample | FusionSpecimen) -> np.ndarray:
    return value.part_owner.astype(np.int32) * 40 + value.material.astype(np.int32) * 4 + value.emission_level.astype(np.int32)


def _score(specimen: FusionSpecimen, parents: tuple[RepairSourceSample, RepairSourceSample], archive: list[RepairSourceSample], audits: list[Mapping[str, Any]]) -> dict[str, float]:
    visible = specimen.part_owner != 0
    padded = np.pad(visible, 1)
    interior = np.logical_and.reduce([padded[1 + dy:49 + dy, 1 + dx:49 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    boundary = float((visible & ~interior).sum() / max(1, visible.sum()))
    emission = float(((specimen.emission_level > 0) & visible).sum() / max(1, visible.sum()))
    symmetry = float(np.mean(np.abs(visible.astype(np.float32) - np.fliplr(visible).astype(np.float32))))
    child = _codes(specimen)
    hamming_a = float(np.mean(child != _codes(parents[0])))
    hamming_b = float(np.mean(child != _codes(parents[1])))
    archive_novelty = min(float(np.mean(child != _codes(other))) for other in archive)
    contribution_a = float(specimen.metrics["parent_a_attributed_pixels"])
    contribution_b = float(specimen.metrics["parent_b_attributed_pixels"])
    ancestry_balance = 1.0 - abs(contribution_a - contribution_b) / max(1.0, contribution_a + contribution_b)
    motion = float(np.mean([float(audit["motion_strength"]) for audit in audits]))
    occupancy = float(visible.mean())
    unique_codes = min(1.0, float(specimen.metrics["unique_codes"]) / 180.0)
    novel_codes = min(1.0, float(specimen.metrics["novel_codes"]) / 90.0)
    values = {
        "ancestry_balance": ancestry_balance,
        "parent_phenotype_novelty": min(hamming_a, hamming_b),
        "archive_novelty": archive_novelty,
        "boundary_readability": 1.0 - min(1.0, abs(boundary - 0.30) / 0.30),
        "emission_balance": 1.0 - min(1.0, abs(emission - 0.18) / 0.18),
        "controlled_asymmetry": 1.0 - min(1.0, abs(symmetry - 0.20) / 0.20),
        "occupancy_balance": 1.0 - min(1.0, abs(occupancy - 0.25) / 0.25),
        "latent_code_diversity": unique_codes,
        "latent_code_novelty": novel_codes,
        "motion_strength": motion,
    }
    total = (
        0.12 * values["ancestry_balance"] + 0.13 * values["parent_phenotype_novelty"]
        + 0.15 * values["archive_novelty"] + 0.10 * values["boundary_readability"]
        + 0.07 * values["emission_balance"] + 0.07 * values["controlled_asymmetry"]
        + 0.06 * values["occupancy_balance"] + 0.09 * values["latent_code_diversity"]
        + 0.09 * values["latent_code_novelty"] + 0.12 * values["motion_strength"]
        - min(0.08, float(specimen.metrics["connective_repair_pixels"]) / 500.0)
    )
    return {**{key: round(float(value), 9) for key, value in values.items()}, "score": round(total, 9)}


def _select(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: (-float(item["score"]["score"]), item["specimen"].genome.specimen_id))
    selected: list[dict[str, Any]] = []
    family_counts = {family_id: 0 for family_id in range(5)}
    diversity_floor = 0.06

    def diverse(candidate: dict[str, Any]) -> bool:
        code = _codes(candidate["specimen"])
        return not selected or all(float(np.mean(code != _codes(other["specimen"]))) >= diversity_floor for other in selected)

    def take(candidate: dict[str, Any]) -> None:
        selected.append(candidate)
        family_id = candidate["specimen"].genome.condition.morphology_id
        family_counts[family_id] += 1

    # One paired champion for each fusion operator. Candidate planning maps
    # every fusion index bijectively to a mutation operator. Prefer a family
    # with fewer than two representatives so operator coverage cannot silently
    # turn into morphology collapse.
    for mode in FUSION_MODES:
        pool = [item for item in ordered if item["fusion_mode"] == mode and item not in selected and diverse(item)]
        champion = next((item for item in pool if family_counts[item["specimen"].genome.condition.morphology_id] < 2), None)
        if champion is None:
            champion = next((item for item in pool if family_counts[item["specimen"].genome.condition.morphology_id] < 3), None)
        if champion is None:
            raise ValueError(f"production evolution lacks diverse fusion operator {mode}")
        take(champion)
    # Bring every morphology to two representatives before optimizing the last
    # two slots. This makes the balance constraint objective rather than a
    # condition-label fig leaf.
    for family_id in range(5):
        while family_counts[family_id] < 2:
            champion = next((item for item in ordered if item not in selected and item["specimen"].genome.condition.morphology_id == family_id and diverse(item)), None)
            if champion is None:
                raise ValueError(f"production evolution lacks two diverse family {family_id} survivors")
            take(champion)
    for candidate in ordered:
        if candidate in selected:
            continue
        family_id = candidate["specimen"].genome.condition.morphology_id
        if family_counts[family_id] >= 3 or not diverse(candidate):
            continue
        take(candidate)
        if len(selected) == SURVIVORS:
            break
    if len(selected) != SURVIVORS:
        raise ValueError("production evolution cannot satisfy balanced diverse survivor census")
    selected.sort(key=lambda item: (-float(item["score"]["score"]), item["specimen"].genome.specimen_id))
    return selected


def _candidate(
    parent_a: RepairSourceSample, parent_b: RepairSourceSample, *, generation: int,
    candidate_index: int, attempt: int, fusion_mode: str, mutation_mode: str,
    alpha: float, archive: list[RepairSourceSample],
) -> dict[str, Any]:
    seed = (0x7A110000 + generation * 0x100000 + candidate_index * 0x9E37 + attempt * 0x10001) & 0xFFFFFFFF
    strength = 0 if mutation_mode == "none" else 1 + ((candidate_index + generation + attempt) % 3)
    specimen = production_latent_fuse(
        parent_a, parent_b, seed=seed, alpha=alpha, fusion_mode=fusion_mode,
        mutation_mode=mutation_mode, mutation_strength=strength,
        dominant_parent="a" if candidate_index < 5 else "auto",
    )
    binding = build_fusion_binding(specimen)
    audits = [compile_motion_clip_audit(binding, motion, facing) for motion, facing in CLIPS]
    score = _score(specimen, (parent_a, parent_b), archive, audits)
    return {
        "specimen": specimen, "binding": binding, "audits": audits, "score": score,
        "parent_ids": [parent_a.sample_id, parent_b.sample_id], "seed": seed,
        "alpha": round(alpha, 6), "fusion_mode": fusion_mode,
        "mutation_mode": mutation_mode, "mutation_strength": strength,
    }


def _render_selected(destination: Path, generation: int, rank: int, record: dict[str, Any]) -> dict[str, Any]:
    specimen: FusionSpecimen = record["specimen"]
    binding = record["binding"]
    rest = render_layers(CategoricalFields(
        part=specimen.part_owner.copy(), material=specimen.material.copy(), emission=specimen.emission_level.copy(),
        aligned_sha256=specimen.fields_sha256,
    ), specimen.genome.condition)
    palette = rest.palette; palette_sha = sha256_bytes(canonical_json_bytes(dict(palette)))
    frame_total = sum(int(audit["frame_count"]) for audit in record["audits"])
    rows = math.ceil(frame_total / 16)
    atlases = {name: np.zeros((rows * 48, 16 * 48, 4), dtype=np.uint8) for name in LAYER_NAMES}
    clips = []; cursor = 0
    for audit in record["audits"]:
        clip = reconstruct_clip(binding, audit); start = cursor
        for frame in clip.frames:
            presentation = render_neural_motion_frame(frame, specimen.genome.condition, specimen.fields_sha256, palette, palette_sha)
            row, column = divmod(cursor, 16); y, x = row * 48, column * 48
            for layer in LAYER_NAMES:
                atlases[layer][y:y + 48, x:x + 48] = presentation.layers[layer]
            cursor += 1
        clips.append({
            "motion": clip.motion, "facing": clip.facing, "start_cell": start,
            "frame_count": len(clip.frames), "fps": clip.fps, "loop": clip.loop,
            "clip_sha256": clip.sha256, "audit_sha256": audit["clip_sha256"],
        })
    prefix = f"generation_{generation}/{rank:02d}_{specimen.genome.specimen_id}"
    artifacts: dict[str, Any] = {}
    for layer, values in atlases.items():
        relative = f"{prefix}/{layer}.png"; payload = png_bytes(values)
        write_exact(destination / relative, payload); artifacts[layer] = artifact_record_from_bytes(relative, payload)
    fields_payload = deterministic_npz_bytes({
        "part_owner": specimen.part_owner, "material": specimen.material,
        "emission_level": specimen.emission_level, "provenance": specimen.provenance,
        "guide": specimen.guide, "genes": specimen.genes,
    })
    fields_relative = f"{prefix}/semantic_fields.npz"; write_exact(destination / fields_relative, fields_payload)
    artifacts["semantic_fields"] = artifact_record_from_bytes(fields_relative, fields_payload)
    binding_payload = canonical_json_bytes(dict(binding.manifest))
    binding_relative = f"{prefix}/binding_manifest.json"; write_exact(destination / binding_relative, binding_payload)
    artifacts["binding"] = artifact_record_from_bytes(binding_relative, binding_payload)
    return {
        "generation": generation, "rank": rank, "specimen_id": specimen.genome.specimen_id,
        "family": specimen.genome.condition.morphology_name,
        "family_id": specimen.genome.condition.morphology_id,
        "subtype_id": specimen.genome.condition.subtype_id, "role_id": specimen.genome.condition.role_id,
        "parent_ids": record["parent_ids"], "seed": record["seed"], "alpha": record["alpha"],
        "fusion_mode": record["fusion_mode"], "mutation_mode": record["mutation_mode"],
        "mutation_strength": record["mutation_strength"], "fields_sha256": specimen.fields_sha256,
        "lineage_sha256": specimen.genome.lineage_sha256, "binding_sha256": binding.sha256,
        "score": record["score"], "metrics": dict(specimen.metrics),
        "layout": {"cell_size": 48, "columns": 16, "rows": rows, "frame_count": frame_total},
        "clips": clips, "artifacts": artifacts,
    }


def _contact(records: list[Mapping[str, Any]], destination: Path) -> bytes:
    scale = 3; tile = 48 * scale; label = 100; top = 62
    image = Image.new("RGB", (label + SURVIVORS * tile, top + 3 * (tile + 26)), (3, 9, 17))
    draw = ImageDraw.Draw(image); font = ImageFont.load_default()
    draw.text((12, 10), "PRODUCTION LATENT EVOLUTION // THREE GENERATIONS", fill=(102, 244, 255), font=font)
    draw.text((12, 27), "EMA-FSQ BREEDING // NOVELTY + READABILITY + MOTION + LINEAGE", fill=(191, 255, 72), font=font)
    for generation in range(1, 4):
        generation_records = sorted((item for item in records if item["generation"] == generation), key=lambda item: item["rank"])
        y = top + (generation - 1) * (tile + 26)
        draw.text((10, y + 8), f"GEN {generation}", fill=(191, 255, 72), font=font)
        draw.text((10, y + 25), "12 ELITES", fill=(91, 124, 142), font=font)
        for column, record in enumerate(generation_records):
            atlas_path = _safe_artifact(destination, record["artifacts"]["composite"])
            with Image.open(atlas_path) as atlas:
                frame = atlas.convert("RGBA").crop((0, 0, 48, 48)).resize((tile, tile), Image.Resampling.NEAREST)
            cell = Image.new("RGBA", (tile, tile), (5, 13, 23, 255)); cell.alpha_composite(frame)
            x = label + column * tile; image.paste(cell.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(27, 62, 78))
            draw.text((x + 5, y + 5), f"{record['score']['score']:.3f}", fill=(196, 218, 229), font=font)
            draw.text((x + 5, y + tile - 27), str(record["fusion_mode"])[:10].upper(), fill=(102, 244, 255), font=font)
            draw.text((x + 5, y + tile - 14), str(record["mutation_mode"])[:10].upper(), fill=(255, 88, 183), font=font)
    output = io.BytesIO(); image.save(output, format="PNG", optimize=False, compress_level=9); return output.getvalue()


def compile_production_evolution(destination: Path = DEFAULT_OUTPUT, *, founders_manifest: Path = DEFAULT_FOUNDERS) -> dict[str, Any]:
    destination = Path(destination).resolve(); require_disk_floor(destination, planned_bytes=4 * 1024**3)
    if destination.exists():
        raise FileExistsError("production latent evolution destination already exists")
    destination.mkdir(parents=True)
    founder_manifest, founders, authority = _load_founders(founders_manifest)
    current = list(founders); archive = list(founders); selected_records = []; lineage_nodes = []
    candidate_ledger = []; failures = []; candidate_count = 0
    for founder, record in zip(founders, founder_manifest["specimens"], strict=True):
        lineage_nodes.append({
            "specimen_id": founder.sample_id, "generation": 0,
            "parent_ids": [str(record["parent_a"]), str(record["parent_b"])],
            "fields_sha256": founder.raw_fields_sha256, "lineage_sha256": str(record["lineage_sha256"]),
        })
    for generation, target_count in enumerate(GENERATION_SIZES, start=1):
        family_champions = []
        for family_id in range(5):
            champion = next((item for item in current if item.family_id == family_id), None)
            if champion is None:
                champion = founders[family_id]
            family_champions.append(champion)
        ordered = family_champions + [item for item in current if item not in family_champions]
        candidates = []
        for candidate_index in range(target_count):
            parent_a = ordered[candidate_index % len(ordered)]
            parent_b = current[(candidate_index * 5 + generation * 3 + 1) % len(current)]
            if parent_b.sample_id == parent_a.sample_id:
                parent_b = current[(candidate_index * 7 + generation + 2) % len(current)]
            fusion_mode = FUSION_MODES[candidate_index % len(FUSION_MODES)]
            mutation_mode = MUTATION_MODES[(candidate_index + generation * 2) % len(MUTATION_MODES)]
            base_alpha = 0.34 if candidate_index < 5 else (0.25, 0.4, 0.5, 0.6, 0.75)[(candidate_index + generation) % 5]
            accepted = None
            for attempt in range(8):
                try:
                    alpha = min(0.82, max(0.18, base_alpha + (attempt // 2) * (0.035 if attempt % 2 else -0.035)))
                    accepted = _candidate(
                        parent_a, parent_b, generation=generation, candidate_index=candidate_index,
                        attempt=attempt, fusion_mode=fusion_mode, mutation_mode=mutation_mode,
                        alpha=alpha, archive=archive,
                    )
                    break
                except ValueError as error:
                    failures.append({
                        "generation": generation, "candidate_index": candidate_index,
                        "attempt": attempt + 1, "parent_ids": [parent_a.sample_id, parent_b.sample_id],
                        "fusion_mode": fusion_mode, "mutation_mode": mutation_mode, "reason": str(error)[:600],
                    })
                    parent_b = current[(candidate_index * 7 + attempt * 3 + generation + 2) % len(current)]
                    if parent_b.sample_id == parent_a.sample_id:
                        parent_b = founders[(candidate_index + attempt + 1) % len(founders)]
            if accepted is not None:
                candidates.append(accepted); candidate_count += 1
        survivors = _select(candidates)
        selected_ids = {item["specimen"].genome.specimen_id for item in survivors}
        for candidate in candidates:
            candidate_ledger.append({
                "generation": generation, "specimen_id": candidate["specimen"].genome.specimen_id,
                "parent_ids": candidate["parent_ids"], "seed": candidate["seed"], "alpha": candidate["alpha"],
                "fusion_mode": candidate["fusion_mode"], "mutation_mode": candidate["mutation_mode"],
                "mutation_strength": candidate["mutation_strength"],
                "fields_sha256": candidate["specimen"].fields_sha256,
                "lineage_sha256": candidate["specimen"].genome.lineage_sha256,
                "score": candidate["score"], "selected": candidate["specimen"].genome.specimen_id in selected_ids,
            })
        generation_records = []
        for rank, survivor in enumerate(survivors):
            output_record = _render_selected(destination, generation, rank, survivor)
            generation_records.append(output_record); selected_records.append(output_record)
            lineage_nodes.append({
                "specimen_id": output_record["specimen_id"], "generation": generation,
                "rank": rank, "parent_ids": output_record["parent_ids"],
                "fields_sha256": output_record["fields_sha256"],
                "lineage_sha256": output_record["lineage_sha256"], "score": output_record["score"],
            })
        current = [_as_parent(item["specimen"], generation * 10_000 + rank, Path(founders_manifest)) for rank, item in enumerate(survivors)]
        archive.extend(current)
    contact_payload = _contact(selected_records, destination)
    contact_relative = "production_evolution_contact_sheet.png"; write_exact(destination / contact_relative, contact_payload)
    manifest: dict[str, Any] = {
        "format": FORMAT, "status": "ready", "quality_tier": "production-learned-latent-evolution-v1",
        "compiler": {
            "source_sha256": evolution_source_hash(), "cpu_only": True, "cuda_used": False,
            "founder_manifest_sha256": sha256_file(Path(founders_manifest)),
            "founder_bank_sha256": founder_manifest["bank_sha256"],
            "production_manifest_sha256": founder_manifest["compiler"]["production_manifest_sha256"],
            "production_checkpoint_sha256": founder_manifest["compiler"]["production_checkpoint_sha256"],
            "production_ema_sha256": founder_manifest["compiler"]["production_ema_sha256"],
            "repair_bank_sha256": authority.bank["bank_sha256"],
        },
        "selection_policy": {
            "generation_sizes": list(GENERATION_SIZES), "survivors_per_generation": SURVIVORS,
            "fusion_operator_floor": 1, "mutation_operator_floor": 1, "family_floor": 1,
            "phenotype_hamming_diversity_floor": 0.06,
            "fitness_components": [
                "ancestry_balance", "parent_phenotype_novelty", "archive_novelty",
                "boundary_readability", "emission_balance", "controlled_asymmetry",
                "occupancy_balance", "latent_code_diversity", "latent_code_novelty", "motion_strength",
            ],
        },
        "counts": {
            "founders": len(founders), "generations": len(GENERATION_SIZES),
            "candidate_plans": sum(GENERATION_SIZES), "accepted_candidates": candidate_count,
            "selected": len(selected_records), "failures": len(failures),
            "motion_clips": sum(len(item["clips"]) for item in selected_records),
            "motion_frames": sum(item["layout"]["frame_count"] for item in selected_records),
            "layer_atlases": len(selected_records) * len(LAYER_NAMES), "lineage_nodes": len(lineage_nodes),
        },
        "candidate_ledger": candidate_ledger, "lineage_nodes": lineage_nodes,
        "selected": selected_records, "failures": failures,
        "artifacts": {"contact_sheet": artifact_record_from_bytes(contact_relative, contact_payload)},
        "gates": {
            "production_codec_authority_valid": True, "three_generations_completed": True,
            "recursive_lineage_closed": True, "all_selected_legal_connected_riggable": True,
            "all_selected_motion_valid": True,
            "all_five_morphologies_each_generation": all(len({item["family_id"] for item in selected_records if item["generation"] == generation}) == 5 for generation in range(1, 4)),
            "balanced_morphology_quota_each_generation": all(all(2 <= sum(item["family_id"] == family_id for item in selected_records if item["generation"] == generation) <= 3 for family_id in range(5)) for generation in range(1, 4)),
            "all_six_fusion_operators_each_generation": all(len({item["fusion_mode"] for item in selected_records if item["generation"] == generation}) == 6 for generation in range(1, 4)),
            "all_six_mutation_operators_each_generation": all(len({item["mutation_mode"] for item in selected_records if item["generation"] == generation}) == 6 for generation in range(1, 4)),
            "all_selected_unique": len({item["fields_sha256"] for item in selected_records}) == len(selected_records),
            "disk_floor_preserved": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise ValueError(f"production evolution gate failure: {[key for key, value in manifest['gates'].items() if not value]}")
    manifest["evolution_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    write_exact(destination / "production_evolution_manifest.json", canonical_json_bytes(manifest))
    return manifest


def validate_production_evolution(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest):
        raise ValueError("production evolution manifest is not canonical JSON")
    unsigned = dict(manifest); stored = unsigned.pop("evolution_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("production evolution self-hash mismatch")
    if manifest.get("format") != FORMAT or manifest.get("status") != "ready":
        raise ValueError("production evolution authority mismatch")
    if manifest.get("compiler", {}).get("source_sha256") != evolution_source_hash():
        raise ValueError("production evolution source hash mismatch")
    if not manifest.get("gates") or not all(value is True for value in manifest["gates"].values()):
        raise ValueError("production evolution gate failure")
    counts = manifest.get("counts", {})
    if counts.get("founders") != 12 or counts.get("generations") != 3 or counts.get("selected") != 36:
        raise ValueError("production evolution count contract")
    selected = manifest.get("selected", [])
    if len(selected) != 36 or len({item["specimen_id"] for item in selected}) != 36 or len({item["fields_sha256"] for item in selected}) != 36:
        raise ValueError("production evolution selected uniqueness")
    known = {node["specimen_id"]: int(node["generation"]) for node in manifest.get("lineage_nodes", [])}
    if len(known) != int(counts.get("lineage_nodes", -1)):
        raise ValueError("production evolution lineage uniqueness")
    for generation in range(1, 4):
        values = [item for item in selected if int(item["generation"]) == generation]
        if len(values) != SURVIVORS or [item["rank"] for item in values] != list(range(SURVIVORS)):
            raise ValueError("production evolution generation rank census")
        if len({item["family_id"] for item in values}) != 5 or {item["fusion_mode"] for item in values} != set(FUSION_MODES) or {item["mutation_mode"] for item in values} != set(MUTATION_MODES):
            raise ValueError("production evolution diversity floors")
        family_counts = [sum(item["family_id"] == family_id for item in values) for family_id in range(5)]
        if any(count < 2 or count > 3 for count in family_counts):
            raise ValueError("production evolution morphology balance")
        field_codes = []
        for item in values:
            fields_path = _safe_artifact(manifest_path.parent, item["artifacts"]["semantic_fields"])
            arrays = _load_npz(fields_path)
            field_codes.append(arrays["part_owner"].astype(np.int32) * 40 + arrays["material"].astype(np.int32) * 4 + arrays["emission_level"])
        if min(float(np.mean(field_codes[left] != field_codes[right])) for left in range(len(field_codes)) for right in range(left)) < 0.06:
            raise ValueError("production evolution phenotype diversity floor")
        for item in values:
            if not math.isfinite(float(item["score"]["score"])):
                raise ValueError("production evolution non-finite score")
            if any(parent not in known or known[parent] >= generation for parent in item["parent_ids"]):
                raise ValueError("production evolution recursive lineage")
            layout = item["layout"]
            if layout["cell_size"] != 48 or layout["columns"] != 16 or layout["rows"] * 16 < layout["frame_count"]:
                raise ValueError("production evolution atlas layout")
            cursor = 0
            for clip in item["clips"]:
                if clip["start_cell"] != cursor:
                    raise ValueError("production evolution clip cursor")
                cursor += clip["frame_count"]
            if cursor != layout["frame_count"]:
                raise ValueError("production evolution frame census")
            for layer in LAYER_NAMES:
                path = _safe_artifact(manifest_path.parent, item["artifacts"][layer])
                with Image.open(path) as image:
                    if image.mode != "RGBA" or image.size != (layout["columns"] * 48, layout["rows"] * 48):
                        raise ValueError("production evolution atlas decode/dimensions")
            fields_path = _safe_artifact(manifest_path.parent, item["artifacts"]["semantic_fields"])
            arrays = _load_npz(fields_path)
            if aligned_fields_hash(arrays["part_owner"], arrays["material"], arrays["emission_level"]) != item["fields_sha256"]:
                raise ValueError("production evolution archived fields hash")
            _safe_artifact(manifest_path.parent, item["artifacts"]["binding"])
    _safe_artifact(manifest_path.parent, manifest["artifacts"]["contact_sheet"])
    if counts.get("motion_clips") != sum(len(item["clips"]) for item in selected) or counts.get("motion_frames") != sum(item["layout"]["frame_count"] for item in selected):
        raise ValueError("production evolution aggregate motion census")
    return manifest
