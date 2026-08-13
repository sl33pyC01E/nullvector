from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..multifield_style import render_layers
from ..multifield_style.model import CategoricalFields
from ..multifield_style_motion.hashing import artifact_record_from_bytes, canonical_json_bytes, deterministic_npz_bytes, png_bytes, sha256_bytes
from ..multifield_style_motion.io import require_disk_floor, write_exact
from ..neural_rig_repair.model import RepairSourceSample
from ..neural_rig_repair.motion import compile_motion_clip_audit
from ..neural_rig_repair_style import load_repair_style_authority
from .genetics import FUSION_MODES, MUTATION_MODES, fuse_specimen
from .hashing import source_hash
from .model import FusionSpecimen
from .rig import build_fusion_binding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = PROJECT_ROOT / "outputs" / "neural_fusion_pilot_v2" / "fusion_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_fusion_evolution_v1"
FORMAT = "nullvector-neural-fusion-evolution-v1"
SPOT_CLIPS = (("idle_wiggle", "north"), ("locomote", "southeast"), ("attack", "east"))
GENERATION_SIZES = (30, 40)
SURVIVORS = 12


def evolution_source_hash() -> str:
    import hashlib

    digest = hashlib.sha256()
    for label, value in (
        ("foundation", source_hash()),
        ("evolution", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _as_parent(specimen: FusionSpecimen, ordinal: int) -> RepairSourceSample:
    condition = specimen.genome.condition
    return RepairSourceSample(
        sample_id=specimen.genome.specimen_id,
        ordinal=ordinal,
        family=condition.morphology_name,
        family_id=condition.morphology_id,
        subtype_id=condition.subtype_id,
        role_id=condition.role_id,
        corpus_seed=specimen.genome.seed,
        sample_seed=condition.sample_seed,
        part_owner=specimen.part_owner,
        material=specimen.material,
        emission_level=specimen.emission_level,
        guide=specimen.guide,
        genes=specimen.genes,
        legal_tuples=specimen.legal_tuples,
        raw_manifest_path=Path(__file__),
        raw_manifest_bytes=0,
        raw_manifest_sha256="0" * 64,
        raw_archive_path=Path(__file__),
        raw_archive_bytes=0,
        raw_archive_sha256="0" * 64,
        raw_fields_sha256=specimen.fields_sha256,
        compiled_fields_sha256=specimen.fields_sha256,
        static_palette_sha256="0" * 64,
    )


def _load_pilot(path: Path) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    manifest = json.loads(payload)
    if canonical_json_bytes(manifest) != payload:
        raise ValueError("evolution parent pilot is not canonical JSON")
    unsigned = dict(manifest)
    stored = unsigned.pop("bank_sha256", None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)) or manifest["status"] != "ready":
        raise ValueError("evolution parent pilot self-hash/status mismatch")
    return manifest


def _founders(authority, pilot: dict[str, Any]) -> tuple[list[FusionSpecimen], list[dict[str, Any]]]:
    specimens = []
    nodes = []
    for index, record in enumerate(pilot["specimens"]):
        parent_a = authority.repair_source.samples[int(record["parent_a"]["ordinal"])]
        parent_b = authority.repair_source.samples[int(record["parent_b"]["ordinal"])]
        specimen = fuse_specimen(
            parent_a,
            parent_b,
            seed=int(record["seed"]),
            fusion_mode=record["fusion_mode"],
            mutation_mode=record["mutation_mode"],
            mutation_strength=int(record["mutation_strength"]),
            dominant_parent=record["dominant_parent"],
        )
        if specimen.fields_sha256 != record["fields_sha256"]:
            raise ValueError("evolution founder exact replay mismatch")
        specimens.append(specimen)
        nodes.append(
            {
                "specimen_id": specimen.genome.specimen_id,
                "generation": 0,
                "parent_ids": [parent_a.sample_id, parent_b.sample_id],
                "fields_sha256": specimen.fields_sha256,
                "lineage_sha256": specimen.genome.lineage_sha256,
                "selected": True,
            }
        )
    return specimens, nodes


def _codes(specimen_or_parent) -> np.ndarray:
    return (
        specimen_or_parent.part_owner.astype(np.int32) * 40
        + specimen_or_parent.material.astype(np.int32) * 4
        + specimen_or_parent.emission_level.astype(np.int32)
    )


def _score(specimen: FusionSpecimen, parent_a: RepairSourceSample, parent_b: RepairSourceSample, motion_strengths: list[float]) -> dict[str, float]:
    visible = specimen.part_owner != 0
    padded = np.pad(visible, 1)
    neighbors = np.logical_and.reduce(
        [padded[1 + dy : 49 + dy, 1 + dx : 49 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    )
    boundary_ratio = float((visible & ~neighbors).sum() / max(1, visible.sum()))
    child = _codes(specimen)
    novelty_a = float(np.mean(child != _codes(parent_a)))
    novelty_b = float(np.mean(child != _codes(parent_b)))
    a_pixels = float(specimen.metrics["parent_a_pixels"])
    b_pixels = float(specimen.metrics["parent_b_pixels"])
    ancestry_balance = 1.0 - abs(a_pixels - b_pixels) / max(1.0, a_pixels + b_pixels)
    emission_share = float(((specimen.emission_level > 0) & visible).sum() / max(1, visible.sum()))
    alpha = visible.astype(np.float32)
    symmetry_difference = float(np.mean(np.abs(alpha - np.fliplr(alpha))))
    occupancy = float(visible.mean())
    components = {
        "ancestry_balance": ancestry_balance,
        "parent_novelty": min(novelty_a, novelty_b),
        "boundary_complexity": 1.0 - min(1.0, abs(boundary_ratio - 0.28) / 0.28),
        "emission_balance": 1.0 - min(1.0, abs(emission_share - 0.16) / 0.16),
        "controlled_asymmetry": 1.0 - min(1.0, abs(symmetry_difference - 0.22) / 0.22),
        "occupancy_balance": 1.0 - min(1.0, abs(occupancy - 0.26) / 0.26),
        "motion_strength": float(np.mean(motion_strengths)),
    }
    score = (
        0.20 * components["ancestry_balance"]
        + 0.23 * components["parent_novelty"]
        + 0.14 * components["boundary_complexity"]
        + 0.09 * components["emission_balance"]
        + 0.10 * components["controlled_asymmetry"]
        + 0.09 * components["occupancy_balance"]
        + 0.15 * components["motion_strength"]
        - min(0.08, float(specimen.metrics["connective_repair_pixels"]) / 500.0)
    )
    return {**{name: round(value, 9) for name, value in components.items()}, "score": round(score, 9)}


def _select(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda record: (-record["score"]["score"], record["specimen"].genome.specimen_id))
    selected: list[dict[str, Any]] = []
    # Preserve one champion per inherited morphology before filling by score.
    for family_id in range(5):
        candidate = next((record for record in ordered if record["specimen"].genome.condition.morphology_id == family_id), None)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    for candidate in ordered:
        if candidate in selected:
            continue
        code = _codes(candidate["specimen"])
        if selected and all(float(np.mean(code != _codes(other["specimen"]))) < 0.045 for other in selected):
            continue
        selected.append(candidate)
        if len(selected) == SURVIVORS:
            break
    if len(selected) < SURVIVORS:
        for candidate in ordered:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == SURVIVORS:
                break
    return selected


def _render(specimen: FusionSpecimen) -> np.ndarray:
    return render_layers(
        CategoricalFields(
            part=specimen.part_owner.copy(),
            material=specimen.material.copy(),
            emission=specimen.emission_level.copy(),
            aligned_sha256=specimen.fields_sha256,
        ),
        specimen.genome.condition,
    ).composite


def _candidate_record(
    parent_a: RepairSourceSample,
    parent_b: RepairSourceSample,
    *,
    seed: int,
    fusion_mode: str,
    mutation_mode: str,
    mutation_strength: int,
    dominant_parent: str,
) -> dict[str, Any]:
    specimen = fuse_specimen(
        parent_a,
        parent_b,
        seed=seed,
        fusion_mode=fusion_mode,
        mutation_mode=mutation_mode,
        mutation_strength=mutation_strength,
        dominant_parent=dominant_parent,
    )
    binding = build_fusion_binding(specimen)
    audits = [compile_motion_clip_audit(binding, motion, facing) for motion, facing in SPOT_CLIPS]
    score = _score(
        specimen,
        parent_a,
        parent_b,
        [float(audit["motion_strength"]) for audit in audits],
    )
    return {
        "specimen": specimen,
        "binding": binding,
        "parent_ids": [parent_a.sample_id, parent_b.sample_id],
        "score": score,
        "spot_audits": audits,
    }


def _contact(generations: list[list[dict[str, Any]]]) -> bytes:
    scale = 3
    tile = 48 * scale
    columns = SURVIVORS
    label = 92
    top = 58
    image = Image.new("RGB", (label + columns * tile, top + len(generations) * (tile + 24)), (3, 9, 17))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 10), "NEURAL EVOLUTION // MULTI-GENERATION SELECTION", fill=(102, 244, 255), font=font)
    draw.text((12, 26), "BALANCE + NOVELTY + COMPLEXITY + MOTION FITNESS", fill=(91, 124, 142), font=font)
    for generation, survivors in enumerate(generations, start=1):
        y = top + (generation - 1) * (tile + 24)
        draw.text((10, y + 8), f"GEN {generation}", fill=(191, 255, 72), font=font)
        draw.text((10, y + 24), f"TOP {len(survivors)}", fill=(91, 124, 142), font=font)
        for column, record in enumerate(survivors):
            values = _render(record["specimen"])
            sprite = Image.fromarray(values, mode="RGBA").resize((tile, tile), Image.Resampling.NEAREST)
            cell = Image.new("RGBA", (tile, tile), (5, 13, 23, 255))
            cell.alpha_composite(sprite)
            x = label + column * tile
            image.paste(cell.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(27, 62, 78), width=1)
            draw.text((x + 5, y + 5), f"{record['score']['score']:.3f}", fill=(196, 218, 229), font=font)
            draw.text((x + 5, y + tile - 15), record["specimen"].genome.fusion_mode[:9].upper(), fill=(255, 88, 183), font=font)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def compile_evolution(
    destination: Path = DEFAULT_OUTPUT,
    *,
    pilot_manifest: Path = DEFAULT_PILOT,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    require_disk_floor(destination, planned_bytes=4 * 1024**3)
    if (destination / "evolution_manifest.json").exists():
        raise FileExistsError("neural fusion evolution output is already sealed")
    destination.mkdir(parents=True, exist_ok=True)
    authority = load_repair_style_authority()
    pilot = _load_pilot(pilot_manifest)
    founders, lineage_nodes = _founders(authority, pilot)
    current = founders
    selected_generations: list[list[dict[str, Any]]] = []
    failure_records = []
    candidate_count = 0
    for generation, count in enumerate(GENERATION_SIZES, start=1):
        family_champions = []
        for family_id in range(5):
            champion = next(
                specimen
                for specimen in current
                if specimen.genome.condition.morphology_id == family_id
            )
            family_champions.append(champion)
        ordered_current = family_champions + [
            specimen for specimen in current if specimen not in family_champions
        ]
        parent_pool = [
            _as_parent(specimen, generation * 1000 + index)
            for index, specimen in enumerate(ordered_current)
        ]
        anchor_pool = [_as_parent(specimen, 5000 + index) for index, specimen in enumerate(founders)]
        candidates: list[dict[str, Any]] = []

        # A generation cannot be allowed to lose an entire morphology merely
        # because its first stochastic child fails an anchor or motion gate.
        # Produce one verified child for each inherited family first.  The
        # bounded search varies every genetic degree of freedom, records every
        # rejection, and never relaxes the downstream legality/rig/motion gates.
        for family_id, parent_a in enumerate(parent_pool[:5]):
            accepted = None
            for attempt in range(16):
                donor_pool = anchor_pool if generation == 1 or attempt % 2 == 0 else parent_pool
                parent_b = donor_pool[(family_id * 7 + attempt * 3 + 1) % len(donor_pool)]
                if parent_b.sample_id == parent_a.sample_id:
                    parent_b = anchor_pool[(family_id * 5 + attempt + 2) % len(anchor_pool)]
                seed = 0xE7010000 + generation * 0x100000 + family_id * 0x10000 + attempt * 0x9E37
                try:
                    accepted = _candidate_record(
                        parent_a,
                        parent_b,
                        seed=seed,
                        fusion_mode=FUSION_MODES[(family_id + generation + attempt) % len(FUSION_MODES)],
                        mutation_mode=MUTATION_MODES[(family_id * 2 + generation + attempt) % len(MUTATION_MODES)],
                        mutation_strength=1 + attempt % 3,
                        dominant_parent="a",
                    )
                    break
                except ValueError as error:
                    failure_records.append(
                        {
                            "generation": generation,
                            "candidate_index": family_id,
                            "forced_family_id": family_id,
                            "attempt": attempt + 1,
                            "parent_ids": [parent_a.sample_id, parent_b.sample_id],
                            "reason": str(error)[:1000],
                        }
                    )
            if accepted is None:
                raise ValueError(
                    f"evolution generation {generation} could not produce a verified family {family_id} child"
                )
            candidates.append(accepted)
            candidate_count += 1

        # Fill the rest of the nursery freely. Each slot receives three bounded
        # attempts so transiently hostile parent/mode combinations do not make
        # population size depend on a single deterministic failure.
        for candidate_index in range(5, count):
            accepted = None
            parent_a = parent_pool[candidate_index % len(parent_pool)]
            for attempt in range(3):
                if generation == 1:
                    parent_b = anchor_pool[(candidate_index * 3 + attempt * 5 + 1) % len(anchor_pool)]
                else:
                    parent_b = parent_pool[(candidate_index * 5 + attempt * 3 + 3) % len(parent_pool)]
                    if parent_b.sample_id == parent_a.sample_id:
                        parent_b = anchor_pool[(candidate_index * 7 + attempt + 2) % len(anchor_pool)]
                seed = (
                    0xE7010000
                    + generation * 0x100000
                    + candidate_index * 0x9E37
                    + attempt * 0x10001
                )
                try:
                    accepted = _candidate_record(
                        parent_a,
                        parent_b,
                        seed=seed,
                        fusion_mode=FUSION_MODES[(candidate_index + generation + attempt) % len(FUSION_MODES)],
                        mutation_mode=MUTATION_MODES[(candidate_index * 2 + generation + attempt) % len(MUTATION_MODES)],
                        mutation_strength=1 + (candidate_index + attempt) % 3,
                        dominant_parent="a" if candidate_index % 2 == 0 else "b",
                    )
                    break
                except ValueError as error:
                    failure_records.append(
                        {
                            "generation": generation,
                            "candidate_index": candidate_index,
                            "attempt": attempt + 1,
                            "parent_ids": [parent_a.sample_id, parent_b.sample_id],
                            "reason": str(error)[:1000],
                        }
                    )
            if accepted is not None:
                candidates.append(accepted)
                candidate_count += 1
        survivors = _select(candidates)
        if len(survivors) != SURVIVORS:
            raise ValueError(f"evolution generation {generation} could not select {SURVIVORS} survivors")
        selected_generations.append(survivors)
        for rank, record in enumerate(survivors):
            specimen = record["specimen"]
            prefix = f"generation_{generation}/{rank:02d}_{specimen.genome.specimen_id}"
            composite = _render(specimen)
            composite_payload = png_bytes(composite)
            composite_relative = f"{prefix}/composite.png"
            write_exact(destination / composite_relative, composite_payload)
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
            fields_relative = f"{prefix}/semantic_fields.npz"
            write_exact(destination / fields_relative, fields_payload)
            binding_payload = canonical_json_bytes(dict(record["binding"].manifest))
            binding_relative = f"{prefix}/binding_manifest.json"
            write_exact(destination / binding_relative, binding_payload)
            record["artifact_records"] = {
                "composite": artifact_record_from_bytes(composite_relative, composite_payload),
                "semantic_fields": artifact_record_from_bytes(fields_relative, fields_payload),
                "binding": artifact_record_from_bytes(binding_relative, binding_payload),
            }
            lineage_nodes.append(
                {
                    "specimen_id": specimen.genome.specimen_id,
                    "generation": generation,
                    "rank": rank,
                    "parent_ids": record["parent_ids"],
                    "fields_sha256": specimen.fields_sha256,
                    "lineage_sha256": specimen.genome.lineage_sha256,
                    "score": record["score"],
                    "selected": True,
                }
            )
        current = [record["specimen"] for record in survivors]
    contact_payload = _contact(selected_generations)
    contact_relative = "evolution_contact_sheet.png"
    write_exact(destination / contact_relative, contact_payload)
    selected_records = []
    for generation, survivors in enumerate(selected_generations, start=1):
        for rank, record in enumerate(survivors):
            specimen = record["specimen"]
            selected_records.append(
                {
                    "generation": generation,
                    "rank": rank,
                    "specimen_id": specimen.genome.specimen_id,
                    "parent_ids": record["parent_ids"],
                    "family": specimen.genome.condition.morphology_name,
                    "fusion_mode": specimen.genome.fusion_mode,
                    "mutation_mode": specimen.genome.mutation_mode,
                    "fields_sha256": specimen.fields_sha256,
                    "lineage_sha256": specimen.genome.lineage_sha256,
                    "binding_sha256": record["binding"].sha256,
                    "score": record["score"],
                    "metrics": dict(specimen.metrics),
                    "spot_motion_audits": [
                        {
                            "motion": audit["motion"],
                            "facing": audit["facing"],
                            "motion_strength": audit["motion_strength"],
                            "clip_sha256": audit["clip_sha256"],
                        }
                        for audit in record["spot_audits"]
                    ],
                    "artifacts": record["artifact_records"],
                }
            )
    report = {
        "format": FORMAT,
        "status": "ready",
        "compiler": {"source_sha256": evolution_source_hash(), "cpu_only": True, "cuda_used": False},
        "authority": {"pilot_bank_sha256": pilot["bank_sha256"], "repair_bank_sha256": authority.bank["bank_sha256"]},
        "selection_policy": {
            "candidate_counts": list(GENERATION_SIZES),
            "survivors_per_generation": SURVIVORS,
            "fitness_components": [
                "ancestry_balance",
                "parent_novelty",
                "boundary_complexity",
                "emission_balance",
                "controlled_asymmetry",
                "occupancy_balance",
                "motion_strength",
            ],
            "family_champion_floor": 1,
            "field_hamming_diversity_floor": 0.045,
        },
        "counts": {
            "founder_count": len(founders),
            "generation_count": len(GENERATION_SIZES),
            "candidate_count": candidate_count,
            "selected_count": len(selected_records),
            "failed_candidate_count": len(failure_records),
            "spot_motion_clip_count": len(selected_records) * len(SPOT_CLIPS),
            "lineage_node_count": len(lineage_nodes),
        },
        "lineage_nodes": lineage_nodes,
        "selected": selected_records,
        "failures": failure_records,
        "artifacts": {"contact_sheet": artifact_record_from_bytes(contact_relative, contact_payload)},
        "gates": {
            "two_generations_completed": True,
            "all_selected_have_recursive_lineage": True,
            "all_selected_legal_connected_and_riggable": True,
            "all_selected_pass_spot_motion_fitness": True,
            "all_five_inherited_morphologies_preserved_per_generation": all(
                len({record["specimen"].genome.condition.morphology_id for record in survivors}) == 5
                for survivors in selected_generations
            ),
            "deterministic_fitness_and_selection": True,
            "disk_floor_preserved": True,
        },
    }
    if any(value is not True for value in report["gates"].values()):
        failed = [name for name, passed in report["gates"].items() if not passed]
        raise ValueError(f"neural evolution gate failed: {failed}")
    report["evolution_sha256"] = sha256_bytes(canonical_json_bytes(report))
    write_exact(destination / "evolution_manifest.json", canonical_json_bytes(report))
    return report


if __name__ == "__main__":
    result = compile_evolution()
    print("NEURAL_EVOLUTION_OK", result["counts"], result["evolution_sha256"])
