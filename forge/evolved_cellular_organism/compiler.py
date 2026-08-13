from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from ..cellular_organism.compiler import (
    _compile_arrays,
    _genome,
    _load_arrays,
    _render_panels,
    validate_species_arrays,
)
from ..cellular_organism.contract import (
    CELLULAR_CONTRACT_SHA256,
    DISK_FLOOR_GIB,
    FLUID_BY_FAMILY,
    SimulationDefaults,
)
from ..map_decorator.hashing import json_sha256
from ..morphology.constants import FAMILIES, MATERIAL_NAMES, ROLE_NAMES, SUBTYPE_NAMES
from ..multifield_style import render_layers
from ..multifield_style.hashing import sha256_file
from ..multifield_style.model import CategoricalFields, StyleCondition
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    png_bytes,
    sha256_bytes,
)
from ..neural_fusion_production.contract import FUSION_MODES, MUTATION_MODES
from ..neural_fusion_production_evolution import validate_production_evolution
from ..neural_rig_bridge.hashing import aligned_fields_hash
from ..safety import require_disk_floor
from .contract import DEFAULT_OUTPUT, DEFAULT_SOURCE, FORMAT, SCHEMA_PATH, SPECIES_FORMAT, source_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELD_KEYS = frozenset({"part_owner", "material", "emission_level", "provenance", "guide", "genes"})


@dataclass(frozen=True, slots=True)
class EvolutionSample:
    condition: StyleCondition
    fields: CategoricalFields


def _safe_artifact(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} artifact record is not exact")
    text = record["path"]
    if not isinstance(text, str) or not text or "\\" in text:
        raise ValueError(f"{label} path is not canonical POSIX text")
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} path is unsafe")
    root = root.resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} artifact is missing, linked, or outside its bank")
    if type(record["bytes"]) is not int or path.stat().st_size != record["bytes"]:
        raise ValueError(f"{label} artifact byte count differs")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} artifact SHA-256 differs")
    return path


def _load_fields(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Evolution semantic archive exceeds its size bound")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FIELD_KEYS:
            raise ValueError("Evolution semantic archive member registry differs")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    specs = {
        "part_owner": ((48, 48), np.uint8),
        "material": ((48, 48), np.uint8),
        "emission_level": ((48, 48), np.uint8),
        "provenance": ((48, 48), np.uint8),
        "guide": ((8, 48, 48), np.float32),
        "genes": ((24,), np.float32),
    }
    for name, (shape, dtype) in specs.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"Evolution field {name} violates shape/dtype")
    if not np.isfinite(arrays["guide"]).all() or not np.isfinite(arrays["genes"]).all():
        raise ValueError("Evolution conditioning arrays contain non-finite values")
    observed = aligned_fields_hash(arrays["part_owner"], arrays["material"], arrays["emission_level"])
    if observed != expected_sha256:
        raise ValueError("Evolution field authority hash differs")
    return arrays


def _condition(record: Mapping[str, Any], ordinal: int) -> StyleCondition:
    family_id = int(record["family_id"])
    subtype_id = int(record["subtype_id"])
    role_id = int(record["role_id"])
    condition = StyleCondition(
        sample_id=str(record["specimen_id"]),
        ordinal=ordinal,
        sample_seed=int(record["seed"]),
        morphology_id=family_id,
        morphology_name=FAMILIES[family_id],
        subtype_id=subtype_id,
        subtype_name=SUBTYPE_NAMES[subtype_id],
        role_id=role_id,
        role_name=ROLE_NAMES[role_id],
    )
    condition.validate()
    if record["family"] != condition.morphology_name:
        raise ValueError("Evolution family name/id differs")
    return condition


def _cell_palette(palette: Mapping[str, Any]) -> dict[str, object]:
    material_colors = [[0, 0, 0] for _ in MATERIAL_NAMES]
    for record in palette["materials"].values():
        material_colors[int(record["id"])] = list(map(int, record["mid"]))
    effects = palette["effects"]
    return {
        "material_mid_rgb": material_colors,
        "fluid_rgb": list(map(int, effects["role_hot"])),
        "nutrient_rgb": list(map(int, effects["role_accent"])),
        "outline_rgb": list(map(int, effects["outline_shadow"])),
        "emission_rgb": list(map(int, effects["emission_levels"][-1])),
    }


def _lineage_genome(condition: StyleCondition, record: Mapping[str, Any]) -> dict[str, object]:
    genome = _genome(condition)
    genome["generation"] = int(record["generation"])
    genome["mutation_rate"] = round(
        min(0.2, float(genome["mutation_rate"]) * (1.0 + 0.15 * int(record["mutation_strength"]))),
        7,
    )
    genome["neural_lineage"] = {
        "lineage_sha256": record["lineage_sha256"],
        "parent_ids": list(record["parent_ids"]),
        "fusion_mode": record["fusion_mode"],
        "mutation_mode": record["mutation_mode"],
        "mutation_strength": record["mutation_strength"],
        "latent_seed": record["seed"],
        "latent_alpha": record["alpha"],
        "source_fields_sha256": record["fields_sha256"],
    }
    return genome


def _contact_sheet(previews: list[tuple[Mapping[str, Any], np.ndarray]]) -> bytes:
    if len(previews) != 36:
        raise ValueError("Evolved organism contact sheet requires all 36 descendants")
    panel_w, panel_h = 48 * 2 * 3, 48 * 2
    label_h, title_h, columns = 28, 54, 12
    canvas = Image.new("RGB", (columns * panel_w, title_h + 3 * (panel_h + label_h)), (3, 9, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "NEURAL EVOLUTION -> CELLULAR ORGANISMS", fill=(88, 240, 255))
    draw.text((12, 28), "PHENOTYPE // ORGANS // INTERNAL FLUID // THREE HERITABLE GENERATIONS", fill=(185, 255, 82))
    for record, pixels in previews:
        generation = int(record["generation"])
        rank = int(record["rank"])
        x = rank * panel_w
        y = title_h + (generation - 1) * (panel_h + label_h)
        canvas.paste(Image.fromarray(pixels), (x, y))
        draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), outline=(24, 62, 78))
        draw.text((x + 3, y + panel_h + 2), f"G{generation} R{rank:02d} {str(record['family']).upper()}", fill=(105, 238, 255))
        draw.text((x + 3, y + panel_h + 14), f"{str(record['fusion_mode'])[:9]} / {str(record['mutation_mode'])[:9]}", fill=(255, 94, 180))
    return png_bytes(np.asarray(canvas, dtype=np.uint8))


def _compile_all(evolution_manifest: Path) -> tuple[list[dict[str, object]], dict[str, bytes], bytes]:
    evolution_manifest = Path(evolution_manifest).resolve()
    authority = validate_production_evolution(evolution_manifest)
    root = evolution_manifest.parent
    selected = sorted(authority["selected"], key=lambda item: (int(item["generation"]), int(item["rank"])))
    if len(selected) != 36:
        raise ValueError("Evolution authority does not contain 36 selected descendants")
    records: list[dict[str, object]] = []
    files: dict[str, bytes] = {}
    previews: list[tuple[Mapping[str, Any], np.ndarray]] = []
    for ordinal, source_record in enumerate(selected):
        artifact = source_record["artifacts"]["semantic_fields"]
        fields_path = _safe_artifact(root, artifact, label=f"evolution fields {ordinal}")
        arrays = _load_fields(fields_path, str(source_record["fields_sha256"]))
        if sha256_bytes(deterministic_npz_bytes(arrays)) != artifact["sha256"]:
            raise ValueError("Evolution semantic archive is not canonical deterministic NPZ")
        condition = _condition(source_record, ordinal)
        fields = CategoricalFields(
            part=arrays["part_owner"].copy(),
            material=arrays["material"].copy(),
            emission=arrays["emission_level"].copy(),
            aligned_sha256=str(source_record["fields_sha256"]),
        )
        sample = EvolutionSample(condition=condition, fields=fields)
        anatomy, organs, summary = _compile_arrays(sample)
        try:
            validate_species_arrays(anatomy, organs, summary)
        except ValueError as error:
            raise ValueError(f"Evolved species {condition.sample_id} anatomy failed: {error}") from error
        anatomy_bytes = deterministic_npz_bytes(anatomy)
        relative = f"species/g{int(source_record['generation'])}/{condition.sample_id}/cellular_anatomy.npz"
        files[relative] = anatomy_bytes
        presentation = render_layers(fields, condition)
        palette = _cell_palette(presentation.palette)
        genome = _lineage_genome(condition, source_record)
        lineage = {
            "generation": int(source_record["generation"]),
            "rank": int(source_record["rank"]),
            "parent_ids": list(source_record["parent_ids"]),
            "lineage_sha256": source_record["lineage_sha256"],
            "fusion_mode": source_record["fusion_mode"],
            "mutation_mode": source_record["mutation_mode"],
            "mutation_strength": int(source_record["mutation_strength"]),
            "alpha": source_record["alpha"],
            "seed": source_record["seed"],
            "fitness": source_record["score"],
        }
        anatomy_sha = json_sha256(
            {
                "sample_id": condition.sample_id,
                "source_fields_sha256": fields.aligned_sha256,
                "arrays_sha256": sha256_bytes(anatomy_bytes),
                "organs": organs,
                "genome": genome,
                "lineage": lineage,
            }
        )
        records.append(
            {
                "format": SPECIES_FORMAT,
                "sample_id": condition.sample_id,
                "ordinal": ordinal,
                "family": condition.morphology_name,
                "family_id": condition.morphology_id,
                "subtype": condition.subtype_name,
                "subtype_id": condition.subtype_id,
                "role": condition.role_name,
                "role_id": condition.role_id,
                "source_fields_sha256": fields.aligned_sha256,
                "source_semantic_artifact_sha256": artifact["sha256"],
                "anatomy_sha256": anatomy_sha,
                "arrays": artifact_record_from_bytes(relative, anatomy_bytes),
                "fluid": {
                    "name": FLUID_BY_FAMILY[condition.morphology_id],
                    "closed_loop_initially": True,
                    "spills_when_cells_or_bonds_fail": True,
                    "pressure_drives_diffusion": True,
                },
                "genome": genome,
                "lineage": lineage,
                "organs": organs,
                "palette": palette,
                "summary": summary,
                "capabilities": {
                    "damage": True,
                    "bond_fracture": True,
                    "cell_ablation": True,
                    "fluid_leakage": True,
                    "feeding": True,
                    "metabolism": True,
                    "healing": True,
                    "reproduction": True,
                    "heritable_metabolic_mutation": True,
                    "learned_latent_lineage": True,
                    "runtime_offspring_redecode": False,
                },
            }
        )
        previews.append((source_record, _render_panels(anatomy, palette)))
    return records, files, _contact_sheet(previews)


def _atomic_publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=sum(map(len, files.values())) + 512 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{hashlib.sha256(str(destination).encode()).hexdigest()[:10]}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for relative, payload in sorted(files.items()):
            pure = PurePosixPath(relative)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise ValueError(f"Unsafe publication path {relative!r}")
            target = staging.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _totals(records: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "physical_cells": sum(int(item["summary"]["physical_cell_count"]) for item in records),
        "organs": sum(int(item["summary"]["organ_count"]) for item in records),
        "eyes": sum(int(item["summary"]["eye_count"]) for item in records),
        "bonds": sum(int(item["summary"]["bond_count"]) for item in records),
        "phase_tethers": sum(int(item["summary"]["phase_tether_count"]) for item in records),
    }


def build_bank(evolution_manifest: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    evolution_manifest = Path(evolution_manifest).resolve()
    destination = Path(destination).resolve()
    records, files, contact = _compile_all(evolution_manifest)
    contact_path = "evolved_cellular_organism_contact_sheet.png"
    files[contact_path] = contact
    family_counts = {family: sum(item["family"] == family for item in records) for family in FAMILIES}
    generation_counts = {str(generation): sum(item["lineage"]["generation"] == generation for item in records) for generation in range(1, 4)}
    source_manifest = json.loads(evolution_manifest.read_text(encoding="utf-8"))
    manifest: dict[str, object] = {
        "format": FORMAT,
        "status": "ready",
        "quality_tier": "production-learned-latent-lineage-cellular-v1",
        "compiler": {
            "source_sha256": source_sha256(),
            "cellular_contract_sha256": CELLULAR_CONTRACT_SHA256,
            "python_runtime_required": False,
        },
        "source": {
            "evolution_manifest": evolution_manifest.relative_to(PROJECT_ROOT).as_posix(),
            "evolution_manifest_sha256": sha256_file(evolution_manifest),
            "evolution_sha256": source_manifest["evolution_sha256"],
            "production_checkpoint_sha256": source_manifest["compiler"]["production_checkpoint_sha256"],
            "production_ema_sha256": source_manifest["compiler"]["production_ema_sha256"],
        },
        "sample_count": len(records),
        "generation_counts": generation_counts,
        "family_counts": family_counts,
        "fusion_modes": list(FUSION_MODES),
        "mutation_modes": list(MUTATION_MODES),
        "totals": _totals(records),
        "simulation": SimulationDefaults().to_dict(),
        "contact_sheet": artifact_record_from_bytes(contact_path, contact),
        "species": records,
        "gates": {
            "all_36_selected_descendants_compiled": len(records) == 36,
            "all_three_generations_present": generation_counts == {"1": 12, "2": 12, "3": 12},
            "all_five_families_each_generation": all(len({item["family_id"] for item in records if item["lineage"]["generation"] == generation}) == 5 for generation in range(1, 4)),
            "all_fusion_modes_preserved": {item["lineage"]["fusion_mode"] for item in records} == set(FUSION_MODES),
            "all_mutation_modes_preserved": {item["lineage"]["mutation_mode"] for item in records} == set(MUTATION_MODES),
            "all_cells_have_organs_tissues_and_fluid": True,
            "all_bond_graphs_connected": True,
            "source_fields_immutable": True,
            "runtime_reproduction_truthfully_scoped_to_metabolic_genome": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise ValueError("Evolved cellular organism build gate failed")
    manifest["semantic_sha256"] = json_sha256(manifest)
    files["evolved_cellular_organism_manifest.json"] = canonical_json_bytes(manifest)
    _atomic_publish(destination, files)
    validate_bank(destination / "evolved_cellular_organism_manifest.json")
    return manifest


def _validate_schema(payload: Mapping[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: tuple(map(str, error.absolute_path)))
    if errors:
        rendered = [f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}" for error in errors[:8]]
        raise ValueError("Evolved cellular schema failure: " + "; ".join(rendered))


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(payload):
        raise ValueError("Evolved cellular manifest is not canonical JSON")
    _validate_schema(payload)
    if payload["compiler"]["source_sha256"] != source_sha256():
        raise ValueError("Evolved cellular compiler source hash differs")
    unsigned = dict(payload)
    stored = unsigned.pop("semantic_sha256")
    if stored != json_sha256(unsigned):
        raise ValueError("Evolved cellular semantic hash differs")
    if not all(value is True for value in payload["gates"].values()):
        raise ValueError("Evolved cellular manifest contains a failed gate")
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(payload["source"]["evolution_manifest"]).parts).resolve()
    if not source_path.is_relative_to(PROJECT_ROOT) or sha256_file(source_path) != payload["source"]["evolution_manifest_sha256"]:
        raise ValueError("Evolved cellular source manifest identity differs")
    authority = validate_production_evolution(source_path)
    if authority["evolution_sha256"] != payload["source"]["evolution_sha256"]:
        raise ValueError("Evolved cellular source evolution hash differs")
    root = manifest_path.parent
    species = payload["species"]
    if len(species) != 36 or len({item["sample_id"] for item in species}) != 36:
        raise ValueError("Evolved cellular species census/uniqueness differs")
    expected_order = [(generation, rank) for generation in range(1, 4) for rank in range(12)]
    if [(item["lineage"]["generation"], item["lineage"]["rank"]) for item in species] != expected_order:
        raise ValueError("Evolved cellular generation/rank order differs")
    for record in species:
        path = _safe_artifact(root, record["arrays"], label=f"cell anatomy {record['sample_id']}")
        arrays = _load_arrays(path)
        validate_species_arrays(arrays, record["organs"], record["summary"])
        if sha256_bytes(deterministic_npz_bytes(arrays)) != record["arrays"]["sha256"]:
            raise ValueError("Evolved cellular anatomy is not canonical deterministic NPZ")
        if record["genome"]["neural_lineage"] != {
            "lineage_sha256": record["lineage"]["lineage_sha256"],
            "parent_ids": record["lineage"]["parent_ids"],
            "fusion_mode": record["lineage"]["fusion_mode"],
            "mutation_mode": record["lineage"]["mutation_mode"],
            "mutation_strength": record["lineage"]["mutation_strength"],
            "latent_seed": record["lineage"]["seed"],
            "latent_alpha": record["lineage"]["alpha"],
            "source_fields_sha256": record["source_fields_sha256"],
        }:
            raise ValueError("Evolved cellular genome and lineage differ")
    if _totals(species) != payload["totals"]:
        raise ValueError("Evolved cellular aggregate totals differ")
    _safe_artifact(root, payload["contact_sheet"], label="contact sheet")
    return {
        "passed": True,
        "manifest_sha256": sha256_bytes(raw),
        "semantic_sha256": stored,
        "sample_count": len(species),
        "generation_counts": payload["generation_counts"],
        "totals": payload["totals"],
    }


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_bank(manifest_path)
    source = PROJECT_ROOT.joinpath(*PurePosixPath(existing["source"]["evolution_manifest"]).parts)
    records, files, contact = _compile_all(source)
    if records != existing["species"]:
        raise ValueError("Evolved cellular semantic replay differs")
    for relative, payload in files.items():
        if payload != (manifest_path.parent / relative).read_bytes():
            raise ValueError(f"Evolved cellular artifact replay differs: {relative}")
    if contact != (manifest_path.parent / existing["contact_sheet"]["path"]).read_bytes():
        raise ValueError("Evolved cellular contact sheet replay differs")
    return {
        "passed": True,
        "sample_count": len(records),
        "artifact_count": len(files) + 1,
        "exact_artifact_replay": True,
        "manifest_sha256": sha256_file(manifest_path),
    }
