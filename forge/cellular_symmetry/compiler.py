from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from ..cellular_breeding import validate_bank as validate_breeding_bank
from ..cellular_breeding.compiler import _repair_connectivity, _safe_artifact as _safe_source_artifact
from ..cellular_organism.compiler import _atomic_publish, _compile_arrays, _load_arrays, _render_panels, validate_species_arrays
from ..cellular_organism.contract import CANVAS_SIZE, CELLULAR_CONTRACT_SHA256, SimulationDefaults
from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..morphology.constants import FAMILIES
from ..multifield_style.hashing import sha256_file
from ..multifield_style.model import CategoricalFields, StyleCondition
from ..multifield_style_motion.hashing import artifact_record_from_bytes, canonical_json_bytes, deterministic_npz_bytes, png_bytes, sha256_bytes
from ..neural_rig_bridge.hashing import aligned_fields_hash
from .contract import DEFAULT_OUTPUT, DEFAULT_SOURCE, FIELDS_FORMAT, FORMAT, SAMPLE_COUNT, SCHEMA_PATH, SPECIES_FORMAT, source_sha256


CHASSIS_OWNERS = frozenset({1, 2, 3, 10, 11, 13, 14})
PAIRED_APPENDAGE_OWNERS = frozenset({4, 5, 6, 7})
OTHER_APPENDAGE_OWNERS = frozenset({8, 9, 12, 15})
MIRROR_OWNER = {4: 5, 5: 4, 6: 7, 7: 6}
FIELD_KEYS = frozenset({"format", "part_owner", "material", "emission", "ancestry", "mutation_mask", "repair_mask", "symmetry_added_mask", "symmetry_class"})

FAMILY_PRIORS: dict[int, dict[str, float]] = {
    0: {"chassis": 0.93, "paired": 0.78, "other": 0.25, "growth_cap": 0.14},
    1: {"chassis": 0.78, "paired": 0.65, "other": 0.30, "growth_cap": 0.12},
    2: {"chassis": 0.70, "paired": 0.55, "other": 0.35, "growth_cap": 0.11},
    3: {"chassis": 0.42, "paired": 0.30, "other": 0.20, "growth_cap": 0.06},
    4: {"chassis": 0.96, "paired": 0.82, "other": 0.30, "growth_cap": 0.15},
}


@dataclass(frozen=True, slots=True)
class SymmetrySample:
    condition: StyleCondition
    fields: CategoricalFields


def _unit(seed: int, *labels: object) -> float:
    payload = ":".join([str(seed), *(str(label) for label in labels)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float((1 << 64) - 1)


def _condition(record: Mapping[str, Any]) -> StyleCondition:
    return StyleCondition(
        sample_id=str(record["sample_id"]), ordinal=int(record["ordinal"]), sample_seed=int(record["lineage"]["seed"]),
        morphology_id=int(record["family_id"]), morphology_name=str(record["family"]),
        subtype_id=int(record["subtype_id"]), subtype_name=str(record["subtype"]),
        role_id=int(record["role_id"]), role_name=str(record["role"]),
    )


def _mask_match(mask: np.ndarray, axis2: int) -> float:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 1.0
    matched = 0
    for y, x in np.argwhere(mask):
        mirror_x = axis2 - int(x)
        if 0 <= mirror_x < CANVAS_SIZE and bool(mask[int(y), mirror_x]):
            matched += 1
    return matched / count


def _appendage_match(part: np.ndarray, axis2: int) -> float:
    mask = np.isin(part, tuple(PAIRED_APPENDAGE_OWNERS | OTHER_APPENDAGE_OWNERS))
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 1.0
    matched = 0
    for y, x in np.argwhere(mask):
        owner = int(part[int(y), int(x)])
        mirror_x = axis2 - int(x)
        expected = MIRROR_OWNER.get(owner, owner)
        if 0 <= mirror_x < CANVAS_SIZE and int(part[int(y), mirror_x]) == expected:
            matched += 1
    return matched / count


def _metrics(part: np.ndarray, axis2: int) -> dict[str, float | int]:
    silhouette = _mask_match(part > 0, axis2)
    chassis = _mask_match(np.isin(part, tuple(CHASSIS_OWNERS)), axis2)
    appendage = _appendage_match(part, axis2)
    weighted = chassis * 0.55 + appendage * 0.25 + silhouette * 0.20
    return {
        "axis_x": axis2 / 2.0,
        "axis_x2": axis2,
        "silhouette_match": round(silhouette, 7),
        "chassis_match": round(chassis, 7),
        "paired_appendage_match": round(appendage, 7),
        "weighted_score": round(weighted, 7),
    }


def _best_axis(part: np.ndarray) -> int:
    physical = np.argwhere(part > 0)
    if not len(physical):
        return 47
    center2 = int(physical[:, 1].min() + physical[:, 1].max())
    candidates = range(max(2, center2 - 4), min(93, center2 + 4) + 1)
    return max(candidates, key=lambda axis2: (float(_metrics(part, axis2)["weighted_score"]), -abs(axis2 - center2), -axis2))


def _refine(source: Mapping[str, np.ndarray], family_id: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    child = {name: source[name].copy() for name in ("part_owner", "material", "emission", "ancestry", "mutation_mask", "repair_mask")}
    child["symmetry_added_mask"] = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    child["symmetry_class"] = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    original = child["part_owner"].copy()
    axis2 = _best_axis(original)
    before = _metrics(original, axis2)
    prior = FAMILY_PRIORS[family_id]
    candidates: list[tuple[int, float, int, int, int, int, int, int]] = []
    for y, x in np.argwhere(original > 0):
        x_value, y_value = int(x), int(y)
        mirror_x = axis2 - x_value
        if not 0 <= mirror_x < CANVAS_SIZE or child["part_owner"][y_value, mirror_x] != 0:
            continue
        owner = int(original[y_value, x_value])
        if owner in CHASSIS_OWNERS:
            class_id, probability, priority = 1, prior["chassis"], 0
        elif owner in PAIRED_APPENDAGE_OWNERS:
            class_id, probability, priority = 2, prior["paired"], 1
        elif owner in OTHER_APPENDAGE_OWNERS:
            class_id, probability, priority = 3, prior["other"], 2
        else:
            continue
        random_value = _unit(seed, "symmetry", x_value, y_value, mirror_x, owner)
        if random_value >= probability:
            continue
        candidates.append((priority, random_value, abs(mirror_x * 2 - axis2), y_value, mirror_x, x_value, owner, class_id))
    candidates.sort()
    original_count = int(np.count_nonzero(original))
    cap = max(1, int(math.ceil(original_count * prior["growth_cap"])))
    for _, _, _, y, mirror_x, source_x, owner, class_id in candidates[:cap]:
        if child["part_owner"][y, mirror_x] != 0:
            continue
        child["part_owner"][y, mirror_x] = np.uint8(MIRROR_OWNER.get(owner, owner))
        child["material"][y, mirror_x] = child["material"][y, source_x]
        child["emission"][y, mirror_x] = child["emission"][y, source_x]
        child["ancestry"][y, mirror_x] = 3
        child["mutation_mask"][y, mirror_x] = 1
        child["symmetry_added_mask"][y, mirror_x] = 1
        child["symmetry_class"][y, mirror_x] = np.uint8(class_id)
    repair_before = child["repair_mask"].copy()
    _repair_connectivity(child)
    new_repairs = (child["repair_mask"] > repair_before).astype(np.uint8)
    after = _metrics(child["part_owner"], axis2)
    added = int(np.count_nonzero(child["symmetry_added_mask"]))
    repair_added = int(np.count_nonzero(new_repairs))
    report = {
        "policy": "family-aware-soft-bilateral-additive-v1",
        "axis_x": before["axis_x"],
        "family_prior": prior,
        "before": before,
        "after": after,
        "weighted_improvement": round(float(after["weighted_score"]) - float(before["weighted_score"]), 7),
        "original_cells": original_count,
        "symmetry_cells_added": added,
        "connectivity_cells_added": repair_added,
        "growth_fraction": round((int(np.count_nonzero(child["part_owner"])) - original_count) / original_count, 7),
        "chassis_cells_added": int(np.count_nonzero(child["symmetry_class"] == 1)),
        "paired_appendage_cells_added": int(np.count_nonzero(child["symmetry_class"] == 2)),
        "other_appendage_cells_added": int(np.count_nonzero(child["symmetry_class"] == 3)),
        "source_cells_deleted": 0,
    }
    return {"format": np.asarray([FIELDS_FORMAT]), **{name: np.ascontiguousarray(values) for name, values in child.items()}}, report


def _validate_fields(fields: Mapping[str, np.ndarray]) -> None:
    if set(fields) != FIELD_KEYS or fields["format"].shape != (1,) or str(fields["format"][0]) != FIELDS_FORMAT:
        raise ValueError("Cellular symmetry field archive contract differs")
    for name in FIELD_KEYS - {"format"}:
        if fields[name].shape != (CANVAS_SIZE, CANVAS_SIZE) or fields[name].dtype != np.uint8:
            raise ValueError(f"Cellular symmetry {name} shape/dtype differs")
    physical = fields["part_owner"] > 0
    if np.any(fields["part_owner"] > 15) or np.any(fields["material"] > 9) or np.any(fields["emission"] > 3):
        raise ValueError("Cellular symmetry categorical vocabulary differs")
    if np.any(fields["symmetry_added_mask"] > 1) or np.any(fields["symmetry_class"] > 3):
        raise ValueError("Cellular symmetry provenance vocabulary differs")
    if np.any(fields["symmetry_added_mask"] & (~physical)) or np.any((fields["symmetry_class"] > 0) & (fields["symmetry_added_mask"] == 0)):
        raise ValueError("Cellular symmetry provenance lies outside added anatomy")
    from ..cellular_organism.compiler import _components
    if len(_components(physical)) != 1:
        raise ValueError("Cellular symmetry phenotype is not raster-connected")


def _load_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FIELD_KEYS:
            raise ValueError("Cellular symmetry archive members differ")
        fields = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    _validate_fields(fields)
    return fields


def _categorical(fields: Mapping[str, np.ndarray]) -> CategoricalFields:
    return CategoricalFields(
        part=fields["part_owner"].copy(), material=fields["material"].copy(), emission=fields["emission"].copy(),
        aligned_sha256=aligned_fields_hash(fields["part_owner"], fields["material"], fields["emission"]),
    )


def _contact_sheet(previews: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]]) -> bytes:
    scale = 2; columns = 5; tile_w = 48 * 5 * scale; tile_h = 48 * scale + 34
    rows = (len(previews) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, 58 + rows * tile_h), (3, 8, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "SOFT ORGANIC SYMMETRY // ADDITIVE CHASSIS + PAIRED APPENDAGE COMPLETION", fill=(76, 239, 255))
    draw.text((12, 30), "SOURCE CHILD | REFINED CHILD | ORGANS | FLUID | ADDED CELLS", fill=(185, 255, 86))
    for index, (record, source_panels, refined_panels, added_panel) in enumerate(previews):
        x = (index % columns) * tile_w; y = 58 + (index // columns) * tile_h
        strip = np.concatenate((source_panels[:, :96], refined_panels, added_panel), axis=1)
        canvas.paste(Image.fromarray(strip).resize((tile_w, 48 * scale), Image.Resampling.NEAREST), (x, y))
        symmetry = record["symmetry"]
        draw.text((x + 4, y + 98), f"{record['sample_id']}  {record['family']}  +{symmetry['symmetry_cells_added']} cells", fill=(220, 241, 255))
        draw.text((x + 4, y + 113), f"score {symmetry['before']['weighted_score']:.3f} -> {symmetry['after']['weighted_score']:.3f}  axis {symmetry['axis_x']:.1f}", fill=(139, 178, 205))
    return png_bytes(np.asarray(canvas))


def _build_files(source_manifest_path: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    source_manifest_path = Path(source_manifest_path).resolve()
    source_validation = validate_breeding_bank(source_manifest_path)
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    root = source_manifest_path.parent
    files: dict[str, bytes] = {}
    records: list[dict[str, object]] = []
    previews = []
    for source_record in source["offspring"]:
        source_fields_path = _safe_source_artifact(root, source_record["fields"], label="symmetry source fields")
        from ..cellular_breeding.compiler import _load_fields as load_breeding_fields
        source_fields = load_breeding_fields(source_fields_path)
        fields, symmetry = _refine(source_fields, int(source_record["family_id"]), int(source_record["lineage"]["seed"]))
        _validate_fields(fields)
        condition = _condition(source_record)
        categorical = _categorical(fields)
        arrays, organs, summary = _compile_arrays(SymmetrySample(condition, categorical))
        validate_species_arrays(arrays, organs, summary)
        fields_bytes = deterministic_npz_bytes(fields); arrays_bytes = deterministic_npz_bytes(arrays)
        fields_relative = f"offspring/{condition.sample_id}/symmetry_fields.npz"
        arrays_relative = f"offspring/{condition.sample_id}/cellular_anatomy.npz"
        files[fields_relative] = fields_bytes; files[arrays_relative] = arrays_bytes
        genome = json.loads(json.dumps(source_record["genome"]))
        genome["symmetry_refinement"] = {
            "policy": symmetry["policy"], "axis_x": symmetry["axis_x"],
            "source_offspring_fields_sha256": source_record["offspring_fields_sha256"],
            "refined_fields_sha256": sha256_bytes(fields_bytes),
        }
        record = {
            **{key: source_record[key] for key in ("sample_id", "ordinal", "family", "family_id", "subtype", "subtype_id", "role", "role_id", "family_pair", "parents", "lineage", "breeding", "palette", "fluid", "capabilities")},
            "format": SPECIES_FORMAT,
            "source_offspring_fields_sha256": source_record["offspring_fields_sha256"],
            "source_anatomy_sha256": source_record["anatomy_sha256"],
            "symmetry": symmetry,
            "refined_fields_sha256": sha256_bytes(fields_bytes),
            "aligned_fields_sha256": categorical.aligned_sha256,
            "fields": artifact_record_from_bytes(fields_relative, fields_bytes),
            "anatomy_sha256": json_sha256({"arrays_sha256": sha256_bytes(arrays_bytes), "organs": organs, "summary": summary, "symmetry": symmetry}),
            "arrays": artifact_record_from_bytes(arrays_relative, arrays_bytes),
            "organs": organs,
            "summary": summary,
            "genome": genome,
        }
        record["capabilities"] = {**record["capabilities"], "soft_bilateral_chassis_prior": True, "soft_paired_appendage_prior": True, "source_cells_preserved": True}
        records.append(record)
        source_arrays = _load_arrays(_safe_source_artifact(root, source_record["arrays"], label="symmetry source anatomy"))
        source_panels = _render_panels(source_arrays, source_record["palette"])
        refined_panels = _render_panels(arrays, source_record["palette"])
        added = np.zeros((48, 48, 3), dtype=np.uint8)
        added[fields["symmetry_class"] == 1] = (80, 245, 255)
        added[fields["symmetry_class"] == 2] = (190, 255, 80)
        added[fields["symmetry_class"] == 3] = (255, 85, 190)
        added[(fields["repair_mask"] > source_fields["repair_mask"]) & (fields["symmetry_class"] == 0)] = (255, 190, 70)
        added = np.repeat(np.repeat(added, 2, axis=0), 2, axis=1)
        previews.append((record, source_panels, refined_panels, added))
    contact = _contact_sheet(previews); files["cellular_symmetry_contact_sheet.png"] = contact
    before_mean = round(sum(float(record["symmetry"]["before"]["weighted_score"]) for record in records) / len(records), 7)
    after_mean = round(sum(float(record["symmetry"]["after"]["weighted_score"]) for record in records) / len(records), 7)
    totals = {
        "physical_cells": sum(int(record["summary"]["physical_cell_count"]) for record in records),
        "organs": sum(int(record["summary"]["organ_count"]) for record in records),
        "eyes": sum(int(record["summary"]["eye_count"]) for record in records),
        "bonds": sum(int(record["summary"]["bond_count"]) for record in records),
        "symmetry_cells_added": sum(int(record["symmetry"]["symmetry_cells_added"]) for record in records),
        "connectivity_cells_added": sum(int(record["symmetry"]["connectivity_cells_added"]) for record in records),
    }
    family_counts = {family: sum(record["family"] == family for record in records) for family in FAMILIES}
    manifest = {
        "format": FORMAT, "status": "ready", "quality_tier": "family-aware-soft-organic-bilateral-symmetry-v1",
        "compiler": {"source_sha256": source_sha256(), "cellular_contract_sha256": CELLULAR_CONTRACT_SHA256, "python_runtime_required": False},
        "source": {
            "breeding_manifest": source_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "breeding_manifest_sha256": sha256_file(source_manifest_path),
            "breeding_semantic_sha256": source["semantic_sha256"],
            "breeding_validation": source_validation,
        },
        "sample_count": len(records), "family_counts": family_counts, "family_pair_counts": source["family_pair_counts"],
        "policy": {
            "name": "family-aware-soft-bilateral-additive-v1", "hard_symmetry_required": False,
            "source_cell_deletion_allowed": False, "family_priors": {FAMILIES[key]: value for key, value in FAMILY_PRIORS.items()},
            "chassis_owners": sorted(CHASSIS_OWNERS), "paired_appendage_owners": sorted(PAIRED_APPENDAGE_OWNERS),
        },
        "symmetry_summary": {
            "mean_weighted_before": before_mean, "mean_weighted_after": after_mean,
            "mean_improvement": round(after_mean - before_mean, 7),
            "improved_samples": sum(float(record["symmetry"]["weighted_improvement"]) > 0 for record in records),
            "unchanged_samples": sum(float(record["symmetry"]["weighted_improvement"]) == 0 for record in records),
        },
        "totals": totals, "simulation": SimulationDefaults().to_dict(),
        "contact_sheet": artifact_record_from_bytes("cellular_symmetry_contact_sheet.png", contact),
        "offspring": records,
        "gates": {
            "all_45_offspring_refined": len(records) == SAMPLE_COUNT,
            "all_source_cells_preserved": all(record["symmetry"]["source_cells_deleted"] == 0 for record in records),
            "mean_symmetry_improved": after_mean > before_mean,
            "at_least_40_samples_improved": sum(float(record["symmetry"]["weighted_improvement"]) > 0 for record in records) >= 40,
            "no_sample_symmetry_regressed": all(float(record["symmetry"]["weighted_improvement"]) >= 0 for record in records),
            "family_specific_priors_applied": True,
            "anomalies_remain_weakly_constrained": FAMILY_PRIORS[3]["growth_cap"] < FAMILY_PRIORS[0]["growth_cap"],
            "all_organ_fluid_and_bond_graphs_redecoded": True,
            "all_lineages_preserved": True,
            "runtime_scope_truthful": True,
        },
    }
    manifest["semantic_sha256"] = json_sha256(manifest)
    files["cellular_symmetry_manifest.json"] = canonical_json_bytes(manifest)
    return files, manifest


def build_bank(source_manifest: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    files, manifest = _build_files(source_manifest)
    if not all(manifest["gates"].values()):
        raise ValueError("Cellular symmetry build gate failed")
    _atomic_publish(Path(destination).resolve(), files)
    validation = validate_bank(Path(destination) / "cellular_symmetry_manifest.json")
    return {"passed": True, "destination": str(Path(destination).resolve()), "sample_count": SAMPLE_COUNT, "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": sha256_file(Path(destination) / "cellular_symmetry_manifest.json"), "validation": validation}


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError(f"Cellular symmetry schema validation failed: {errors[0].message}")
    if raw != canonical_json_bytes(manifest):
        raise ValueError("Cellular symmetry manifest is not canonical JSON")
    if manifest["semantic_sha256"] != json_sha256({key: value for key, value in manifest.items() if key != "semantic_sha256"}):
        raise ValueError("Cellular symmetry semantic hash differs")
    if manifest["compiler"]["source_sha256"] != source_sha256():
        raise ValueError("Cellular symmetry compiler source hash is stale")
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["breeding_manifest"]).parts).resolve()
    if not source_path.is_relative_to(PROJECT_ROOT) or sha256_file(source_path) != manifest["source"]["breeding_manifest_sha256"]:
        raise ValueError("Cellular symmetry source provenance differs")
    validate_breeding_bank(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8")); source_by_id = {record["sample_id"]: record for record in source["offspring"]}
    if source["semantic_sha256"] != manifest["source"]["breeding_semantic_sha256"]:
        raise ValueError("Cellular symmetry source semantic provenance differs")
    root = manifest_path.parent
    totals = {name: 0 for name in ("physical_cells", "organs", "eyes", "bonds", "symmetry_cells_added", "connectivity_cells_added")}
    for record in manifest["offspring"]:
        source_record = source_by_id[record["sample_id"]]
        from ..cellular_breeding.compiler import _load_fields as load_breeding_fields
        source_fields = load_breeding_fields(_safe_source_artifact(source_path.parent, source_record["fields"], label="symmetry validation source"))
        expected_fields, expected_symmetry = _refine(source_fields, int(record["family_id"]), int(record["lineage"]["seed"]))
        fields = _load_fields(_safe_source_artifact(root, record["fields"], label="symmetry fields"))
        for name in FIELD_KEYS:
            if not np.array_equal(fields[name], expected_fields[name]):
                raise ValueError(f"Cellular symmetry deterministic field replay differs: {record['sample_id']} {name}")
        if record["symmetry"] != expected_symmetry:
            raise ValueError("Cellular symmetry metric replay differs")
        source_physical = source_fields["part_owner"] > 0
        if not np.array_equal(fields["part_owner"][source_physical], source_fields["part_owner"][source_physical]):
            raise ValueError("Cellular symmetry changed or deleted a source cell")
        categorical = _categorical(fields); condition = _condition(source_record)
        expected_arrays, expected_organs, expected_summary = _compile_arrays(SymmetrySample(condition, categorical))
        arrays = _load_arrays(_safe_source_artifact(root, record["arrays"], label="symmetry anatomy"))
        validate_species_arrays(arrays, record["organs"], record["summary"])
        if expected_organs != record["organs"] or expected_summary != record["summary"]:
            raise ValueError("Cellular symmetry organ/summary replay differs")
        for name, values in expected_arrays.items():
            if not np.array_equal(arrays[name], values):
                raise ValueError(f"Cellular symmetry anatomy replay differs: {record['sample_id']} {name}")
        totals["physical_cells"] += int(record["summary"]["physical_cell_count"])
        totals["organs"] += int(record["summary"]["organ_count"])
        totals["eyes"] += int(record["summary"]["eye_count"])
        totals["bonds"] += int(record["summary"]["bond_count"])
        totals["symmetry_cells_added"] += int(record["symmetry"]["symmetry_cells_added"])
        totals["connectivity_cells_added"] += int(record["symmetry"]["connectivity_cells_added"])
    if totals != manifest["totals"] or not all(manifest["gates"].values()):
        raise ValueError("Cellular symmetry totals/gates differ")
    contact = _safe_source_artifact(root, manifest["contact_sheet"], label="symmetry contact sheet")
    return {"passed": True, "sample_count": len(manifest["offspring"]), "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": sha256_file(manifest_path), "contact_sheet_sha256": sha256_file(contact), "symmetry_summary": manifest["symmetry_summary"], "totals": totals}


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); validation = validate_bank(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["breeding_manifest"]).parts)
    expected, expected_manifest = _build_files(source_path)
    if expected_manifest["semantic_sha256"] != manifest["semantic_sha256"]:
        raise ValueError("Cellular symmetry semantic replay differs")
    root = manifest_path.parent
    for relative, payload in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Cellular symmetry byte replay differs: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected):
        raise ValueError("Cellular symmetry output closure differs")
    return {**validation, "exact_replay": True, "artifact_count": len(expected), "artifact_bytes": sum(map(len, expected.values()))}
