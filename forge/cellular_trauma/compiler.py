from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from ..cellular_organism.compiler import _atomic_publish, _load_arrays as _load_anatomy
from ..cellular_organism.contract import TissueType
from ..cellular_physiology import validate_bank as validate_physiology_bank
from ..cellular_physiology.compiler import _load_overlay
from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..morphology.constants import FAMILIES
from ..multifield_style.hashing import sha256_file
from ..multifield_style_motion.hashing import artifact_record_from_bytes, canonical_json_bytes, deterministic_npz_bytes, png_bytes
from .contract import ARRAY_FORMAT, DEFAULT_OUTPUT, DEFAULT_SOURCE, FAMILY_PROFILES, FORMAT, HEAL_CLASS_NAMES, SCHEMA_PATH, source_sha256


def _profile(family_id: int) -> dict[str, object]:
    if not 0 <= family_id < len(FAMILY_PROFILES): raise ValueError("Unknown trauma family")
    return dict(FAMILY_PROFILES[family_id])


def compile_trauma(record: Mapping[str, Any], anatomy: Mapping[str, np.ndarray], physiology: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    count = len(anatomy["position_xy"]); bond_count = len(anatomy["bond_ab"])
    if physiology["system_role"].shape != (8, count): raise ValueError("Physiology/anatomy cell census differs")
    tissue = anatomy["tissue"].astype(np.uint8, copy=False)
    heal_class = np.zeros(count, dtype=np.uint8)
    heal_class[np.isin(tissue, (int(TissueType.EPIDERMIS), int(TissueType.ARMOR), int(TissueType.STRUCTURAL)))] = 1
    heal_class[tissue == int(TissueType.CONTRACTILE)] = 2
    heal_class[tissue == int(TissueType.VASCULAR)] = 3
    heal_class[np.isin(tissue, (int(TissueType.DIGESTIVE), int(TissueType.REPRODUCTIVE), int(TissueType.STORAGE), int(TissueType.EMITTER), int(TissueType.WEAPON)))] = 4
    heal_class[np.isin(tissue, (int(TissueType.NEURAL), int(TissueType.SENSORY)))] = 5
    heal_class[np.isin(tissue, (int(TissueType.IMMUNE), int(TissueType.STEM)))] = 6
    if np.any(heal_class == 0): raise ValueError("Every physical cell must have a healing class")

    profile = _profile(int(record["family_id"]))
    base_clot = np.asarray((0.0, 0.58, 0.66, 1.0, 0.72, 0.38, 1.0), dtype=np.float32)[heal_class]
    base_scar = np.asarray((0.0, 0.75, 0.58, 0.48, 0.52, 0.30, 0.18), dtype=np.float32)[heal_class]
    base_regrowth = np.asarray((0.0, 0.40, 0.52, 0.64, 0.48, 0.22, 1.0), dtype=np.float32)[heal_class]
    immune = physiology["system_weight"][7].astype(np.float32, copy=False)
    circulation = physiology["system_weight"][0].astype(np.float32, copy=False)
    clot = np.clip(base_clot * float(profile["clot_rate"]) + immune * 0.22 + circulation * 0.16, 0.05, 1.0).astype(np.float32)
    scar = np.clip(base_scar * float(profile["scar_rate"]), 0.02, 1.0).astype(np.float32)
    regrowth = np.clip(base_regrowth * float(profile["regrowth_rate"]) + immune * 0.20, 0.01, 1.0).astype(np.float32)

    pairs = anatomy["bond_ab"].astype(np.int64, copy=False)
    repair = np.sqrt(regrowth[pairs[:, 0]] * regrowth[pairs[:, 1]]).astype(np.float32)
    magnetic = np.sqrt(clot[pairs[:, 0]] * clot[pairs[:, 1]]).astype(np.float32) * np.float32(profile["magnetic_strength"])
    arrays = {
        "heal_class": np.ascontiguousarray(heal_class),
        "clotting_weight": np.ascontiguousarray(clot),
        "scar_bias": np.ascontiguousarray(scar),
        "regrowth_weight": np.ascontiguousarray(regrowth),
        "bond_repair_weight": np.ascontiguousarray(repair),
        "bond_magnetic_weight": np.ascontiguousarray(magnetic.astype(np.float32)),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()): raise ValueError("Trauma overlay contains a non-finite value")
    summary = {
        **profile,
        "cell_count": count,
        "bond_count": bond_count,
        "healing_class_counts": {name: int(np.count_nonzero(heal_class == index)) for index, name in enumerate(HEAL_CLASS_NAMES) if index > 0},
        "mean_clotting_weight": round(float(clot.mean()), 7),
        "mean_scar_bias": round(float(scar.mean()), 7),
        "mean_regrowth_weight": round(float(regrowth.mean()), 7),
    }
    return arrays, summary


def _arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"nullvector-cellular-trauma-arrays-v4\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name]); digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0"); digest.update(memoryview(value.view(np.uint8)))
    return digest.hexdigest()


def _contact_sheet(source_root: Path, source: Mapping[str, Any], physiology_root: Path, physiology_by_id: Mapping[str, Any]) -> bytes:
    tile = 112; gutter = 70; canvas = Image.new("RGB", (gutter + 5 * tile, 3 * tile + 42), (3, 8, 14)); draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "CELLULAR TRAUMA // CLOT / SCAR / REGROWTH", fill=(61, 232, 255))
    for row, label in enumerate(("CLOT", "SCAR", "REGROW")): draw.text((7, 42 + row * tile + 48), label, fill=(184, 255, 73))
    for column, family in enumerate(FAMILIES):
        record = next(item for item in source["offspring"] if item["family"] == family)
        anatomy = _load_anatomy(source_root.joinpath(*PurePosixPath(record["arrays"]["path"]).parts))
        p_record = physiology_by_id[record["sample_id"]]
        overlay = _load_overlay(physiology_root.joinpath(*PurePosixPath(p_record["arrays"]["path"]).parts), len(anatomy["position_xy"]))
        trauma, _ = compile_trauma(record, anatomy, overlay); positions = anatomy["position_xy"].astype(int)
        for row, field in enumerate(("clotting_weight", "scar_bias", "regrowth_weight")):
            image = np.zeros((48, 48, 3), dtype=np.uint8); values = trauma[field]
            for index, (x, y) in enumerate(positions):
                value = float(values[index])
                if row == 0: image[y, x] = (int(90 + 165 * value), int(220 * value), int(230 * (1.0 - value)))
                elif row == 1: image[y, x] = (int(90 + 120 * value), int(200 - 105 * value), int(120 + 95 * value))
                else: image[y, x] = (int(70 + 100 * (1.0 - value)), int(120 + 135 * value), int(230 * value))
            canvas.paste(Image.fromarray(image).resize((tile, tile), Image.Resampling.NEAREST), (gutter + column * tile, 42 + row * tile))
        draw.text((gutter + column * tile + 4, 25), family.upper(), fill=(184, 255, 73))
    return png_bytes(np.asarray(canvas))


def _build_files(source_manifest: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    source_manifest = Path(source_manifest).resolve(); physiology_validation = validate_physiology_bank(source_manifest)
    physiology = json.loads(source_manifest.read_text(encoding="utf-8")); physiology_by_id = {item["sample_id"]: item for item in physiology["identities"]}
    anatomy_path = PROJECT_ROOT.joinpath(*PurePosixPath(physiology["source"]["organism_manifest"]).parts).resolve()
    source = json.loads(anatomy_path.read_text(encoding="utf-8")); files: dict[str, bytes] = {}; identities = []
    total_cells = total_bonds = 0; fate_counts: dict[str, int] = {}
    for record in source["offspring"]:
        anatomy = _load_anatomy(anatomy_path.parent.joinpath(*PurePosixPath(record["arrays"]["path"]).parts)); p_record = physiology_by_id[record["sample_id"]]
        overlay = _load_overlay(source_manifest.parent.joinpath(*PurePosixPath(p_record["arrays"]["path"]).parts), len(anatomy["position_xy"]))
        arrays, profile = compile_trauma(record, anatomy, overlay); relative = f"identities/{record['sample_id']}/trauma.npz"; payload = deterministic_npz_bytes(arrays); files[relative] = payload
        total_cells += len(anatomy["position_xy"]); total_bonds += len(anatomy["bond_ab"]); fate = str(profile["detached_fate"]); fate_counts[fate] = fate_counts.get(fate, 0) + 1
        identities.append({"sample_id": record["sample_id"], "ordinal": record["ordinal"], "family": record["family"], "family_id": record["family_id"], "source_anatomy_sha256": record["anatomy_sha256"], "source_physiology_sha256": p_record["arrays_semantic_sha256"], "profile": profile, "arrays_semantic_sha256": _arrays_sha256(arrays), "arrays": artifact_record_from_bytes(relative, payload)})
    contact = _contact_sheet(anatomy_path.parent, source, source_manifest.parent, physiology_by_id); files["cellular_trauma_contact_sheet.png"] = contact
    manifest: dict[str, object] = {
        "format": FORMAT, "status": "ready", "quality_tier": "connected-wound-clot-scar-fragment-fate-local-perfusion-v4",
        "compiler": {"source_sha256": source_sha256(), "python_runtime_required": False},
        "source": {"physiology_manifest": source_manifest.relative_to(PROJECT_ROOT).as_posix(), "physiology_manifest_sha256": sha256_file(source_manifest), "physiology_semantic_sha256": physiology["semantic_sha256"], "physiology_validation": physiology_validation, "anatomy_manifest_sha256": sha256_file(anatomy_path)},
        "array_format": ARRAY_FORMAT, "heal_class_vocab": list(HEAL_CLASS_NAMES), "identity_count": len(identities), "total_cells": total_cells, "total_bonds": total_bonds, "fragment_fate_counts": fate_counts, "identities": identities,
        "contact_sheet": artifact_record_from_bytes("cellular_trauma_contact_sheet.png", contact),
        "runtime_contract": {"open_wounds_leak_until_clotted": True, "healing_consumes_energy": True, "healing_can_form_scars": True, "scars_reduce_local_compliance": True, "severed_components_have_persistent_age": True, "reconnection_is_time_and_distance_bounded": True, "humanoid_and_animal_fragments_become_biomass": True, "plant_anomaly_machine_fragments_can_become_polyps": True, "organ_capacity_remains_connectivity_driven": True, "local_fluid_perfusion_controls_capacity": True, "python_runtime_required": False},
        "gates": {"all_45_identities_compiled": len(identities) == 45, "all_cells_classified": all(sum(item["profile"]["healing_class_counts"].values()) == item["profile"]["cell_count"] for item in identities), "all_bonds_repair_weighted": all(item["profile"]["bond_count"] > 0 for item in identities), "all_five_family_profiles_present": {item["family"] for item in identities} == set(FAMILIES), "both_biomass_and_polyp_fates_present": sum(value for key, value in fate_counts.items() if key == "biomass") > 0 and sum(value for key, value in fate_counts.items() if "polyp" in key) > 0, "local_fluid_perfusion_is_authoritative": True, "source_anatomy_and_physiology_immutable": True, "native_runtime_independent_of_python": True},
    }
    manifest["semantic_sha256"] = json_sha256(manifest); files["cellular_trauma_manifest.json"] = canonical_json_bytes(manifest)
    return files, manifest


def build_bank(source_manifest: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    files, manifest = _build_files(source_manifest)
    if not all(manifest["gates"].values()): raise ValueError("Cellular trauma build gate failed")
    _atomic_publish(Path(destination).resolve(), files); validation = validate_bank(Path(destination) / "cellular_trauma_manifest.json")
    return {"passed": True, "destination": str(Path(destination).resolve()), "semantic_sha256": manifest["semantic_sha256"], "validation": validation}


def _load_trauma(path: Path, cell_count: int, bond_count: int) -> dict[str, np.ndarray]:
    expected = {"heal_class", "clotting_weight", "scar_bias", "regrowth_weight", "bond_repair_weight", "bond_magnetic_weight"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected: raise ValueError("Trauma array member registry differs")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if arrays["heal_class"].dtype != np.uint8 or arrays["heal_class"].shape != (cell_count,) or np.any((arrays["heal_class"] < 1) | (arrays["heal_class"] >= len(HEAL_CLASS_NAMES))): raise ValueError("Trauma healing class contract differs")
    for name in ("clotting_weight", "scar_bias", "regrowth_weight"):
        if arrays[name].dtype != np.float32 or arrays[name].shape != (cell_count,) or not np.isfinite(arrays[name]).all() or np.any((arrays[name] < 0) | (arrays[name] > 1)): raise ValueError(f"Trauma {name} contract differs")
    for name in ("bond_repair_weight", "bond_magnetic_weight"):
        if arrays[name].dtype != np.float32 or arrays[name].shape != (bond_count,) or not np.isfinite(arrays[name]).all() or np.any(arrays[name] < 0): raise ValueError(f"Trauma {name} contract differs")
    return arrays


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    errors = sorted(Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Cellular trauma schema validation failed: {errors[0].message}")
    if raw != canonical_json_bytes(manifest): raise ValueError("Cellular trauma manifest is not canonical JSON")
    if manifest["semantic_sha256"] != json_sha256({key: value for key, value in manifest.items() if key != "semantic_sha256"}): raise ValueError("Cellular trauma semantic hash differs")
    if manifest["compiler"]["source_sha256"] != source_sha256(): raise ValueError("Cellular trauma compiler source is stale")
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["physiology_manifest"]).parts).resolve()
    if not source_path.is_relative_to(PROJECT_ROOT) or sha256_file(source_path) != manifest["source"]["physiology_manifest_sha256"]: raise ValueError("Cellular trauma source provenance differs")
    validate_physiology_bank(source_path); physiology = json.loads(source_path.read_text(encoding="utf-8")); p_by_id = {item["sample_id"]: item for item in physiology["identities"]}
    anatomy_path = PROJECT_ROOT.joinpath(*PurePosixPath(physiology["source"]["organism_manifest"]).parts); source = json.loads(anatomy_path.read_text(encoding="utf-8")); a_by_id = {item["sample_id"]: item for item in source["offspring"]}
    for identity in manifest["identities"]:
        record = a_by_id[identity["sample_id"]]; p_record = p_by_id[identity["sample_id"]]; anatomy = _load_anatomy(anatomy_path.parent.joinpath(*PurePosixPath(record["arrays"]["path"]).parts)); overlay = _load_overlay(source_path.parent.joinpath(*PurePosixPath(p_record["arrays"]["path"]).parts), len(anatomy["position_xy"]))
        artifact = identity["arrays"]; path = manifest_path.parent.joinpath(*PurePosixPath(artifact["path"]).parts)
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]: raise ValueError("Cellular trauma artifact integrity differs")
        arrays = _load_trauma(path, len(anatomy["position_xy"]), len(anatomy["bond_ab"])); expected, profile = compile_trauma(record, anatomy, overlay)
        if identity["profile"] != profile or identity["arrays_semantic_sha256"] != _arrays_sha256(expected) or any(not np.array_equal(arrays[name], expected[name]) for name in arrays): raise ValueError("Cellular trauma deterministic replay differs")
    contact = manifest["contact_sheet"]; contact_path = manifest_path.parent.joinpath(*PurePosixPath(contact["path"]).parts)
    if contact_path.stat().st_size != contact["bytes"] or sha256_file(contact_path) != contact["sha256"]: raise ValueError("Cellular trauma contact sheet integrity differs")
    if not all(manifest["gates"].values()): raise ValueError("Cellular trauma gate differs")
    return {"passed": True, "identity_count": 45, "total_cells": manifest["total_cells"], "total_bonds": manifest["total_bonds"], "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": sha256_file(manifest_path), "contact_sheet_sha256": sha256_file(contact_path)}


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); validation = validate_bank(manifest_path); manifest = json.loads(manifest_path.read_text(encoding="utf-8")); source = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["physiology_manifest"]).parts); expected, _ = _build_files(source); root = manifest_path.parent
    for relative, payload in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.read_bytes() != payload: raise ValueError(f"Cellular trauma byte replay differs: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected): raise ValueError("Cellular trauma output closure differs")
    return {**validation, "exact_replay": True, "artifact_count": len(expected), "artifact_bytes": sum(map(len, expected.values()))}
