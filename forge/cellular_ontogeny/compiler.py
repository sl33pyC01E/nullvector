from __future__ import annotations

from collections import deque
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from ..cellular_organism.compiler import _load_arrays as load_anatomy_arrays, validate_species_arrays
from ..cellular_symmetry import validate_bank as validate_symmetry_bank
from ..config import PROJECT_ROOT
from ..maps.io import array_digest, file_sha256
from ..multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from ..safety import require_disk_floor
from .contract import ARRAY_FORMAT, ARRAY_NAMES, DEFAULT_OUTPUT, DEFAULT_SOURCE, FORMAT, PROGRAM_FORMAT, SCHEMA_PATH, STAGE_FRACTIONS, STAGES, source_sha256


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT_ROOT): raise ValueError("Ontogeny source must be inside project root")
    return path.relative_to(PROJECT_ROOT).as_posix()


def _artifact(root: Path, record: Mapping[str, object]) -> Path:
    relative = str(record["path"]); pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative: raise ValueError("Unsafe ontogeny artifact path")
    path = root.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError("Ontogeny artifact missing/outside root")
    if path.stat().st_size != int(record["bytes"]) or file_sha256(path) != record["sha256"]: raise ValueError("Ontogeny artifact size/hash differs")
    return path


def _source_artifact(manifest_path: Path, record: Mapping[str, object]) -> Path:
    return _artifact(manifest_path.parent, record)


def _adjacency(cell_count: int, pairs: np.ndarray) -> list[list[int]]:
    graph = [[] for _ in range(cell_count)]
    for a_value, b_value in pairs:
        a, b = int(a_value), int(b_value); graph[a].append(b); graph[b].append(a)
    for neighbors in graph: neighbors.sort()
    return graph


def _lineages(tissue: np.ndarray) -> np.ndarray:
    # ectoderm / mesoderm / endoderm / germline / exotic-specialized
    result = np.full(len(tissue), 1, dtype=np.uint8)
    result[np.isin(tissue, (2, 3, 6, 10, 13, 14))] = 2
    result[np.isin(tissue, (7, 9))] = 3
    result[tissue == 8] = 4
    result[np.isin(tissue, (11, 12))] = 5
    return result


def _mirror_partners(positions: np.ndarray, axis_x: float) -> np.ndarray:
    partners = np.full(len(positions), -1, dtype=np.int32)
    for index, (x, y) in enumerate(positions):
        target_x = 2.0 * axis_x - float(x)
        delta = np.abs(positions[:, 0].astype(np.float64) - target_x) + np.abs(positions[:, 1].astype(np.float64) - float(y)) * 1.25
        candidate = int(np.argmin(delta))
        if float(delta[candidate]) <= 1.5 and candidate != index: partners[index] = candidate
    return partners


def _connected(mask: np.ndarray, graph: list[list[int]]) -> bool:
    indices = np.flatnonzero(mask)
    if not len(indices): return False
    seen = {int(indices[0])}; queue = deque(seen)
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if mask[neighbor] and neighbor not in seen: seen.add(neighbor); queue.append(neighbor)
    return len(seen) == len(indices)


def _symmetry_score(positions: np.ndarray, mask: np.ndarray, axis_x: float) -> float:
    active = {tuple(map(int, positions[index])) for index in np.flatnonzero(mask)}
    if not active: return 0.0
    matched = sum((int(round(2.0 * axis_x - x)), y) in active for x, y in active)
    return round(matched / len(active), 7)


def _program(arrays: dict[str, np.ndarray], organs: list[dict[str, object]], seed: int) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, object]]:
    positions = arrays["position_xy"]; organ_ids = arrays["organ_id"]; pairs = arrays["bond_ab"]
    count = len(positions); graph = _adjacency(count, pairs); center = positions.astype(np.float64).mean(axis=0)
    organ_kind = {int(record["id"]): str(record["kind"]) for record in organs}
    priority = {"reproductive": 0, "circulatory": 1, "neural": 1, "digestive": 1, "vascular": 1, "integument": 2, "contractile": 2, "armor": 3, "sensory": 3, "appendage": 4, "locomotor": 4, "emitter": 5, "weapon": 5}
    repro = np.flatnonzero(np.asarray([organ_kind.get(int(value), "") == "reproductive" for value in organ_ids]))
    candidates = repro if len(repro) else np.arange(count)
    root = int(candidates[np.argmin(np.sum((positions[candidates].astype(np.float64) - center) ** 2, axis=1))])
    axis_x = float(center[0]); partners = _mirror_partners(positions, axis_x)
    order: list[int] = [root]; selected = {root}; parents = np.full(count, -1, dtype=np.int32); frontier = set(graph[root]); last = root
    stable = lambda index: int.from_bytes(hashlib.sha256(f"{seed}:ontogeny:{index}".encode("ascii")).digest()[:4], "big")
    while len(order) < count:
        if not frontier: raise ValueError("Adult bond graph is disconnected during ontogeny")
        preferred_partner = int(partners[last])
        chosen = min(frontier, key=lambda index: (
            0 if index == preferred_partner else 1,
            priority.get(organ_kind.get(int(organ_ids[index]), ""), 3),
            round(float(np.sum((positions[index].astype(np.float64) - center) ** 2)), 6),
            stable(index), index,
        ))
        earlier_neighbors = [neighbor for neighbor in graph[chosen] if neighbor in selected]
        parents[chosen] = min(earlier_neighbors, key=lambda item: (order.index(item), item))
        selected.add(chosen); order.append(chosen); frontier.remove(chosen)
        frontier.update(neighbor for neighbor in graph[chosen] if neighbor not in selected); last = chosen
    birth_order = np.empty(count, dtype=np.uint16)
    for rank, index in enumerate(order): birth_order[index] = rank
    thresholds = []
    previous = 0
    for fraction in STAGE_FRACTIONS:
        threshold = min(count, max(previous + 1, 4 if not thresholds else 1, int(math.ceil(count * fraction))))
        thresholds.append(threshold); previous = threshold
    thresholds[-1] = count
    activation = np.empty(count, dtype=np.uint8)
    for index in range(count): activation[index] = next(stage for stage, threshold in enumerate(thresholds) if int(birth_order[index]) < threshold)
    lineage = _lineages(arrays["tissue"])
    differentiation = np.asarray((birth_order.astype(np.float64) + 0.5) / count, dtype=np.float32)
    bond_stage = np.maximum(activation[pairs[:, 0]], activation[pairs[:, 1]]).astype(np.uint8)
    x = positions[:, 0].astype(np.float32); y = positions[:, 1].astype(np.float32)
    x_span = max(float(x.max() - x.min()), 1.0); y_span = max(float(y.max() - y.min()), 1.0)
    morph_lr = np.asarray((x - float(x.min())) / x_span, dtype=np.float32)
    morph_ap = np.asarray((y - float(y.min())) / y_span, dtype=np.float32)
    distance = np.sqrt((x - float(center[0])) ** 2 + (y - float(center[1])) ** 2); morph_core = np.asarray(1.0 - distance / max(float(distance.max()), 1.0), dtype=np.float32)
    program_arrays = {
        "birth_order": birth_order, "activation_stage": activation, "parent_cell": parents,
        "lineage_id": lineage, "differentiation_time": differentiation, "bond_activation_stage": bond_stage,
        "morphogen_lr": morph_lr, "morphogen_ap": morph_ap, "morphogen_core": morph_core,
    }
    stages = []
    essential_ids = {int(record["id"]) for record in organs if bool(record.get("essential", False))}
    for stage, name in enumerate(STAGES):
        mask = activation <= stage; active_bonds = bond_stage <= stage; visible_organs = set(map(int, organ_ids[mask].tolist()))
        functional = []
        for record in organs:
            members = organ_ids == int(record["id"])
            if np.count_nonzero(mask & members) / max(1, np.count_nonzero(members)) >= 0.60: functional.append(int(record["id"]))
        stages.append({
            "id": stage, "name": name, "cell_count": int(mask.sum()), "cell_fraction": round(float(mask.mean()), 7),
            "bond_count": int(active_bonds.sum()), "visible_organ_count": len(visible_organs), "functional_organ_count": len(functional),
            "functional_essential_count": len(essential_ids & set(functional)), "symmetry_score": _symmetry_score(positions, mask, axis_x),
        })
    paired = [(index, int(partner)) for index, partner in enumerate(partners) if partner > index]
    pair_stage_delta = [abs(int(activation[a]) - int(activation[b])) for a, b in paired]
    metrics = {
        "root_cell": root, "axis_x": round(axis_x, 7), "lineage_count": int(len(set(map(int, lineage)))),
        "paired_cell_count": len(paired) * 2, "mean_paired_stage_delta": round(float(np.mean(pair_stage_delta)) if pair_stage_delta else 0.0, 7),
        "maximum_paired_stage_delta": max(pair_stage_delta, default=0), "adult_cell_count": count, "adult_bond_count": len(pairs),
    }
    # Hard lineage invariants: every non-root cell descends through a real earlier bond.
    pair_set = {tuple(sorted(map(int, pair))) for pair in pairs}
    if sorted(map(int, birth_order)) != list(range(count)): raise ValueError("Ontogeny birth order is not a permutation")
    for cell in range(count):
        parent = int(parents[cell])
        if cell == root:
            if parent != -1: raise ValueError("Ontogeny root has a parent")
        elif parent < 0 or int(birth_order[parent]) >= int(birth_order[cell]) or tuple(sorted((cell, parent))) not in pair_set:
            raise ValueError("Ontogeny parent is not an earlier bonded cell")
    for stage in range(len(STAGES)):
        if not _connected(activation <= stage, graph): raise ValueError(f"Ontogeny stage {stage} is disconnected")
    return program_arrays, stages, metrics


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = BytesIO(); np.savez_compressed(buffer, format=np.asarray([ARRAY_FORMAT]), **{name: arrays[name] for name in ARRAY_NAMES}); return buffer.getvalue()


def _load_program_arrays(path: Path, cell_count: int, bond_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"format", *ARRAY_NAMES} or archive["format"].shape != (1,) or str(archive["format"][0]) != ARRAY_FORMAT: raise ValueError("Ontogeny archive member contract differs")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in ARRAY_NAMES}
    cell_types = {"birth_order": np.uint16, "activation_stage": np.uint8, "parent_cell": np.int32, "lineage_id": np.uint8, "differentiation_time": np.float32, "morphogen_lr": np.float32, "morphogen_ap": np.float32, "morphogen_core": np.float32}
    for name, dtype in cell_types.items():
        if arrays[name].shape != (cell_count,) or arrays[name].dtype != dtype: raise ValueError(f"Ontogeny cell array contract differs: {name}")
    if arrays["bond_activation_stage"].shape != (bond_count,) or arrays["bond_activation_stage"].dtype != np.uint8: raise ValueError("Ontogeny bond stage contract differs")
    if np.any(arrays["activation_stage"] > 5) or np.any(arrays["bond_activation_stage"] > 5) or np.any(arrays["lineage_id"] < 1) or np.any(arrays["lineage_id"] > 5): raise ValueError("Ontogeny categorical range differs")
    for name in ("differentiation_time", "morphogen_lr", "morphogen_ap", "morphogen_core"):
        if not np.isfinite(arrays[name]).all() or np.any(arrays[name] < 0) or np.any(arrays[name] > 1): raise ValueError(f"Ontogeny continuous range differs: {name}")
    return arrays


def _render_stage(anatomy: dict[str, np.ndarray], program: dict[str, np.ndarray], stage: int, scale: int = 2) -> Image.Image:
    canvas = np.zeros((48, 48, 3), dtype=np.uint8); positions = anatomy["position_xy"]; active = program["activation_stage"] <= stage
    lineage_colors = np.asarray(((0, 0, 0), (74, 213, 255), (255, 99, 143), (255, 183, 68), (199, 100, 255), (104, 255, 126)), dtype=np.uint8)
    for index in np.flatnonzero(active):
        x, y = map(int, positions[index]); base = lineage_colors[int(program["lineage_id"][index])]
        strength = 0.48 + 0.52 * float(program["morphogen_core"][index]); canvas[y, x] = np.asarray(base * strength, dtype=np.uint8)
    image = Image.fromarray(canvas).resize((48 * scale, 48 * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image); draw.text((3, 3), STAGES[stage].upper(), fill=(225, 247, 255)); return image


def _contact_sheet(compiled: list[tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]]) -> bytes:
    scale = 2; tile_w = 48 * scale * len(STAGES); tile_h = 48 * scale + 18; columns = 3; rows = math.ceil(len(compiled) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (2, 7, 13)); draw = ImageDraw.Draw(sheet)
    for ordinal, (record, anatomy, program) in enumerate(compiled):
        x0 = (ordinal % columns) * tile_w; y0 = (ordinal // columns) * tile_h
        draw.text((x0 + 3, y0 + 3), f"{record['sample_id']} // {record['family'].upper()} // SOFT SYMMETRY", fill=(125, 224, 239))
        for stage in range(len(STAGES)): sheet.paste(_render_stage(anatomy, program, stage, scale), (x0 + stage * 48 * scale, y0 + 18))
    buffer = BytesIO(); sheet.save(buffer, format="PNG", optimize=False, compress_level=9); return buffer.getvalue()


def _build_files(source_manifest: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    source_manifest = Path(source_manifest).resolve(); validation = validate_symmetry_bank(source_manifest); source = json.loads(source_manifest.read_text(encoding="utf-8"))
    files: dict[str, bytes] = {}; records = []; compiled = []; totals = {"cells": 0, "bonds": 0, "lineage_edges": 0, "paired_cells": 0}
    for source_record in source["offspring"]:
        anatomy_path = _source_artifact(source_manifest, source_record["arrays"]); anatomy = load_anatomy_arrays(anatomy_path)
        validate_species_arrays(anatomy, source_record["organs"], source_record["summary"])
        seed = int(source_record["lineage"]["seed"]); program, stages, metrics = _program(anatomy, source_record["organs"], seed)
        relative = f"programs/{source_record['sample_id']}/ontogeny.npz"; payload = _npz_bytes(program); files[relative] = payload
        record = {
            "format": PROGRAM_FORMAT, "sample_id": source_record["sample_id"], "ordinal": source_record["ordinal"],
            "family": source_record["family"], "family_id": source_record["family_id"], "role": source_record["role"], "role_id": source_record["role_id"],
            "source_anatomy_sha256": source_record["anatomy_sha256"], "source_arrays_sha256": source_record["arrays"]["sha256"],
            "cell_count": source_record["summary"]["physical_cell_count"], "bond_count": source_record["summary"]["bond_count"],
            "development_seconds": round(float(source_record["genome"]["gestation_seconds"]) * 1.8, 6),
            "stages": stages, "metrics": metrics,
            "artifact": {"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload), "array_sha256": array_digest(program)},
            "gates": {"adult_anatomy_immutable": True, "birth_order_permutation": True, "every_cell_has_bonded_earlier_parent": True, "all_stages_connected": True, "adult_stage_exact": True, "soft_paired_growth_not_hard_mirroring": metrics["mean_paired_stage_delta"] <= 1.5},
        }
        if not all(record["gates"].values()): raise ValueError(f"Ontogeny gates failed: {record['sample_id']}")
        records.append(record); compiled.append((record, anatomy, program)); totals["cells"] += len(anatomy["position_xy"]); totals["bonds"] += len(anatomy["bond_ab"]); totals["lineage_edges"] += len(anatomy["position_xy"]) - 1; totals["paired_cells"] += int(metrics["paired_cell_count"])
    contact = _contact_sheet(compiled); files["cellular_ontogeny_contact_sheet.png"] = contact
    manifest: dict[str, object] = {
        "format": FORMAT, "status": "ready", "quality_tier": "deterministic-connected-soft-symmetric-cell-lineage-v1",
        "compiler": {"source_sha256": source_sha256(), "growth_policy": "bond-frontier-organ-priority-mirror-cohort-v1"},
        "source": {"manifest": _relative(source_manifest), "manifest_sha256": file_sha256(source_manifest), "semantic_sha256": source["semantic_sha256"], "sample_count": validation["sample_count"]},
        "stages": [{"id": index, "name": name, "target_fraction": STAGE_FRACTIONS[index]} for index, name in enumerate(STAGES)],
        "lineages": [{"id": 1, "name": "ectoderm"}, {"id": 2, "name": "mesoderm"}, {"id": 3, "name": "endoderm"}, {"id": 4, "name": "germline"}, {"id": 5, "name": "specialized"}],
        "program_count": len(records), "programs": records, "totals": totals,
        "contact_sheet": {"path": "cellular_ontogeny_contact_sheet.png", "bytes": len(contact), "sha256": sha256_bytes(contact)},
        "runtime_contract": {"children_begin_as_connected_zygotes": True, "cells_activate_from_real_bonded_parents": True, "organs_differentiate_progressively": True, "symmetry_emerges_as_soft_paired_cohorts": True, "adult_arrays_are_never_modified": True, "growth_uses_existing_spring_physics": True, "damage_can_interrupt_development": True},
        "gates": {"all_45_programs": len(records) == 45, "all_adults_source_exact": True, "all_lineage_trees_connected": True, "all_stage_masks_connected": True, "all_programs_exact_replayable": True, "soft_symmetry_preserved": True},
    }
    manifest["semantic_sha256"] = sha256_bytes(canonical_json_bytes(manifest)); files["cellular_ontogeny_manifest.json"] = canonical_json_bytes(manifest)
    return files, manifest


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent, planned_bytes=sum(map(len, files.values())) + 256 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"; staging.mkdir(parents=True)
    for relative, payload in sorted(files.items()):
        target = staging.joinpath(*PurePosixPath(relative).parts); target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, target)
    os.replace(staging, destination)


def build_bank(source_manifest: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    first, manifest = _build_files(source_manifest); second, _ = _build_files(source_manifest)
    if first != second: raise ValueError("Ontogeny build is not byte deterministic")
    _publish(destination, first); validation = validate_bank(Path(destination) / "cellular_ontogeny_manifest.json")
    return {"passed": True, "destination": str(Path(destination).resolve()), "program_count": 45, "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": file_sha256(Path(destination) / "cellular_ontogeny_manifest.json"), "validation": validation}


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8")); errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Ontogeny schema validation failed: {errors[0].message}")
    if raw != canonical_json_bytes(manifest): raise ValueError("Ontogeny manifest is not canonical JSON")
    semantic = {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    if manifest["semantic_sha256"] != sha256_bytes(canonical_json_bytes(semantic)): raise ValueError("Ontogeny semantic hash differs")
    if manifest["compiler"]["source_sha256"] != source_sha256(): raise ValueError("Ontogeny compiler source is stale")
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["manifest"]).parts); validate_symmetry_bank(source_path)
    if file_sha256(source_path) != manifest["source"]["manifest_sha256"]: raise ValueError("Ontogeny adult source differs")
    source = json.loads(source_path.read_text(encoding="utf-8")); source_by_id = {record["sample_id"]: record for record in source["offspring"]}; root = manifest_path.parent
    totals = {"cells": 0, "bonds": 0, "lineage_edges": 0, "paired_cells": 0}
    for record in manifest["programs"]:
        adult = source_by_id[record["sample_id"]]; anatomy = load_anatomy_arrays(_source_artifact(source_path, adult["arrays"])); expected, stages, metrics = _program(anatomy, adult["organs"], int(adult["lineage"]["seed"])); observed = _load_program_arrays(_artifact(root, record["artifact"]), len(anatomy["position_xy"]), len(anatomy["bond_ab"]))
        for name in ARRAY_NAMES:
            if not np.array_equal(expected[name], observed[name]): raise ValueError(f"Ontogeny array replay differs: {record['sample_id']} {name}")
        if record["stages"] != stages or record["metrics"] != metrics or not all(record["gates"].values()): raise ValueError(f"Ontogeny record replay differs: {record['sample_id']}")
        totals["cells"] += len(anatomy["position_xy"]); totals["bonds"] += len(anatomy["bond_ab"]); totals["lineage_edges"] += len(anatomy["position_xy"]) - 1; totals["paired_cells"] += int(metrics["paired_cell_count"])
    if totals != manifest["totals"] or len(manifest["programs"]) != 45 or not all(manifest["gates"].values()): raise ValueError("Ontogeny totals/gates differ")
    _artifact(root, manifest["contact_sheet"])
    return {"passed": True, "program_count": 45, "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": file_sha256(manifest_path), "contact_sheet_sha256": manifest["contact_sheet"]["sha256"], "totals": totals}


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); validation = validate_bank(manifest_path); manifest = json.loads(manifest_path.read_text(encoding="utf-8")); source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["manifest"]).parts)
    expected, expected_manifest = _build_files(source_path)
    if expected_manifest["semantic_sha256"] != manifest["semantic_sha256"]: raise ValueError("Ontogeny semantic replay differs")
    root = manifest_path.parent
    for relative, payload in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.read_bytes() != payload: raise ValueError(f"Ontogeny byte replay differs: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected): raise ValueError("Ontogeny output closure differs")
    return {**validation, "exact_replay": True, "artifact_count": len(expected), "artifact_bytes": sum(map(len, expected.values()))}
