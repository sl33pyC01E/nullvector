from __future__ import annotations

from collections import deque
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from ..cellular_organism.compiler import _atomic_publish, _load_arrays
from ..cellular_organism.contract import CellFlag, TissueType
from ..cellular_symmetry import validate_bank as validate_symmetry_bank
from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..morphology.constants import FAMILIES
from ..multifield_style.hashing import sha256_file
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    png_bytes,
)
from .contract import (
    ARRAY_FORMAT,
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    DEPENDENCIES,
    FORMAT,
    ROLE_NAMES,
    SCHEMA_PATH,
    SYSTEM_NAMES,
    source_sha256,
)


ROLE_CORE = 1
ROLE_CONDUIT = 2
ROLE_EFFECTOR = 3
ROLE_COLORS = np.asarray(((0, 0, 0), (255, 83, 116), (55, 216, 255), (184, 255, 73)), dtype=np.uint8)


def _adjacency(arrays: Mapping[str, np.ndarray]) -> list[list[int]]:
    result = [[] for _ in range(len(arrays["position_xy"]))]
    for a_raw, b_raw in arrays["bond_ab"]:
        a, b = int(a_raw), int(b_raw)
        result[a].append(b); result[b].append(a)
    for values in result:
        values.sort()
    return result


def _shortest_path(adjacency: list[list[int]], starts: set[int], targets: set[int]) -> list[int]:
    if starts & targets:
        return [min(starts & targets)]
    queue = deque(sorted(starts)); parent = {index: -1 for index in starts}; found = -1
    while queue and found < 0:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor in targets:
                found = neighbor; break
            queue.append(neighbor)
    if found < 0:
        raise ValueError("Anatomy graph cannot connect required organ systems")
    path = [found]
    while parent[path[-1]] >= 0:
        path.append(parent[path[-1]])
    return list(reversed(path))


def _organ_cells(record: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, set[int]]:
    organ_id = arrays["organ_id"]
    result: dict[str, set[int]] = {}
    for organ in record["organs"]:
        result.setdefault(str(organ["kind"]), set()).update(
            map(int, np.flatnonzero(organ_id == int(organ["id"])))
        )
    return result


def _surface_cells(arrays: Mapping[str, np.ndarray]) -> set[int]:
    positions = arrays["position_xy"].astype(np.int64)
    lookup = {tuple(map(int, position)): index for index, position in enumerate(positions)}
    result: set[int] = set()
    for index, (x_raw, y_raw) in enumerate(positions):
        x, y = int(x_raw), int(y_raw)
        if any((x + dx, y + dy) not in lookup for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            result.add(index)
    return result


def _path_union(adjacency: list[list[int]], starts: set[int], targets: set[int]) -> set[int]:
    result: set[int] = set()
    for start in sorted(starts):
        result.update(_shortest_path(adjacency, {start}, targets))
    return result


def _respiratory_exchange(
    record: Mapping[str, Any], arrays: Mapping[str, np.ndarray], surface: set[int], heart: set[int]
) -> set[int]:
    tissue = arrays["tissue"]; flags = arrays["cell_flags"]; emission = arrays["emission"]
    positions = arrays["position_xy"].astype(np.float64)
    family = str(record["family"])
    candidates = set(surface)
    if family == "plantlike":
        preferred = {index for index in candidates if int(flags[index]) & int(CellFlag.PHOTOSYNTHETIC)}
    elif family == "anomaly":
        preferred = {index for index in candidates if int(emission[index]) > 0 or int(tissue[index]) == int(TissueType.EMITTER)}
    elif family == "machine":
        preferred = {index for index in candidates if int(tissue[index]) in (int(TissueType.VASCULAR), int(TissueType.EMITTER))}
    else:
        y_values = positions[:, 1]; low, high = np.quantile(y_values, (0.18, 0.62))
        preferred = {index for index in candidates if low <= positions[index, 1] <= high}
    candidates = preferred or candidates
    heart_center = positions[sorted(heart)].mean(axis=0)
    target_count = min(len(candidates), max(4, min(16, len(positions) // 24)))
    ordered = sorted(
        candidates,
        key=lambda index: (
            float(np.square(positions[index] - heart_center).sum()),
            abs(float(positions[index, 0] - heart_center[0])),
            index,
        ),
    )
    # Alternate left/right where possible, then fill deterministically.
    left = [index for index in ordered if positions[index, 0] <= heart_center[0]]
    right = [index for index in ordered if positions[index, 0] > heart_center[0]]
    selected: list[int] = []
    for offset in range(max(len(left), len(right))):
        if offset < len(left): selected.append(left[offset])
        if offset < len(right): selected.append(right[offset])
        if len(selected) >= target_count: break
    return set(selected[:target_count])


def compile_systems(
    record: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    count = len(arrays["position_xy"]); adjacency = _adjacency(arrays)
    organ_cells = _organ_cells(record, arrays); tissue = arrays["tissue"]
    surface = _surface_cells(arrays)
    heart = set(organ_cells.get("circulatory", set()))
    brain = set(organ_cells.get("neural", set()))
    gut = set(organ_cells.get("digestive", set()))
    reproductive = set(organ_cells.get("reproductive", set()))
    sensory = set(organ_cells.get("sensory", set()))
    locomotor = set().union(*(organ_cells.get(kind, set()) for kind in ("contractile", "appendage", "locomotor", "weapon")))
    if not heart or not brain or not gut or not reproductive or not sensory or not locomotor:
        raise ValueError(f"Species {record['sample_id']} lacks a required organ substrate")

    vascular = set(map(int, np.flatnonzero(tissue == int(TissueType.VASCULAR))))
    respiration = _respiratory_exchange(record, arrays, surface, heart)
    immune_seed = set(map(int, np.flatnonzero(np.isin(tissue, (int(TissueType.IMMUNE), int(TissueType.STEM))))))
    if not immune_seed:
        positions = arrays["position_xy"].astype(np.float64); center = positions[sorted(heart)].mean(axis=0)
        immune_seed = set(sorted(vascular or surface, key=lambda index: (float(np.square(positions[index] - center).sum()), index))[:max(2, min(6, count // 100 + 2))])

    definitions: dict[str, tuple[set[int], set[int], set[int]]] = {}
    definitions["circulation"] = (heart, vascular | heart, set())
    respiration_paths = _path_union(adjacency, respiration, heart)
    definitions["respiration"] = (respiration, respiration | respiration_paths, respiration)
    digestion_paths = _path_union(adjacency, gut, heart)
    definitions["digestion"] = (gut, gut | digestion_paths, {index for index in gut if int(arrays["cell_flags"][index]) & int(CellFlag.MOUTH)})
    neural_paths = _path_union(adjacency, brain, heart)
    definitions["neural"] = (brain, brain | neural_paths, set())
    sensory_paths = _path_union(adjacency, sensory, brain)
    definitions["sensory"] = (brain, brain | sensory | sensory_paths, sensory)
    locomotor_roots = {index for index in locomotor if any(neighbor not in locomotor for neighbor in adjacency[index])} or {min(locomotor)}
    locomotor_paths = _path_union(adjacency, locomotor_roots, brain)
    definitions["locomotion"] = (brain, brain | locomotor | locomotor_paths, locomotor)
    reproduction_paths = _path_union(adjacency, reproductive, heart)
    definitions["reproduction"] = (reproductive, reproductive | reproduction_paths, reproductive)
    immune_paths = _path_union(adjacency, immune_seed, heart)
    definitions["immune"] = (immune_seed, immune_seed | immune_paths, immune_seed)

    roles = np.zeros((len(SYSTEM_NAMES), count), dtype=np.uint8)
    weights = np.zeros((len(SYSTEM_NAMES), count), dtype=np.float32)
    membership = np.zeros(count, dtype=np.uint16)
    records: list[dict[str, object]] = []
    for system_id, name in enumerate(SYSTEM_NAMES):
        core, members, effectors = definitions[name]
        if not core or not members or not core <= members or not effectors <= members:
            raise ValueError(f"System {name} violates its cell-role contract")
        conduit = members - core - effectors
        roles[system_id, sorted(conduit)] = ROLE_CONDUIT
        roles[system_id, sorted(effectors)] = ROLE_EFFECTOR
        roles[system_id, sorted(core)] = ROLE_CORE
        weights[system_id, sorted(conduit)] = 0.55
        weights[system_id, sorted(effectors)] = 0.85
        weights[system_id, sorted(core)] = 1.0
        membership[sorted(members)] |= np.uint16(1 << system_id)
        records.append({
            "id": system_id, "name": name, "dependencies": list(DEPENDENCIES[name]),
            "core_count": len(core), "conduit_count": len(conduit),
            "effector_count": len(effectors), "member_count": len(members),
            "baseline_weight": round(float(weights[system_id].sum()), 7),
        })
    return {
        "system_membership": membership,
        "system_role": np.ascontiguousarray(roles),
        "system_weight": np.ascontiguousarray(weights),
    }, records


def _overlay_sha(arrays: Mapping[str, np.ndarray]) -> str:
    import hashlib
    digest = hashlib.sha256(b"nullvector-connected-physiology-arrays-v1\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(memoryview(value.view(np.uint8)))
    return digest.hexdigest()


def _contact_sheet(source_root: Path, source: Mapping[str, Any]) -> bytes:
    cell = 96; header = 46; canvas = Image.new("RGB", (len(SYSTEM_NAMES) * cell, header + 5 * cell), (3, 8, 14)); draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "CONNECTED PIXEL-CELL PHYSIOLOGY // CORE / CONDUIT / EXCHANGE", fill=(61, 232, 255))
    for column, name in enumerate(SYSTEM_NAMES): draw.text((column * cell + 3, 28), name[:12], fill=(188, 255, 83))
    for row, family in enumerate(FAMILIES):
        record = next(item for item in source["offspring"] if item["family"] == family)
        arrays = _load_arrays(source_root.joinpath(*PurePosixPath(record["arrays"]["path"]).parts)); overlay, _ = compile_systems(record, arrays)
        positions = arrays["position_xy"].astype(int); roles = overlay["system_role"]
        for column in range(len(SYSTEM_NAMES)):
            image = np.zeros((48, 48, 3), dtype=np.uint8)
            for index, (x, y) in enumerate(positions): image[y, x] = ROLE_COLORS[int(roles[column, index])]
            tile = Image.fromarray(image).resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(tile, (column * cell, header + row * cell))
        draw.text((3, header + row * cell + 2), family.upper(), fill=(255, 255, 255))
    return png_bytes(np.asarray(canvas))


def _build_files(source_manifest: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    source_manifest = Path(source_manifest).resolve(); validation = validate_symmetry_bank(source_manifest)
    source = json.loads(source_manifest.read_text(encoding="utf-8")); files: dict[str, bytes] = {}; identities = []
    total_memberships = 0; overlapping_cells = 0; identities_with_overlap = 0
    for record in source["offspring"]:
        arrays = _load_arrays(source_manifest.parent.joinpath(*PurePosixPath(record["arrays"]["path"]).parts)); overlay, systems = compile_systems(record, arrays)
        relative = f"identities/{record['sample_id']}/physiology.npz"; payload = deterministic_npz_bytes(overlay); files[relative] = payload
        total_memberships += int(sum(item["member_count"] for item in systems))
        overlap = int(np.count_nonzero(overlay["system_membership"] & (overlay["system_membership"] - 1)))
        overlapping_cells += overlap; identities_with_overlap += int(overlap > 0)
        identities.append({
            "sample_id": record["sample_id"], "ordinal": record["ordinal"], "family": record["family"], "family_id": record["family_id"],
            "source_anatomy_sha256": record["anatomy_sha256"], "physical_cell_count": record["summary"]["physical_cell_count"],
            "systems": systems, "arrays_semantic_sha256": _overlay_sha(overlay), "arrays": artifact_record_from_bytes(relative, payload),
        })
    contact_payload = _contact_sheet(source_manifest.parent, source); files["cellular_physiology_contact_sheet.png"] = contact_payload
    manifest: dict[str, object] = {
        "format": FORMAT, "status": "ready", "quality_tier": "overlapping-connected-damage-responsive-organ-systems-v1",
        "compiler": {"source_sha256": source_sha256(), "python_runtime_required": False},
        "source": {"organism_manifest": source_manifest.relative_to(PROJECT_ROOT).as_posix(), "organism_manifest_sha256": sha256_file(source_manifest), "organism_semantic_sha256": source["semantic_sha256"], "organism_validation": validation},
        "array_format": ARRAY_FORMAT, "system_vocab": list(SYSTEM_NAMES), "role_vocab": list(ROLE_NAMES),
        "identity_count": len(identities), "system_count": len(SYSTEM_NAMES), "total_system_memberships": total_memberships, "overlapping_cell_count": overlapping_cells,
        "identities": identities, "contact_sheet": artifact_record_from_bytes("cellular_physiology_contact_sheet.png", contact_payload),
        "runtime_contract": {"overlapping_system_membership": True, "capacity_depends_on_live_cells": True, "capacity_depends_on_bond_connectivity": True, "core_loss_is_catastrophic": True, "dependency_cascade_is_explicit": True, "respiratory_exchange_is_family_specific": True, "python_runtime_required": False},
        "gates": {"all_45_identities_compiled": len(identities) == 45, "all_8_systems_every_identity": all(len(item["systems"]) == 8 for item in identities), "all_systems_have_core": all(system["core_count"] > 0 for item in identities for system in item["systems"]), "all_systems_have_members": all(system["member_count"] >= system["core_count"] for item in identities for system in item["systems"]), "all_identities_have_overlapping_membership": identities_with_overlap == 45, "source_anatomy_immutable": True, "native_runtime_independent_of_python": True},
    }
    manifest["semantic_sha256"] = json_sha256(manifest); files["cellular_physiology_manifest.json"] = canonical_json_bytes(manifest)
    return files, manifest


def build_bank(source_manifest: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    files, manifest = _build_files(source_manifest)
    if not all(manifest["gates"].values()): raise ValueError("Cellular physiology build gate failed")
    _atomic_publish(Path(destination).resolve(), files)
    validation = validate_bank(Path(destination) / "cellular_physiology_manifest.json")
    return {"passed": True, "destination": str(Path(destination).resolve()), "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": validation["manifest_sha256"], "validation": validation}


def _load_overlay(path: Path, cell_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"system_membership", "system_role", "system_weight"}: raise ValueError("Physiology array members differ")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if arrays["system_membership"].dtype != np.uint16 or arrays["system_membership"].shape != (cell_count,): raise ValueError("Physiology membership array differs")
    if arrays["system_role"].dtype != np.uint8 or arrays["system_role"].shape != (8, cell_count): raise ValueError("Physiology role array differs")
    if arrays["system_weight"].dtype != np.float32 or arrays["system_weight"].shape != (8, cell_count) or not np.isfinite(arrays["system_weight"]).all(): raise ValueError("Physiology weight array differs")
    if np.any(arrays["system_role"] > ROLE_EFFECTOR) or np.any(arrays["system_membership"] >= 1 << len(SYSTEM_NAMES)): raise ValueError("Physiology categorical bounds differ")
    return arrays


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8")); errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Cellular physiology schema validation failed: {errors[0].message}")
    if raw != canonical_json_bytes(manifest): raise ValueError("Cellular physiology manifest is not canonical JSON")
    if manifest["semantic_sha256"] != json_sha256({key: value for key, value in manifest.items() if key != "semantic_sha256"}): raise ValueError("Cellular physiology semantic hash differs")
    if manifest["compiler"]["source_sha256"] != source_sha256(): raise ValueError("Cellular physiology compiler source hash is stale")
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["organism_manifest"]).parts).resolve()
    if not source_path.is_relative_to(PROJECT_ROOT) or sha256_file(source_path) != manifest["source"]["organism_manifest_sha256"]: raise ValueError("Cellular physiology source provenance differs")
    validate_symmetry_bank(source_path); source = json.loads(source_path.read_text(encoding="utf-8")); source_by_id = {item["sample_id"]: item for item in source["offspring"]}
    if manifest["source"]["organism_semantic_sha256"] != source["semantic_sha256"]: raise ValueError("Cellular physiology source semantic identity differs")
    for identity in manifest["identities"]:
        record = source_by_id[identity["sample_id"]]; source_arrays = _load_arrays(source_path.parent.joinpath(*PurePosixPath(record["arrays"]["path"]).parts))
        artifact = identity["arrays"]; path = manifest_path.parent.joinpath(*PurePosixPath(artifact["path"]).parts)
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]: raise ValueError("Cellular physiology artifact integrity differs")
        arrays = _load_overlay(path, identity["physical_cell_count"]); expected_arrays, expected_systems = compile_systems(record, source_arrays)
        if identity["systems"] != expected_systems or identity["arrays_semantic_sha256"] != _overlay_sha(expected_arrays): raise ValueError("Cellular physiology deterministic semantic replay differs")
        if any(not np.array_equal(arrays[name], expected_arrays[name]) for name in arrays): raise ValueError("Cellular physiology array replay differs")
    contact = manifest["contact_sheet"]; contact_path = manifest_path.parent.joinpath(*PurePosixPath(contact["path"]).parts)
    if contact_path.stat().st_size != contact["bytes"] or sha256_file(contact_path) != contact["sha256"]: raise ValueError("Cellular physiology contact integrity differs")
    if not all(manifest["gates"].values()): raise ValueError("Cellular physiology gate differs")
    return {"passed": True, "identity_count": 45, "system_count": 8, "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": sha256_file(manifest_path), "contact_sheet_sha256": sha256_file(contact_path)}


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); validation = validate_bank(manifest_path); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["organism_manifest"]).parts); expected, _ = _build_files(source_path); root = manifest_path.parent
    for relative, payload in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.read_bytes() != payload: raise ValueError(f"Cellular physiology byte replay differs: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected): raise ValueError("Cellular physiology output closure differs")
    return {**validation, "exact_replay": True, "artifact_count": len(expected), "artifact_bytes": sum(map(len, expected.values()))}
