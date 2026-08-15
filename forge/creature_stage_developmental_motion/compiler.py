from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_developmental import TISSUES as DEVELOPMENTAL_TISSUES
from ..creature_stage_developmental import TRAITS, develop, review_genomes, simulate_cycle
from ..creature_stage_developmental.contract import source_sha256 as developmental_source_sha256
from ..creature_stage_developmental.review import validate_review
from ..creature_stage_neural_motion.contract import GENE_NAMES, ORGANS, STATIC_FEATURES, TISSUES
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    sha256_bytes,
    sha256_file,
)
from .contract import (
    APPROVAL,
    CORPUS_SCHEMA,
    FIXED_HZ,
    FORMAT,
    FRAME_COUNT,
    MAX_APPENDAGES,
    MAX_CELLS,
    MAX_DISPLACEMENT,
    MAX_MUSCLES,
    MAX_NODES,
    MUSCLE_FEATURES,
    NODE_FEATURES,
    corpus_source_sha256,
)


ARRAY_FILE = "developmental_motion.npz"
MANIFEST_FILE = "manifest.json"
MAX_ARCHIVE_BYTES = 256 * 1024**2

TISSUE_MAP = {
    "skin": "skin",
    "bone": "structure",
    "muscle": "locomotor",
    "tendon": "locomotor",
    "armor": "armor",
    "neural": "neural",
    "vascular": "circulatory",
    "respiratory": "respiratory",
    "digestive": "digestive",
    "sensor": "sensor",
    "storage": "storage",
    "root": "root",
    "phase": "phase",
    "machine": "structure",
    "weapon": "weapon",
}
ORGAN_MAP = {"heart": "heart", "jaw": "none", "none": "none"}


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("developmental motion authority must remain inside the project") from error


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(b"nullvector-developmental-motion-array-v1\0")
    digest.update(name.encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _schema_validate(payload: dict[str, Any]) -> None:
    schema = json.loads(CORPUS_SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = "/".join(str(item) for item in error.absolute_path) or "<root>"
        raise ValueError(f"developmental motion corpus schema failed at {location}: {error.message}")


def _cell_static(organism) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = organism.cell_count
    if not 1 <= count <= MAX_CELLS:
        raise ValueError("developmental motion cell count escaped its padded bound")
    features = np.zeros((MAX_CELLS, STATIC_FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_CELLS, dtype=np.bool_)
    adjacency = np.zeros((MAX_CELLS, MAX_CELLS), dtype=np.bool_)
    mask[:count] = True
    coordinates = organism.cell_xy.astype(np.float32)
    dominant_component = np.argmax(organism.component_weights, axis=1)
    trait_index = {name: TRAITS.index(name) for name in TRAITS}
    span = np.ptp(coordinates, axis=0)
    genes = {
        "width": min(1.0, float(span[0]) / 48.0),
        "height": min(1.0, float(span[1]) / 48.0),
        "asymmetry": 1.0 - float(organism.genome.traits[trait_index["symmetry"]]),
        "symmetry": float(organism.genome.traits[trait_index["symmetry"]]),
        "repair": float(organism.genome.traits[trait_index["regeneration"]]),
        "metabolism": float(organism.genome.traits[trait_index["metabolism"]]),
        "fertility": float(np.clip(.25 + organism.genome.traits[trait_index["metabolism"]] * .35, 0.0, 1.0)),
        "bond_strength": float(np.clip(
            organism.genome.traits[trait_index["stiffness"]] * .55
            + organism.genome.traits[trait_index["bone_density"]] * .45,
            0.0,
            1.0,
        )),
    }
    for index in range(count):
        x, y = map(float, coordinates[index])
        features[index, 0:4] = (x / 16.0, y / 16.0, float(np.hypot(x, y)) / 24.0, y / 16.0)
        source_tissue = DEVELOPMENTAL_TISSUES[int(organism.tissue[index])]
        target_tissue = TISSUE_MAP[source_tissue]
        features[index, 4 + TISSUES.index(target_tissue)] = 1.0
        component = organism.genome.components[int(dominant_component[index])]
        organ = ORGAN_MAP.get(component.organ, component.organ if component.organ in ORGANS else "none")
        features[index, 17 + ORGANS.index(organ)] = 1.0
        side = int(organism.side[index])
        features[index, 46 + side + 1] = 1.0
        features[index, 49] = max(0.0, 1.0 - float(np.hypot(x, y)) / 28.0)
        appendage = int(organism.appendage_index[index])
        if appendage >= 0:
            features[index, 50] = 1.0
            features[index, 51] = np.sin(appendage * np.pi * 2.0 / 32.0)
            features[index, 52] = np.cos(appendage * np.pi * 2.0 / 32.0)
        for gene_index, name in enumerate(GENE_NAMES):
            features[index, 53 + gene_index] = genes[name]
    grids = organism.cell_xy.astype(np.int16)
    delta = grids[:, None, :] - grids[None, :, :]
    local = np.max(np.abs(delta), axis=2) <= 1
    adjacency[:count, :count] = local
    seen = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in np.flatnonzero(local[current]).tolist():
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    if len(seen) != count:
        raise ValueError("developmental motion cellular graph is disconnected")
    return features, mask, adjacency


def _muscle_features(organism) -> tuple[np.ndarray, np.ndarray]:
    count = len(organism.muscles)
    if not 1 <= count <= MAX_MUSCLES:
        raise ValueError("developmental motion muscle count escaped its padded bound")
    values = np.zeros((MAX_MUSCLES, MUSCLE_FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_MUSCLES, dtype=np.bool_)
    mask[:count] = True
    for index, muscle in enumerate(organism.muscles):
        origin = organism.skeleton_nodes[int(muscle[0]), :2] / 16.0
        insertion = organism.skeleton_nodes[int(muscle[1]), :2] / 16.0
        appendage = int(muscle[2])
        values[index] = (
            float(origin[0]), float(origin[1]), float(insertion[0]), float(insertion[1]),
            float(muscle[3]), float(muscle[4]), float(muscle[5]), float(muscle[6]) / 4.0,
            float(np.sin(appendage * np.pi * 2.0 / 32.0)),
            float(np.cos(appendage * np.pi * 2.0 / 32.0)),
        )
    return values, mask


def _structure_arrays(organism) -> dict[str, np.ndarray]:
    """Compile the exact sparse anatomy used by the actuator and skinning head."""

    node_count = len(organism.skeleton_nodes)
    muscle_count = len(organism.muscles)
    appendage_count = len(organism.genome.appendages)
    if node_count > MAX_NODES or muscle_count > MAX_MUSCLES or appendage_count > MAX_APPENDAGES:
        raise ValueError("developmental motion structural bound drifted")

    node_features = np.zeros((MAX_NODES, NODE_FEATURES), dtype=np.float32)
    node_adjacency = np.zeros((MAX_NODES, MAX_NODES), dtype=np.bool_)
    muscle_incidence = np.zeros((MAX_MUSCLES, MAX_NODES), dtype=np.float32)
    skinning = np.zeros((MAX_CELLS, MAX_NODES), dtype=np.float32)
    appendage_mask = np.zeros(MAX_APPENDAGES, dtype=np.bool_)
    appendage_mask[:appendage_count] = True

    nodes = organism.skeleton_nodes[:, :2].astype(np.float32)
    degree = np.zeros(node_count, dtype=np.float32)
    for left_raw, right_raw in organism.skeleton_edges:
        left, right = int(left_raw), int(right_raw)
        node_adjacency[left, right] = True
        node_adjacency[right, left] = True
        degree[left] += 1.0
        degree[right] += 1.0
    np.fill_diagonal(node_adjacency[:node_count, :node_count], True)

    node_appendage = np.full(node_count, -1, dtype=np.int16)
    node_side = np.zeros(node_count, dtype=np.int8)
    for edge_index, (_, right_raw) in enumerate(organism.skeleton_edges):
        right = int(right_raw)
        appendage = int(organism.skeleton_edge_appendage[edge_index])
        if appendage >= 0:
            node_appendage[right] = appendage
        node_side[right] = int(organism.skeleton_edge_side[edge_index])
    component_count = len(organism.genome.components)
    node_features[:node_count, 0:2] = nodes / 16.0
    node_features[:node_count, 2] = organism.skeleton_nodes[:, 2] / 12.0
    node_features[:node_count, 3] = np.clip(degree / 8.0, 0.0, 1.0)
    node_features[:component_count, 4] = 1.0
    node_features[:node_count, 5] = (node_appendage >= 0).astype(np.float32)
    node_features[:node_count, 6] = node_side.astype(np.float32)
    node_features[:node_count, 7] = np.where(
        node_appendage >= 0,
        np.sin(np.maximum(node_appendage, 0) * np.pi * 2.0 / MAX_APPENDAGES),
        0.0,
    )

    for muscle_index, muscle in enumerate(organism.muscles):
        origin, insertion = int(muscle[0]), int(muscle[1])
        muscle_incidence[muscle_index, origin] = -float(muscle[3])
        muscle_incidence[muscle_index, insertion] = float(muscle[3])

    points = organism.cell_xy.astype(np.float32)
    distance = np.linalg.norm(points[:, None, :] - nodes[None, :, :], axis=2)
    nearest = np.argpartition(distance, kth=min(2, node_count - 1), axis=1)[:, :3]
    selected = np.take_along_axis(distance, nearest, axis=1)
    weights = np.exp(-selected * .72)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    rows = np.arange(organism.cell_count)[:, None]
    skinning[rows, nearest] = weights
    if not np.allclose(skinning[:organism.cell_count].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("developmental motion skinning weights drifted")
    return {
        "node_features": node_features,
        "node_adjacency": node_adjacency,
        "muscle_incidence": muscle_incidence,
        "cell_node_weights": skinning,
        "appendage_mask": appendage_mask,
    }


def compile_candidate_arrays() -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    genomes = review_genomes()
    organisms = [develop(genome) for genome in genomes]
    cycles = [simulate_cycle(organism, frame_count=FRAME_COUNT, settle_cycles=12) for organism in organisms]
    count = len(organisms)
    arrays = {
        "static": np.zeros((count, MAX_CELLS, STATIC_FEATURES), dtype=np.float32),
        "mask": np.zeros((count, MAX_CELLS), dtype=np.bool_),
        "adjacency": np.zeros((count, MAX_CELLS, MAX_CELLS), dtype=np.bool_),
        "rest_xy": np.zeros((count, MAX_CELLS, 2), dtype=np.float32),
        "trajectory": np.zeros((count, FRAME_COUNT, MAX_CELLS, 2), dtype=np.float32),
        "muscle_features": np.zeros((count, MAX_MUSCLES, MUSCLE_FEATURES), dtype=np.float32),
        "muscle_mask": np.zeros((count, MAX_MUSCLES), dtype=np.bool_),
        "muscle_activation": np.zeros((count, FRAME_COUNT, MAX_MUSCLES), dtype=np.float32),
        "muscle_incidence": np.zeros((count, MAX_MUSCLES, MAX_NODES), dtype=np.float32),
        "node_mask": np.zeros((count, MAX_NODES), dtype=np.bool_),
        "node_features": np.zeros((count, MAX_NODES, NODE_FEATURES), dtype=np.float32),
        "node_adjacency": np.zeros((count, MAX_NODES, MAX_NODES), dtype=np.bool_),
        "node_rest": np.zeros((count, MAX_NODES, 2), dtype=np.float32),
        "node_trajectory": np.zeros((count, FRAME_COUNT, MAX_NODES, 2), dtype=np.float32),
        "cell_node_weights": np.zeros((count, MAX_CELLS, MAX_NODES), dtype=np.float32),
        "appendage_mask": np.zeros((count, MAX_APPENDAGES), dtype=np.bool_),
        "planted_contacts": np.zeros((count, FRAME_COUNT, MAX_APPENDAGES), dtype=np.bool_),
        "family": np.zeros(count, dtype=np.uint8),
        "morphotype": np.zeros(count, dtype=np.uint8),
        "family_mix": np.zeros((count, 5), dtype=np.float32),
        "traits": np.zeros((count, len(TRAITS)), dtype=np.float32),
        "cell_count": np.zeros(count, dtype=np.uint16),
        "node_count": np.zeros(count, dtype=np.uint8),
        "muscle_count": np.zeros(count, dtype=np.uint8),
    }
    records: list[dict[str, Any]] = []
    for index, (organism, cycle) in enumerate(zip(organisms, cycles, strict=True)):
        static, mask, adjacency = _cell_static(organism)
        muscle_features, muscle_mask = _muscle_features(organism)
        structure = _structure_arrays(organism)
        cell_count = organism.cell_count
        node_count = len(organism.skeleton_nodes)
        muscle_count = len(organism.muscles)
        if node_count > MAX_NODES:
            raise ValueError("developmental motion node count escaped its padded bound")
        arrays["static"][index] = static
        arrays["mask"][index] = mask
        arrays["adjacency"][index] = adjacency
        arrays["rest_xy"][index, :cell_count] = organism.cell_xy
        arrays["muscle_features"][index] = muscle_features
        arrays["muscle_mask"][index] = muscle_mask
        arrays["muscle_incidence"][index] = structure["muscle_incidence"]
        arrays["node_mask"][index, :node_count] = True
        arrays["node_features"][index] = structure["node_features"]
        arrays["node_adjacency"][index] = structure["node_adjacency"]
        arrays["node_rest"][index, :node_count] = organism.skeleton_nodes[:, :2]
        arrays["cell_node_weights"][index] = structure["cell_node_weights"]
        arrays["appendage_mask"][index] = structure["appendage_mask"]
        arrays["family"][index] = index // 2
        arrays["morphotype"][index] = (index // 2) * 4 + (index % 2)
        arrays["family_mix"][index] = organism.genome.family_mix
        arrays["traits"][index] = organism.genome.traits
        arrays["cell_count"][index] = cell_count
        arrays["node_count"][index] = node_count
        arrays["muscle_count"][index] = muscle_count
        for frame, dynamic in enumerate(cycle.frames):
            displacement = dynamic.cells - organism.cell_xy.astype(np.float32)
            if float(np.max(np.abs(displacement))) > MAX_DISPLACEMENT:
                raise ValueError("developmental motion exceeded inherited displacement range")
            arrays["trajectory"][index, frame, :cell_count] = displacement
            arrays["muscle_activation"][index, frame, :muscle_count] = dynamic.muscle_activation
            arrays["node_trajectory"][index, frame, :node_count] = dynamic.nodes[:, :2] - organism.skeleton_nodes[:, :2]
            arrays["planted_contacts"][index, frame, :len(dynamic.planted_contacts)] = dynamic.planted_contacts
        records.append({
            "index": index,
            "genome_id": organism.genome.genome_id,
            "genome": asdict(organism.genome),
            "identity_sha256": organism.identity_sha256,
            "family_id": index // 2,
            "morphotype_id": (index // 2) * 4 + (index % 2),
            "role": "base-prior" if index % 2 == 0 else "component-graft",
            "cell_count": cell_count,
            "node_count": node_count,
            "muscle_count": muscle_count,
            "loop_seam_max_abs": cycle.loop_seam_max_abs,
            "maximum_edge_strain": cycle.maximum_edge_strain,
        })
    for name, value in arrays.items():
        if (
            not value.flags.c_contiguous
            or value.dtype.hasobject
            or (value.dtype.kind == "f" and not np.isfinite(value).all())
        ):
            raise ValueError(f"developmental motion array {name} drifted")
    return arrays, records


def _manifest(review: Path, review_validation: dict[str, Any], records: list[dict[str, Any]], arrays: dict[str, np.ndarray], archive: bytes) -> dict[str, Any]:
    review_manifest = review / "review_manifest.json"
    payload: dict[str, Any] = {
        "format": FORMAT,
        "status": "approved",
        "source_sha256": corpus_source_sha256(),
        "developmental_source_sha256": developmental_source_sha256(),
        "approval": APPROVAL,
        "review": {
            "path": _relative(review),
            "manifest_sha256": sha256_file(review_manifest),
            "source_sha256": review_validation["source_sha256"],
            "motion_semantic_sha256": review_validation["motion_semantic_sha256"],
            "motion_frame_stream_sha256": review_validation["motion_frame_stream_sha256"],
        },
        "contract": {
            "specimen_count": len(records),
            "frame_count": FRAME_COUNT,
            "fixed_hz": FIXED_HZ,
            "motion": "locomote",
            "loop": True,
            "max_cells": MAX_CELLS,
            "max_nodes": MAX_NODES,
            "max_muscles": MAX_MUSCLES,
            "muscle_features": MUSCLE_FEATURES,
            "max_displacement": MAX_DISPLACEMENT,
        },
        "specimens": records,
        "arrays": {
            "artifact": artifact_record_from_bytes(ARRAY_FILE, archive),
            "members": {
                name: {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _array_sha256(name, value)}
                for name, value in sorted(arrays.items())
            },
        },
    }
    payload["semantic_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def build_candidate_corpus(output: Path, *, review: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    review = Path(review).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=512 * 1024**2)
    review_validation = validate_review(review)
    arrays, records = compile_candidate_arrays()
    archive = deterministic_npz_bytes(arrays)
    if not 0 < len(archive) <= MAX_ARCHIVE_BYTES:
        raise ValueError("developmental motion archive size drifted")
    manifest = _manifest(review, review_validation, records, arrays, archive)
    _schema_validate(manifest)
    staging = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    (staging / ARRAY_FILE).write_bytes(archive)
    (staging / MANIFEST_FILE).write_bytes(canonical_json_bytes(manifest))
    os.replace(staging, output)
    return validate_candidate_corpus(output, replay=False)


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("developmental motion archive is missing, linked, or oversized")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    return arrays


def validate_candidate_corpus(output: Path, *, replay: bool = True) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / MANIFEST_FILE
    archive_path = output / ARRAY_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 4 * 1024**2:
        raise ValueError("developmental motion manifest is missing, linked, or oversized")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest):
        raise ValueError("developmental motion manifest is not canonical")
    _schema_validate(manifest)
    if (
        manifest["format"] != FORMAT
        or manifest["status"] != "approved"
        or manifest["source_sha256"] != corpus_source_sha256()
        or manifest["developmental_source_sha256"] != developmental_source_sha256()
        or manifest["approval"] != APPROVAL
        or manifest["semantic_sha256"] != sha256_bytes(canonical_json_bytes({key: value for key, value in manifest.items() if key != "semantic_sha256"}))
    ):
        raise ValueError("developmental motion corpus authority drifted")
    review = PROJECT_ROOT / manifest["review"]["path"]
    review_manifest_path = review / "review_manifest.json"
    if replay:
        review_validation = validate_review(review)
    else:
        review_payload = json.loads(review_manifest_path.read_bytes())
        if not isinstance(review_payload, dict) or review_payload.get("passed") is not True:
            raise ValueError("developmental motion review manifest structure drifted")
        review_validation = {
            "source_sha256": review_payload.get("source_sha256"),
            "motion_semantic_sha256": review_payload.get("motion", {}).get("semantic_sha256"),
            "motion_frame_stream_sha256": review_payload.get("motion", {}).get("frame_stream_sha256"),
        }
    if (
        sha256_file(review_manifest_path) != manifest["review"]["manifest_sha256"]
        or review_validation["source_sha256"] != manifest["review"]["source_sha256"]
        or review_validation["motion_semantic_sha256"] != manifest["review"]["motion_semantic_sha256"]
        or review_validation["motion_frame_stream_sha256"] != manifest["review"]["motion_frame_stream_sha256"]
    ):
        raise ValueError("developmental motion review provenance drifted")
    artifact = manifest["arrays"]["artifact"]
    if archive_path.stat().st_size != artifact["bytes"] or sha256_file(archive_path) != artifact["sha256"]:
        raise ValueError("developmental motion archive artifact drifted")
    arrays = _load_archive(archive_path)
    if set(arrays) != set(manifest["arrays"]["members"]):
        raise ValueError("developmental motion array registry drifted")
    for name, array in arrays.items():
        record = manifest["arrays"]["members"][name]
        if record != {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": _array_sha256(name, array)}:
            raise ValueError(f"developmental motion array {name} drifted")
    if replay:
        expected_arrays, records = compile_candidate_arrays()
        expected_archive = deterministic_npz_bytes(expected_arrays)
        canonical_records = json.loads(json.dumps(records, ensure_ascii=True, allow_nan=False))
        if expected_archive != archive_path.read_bytes() or canonical_records != manifest["specimens"]:
            raise ValueError("developmental motion exact replay drifted")
    return {
        "passed": True,
        "status": manifest["status"],
        "training_permitted": manifest["approval"]["training_permitted"],
        "specimens": manifest["contract"]["specimen_count"],
        "frames": manifest["contract"]["frame_count"],
        "semantic_sha256": manifest["semantic_sha256"],
        "archive_sha256": artifact["sha256"],
    }
