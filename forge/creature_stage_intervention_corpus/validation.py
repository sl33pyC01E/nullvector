from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np


FORMAT = "nullvector-creature-stage-intervention-corpus-v1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_BINARY_BYTES = 160 * 1024 * 1024
FRAMES = 180
INTERVENTION_FRAME = 15
HEAL_FRAME = 75
MAX_FLUIDS = 160
POSITION_SCALE = 256.0
POSITION_BIAS = 32768
UNIT_SCALE = 65535
FLUID_SCALE = 1024.0
FAMILIES = ["humanoid", "animalian", "plantlike", "anomaly", "machine"]
MORPHOTYPES = [
    ["balanced", "longarm", "sixlimb", "crowned"],
    ["quadruped", "crawler", "longtail", "horned"],
    ["treeform", "rosette", "runner", "twin_stem"],
    ["triad", "cross", "pentad", "halo"],
    ["tracked", "walker", "hover", "crab"],
]
INTERVENTIONS = [
    {"name": "control", "target": "none", "event_frames": []},
    {"name": "wound", "target": "central_nonorgan_patch", "event_frames": [15]},
    {"name": "heal", "target": "central_nonorgan_patch", "event_frames": [15, 75]},
    {"name": "cut", "target": "lower_appendage_plane", "event_frames": [15]},
    {"name": "neural_ablation", "target": "neural", "event_frames": [15]},
    {"name": "circulation_ablation", "target": "circulation", "event_frames": [15]},
    {"name": "respiration_ablation", "target": "respiration", "event_frames": [15]},
    {"name": "digestion_ablation", "target": "digestion", "event_frames": [15]},
    {"name": "sensory_ablation", "target": "senses", "event_frames": [15]},
]
ORGAN_GROUPS = {
    "neural": ["brain", "meristem", "phase_brain", "processor"],
    "circulation": ["heart", "vascular", "flux", "coolant_pump"],
    "respiration": ["lung", "frond", "orbital", "radiator"],
    "digestion": ["gut", "bulb", "transmuter", "battery"],
    "senses": ["eye", "photoreceptor", "singularity", "optic"],
}
SUMMARY_FIELDS = [
    "integrity", "neural", "circulation", "respiration", "digestion",
    "senses", "energy", "hydration", "dead", "fluid_count",
]
SOURCE_PATHS = [
    "scripts/creature_stage/creature_neural.gd",
    "scripts/creature_stage/neural_creature.gd",
    "scripts/creature_stage/intervention_corpus.gd",
]
ROOT = Path(__file__).resolve().parents[2]
GAME_ROOT = ROOT / "game"
SCHEMA_PATH = ROOT / "shared" / "schema" / "creature_stage_intervention_corpus.schema.json"


class InterventionCorpusValidationError(ValueError):
    """Raised when an intervention corpus fails closed."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InterventionCorpusValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise InterventionCorpusValidationError(f"non-finite JSON number: {value}")


def _sha256(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve(path: str | Path) -> tuple[Path, Path]:
    supplied = Path(path).resolve()
    manifest = supplied / "manifest.json" if supplied.is_dir() else supplied
    if manifest.name != "manifest.json" or not manifest.is_file():
        raise InterventionCorpusValidationError("corpus must contain manifest.json")
    if manifest.is_symlink() or manifest.parent.is_symlink():
        raise InterventionCorpusValidationError("symlinked corpus paths are rejected")
    return manifest.parent, manifest


def _load_manifest(path: str | Path) -> tuple[Path, Path, dict[str, Any], bytes]:
    root, manifest = _resolve(path)
    size = manifest.stat().st_size
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise InterventionCorpusValidationError("manifest exceeds bounded size")
    raw = manifest.read_bytes()
    try:
        document = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InterventionCorpusValidationError(f"invalid UTF-8 JSON: {exc}") from exc
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=str)
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path)
        raise InterventionCorpusValidationError(
            f"schema violation at {location or '<root>'}: {first.message}"
        )
    return root, manifest, document, raw


def _cell_identity(cells: list[dict[str, Any]]) -> str:
    material = "|".join(
        f"{cell['grid'][0]},{cell['grid'][1]},{cell['tissue']},{cell['organ']},"
        f"{cell['appendage']},{cell['side']}"
        for cell in cells
    )
    return _sha256(material.encode("utf-8"))


def _rounded_unit(values: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(values, 0.0, 1.0) * UNIT_SCALE + 0.5).astype(np.uint16)


def _validate_clip_semantics(
    clip: dict[str, Any],
    chassis: dict[str, Any],
    frames: np.ndarray,
) -> dict[str, float]:
    cell_count = int(clip["cell_count"])
    summaries = frames[:, :10]
    cell_values = frames[:, 10 : 10 + cell_count * 4].reshape(FRAMES, cell_count, 4)
    fluids = frames[:, 10 + cell_count * 4 :].reshape(FRAMES, MAX_FLUIDS, 6)

    alive = cell_values[:, :, 3]
    health = cell_values[:, :, 2]
    if not np.isin(alive, np.array([0, UNIT_SCALE], dtype=np.uint16)).all():
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} has non-binary alive flags")
    if np.any((alive == 0) & (health != 0)):
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} has health on dead cells")
    expected_integrity = _rounded_unit((alive == UNIT_SCALE).mean(axis=1))
    if not np.array_equal(summaries[:, 0], expected_integrity):
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} integrity summary mismatch")

    organ_names = np.array([cell["organ"] for cell in chassis["cells"]], dtype=object)
    for summary_index, group_name in enumerate(
        ["neural", "circulation", "respiration", "digestion", "senses"], start=1
    ):
        mask = np.isin(organ_names, ORGAN_GROUPS[group_name])
        if not mask.any():
            raise InterventionCorpusValidationError(
                f"chassis {chassis['chassis_id']} lacks {group_name} cells"
            )
        expected = _rounded_unit((alive[:, mask] == UNIT_SCALE).mean(axis=1))
        if not np.array_equal(summaries[:, summary_index], expected):
            raise InterventionCorpusValidationError(
                f"clip {clip['clip_id']} {group_name} capacity mismatch"
            )

    dead = summaries[:, 8]
    if not np.isin(dead, np.array([0, UNIT_SCALE], dtype=np.uint16)).all():
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} dead flag is invalid")
    if np.any(np.diff((dead == UNIT_SCALE).astype(np.int8)) < 0):
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} revives after death")
    fluid_counts = summaries[:, 9].astype(np.int64)
    active_fluid_mask = np.any(fluids != 0, axis=2)
    active_fluid_counts = active_fluid_mask.sum(axis=1)
    if not np.array_equal(fluid_counts, active_fluid_counts):
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} fluid count mismatch")
    for frame_index, count in enumerate(fluid_counts.tolist()):
        if count > MAX_FLUIDS or np.any(fluids[frame_index, count:] != 0):
            raise InterventionCorpusValidationError(
                f"clip {clip['clip_id']} has noncanonical unused fluid slots"
            )
    if int(fluid_counts.max()) != int(clip["maximum_fluid_count"]):
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} maximum fluids mismatch")

    intervention = str(clip["intervention"])
    before = INTERVENTION_FRAME - 1
    after = INTERVENTION_FRAME
    if intervention == "control":
        if clip["hit_count"] != 0 or fluid_counts.max() != 0 or np.any(alive != UNIT_SCALE):
            raise InterventionCorpusValidationError("control clip changes cell integrity")
        if np.any(summaries[:, 1:6] != UNIT_SCALE) or np.any(dead != 0):
            raise InterventionCorpusValidationError("control clip changes organ capacity or death")
    else:
        if int(clip["hit_count"]) <= 0:
            raise InterventionCorpusValidationError(f"clip {clip['clip_id']} intervention hit nothing")
        if int(fluid_counts.max()) <= 0:
            raise InterventionCorpusValidationError(f"clip {clip['clip_id']} emitted no fluid")
        active_radius = fluids[:, :, 4][active_fluid_mask].astype(np.float64) / FLUID_SCALE
        if active_radius.size == 0 or float(active_radius.max()) <= 0.8:
            raise InterventionCorpusValidationError(f"clip {clip['clip_id']} fluid never diffused")
    if intervention in {"wound", "heal"}:
        if not np.any((health[after] > 0) & (health[after] < UNIT_SCALE)):
            raise InterventionCorpusValidationError(f"clip {clip['clip_id']} lacks partial wound state")
        if np.any(alive[after] != UNIT_SCALE):
            raise InterventionCorpusValidationError(f"clip {clip['clip_id']} wound unexpectedly kills")
    if intervention == "heal":
        if float(health[HEAL_FRAME].mean()) <= float(health[HEAL_FRAME - 1].mean()):
            raise InterventionCorpusValidationError(f"clip {clip['clip_id']} healing has no effect")
    if intervention == "cut" and not np.any(alive[after] < alive[before]):
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} cut removes no cells")
    target_summary = {
        "neural_ablation": 1,
        "circulation_ablation": 2,
        "respiration_ablation": 3,
        "digestion_ablation": 4,
        "sensory_ablation": 5,
    }.get(intervention)
    if target_summary is not None and summaries[after, target_summary] != 0:
        raise InterventionCorpusValidationError(
            f"clip {clip['clip_id']} does not ablate target capacity"
        )
    if intervention == "neural_ablation" and not np.any(dead[after:] == UNIT_SCALE):
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} neural ablation is not terminal")

    deltas = (cell_values[:, :, :2].astype(np.float64) - POSITION_BIAS) / POSITION_SCALE
    displacement = np.linalg.norm(deltas, axis=2)
    if not np.isfinite(displacement).all() or float(displacement.max()) > 18.01:
        raise InterventionCorpusValidationError(f"clip {clip['clip_id']} displacement is invalid")
    return {
        "maximum_displacement": float(displacement.max()),
        "minimum_integrity": float(expected_integrity.min()) / UNIT_SCALE,
        "maximum_fluids": float(fluid_counts.max()),
    }


def validate_intervention_corpus(path: str | Path) -> dict[str, Any]:
    root, manifest_path, document, manifest_bytes = _load_manifest(path)
    names = {entry.name for entry in root.iterdir()}
    if names != {"manifest.json", "intervention_frames.u16le"}:
        raise InterventionCorpusValidationError(f"unexpected corpus members: {sorted(names)}")
    if document["interventions"] != INTERVENTIONS:
        raise InterventionCorpusValidationError("intervention vocabulary or schedule mismatch")
    if document["organ_groups"] != ORGAN_GROUPS or document["summary_fields"] != SUMMARY_FIELDS:
        raise InterventionCorpusValidationError("organ groups or summary order mismatch")

    current_hashes: dict[str, str] = {}
    source_material: list[str] = []
    for relative in SOURCE_PATHS:
        digest = _sha256((GAME_ROOT / relative).read_bytes())
        current_hashes[relative] = digest
        source_material.append(f"{relative}:{digest}")
    if document["source"]["files"] != current_hashes:
        raise InterventionCorpusValidationError("producer source hashes are stale")
    if document["source"]["combined_sha256"] != _sha256("|".join(source_material).encode()):
        raise InterventionCorpusValidationError("combined producer source hash mismatch")

    chassis_records: list[dict[str, Any]] = document["chassis"]
    chassis_identities: list[str] = []
    for chassis_id, chassis in enumerate(chassis_records):
        family_id, morphotype_id = divmod(chassis_id, 4)
        if (
            chassis["chassis_id"] != chassis_id
            or chassis["family_id"] != family_id
            or chassis["morphotype_id"] != morphotype_id
            or chassis["family"] != FAMILIES[family_id]
            or chassis["morphotype"] != MORPHOTYPES[family_id][morphotype_id]
            or chassis["seed"] != 0x710D0000 + family_id * 0x100 + morphotype_id
            or chassis["cell_count"] != len(chassis["cells"])
        ):
            raise InterventionCorpusValidationError(f"chassis identity mismatch at {chassis_id}")
        grids = [tuple(cell["grid"]) for cell in chassis["cells"]]
        if len(grids) != len(set(grids)):
            raise InterventionCorpusValidationError(f"duplicate cell grid at chassis {chassis_id}")
        identity = _cell_identity(chassis["cells"])
        if identity != chassis["cell_identity_sha256"]:
            raise InterventionCorpusValidationError(f"cell identity mismatch at chassis {chassis_id}")
        chassis_identities.append(identity)

    artifact = document["artifacts"]["intervention_frames"]
    binary_path = root / artifact["path"]
    if binary_path.is_symlink() or not binary_path.is_file():
        raise InterventionCorpusValidationError("intervention binary is missing or symlinked")
    binary_size = binary_path.stat().st_size
    if binary_size <= 0 or binary_size > MAX_BINARY_BYTES or binary_size != artifact["bytes"]:
        raise InterventionCorpusValidationError("intervention binary size mismatch or exceeds bound")

    expected_offset = 0
    expected_cell_samples = 0
    identity_material: list[str] = []
    maximum_displacement = 0.0
    minimum_integrity = 1.0
    maximum_fluid_count = 0
    control_prefixes: dict[int, bytes] = {}
    binary_bytes = binary_path.read_bytes()
    if _sha256(binary_bytes) != artifact["sha256"]:
        raise InterventionCorpusValidationError("intervention binary SHA-256 mismatch")
    for clip_id, clip in enumerate(document["clips"]):
        chassis_id, intervention_id = divmod(clip_id, len(INTERVENTIONS))
        family_id, morphotype_id = divmod(chassis_id, 4)
        expected_intervention = INTERVENTIONS[intervention_id]
        chassis = chassis_records[chassis_id]
        frame_stride = 20 + int(chassis["cell_count"]) * 8 + MAX_FLUIDS * 12
        byte_length = frame_stride * FRAMES
        if (
            clip["clip_id"] != clip_id
            or clip["chassis_id"] != chassis_id
            or clip["family_id"] != family_id
            or clip["morphotype_id"] != morphotype_id
            or clip["intervention_id"] != intervention_id
            or clip["intervention"] != expected_intervention["name"]
            or clip["target"] != expected_intervention["target"]
            or clip["event_frames"] != expected_intervention["event_frames"]
            or clip["cell_count"] != chassis["cell_count"]
            or clip["frame_stride_bytes"] != frame_stride
            or clip["byte_offset"] != expected_offset
            or clip["byte_length"] != byte_length
        ):
            raise InterventionCorpusValidationError(f"clip contract mismatch at {clip_id}")
        clip_bytes = binary_bytes[expected_offset : expected_offset + byte_length]
        clip_sha = _sha256(clip_bytes)
        if clip_sha != clip["trajectory_sha256"]:
            raise InterventionCorpusValidationError(f"trajectory SHA mismatch at clip {clip_id}")
        prefix = clip_bytes[: INTERVENTION_FRAME * frame_stride]
        if intervention_id == 0:
            control_prefixes[chassis_id] = prefix
        elif prefix != control_prefixes[chassis_id]:
            raise InterventionCorpusValidationError(
                f"clip {clip_id} baseline differs before intervention"
            )
        frames = np.frombuffer(clip_bytes, dtype="<u2").reshape(FRAMES, frame_stride // 2)
        metrics = _validate_clip_semantics(clip, chassis, frames)
        maximum_displacement = max(maximum_displacement, metrics["maximum_displacement"])
        minimum_integrity = min(minimum_integrity, metrics["minimum_integrity"])
        maximum_fluid_count = max(maximum_fluid_count, int(metrics["maximum_fluids"]))
        expected_cell_samples += int(chassis["cell_count"]) * FRAMES
        identity_material.append(f"clip:{clip_id}:{clip_sha}")
        expected_offset += byte_length
    if expected_offset != binary_size:
        raise InterventionCorpusValidationError("binary has a gap or trailing bytes")

    if expected_cell_samples != document["total_cell_samples"]:
        raise InterventionCorpusValidationError("total cell sample count mismatch")
    corpus_material: list[str] = []
    clip_index = 0
    for chassis_id, identity in enumerate(chassis_identities):
        corpus_material.append(f"chassis:{chassis_id}:{identity}")
        for _ in INTERVENTIONS:
            corpus_material.append(identity_material[clip_index])
            clip_index += 1
    corpus_identity = _sha256("|".join(corpus_material).encode())
    if corpus_identity != document["corpus_identity_sha256"]:
        raise InterventionCorpusValidationError("corpus identity mismatch")

    return {
        "passed": True,
        "format": FORMAT,
        "root": str(root),
        "manifest_sha256": _sha256(manifest_bytes),
        "binary_sha256": artifact["sha256"],
        "source_sha256": document["source"]["combined_sha256"],
        "corpus_identity_sha256": corpus_identity,
        "chassis_count": 20,
        "clip_count": 180,
        "total_frames": 32400,
        "total_cell_samples": expected_cell_samples,
        "binary_bytes": binary_size,
        "maximum_displacement": maximum_displacement,
        "minimum_integrity": minimum_integrity,
        "maximum_fluid_count": maximum_fluid_count,
    }


def load_intervention_clip(path: str | Path, clip_id: int) -> dict[str, np.ndarray]:
    validate_intervention_corpus(path)
    root, _, document, _ = _load_manifest(path)
    if isinstance(clip_id, bool) or not isinstance(clip_id, int) or not 0 <= clip_id < 180:
        raise InterventionCorpusValidationError("clip_id must be an integer in [0, 180)")
    clip = document["clips"][clip_id]
    cell_count = int(clip["cell_count"])
    with (root / "intervention_frames.u16le").open("rb") as handle:
        handle.seek(int(clip["byte_offset"]))
        raw = handle.read(int(clip["byte_length"]))
    frame_stride_words = int(clip["frame_stride_bytes"]) // 2
    frames = np.frombuffer(raw, dtype="<u2").reshape(FRAMES, frame_stride_words)
    summaries = frames[:, :10].astype(np.float32)
    summaries[:, :9] /= float(UNIT_SCALE)
    cells = frames[:, 10 : 10 + cell_count * 4].reshape(FRAMES, cell_count, 4)
    positions = (cells[:, :, :2].astype(np.float32) - POSITION_BIAS) / POSITION_SCALE
    health = cells[:, :, 2].astype(np.float32) / float(UNIT_SCALE)
    alive = cells[:, :, 3] == UNIT_SCALE
    fluid_words = frames[:, 10 + cell_count * 4 :].reshape(FRAMES, MAX_FLUIDS, 6)
    fluid = np.empty((FRAMES, MAX_FLUIDS, 6), dtype=np.float32)
    fluid[:, :, :4] = (fluid_words[:, :, :4].astype(np.float32) - POSITION_BIAS) / POSITION_SCALE
    fluid[:, :, 4:] = fluid_words[:, :, 4:].astype(np.float32) / FLUID_SCALE
    counts = frames[:, 9].astype(np.int32)
    result = {
        "summaries": summaries,
        "position_deltas": positions,
        "health": health,
        "alive": alive,
        "fluid": fluid,
        "fluid_counts": counts,
    }
    for value in result.values():
        value.setflags(write=False)
    return result
