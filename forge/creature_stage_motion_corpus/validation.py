from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np


FORMAT = "nullvector-creature-stage-motion-corpus-v1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_BINARY_BYTES = 40 * 1024 * 1024
FAMILIES = ["humanoid", "animalian", "plantlike", "anomaly", "machine"]
MORPHOTYPES = [
    ["balanced", "longarm", "sixlimb", "crowned"],
    ["quadruped", "crawler", "longtail", "horned"],
    ["treeform", "rosette", "runner", "twin_stem"],
    ["triad", "cross", "pentad", "halo"],
    ["tracked", "walker", "hover", "crab"],
]
MOTIONS = [
    "idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear",
    "confused", "sleep", "taunt", "attack", "cast", "hit", "death",
]
MOTION_SPECS = {
    "idle_breathe": {"cycle": 2.4, "loop": True, "priority": 0},
    "idle_wiggle": {"cycle": 1.8, "loop": True, "priority": 0},
    "locomote": {"cycle": 0.78, "loop": True, "priority": 1},
    "joy": {"cycle": 1.0, "loop": False, "priority": 2},
    "anger": {"cycle": 0.8, "loop": False, "priority": 2},
    "fear": {"cycle": 0.9, "loop": False, "priority": 2},
    "confused": {"cycle": 1.1, "loop": False, "priority": 2},
    "sleep": {"cycle": 2.4, "loop": False, "priority": 2},
    "taunt": {"cycle": 0.9, "loop": False, "priority": 2},
    "attack": {"cycle": 0.55, "loop": False, "priority": 3},
    "cast": {"cycle": 0.72, "loop": False, "priority": 3},
    "hit": {"cycle": 0.42, "loop": False, "priority": 4},
    "death": {"cycle": 1.2, "loop": False, "priority": 5},
}
SOURCE_PATHS = [
    "scripts/creature_stage/creature_neural.gd",
    "scripts/creature_stage/neural_creature.gd",
    "scripts/creature_stage/motion_corpus.gd",
]
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "schema"
    / "creature_stage_motion_corpus.schema.json"
)
GAME_ROOT = Path(__file__).resolve().parents[2] / "game"


class MotionCorpusValidationError(ValueError):
    """Raised when a cellular motion corpus fails closed."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MotionCorpusValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise MotionCorpusValidationError(f"non-finite JSON number: {value}")


def _sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_manifest(path: str | Path) -> tuple[Path, Path]:
    supplied = Path(path).resolve()
    manifest_path = supplied / "manifest.json" if supplied.is_dir() else supplied
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise MotionCorpusValidationError("corpus must contain manifest.json")
    root = manifest_path.parent
    if manifest_path.is_symlink() or root.is_symlink():
        raise MotionCorpusValidationError("symlinked corpus paths are not accepted")
    return root, manifest_path


def _load_manifest(path: str | Path) -> tuple[Path, Path, dict[str, Any], bytes]:
    root, manifest_path = _resolve_manifest(path)
    size = manifest_path.stat().st_size
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise MotionCorpusValidationError(
            f"manifest size {size} is outside (0, {MAX_MANIFEST_BYTES}]"
        )
    raw = manifest_path.read_bytes()
    try:
        document = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MotionCorpusValidationError(f"invalid UTF-8 JSON: {exc}") from exc
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=str)
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path)
        raise MotionCorpusValidationError(
            f"schema violation at {location or '<root>'}: {first.message}"
        )
    return root, manifest_path, document, raw


def _cell_identity(cells: list[dict[str, Any]]) -> str:
    material = "|".join(
        f"{cell['grid'][0]},{cell['grid'][1]},{cell['tissue']},{cell['organ']},"
        f"{cell['appendage']},{cell['side']},{cell['initial_health_q']}"
        for cell in cells
    )
    return _sha256_bytes(material.encode("utf-8"))


def _expected_controls(motion: str) -> tuple[bool, bool, str]:
    return (
        motion == "attack",
        motion == "cast",
        "impact" if motion == "hit" else ("terminal" if motion == "death" else "none"),
    )


def _rms_spread(
    values: array,
    value_offset: int,
    cells: list[dict[str, Any]],
    scale: float,
    bias: int,
) -> float:
    positions: list[tuple[float, float]] = []
    for cell_index, cell in enumerate(cells):
        index = value_offset + cell_index * 2
        x = float(cell["grid"][0]) * 3.0 + (values[index] - bias) / scale
        y = float(cell["grid"][1]) * 3.0 + (values[index + 1] - bias) / scale
        positions.append((x, y))
    center_x = sum(position[0] for position in positions) / len(positions)
    center_y = sum(position[1] for position in positions) / len(positions)
    return math.sqrt(
        sum(
            (position[0] - center_x) ** 2 + (position[1] - center_y) ** 2
            for position in positions
        )
        / len(positions)
    )


def validate_motion_corpus(path: str | Path) -> dict[str, Any]:
    root, manifest_path, document, manifest_bytes = _load_manifest(path)
    names = {entry.name for entry in root.iterdir()}
    if names != {"manifest.json", "motion_frames.u16le"}:
        raise MotionCorpusValidationError(f"unexpected corpus members: {sorted(names)}")

    artifact = document["artifacts"]["motion_frames"]
    binary_path = root / artifact["path"]
    if binary_path.is_symlink() or not binary_path.is_file():
        raise MotionCorpusValidationError("motion binary is missing or symlinked")
    binary_size = binary_path.stat().st_size
    if binary_size <= 0 or binary_size > MAX_BINARY_BYTES:
        raise MotionCorpusValidationError("motion binary exceeds bounded size")
    if binary_size != int(artifact["bytes"]):
        raise MotionCorpusValidationError("motion binary size mismatch")
    binary_bytes = binary_path.read_bytes()
    binary_sha256 = _sha256_bytes(binary_bytes)
    if binary_sha256 != artifact["sha256"]:
        raise MotionCorpusValidationError("motion binary SHA-256 mismatch")

    current_hashes: dict[str, str] = {}
    source_material: list[str] = []
    for relative in SOURCE_PATHS:
        source_path = GAME_ROOT / relative
        digest = _sha256_bytes(source_path.read_bytes())
        current_hashes[relative] = digest
        source_material.append(f"{relative}:{digest}")
    if document["source"]["files"] != current_hashes:
        raise MotionCorpusValidationError("producer source hashes are stale")
    combined_source = _sha256_bytes("|".join(source_material).encode("utf-8"))
    if document["source"]["combined_sha256"] != combined_source:
        raise MotionCorpusValidationError("combined producer source hash mismatch")

    if document["motion_order"] != MOTIONS or document["motion_specs"] != MOTION_SPECS:
        raise MotionCorpusValidationError("motion vocabulary or specs mismatch")

    chassis: list[dict[str, Any]] = document["chassis"]
    chassis_identities: list[str] = []
    for chassis_id, record in enumerate(chassis):
        family_id = chassis_id // 4
        morphotype_id = chassis_id % 4
        if (
            record["chassis_id"] != chassis_id
            or record["family_id"] != family_id
            or record["morphotype_id"] != morphotype_id
            or record["family"] != FAMILIES[family_id]
            or record["morphotype"] != MORPHOTYPES[family_id][morphotype_id]
            or record["seed"] != 0x6D0F0000 + family_id * 0x100 + morphotype_id
            or record["cell_count"] != len(record["cells"])
        ):
            raise MotionCorpusValidationError(f"chassis identity mismatch at {chassis_id}")
        grids = [tuple(cell["grid"]) for cell in record["cells"]]
        if len(set(grids)) != len(grids):
            raise MotionCorpusValidationError(f"duplicate cell grid in chassis {chassis_id}")
        identity = _cell_identity(record["cells"])
        if identity != record["cell_identity_sha256"]:
            raise MotionCorpusValidationError(f"cell identity mismatch at chassis {chassis_id}")
        chassis_identities.append(identity)

    values = array("H")
    values.frombytes(binary_bytes)
    if sys.byteorder != "little":
        values.byteswap()
    actual_minimum = min(values)
    actual_maximum = max(values)
    quantization = document["quantization"]
    if (
        actual_minimum != quantization["minimum_encoded"]
        or actual_maximum != quantization["maximum_encoded"]
    ):
        raise MotionCorpusValidationError("quantization extrema mismatch")

    clips: list[dict[str, Any]] = document["clips"]
    expected_offset = 0
    total_cell_samples = 0
    maximum_displacement = 0.0
    minimum_motion = math.inf
    trajectory_hashes: list[str] = []
    for clip_id, clip in enumerate(clips):
        chassis_id = clip_id // len(MOTIONS)
        motion_id = clip_id % len(MOTIONS)
        family_id = chassis_id // 4
        morphotype_id = chassis_id % 4
        motion = MOTIONS[motion_id]
        cell_count = int(chassis[chassis_id]["cell_count"])
        expected_length = 72 * cell_count * 4
        if (
            clip["clip_id"] != clip_id
            or clip["chassis_id"] != chassis_id
            or clip["family_id"] != family_id
            or clip["morphotype_id"] != morphotype_id
            or clip["motion_id"] != motion_id
            or clip["motion"] != motion
            or clip["cell_count"] != cell_count
            or clip["frame_stride_bytes"] != cell_count * 4
            or clip["byte_offset"] != expected_offset
            or clip["byte_length"] != expected_length
        ):
            raise MotionCorpusValidationError(f"clip layout mismatch at {clip_id}")
        end = expected_offset + expected_length
        trajectory = binary_bytes[expected_offset:end]
        trajectory_sha256 = _sha256_bytes(trajectory)
        if trajectory_sha256 != clip["trajectory_sha256"]:
            raise MotionCorpusValidationError(f"trajectory hash mismatch at clip {clip_id}")
        trajectory_hashes.append(trajectory_sha256)

        controls = clip["controls"]
        attack, utility, event = _expected_controls(motion)
        move_length = math.hypot(*controls["move"])
        aim_length = math.hypot(*controls["aim"])
        if not math.isclose(aim_length, 1.0, rel_tol=0.0, abs_tol=2e-6):
            raise MotionCorpusValidationError(f"aim is not normalized at clip {clip_id}")
        if (motion == "locomote") != (move_length > 0.99):
            raise MotionCorpusValidationError(f"locomotion control mismatch at clip {clip_id}")
        if (
            bool(controls["attack"] > 0.5) != attack
            or bool(controls["utility"] > 0.5) != utility
            or controls["external_event"] != event
        ):
            raise MotionCorpusValidationError(f"action control mismatch at clip {clip_id}")

        value_offset = expected_offset // 2
        appendage_max = 0.0
        core_max = 0.0
        clip_max = 0.0
        cells = chassis[chassis_id]["cells"]
        for frame in range(72):
            frame_offset = value_offset + frame * cell_count * 2
            for cell_index, cell in enumerate(cells):
                index = frame_offset + cell_index * 2
                dx = (values[index] - 32768) / 256.0
                dy = (values[index + 1] - 32768) / 256.0
                displacement = math.hypot(dx, dy)
                clip_max = max(clip_max, displacement)
                if cell["appendage"] >= 0:
                    appendage_max = max(appendage_max, displacement)
                else:
                    core_max = max(core_max, displacement)
        meaningful = core_max if motion == "idle_breathe" else appendage_max
        if motion == "death":
            meaningful = clip_max
            first_spread = _rms_spread(values, value_offset, cells, 256.0, 32768)
            last_spread = _rms_spread(
                values,
                value_offset + 71 * cell_count * 2,
                cells,
                256.0,
                32768,
            )
            if abs(last_spread - first_spread) > 1.5:
                raise MotionCorpusValidationError(f"death spread exploded at clip {clip_id}")
        if meaningful < 0.02:
            raise MotionCorpusValidationError(f"motion collapsed at clip {clip_id}")
        if clip_max > 14.01:
            raise MotionCorpusValidationError(f"motion bound exceeded at clip {clip_id}")
        maximum_displacement = max(maximum_displacement, clip_max)
        minimum_motion = min(minimum_motion, meaningful)
        expected_offset = end
        total_cell_samples += 72 * cell_count

    if expected_offset != binary_size:
        raise MotionCorpusValidationError("binary coverage does not end at EOF")
    if total_cell_samples != document["total_cell_samples"]:
        raise MotionCorpusValidationError("total cell sample count mismatch")
    corpus_identity: list[str] = []
    for chassis_id, chassis_identity in enumerate(chassis_identities):
        corpus_identity.append(f"chassis:{chassis_id}:{chassis_identity}")
        first_clip = chassis_id * len(MOTIONS)
        for clip_id in range(first_clip, first_clip + len(MOTIONS)):
            corpus_identity.append(f"clip:{clip_id}:{trajectory_hashes[clip_id]}")
    identity = _sha256_bytes("|".join(corpus_identity).encode("utf-8"))
    if identity != document["corpus_identity_sha256"]:
        raise MotionCorpusValidationError("corpus identity SHA-256 mismatch")

    return {
        "passed": True,
        "format": FORMAT,
        "root": str(root),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "binary_sha256": binary_sha256,
        "source_sha256": combined_source,
        "corpus_identity_sha256": identity,
        "chassis_count": len(chassis),
        "clip_count": len(clips),
        "total_frames": len(clips) * 72,
        "total_cell_samples": total_cell_samples,
        "binary_bytes": binary_size,
        "maximum_displacement": maximum_displacement,
        "minimum_meaningful_displacement": minimum_motion,
    }


def assert_valid_motion_corpus(path: str | Path) -> dict[str, Any]:
    return validate_motion_corpus(path)


def load_clip_deltas(path: str | Path, clip_id: int) -> np.ndarray:
    """Validate the corpus, then return one immutable [frame, cell, xy] tensor."""
    assert_valid_motion_corpus(path)
    root, _manifest_path, document, _raw = _load_manifest(path)
    if isinstance(clip_id, bool) or not 0 <= int(clip_id) < len(document["clips"]):
        raise MotionCorpusValidationError("clip_id is outside the corpus")
    clip = document["clips"][int(clip_id)]
    binary_path = root / document["artifacts"]["motion_frames"]["path"]
    with binary_path.open("rb") as handle:
        handle.seek(int(clip["byte_offset"]))
        payload = handle.read(int(clip["byte_length"]))
    encoded = np.frombuffer(payload, dtype="<u2").reshape(
        int(clip["frames"]), int(clip["cell_count"]), 2
    )
    result = (encoded.astype(np.float32) - 32768.0) / 256.0
    result.setflags(write=False)
    return result
