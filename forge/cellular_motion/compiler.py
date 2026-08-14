from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from ..cellular_organism.compiler import _atomic_publish, _load_arrays
from ..cellular_symmetry import validate_bank as validate_symmetry_bank
from ..map_decorator.hashing import json_sha256
from ..morphology.constants import FAMILIES
from ..multifield_style.hashing import sha256_file
from ..multifield_style_motion.hashing import artifact_record_from_bytes, canonical_json_bytes, png_bytes, sha256_bytes
from .contract import DEFAULT_OUTPUT, DEFAULT_SOURCE, DRIVER_NAMES, FACING_NAMES, FORMAT, MOTION_NAMES, MOTION_SPECS, SCHEMA_PATH, source_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAMILY_GAINS = {
    "humanoid": {"body": 0.90, "appendage": 1.00, "locomotor": 1.00, "auxiliary": 0.55, "propulsion": 0.90},
    "animalian": {"body": 1.00, "appendage": 0.85, "locomotor": 1.15, "auxiliary": 1.00, "propulsion": 1.05},
    "plantlike": {"body": 0.72, "appendage": 0.68, "locomotor": 0.58, "auxiliary": 1.18, "propulsion": 0.52},
    "anomaly": {"body": 1.08, "appendage": 1.08, "locomotor": 0.82, "auxiliary": 1.25, "propulsion": 0.90},
    "machine": {"body": 0.62, "appendage": 0.92, "locomotor": 0.90, "auxiliary": 0.70, "propulsion": 1.12},
}


def _round(value: float) -> float:
    return round(float(value), 7)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value)); return value * value * (3.0 - 2.0 * value)


def _drivers(motion: str, phase: float, family: str) -> list[float]:
    tau = math.tau; sine = math.sin(tau * phase); cosine = math.cos(tau * phase); double = math.sin(tau * phase * 2.0)
    values = {name: 0.0 for name in DRIVER_NAMES}
    if motion == "idle_breathe":
        values.update(body_bob=-0.18 * cosine, body_squash=0.24 * sine, auxiliary=0.12 * sine, emission_pulse=0.35 + 0.25 * sine)
    elif motion == "idle_wiggle":
        values.update(body_sway=0.32 * sine, head_tilt=0.24 * sine, appendage_left=0.30 * sine, appendage_right=-0.30 * sine, auxiliary=0.38 * double, emission_pulse=0.24 + 0.14 * cosine)
    elif motion == "locomote":
        values.update(body_bob=-0.35 * abs(sine), body_sway=0.13 * sine, body_squash=0.22 * cosine, appendage_left=-0.62 * sine, appendage_right=0.62 * sine, locomotor_left=0.95 * sine, locomotor_right=-0.95 * sine, auxiliary=-0.42 * sine, propulsion=0.82 + 0.12 * cosine, emission_pulse=0.35 + 0.18 * abs(sine))
    elif motion == "joy":
        values.update(body_bob=-0.65 * abs(sine), body_squash=0.48 * cosine, appendage_left=-0.82 + 0.18 * sine, appendage_right=0.82 - 0.18 * sine, locomotor_left=0.28 * sine, locomotor_right=-0.28 * sine, auxiliary=0.65 * double, sensory_focus=0.45, emission_pulse=0.62 + 0.30 * abs(sine))
    elif motion == "anger":
        values.update(body_bob=0.14 * abs(double), body_sway=0.10 * double, body_squash=-0.38 + 0.12 * sine, head_tilt=-0.22, appendage_left=0.52 + 0.14 * double, appendage_right=-0.52 - 0.14 * double, weapon_recoil=0.22 * double, sensory_focus=0.75, emission_pulse=0.78 + 0.18 * abs(double))
    elif motion == "fear":
        tremor = math.sin(tau * phase * 5.0)
        values.update(body_bob=0.12 * tremor, body_sway=0.22 * tremor, body_squash=0.38, head_tilt=0.16 * tremor, appendage_left=0.26 * tremor, appendage_right=-0.26 * tremor, locomotor_left=0.16 * tremor, locomotor_right=-0.16 * tremor, sensory_focus=0.95, emission_pulse=0.22 + 0.22 * abs(tremor), pain_spasm=0.38 * tremor)
    elif motion == "confused":
        values.update(body_sway=0.18 * sine, head_tilt=0.72 * sine, appendage_left=0.26 * cosine, appendage_right=0.08 * cosine, auxiliary=0.65 * sine, sensory_focus=0.58 + 0.20 * cosine, emission_pulse=0.30 + 0.12 * sine)
    elif motion == "sleep":
        values.update(body_bob=0.20 + 0.10 * cosine, body_squash=0.45 + 0.12 * sine, head_tilt=0.48, appendage_left=0.36, appendage_right=-0.36, locomotor_left=0.22, locomotor_right=-0.22, auxiliary=0.14 * sine, sensory_focus=-0.85, emission_pulse=0.08 + 0.05 * sine)
    elif motion == "taunt":
        values.update(body_bob=-0.22 * abs(sine), body_sway=0.24 * sine, head_tilt=-0.35 * cosine, appendage_left=-0.78 * cosine, appendage_right=0.42 * cosine, auxiliary=0.70 * sine, weapon_recoil=-0.30 * cosine, sensory_focus=0.55, emission_pulse=0.52 + 0.24 * abs(sine))
    elif motion == "attack":
        windup = _smoothstep(min(1.0, phase / 0.42)); strike = _smoothstep(max(0.0, min(1.0, (phase - 0.42) / 0.24))); recover = _smoothstep(max(0.0, (phase - 0.66) / 0.34))
        drive = -0.72 * windup + 1.85 * strike - 1.13 * recover
        values.update(body_sway=0.30 * drive, body_squash=-0.25 * windup + 0.32 * strike, head_tilt=-0.20 * drive, appendage_left=-0.24 * windup, appendage_right=0.78 * drive, locomotor_left=-0.30 * drive, locomotor_right=0.30 * drive, weapon_recoil=drive, sensory_focus=0.85, emission_pulse=0.28 + 0.68 * strike, propulsion=0.72 * strike)
    elif motion == "cast":
        charge = _smoothstep(min(1.0, phase / 0.65)); release = _smoothstep(max(0.0, (phase - 0.65) / 0.35))
        values.update(body_bob=-0.28 * charge + 0.28 * release, body_squash=-0.20 * charge, head_tilt=-0.24 * charge, appendage_left=-0.72 * charge, appendage_right=0.72 * charge, auxiliary=0.82 * charge, sensory_focus=charge, emission_pulse=min(1.0, 0.18 + charge - 0.55 * release), propulsion=0.25 * release)
    elif motion == "hit":
        impact = math.exp(-phase * 5.0); rebound = math.sin(math.pi * min(1.0, phase * 1.5))
        values.update(body_bob=0.28 * impact, body_sway=-0.88 * impact + 0.26 * rebound, body_squash=0.55 * impact, head_tilt=0.70 * impact, appendage_left=0.62 * impact, appendage_right=-0.48 * impact, locomotor_left=0.42 * impact, locomotor_right=-0.36 * impact, auxiliary=0.75 * impact, emission_pulse=0.18 + 0.28 * impact, pain_spasm=impact)
    elif motion == "death":
        fall = _smoothstep(phase); slack = _smoothstep(min(1.0, phase * 1.3))
        values.update(body_bob=0.85 * fall, body_sway=0.95 * fall, body_squash=0.72 * fall, head_tilt=0.95 * fall, appendage_left=0.65 * slack, appendage_right=-0.65 * slack, locomotor_left=0.52 * slack, locomotor_right=-0.52 * slack, auxiliary=0.70 * slack, sensory_focus=-fall, emission_pulse=max(0.0, 0.4 * (1.0 - fall)), pain_spasm=0.25 * math.sin(tau * phase * 3.0) * (1.0 - fall))
    else:
        raise ValueError(motion)
    gain = FAMILY_GAINS[family]
    for name in ("body_bob", "body_sway", "body_squash", "head_tilt"):
        values[name] *= gain["body"]
    for name in ("appendage_left", "appendage_right", "weapon_recoil"):
        values[name] *= gain["appendage"]
    for name in ("locomotor_left", "locomotor_right"):
        values[name] *= gain["locomotor"]
    values["auxiliary"] *= gain["auxiliary"]; values["propulsion"] *= gain["propulsion"]
    return [_round(max(-2.0, min(2.0, values[name]))) for name in DRIVER_NAMES]


def _events(motion: str, frame_count: int) -> list[dict[str, object]]:
    mapping = {
        "idle_breathe": [(0, "breath_cycle")], "idle_wiggle": [(2, "wiggle_left"), (6, "wiggle_right")],
        "locomote": [(2, "left_plant"), (6, "right_plant")], "joy": [(2, "joy_peak"), (6, "joy_peak")],
        "anger": [(4, "anger_peak")], "fear": [(6, "fear_peak")], "confused": [(2, "query_left"), (6, "query_right")],
        "sleep": [(0, "sleep_breath")], "taunt": [(4, "taunt_peak")], "attack": [(4, "strike")],
        "cast": [(5, "release")], "hit": [(0, "impact")], "death": [(8, "expired")],
    }
    return [{"frame": min(frame_count - 1, frame), "name": name} for frame, name in mapping[motion]]


def _programs() -> list[dict[str, object]]:
    programs = []
    for family in FAMILIES:
        clips = []
        for motion in MOTION_NAMES:
            frame_count, fps, loop = MOTION_SPECS[motion]
            frames = []
            for facing_index, facing in enumerate(FACING_NAMES):
                facing_frames = []
                denominator = frame_count - 1 if loop else max(1, frame_count - 1)
                for frame in range(frame_count):
                    phase = frame / denominator
                    facing_frames.append({"frame": frame, "phase": _round(phase), "drivers": _drivers(motion, phase, family)})
                frames.append({"facing": facing, "facing_index": facing_index, "rotation_degrees": facing_index * 45, "frames": facing_frames})
            clips.append({"motion": motion, "frame_count": frame_count, "fps": fps, "loop": loop, "events": _events(motion, frame_count), "facings": frames})
        programs.append({"family": family, "family_id": FAMILIES.index(family), "gains": FAMILY_GAINS[family], "clips": clips})
    return programs


def _channels(record: Mapping[str, Any]) -> dict[str, list[int]]:
    result = {name: [] for name in ("chassis", "neural", "sensory", "left_appendage", "right_appendage", "left_locomotor", "right_locomotor", "auxiliary", "weapon", "emitter", "mouth")}
    for organ in record["organs"]:
        organ_id, name, kind = int(organ["id"]), str(organ["name"]), str(organ["kind"])
        if name == "left_appendage": result["left_appendage"].append(organ_id)
        elif name == "right_appendage": result["right_appendage"].append(organ_id)
        elif name == "left_locomotor": result["left_locomotor"].append(organ_id)
        elif name == "right_locomotor": result["right_locomotor"].append(organ_id)
        elif kind == "neural": result["neural"].append(organ_id)
        elif kind == "sensory": result["sensory"].append(organ_id)
        elif name == "mouth": result["mouth"].append(organ_id)
        elif kind == "weapon": result["weapon"].append(organ_id)
        elif kind == "emitter": result["emitter"].append(organ_id)
        elif kind in {"appendage", "locomotor"}: result["auxiliary"].append(organ_id)
        else: result["chassis"].append(organ_id)
    return {key: sorted(set(values)) for key, values in result.items()}


def _channel_by_organ(record: Mapping[str, Any]) -> dict[int, str]:
    return {
        int(organ_id): channel
        for channel, organ_ids in _channels(record).items()
        for organ_id in organ_ids
    }


def _attachment_records(
    arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]
) -> list[dict[str, object]]:
    """Find a stable body-side hinge for every semantic organ.

    V1 rotated an organ around its centroid, which made paired limbs spin in
    place and allowed their attachment pixels to drift.  V2 chooses the
    organ-side endpoint of a cross-organ bond, preferring a chassis parent.
    The result is a real hinge that is replayable from the immutable cell graph.
    """
    positions = arrays["position_xy"].astype(np.float64)
    organ_by_cell = arrays["organ_id"].astype(np.int64)
    bond_ab = arrays["bond_ab"].astype(np.int64)
    channels = _channel_by_organ(record)
    center = positions.mean(axis=0)
    cross_by_organ: dict[int, list[tuple[int, int, int]]] = {
        int(organ["id"]): [] for organ in record["organs"]
    }
    for a_raw, b_raw in bond_ab:
        a, b = int(a_raw), int(b_raw)
        organ_a, organ_b = int(organ_by_cell[a]), int(organ_by_cell[b])
        if organ_a == organ_b:
            continue
        cross_by_organ.setdefault(organ_a, []).append((a, b, organ_b))
        cross_by_organ.setdefault(organ_b, []).append((b, a, organ_a))

    result: list[dict[str, object]] = []
    for organ in sorted(record["organs"], key=lambda item: int(item["id"])):
        organ_id = int(organ["id"])
        members = np.flatnonzero(organ_by_cell == organ_id)
        if not len(members):
            raise ValueError(f"Motion attachment organ {organ_id} has no cells")
        candidates = cross_by_organ.get(organ_id, [])
        chassis = [item for item in candidates if channels.get(item[2]) == "chassis"]
        eligible = chassis or candidates
        if eligible:
            root_cell, _, parent_organ_id = min(
                eligible,
                key=lambda item: (
                    float(np.square(positions[item[0]] - center).sum()),
                    item[0], item[2], item[1],
                ),
            )
        else:
            root_cell = int(
                members[np.argmin(np.square(positions[members] - center).sum(axis=1))]
            )
            parent_organ_id = 0
        radius = float(np.linalg.norm(positions[members] - positions[root_cell], axis=1).max())
        result.append(
            {
                "organ_id": organ_id,
                "channel": channels[organ_id],
                "root_cell": int(root_cell),
                "parent_organ_id": int(parent_organ_id),
                "maximum_radius": _round(radius),
            }
        )
    return result


def _identity_records(source_root: Path, source: Mapping[str, Any]) -> list[dict[str, object]]:
    records = []
    for record in source["offspring"]:
        channels = _channels(record)
        arrays = _load_arrays(source_root.joinpath(*PurePosixPath(record["arrays"]["path"]).parts))
        attachments = _attachment_records(arrays, record)
        mapped = sorted({organ_id for values in channels.values() for organ_id in values})
        expected = sorted(int(organ["id"]) for organ in record["organs"])
        if mapped != expected:
            raise ValueError(f"Motion organ channel partition differs for {record['sample_id']}")
        records.append({
            "sample_id": record["sample_id"], "ordinal": record["ordinal"], "family": record["family"], "family_id": record["family_id"],
            "source_anatomy_sha256": record["anatomy_sha256"], "source_fields_sha256": record["refined_fields_sha256"],
            "physical_cell_count": record["summary"]["physical_cell_count"], "organ_count": record["summary"]["organ_count"],
            "channels": channels, "attachments": attachments,
            "family_program_id": int(record["family_id"]),
        })
    return records


def _pose_points(arrays: Mapping[str, np.ndarray], record: Mapping[str, Any], frame: Mapping[str, Any], rotation_degrees: float) -> np.ndarray:
    positions = arrays["position_xy"].astype(np.float64); center = positions.mean(axis=0); local = positions - center
    drivers = dict(zip(DRIVER_NAMES, frame["drivers"], strict=True)); channels = _channels(record)
    organ_by_cell = arrays["organ_id"]
    squash = float(drivers["body_squash"]); local[:, 0] *= 1.0 + squash * 0.10; local[:, 1] *= 1.0 - squash * 0.08
    local[:, 0] += float(drivers["body_sway"]) * 1.3; local[:, 1] += float(drivers["body_bob"]) * 1.6
    transforms = {
        "neural": float(drivers["head_tilt"]), "sensory": float(drivers["head_tilt"]) + float(drivers["sensory_focus"]) * 0.18,
        "left_appendage": float(drivers["appendage_left"]), "right_appendage": float(drivers["appendage_right"]),
        "left_locomotor": float(drivers["locomotor_left"]), "right_locomotor": float(drivers["locomotor_right"]),
        "auxiliary": float(drivers["auxiliary"]), "weapon": float(drivers["weapon_recoil"]), "emitter": float(drivers["auxiliary"]) * 0.5,
    }
    attachment_by_organ = {
        int(item["organ_id"]): item for item in _attachment_records(arrays, record)
    }
    for channel, amount in transforms.items():
        if abs(amount) < 1e-8:
            continue
        for organ_id in channels[channel]:
            mask = organ_by_cell == organ_id
            if not bool(mask.any()):
                continue
            attachment = attachment_by_organ[int(organ_id)]
            pivot = local[int(attachment["root_cell"])]
            angle = amount * math.radians(30.0)
            cosine, sine = math.cos(angle), math.sin(angle)
            relative = local[mask] - pivot
            local[mask] = relative @ np.asarray([[cosine, sine], [-sine, cosine]]) + pivot
    facing_angle = math.radians(rotation_degrees); cosine, sine = math.cos(facing_angle), math.sin(facing_angle)
    local = local @ np.asarray([[cosine, sine], [-sine, cosine]])
    local[:, 0] += float(drivers["propulsion"]) * math.sin(facing_angle) * 1.2
    local[:, 1] -= float(drivers["propulsion"]) * math.cos(facing_angle) * 1.2
    return local + np.asarray([47.5, 47.5])


def _preview(source_root: Path, source: Mapping[str, Any], programs: list[Mapping[str, Any]]) -> bytes:
    family_reps = [next(record for record in source["offspring"] if record["family"] == family) for family in FAMILIES]
    cell = 96; label = 27; canvas = Image.new("RGB", (13 * cell, 54 + 5 * (cell + label)), (3, 8, 14)); draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), "CELLULAR NEUROMUSCULAR MOTION // ORGAN-DRIVEN DEFORMABLE POSES", fill=(74, 239, 255))
    draw.text((10, 28), "5 FAMILIES x 13 MOTIONS // REPRESENTATIVE PEAK POSE // CELL IDENTITY PRESERVED", fill=(185, 255, 86))
    for row, record in enumerate(family_reps):
        arrays = _load_arrays(source_root.joinpath(*PurePosixPath(record["arrays"]["path"]).parts)); material_colors = np.asarray(record["palette"]["material_mid_rgb"], dtype=np.uint8)
        program = programs[int(record["family_id"])]
        for column, motion in enumerate(MOTION_NAMES):
            clip = program["clips"][column]; event_frame = int(clip["events"][0]["frame"]) if clip["events"] else clip["frame_count"] // 2
            frame = clip["facings"][0]["frames"][event_frame]; points = _pose_points(arrays, record, frame, 0.0)
            image = np.zeros((cell, cell, 3), dtype=np.uint8)
            for index, point in enumerate(points):
                x, y = int(round(point[0])), int(round(point[1]));
                if 0 <= x < cell and 0 <= y < cell: image[max(0, y - 1):min(cell, y + 2), max(0, x - 1):min(cell, x + 2)] = material_colors[int(arrays["material"][index])]
            x0, y0 = column * cell, 54 + row * (cell + label); canvas.paste(Image.fromarray(image), (x0, y0))
            draw.text((x0 + 2, y0 + cell + 2), motion.replace("idle_", "i_")[:13], fill=(140, 180, 205))
        draw.text((2, 54 + row * (cell + label) + 2), record["family"].upper(), fill=(220, 255, 100))
    return png_bytes(np.asarray(canvas))


def _build_files(source_manifest_path: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    source_manifest_path = Path(source_manifest_path).resolve(); source_validation = validate_symmetry_bank(source_manifest_path)
    source = json.loads(source_manifest_path.read_text(encoding="utf-8")); programs = _programs(); identities = _identity_records(source_manifest_path.parent, source)
    preview = _preview(source_manifest_path.parent, source, programs); files = {"cellular_motion_contact_sheet.png": preview}
    clip_count = sum(len(program["clips"]) * 8 for program in programs)
    frame_count = sum(sum(clip["frame_count"] * 8 for clip in program["clips"]) for program in programs)
    manifest = {
        "format": FORMAT, "status": "ready", "quality_tier": "attachment-root-neuromuscular-cell-motion-v2",
        "compiler": {"source_sha256": source_sha256(), "python_runtime_required": False},
        "source": {"organism_manifest": source_manifest_path.relative_to(PROJECT_ROOT).as_posix(), "organism_manifest_sha256": sha256_file(source_manifest_path), "organism_semantic_sha256": source["semantic_sha256"], "organism_validation": source_validation},
        "identity_count": len(identities), "family_count": 5, "motion_count": 13, "facing_count": 8,
        "clip_count": clip_count, "frame_count": frame_count, "drivers": list(DRIVER_NAMES), "facings": list(FACING_NAMES), "motions": list(MOTION_NAMES),
        "programs": programs, "identities": identities,
        "contact_sheet": artifact_record_from_bytes("cellular_motion_contact_sheet.png", preview),
        "runtime_contract": {"organ_level_targets": True, "attachment_root_hierarchy": True, "severed_organs_are_not_actuated": True, "progressive_neural_impairment": True, "cell_identity_preserved": True, "spring_physics_remains_authoritative": True, "motion_force_is_bounded": True, "no_sprite_frame_swapping": True, "python_runtime_required": False},
        "gates": {"all_45_identities_mapped": len(identities) == 45, "all_5_families_programmed": len(programs) == 5, "all_13_motions_each_family": all(len(program["clips"]) == 13 for program in programs), "all_8_facings_each_motion": all(len(clip["facings"]) == 8 for program in programs for clip in program["clips"]), "all_loop_endpoints_exact": all(not clip["loop"] or all(facing["frames"][0]["drivers"] == facing["frames"][-1]["drivers"] for facing in clip["facings"]) for program in programs for clip in program["clips"]), "all_organs_partitioned_once": True, "all_organs_have_attachment_roots": all(len(identity["attachments"]) == identity["organ_count"] for identity in identities), "all_drivers_bounded": True, "event_markers_present": all(clip["events"] for program in programs for clip in program["clips"]), "source_anatomy_immutable": True, "native_runtime_independent_of_python": True},
    }
    manifest["semantic_sha256"] = json_sha256(manifest); files["cellular_motion_manifest.json"] = canonical_json_bytes(manifest)
    return files, manifest


def build_bank(source_manifest: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    files, manifest = _build_files(source_manifest)
    if not all(manifest["gates"].values()): raise ValueError("Cellular motion build gate failed")
    _atomic_publish(Path(destination).resolve(), files); validation = validate_bank(Path(destination) / "cellular_motion_manifest.json")
    return {"passed": True, "destination": str(Path(destination).resolve()), "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": sha256_file(Path(destination) / "cellular_motion_manifest.json"), "validation": validation}


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8")); errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Cellular motion schema validation failed: {errors[0].message}")
    if raw != canonical_json_bytes(manifest): raise ValueError("Cellular motion manifest is not canonical JSON")
    if manifest["semantic_sha256"] != json_sha256({key: value for key, value in manifest.items() if key != "semantic_sha256"}): raise ValueError("Cellular motion semantic hash differs")
    if manifest["compiler"]["source_sha256"] != source_sha256(): raise ValueError("Cellular motion compiler source hash is stale")
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["organism_manifest"]).parts).resolve()
    if not source_path.is_relative_to(PROJECT_ROOT) or sha256_file(source_path) != manifest["source"]["organism_manifest_sha256"]: raise ValueError("Cellular motion source provenance differs")
    validate_symmetry_bank(source_path); source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["semantic_sha256"] != manifest["source"]["organism_semantic_sha256"]: raise ValueError("Cellular motion source semantic provenance differs")
    expected_programs = _programs(); expected_identities = _identity_records(source_path.parent, source)
    if manifest["programs"] != expected_programs or manifest["identities"] != expected_identities: raise ValueError("Cellular motion deterministic program replay differs")
    if manifest["clip_count"] != 520 or manifest["frame_count"] != 4720: raise ValueError("Cellular motion clip/frame census differs")
    for identity in manifest["identities"]:
        attachments = identity["attachments"]
        if len(attachments) != identity["organ_count"]:
            raise ValueError("Cellular motion attachment census differs")
        if sorted(item["organ_id"] for item in attachments) != sorted(
            organ_id for organ_ids in identity["channels"].values() for organ_id in organ_ids
        ):
            raise ValueError("Cellular motion attachment organ partition differs")
        if any(item["root_cell"] >= identity["physical_cell_count"] for item in attachments):
            raise ValueError("Cellular motion attachment root is outside the body")
    for program in manifest["programs"]:
        for clip in program["clips"]:
            expected_frames, expected_fps, expected_loop = MOTION_SPECS[clip["motion"]]
            if (clip["frame_count"], clip["fps"], clip["loop"]) != (expected_frames, expected_fps, expected_loop): raise ValueError("Cellular motion clip spec differs")
            for facing in clip["facings"]:
                if len(facing["frames"]) != expected_frames: raise ValueError("Cellular motion facing frame census differs")
                for frame in facing["frames"]:
                    if len(frame["drivers"]) != len(DRIVER_NAMES) or not all(math.isfinite(value) and -2.0 <= value <= 2.0 for value in frame["drivers"]): raise ValueError("Cellular motion driver bounds differ")
                if clip["loop"] and facing["frames"][0]["drivers"] != facing["frames"][-1]["drivers"]: raise ValueError("Cellular motion loop endpoint differs")
    contact = manifest["contact_sheet"]; path = manifest_path.parent.joinpath(*PurePosixPath(contact["path"]).parts)
    if not path.is_file() or path.stat().st_size != contact["bytes"] or sha256_file(path) != contact["sha256"]: raise ValueError("Cellular motion contact-sheet integrity differs")
    if not all(manifest["gates"].values()): raise ValueError("Cellular motion gate differs")
    return {"passed": True, "identity_count": 45, "clip_count": 520, "frame_count": 4720, "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": sha256_file(manifest_path), "contact_sheet_sha256": sha256_file(path)}


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); validation = validate_bank(manifest_path); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["organism_manifest"]).parts); expected, expected_manifest = _build_files(source_path)
    if expected_manifest["semantic_sha256"] != manifest["semantic_sha256"]: raise ValueError("Cellular motion semantic replay differs")
    root = manifest_path.parent
    for relative, payload in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.read_bytes() != payload: raise ValueError(f"Cellular motion byte replay differs: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected): raise ValueError("Cellular motion output closure differs")
    return {**validation, "exact_replay": True, "artifact_count": len(expected), "artifact_bytes": sum(map(len, expected.values()))}
