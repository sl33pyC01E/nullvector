from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from forge.cellular_motion import replay_bank, validate_bank
from forge.cellular_motion.compiler import _attachment_records, _channels, _pose_points
from forge.cellular_motion.contract import DRIVER_NAMES, FACING_NAMES, MOTION_NAMES, MOTION_SPECS
from forge.cellular_organism.compiler import _load_arrays


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/cellular_motion_v2/cellular_motion_manifest.json"
ORGANISMS = ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(BANK.read_text(encoding="utf-8"))


def test_cellular_motion_bank_validates_and_exact_replays() -> None:
    validation = validate_bank(BANK); replay = replay_bank(BANK)
    assert validation["passed"] is True
    assert (validation["identity_count"], validation["clip_count"], validation["frame_count"]) == (45, 520, 4720)
    assert replay["exact_replay"] is True and replay["artifact_count"] == 2


def test_every_family_motion_facing_frame_and_loop_contract_is_exact() -> None:
    manifest = _manifest()
    assert manifest["drivers"] == list(DRIVER_NAMES)
    assert manifest["motions"] == list(MOTION_NAMES) and manifest["facings"] == list(FACING_NAMES)
    for family_id, program in enumerate(manifest["programs"]):
        assert program["family_id"] == family_id and len(program["clips"]) == 13
        for clip, motion in zip(program["clips"], MOTION_NAMES, strict=True):
            frames, fps, loop = MOTION_SPECS[motion]
            assert (clip["frame_count"], clip["fps"], clip["loop"]) == (frames, fps, loop)
            assert clip["events"] and len(clip["facings"]) == 8
            for facing_index, facing in enumerate(clip["facings"]):
                assert facing["facing"] == FACING_NAMES[facing_index]
                assert facing["rotation_degrees"] == facing_index * 45
                assert len(facing["frames"]) == frames
                if loop:
                    assert facing["frames"][0]["drivers"] == facing["frames"][-1]["drivers"]


def test_motion_profiles_are_non_degenerate_and_family_specific() -> None:
    programs = _manifest()["programs"]
    driver_index = {name: index for index, name in enumerate(DRIVER_NAMES)}
    for program in programs:
        locomote = next(clip for clip in program["clips"] if clip["motion"] == "locomote")
        frames = locomote["facings"][0]["frames"]
        left = [frame["drivers"][driver_index["locomotor_left"]] for frame in frames]
        right = [frame["drivers"][driver_index["locomotor_right"]] for frame in frames]
        assert max(left) - min(left) > 0.5
        assert np.allclose(left, np.negative(right), atol=1e-7)
        assert max(frame["drivers"][driver_index["propulsion"]] for frame in frames) > 0.4
    gains = [program["gains"] for program in programs]
    assert gains[2]["auxiliary"] > gains[0]["auxiliary"]
    assert gains[1]["locomotor"] > gains[4]["locomotor"]
    assert gains[4]["body"] < gains[3]["body"]


def test_every_identity_partitions_all_organs_exactly_once() -> None:
    source = json.loads(ORGANISMS.read_text(encoding="utf-8")); source_by_id = {record["sample_id"]: record for record in source["offspring"]}
    for identity in _manifest()["identities"]:
        record = source_by_id[identity["sample_id"]]
        assert identity["channels"] == _channels(record)
        mapped = [organ_id for values in identity["channels"].values() for organ_id in values]
        assert len(mapped) == len(set(mapped)) == identity["organ_count"]
        assert sorted(mapped) == sorted(organ["id"] for organ in record["organs"])
        assert identity["source_anatomy_sha256"] == record["anatomy_sha256"]


def test_every_organ_has_a_replayable_attachment_root_and_parent() -> None:
    source = json.loads(ORGANISMS.read_text(encoding="utf-8"))
    for record in source["offspring"]:
        arrays = _load_arrays(ORGANISMS.parent / record["arrays"]["path"])
        attachments = _attachment_records(arrays, record)
        assert len(attachments) == len(record["organs"])
        assert {item["organ_id"] for item in attachments} == {item["id"] for item in record["organs"]}
        for item in attachments:
            assert 0 <= item["root_cell"] < len(arrays["position_xy"])
            assert arrays["organ_id"][item["root_cell"]] == item["organ_id"]
            assert math.isfinite(item["maximum_radius"]) and item["maximum_radius"] >= 0


def test_locomotor_and_action_organs_swing_from_attachment_not_centroid() -> None:
    source = json.loads(ORGANISMS.read_text(encoding="utf-8")); manifest = _manifest()
    driver_index = {name: index for index, name in enumerate(DRIVER_NAMES)}
    for family_id in range(5):
        record = next(record for record in source["offspring"] if record["family_id"] == family_id)
        arrays = _load_arrays(ORGANISMS.parent / record["arrays"]["path"])
        attachments = {item["organ_id"]: item for item in _attachment_records(arrays, record)}
        for motion, channel_drivers in {
            "locomote": {
                "left_appendage": "appendage_left", "right_appendage": "appendage_right",
                "left_locomotor": "locomotor_left", "right_locomotor": "locomotor_right",
            },
            "attack": {"right_appendage": "appendage_right", "weapon": "weapon_recoil"},
        }.items():
            clip = next(item for item in manifest["programs"][family_id]["clips"] if item["motion"] == motion)
            for channel, driver in channel_drivers.items():
                values = clip["facings"][0]["frames"]
                low = min(values, key=lambda item: abs(item["drivers"][driver_index[driver]]))
                peak = max(values, key=lambda item: abs(item["drivers"][driver_index[driver]]))
                if abs(peak["drivers"][driver_index[driver]] - low["drivers"][driver_index[driver]]) < 0.1:
                    continue
                base = _pose_points(arrays, record, low, 0.0); posed = _pose_points(arrays, record, peak, 0.0)
                for organ_id in _channels(record)[channel]:
                    members = np.flatnonzero(arrays["organ_id"] == organ_id)
                    if len(members) < 2: continue
                    root = attachments[organ_id]["root_cell"]
                    distal = members[np.argmax(np.linalg.norm(arrays["position_xy"][members] - arrays["position_xy"][root], axis=1))]
                    rest_vector = base[distal] - base[root]; posed_vector = posed[distal] - posed[root]
                    assert np.linalg.norm(posed_vector - rest_vector) > 0.01

        breathe = next(item for item in manifest["programs"][family_id]["clips"] if item["motion"] == "idle_breathe")
        spans = []
        for frame in breathe["facings"][0]["frames"]:
            points = _pose_points(arrays, record, frame, 0.0)
            spans.append(float(np.ptp(points[:, 1])))
        assert max(spans) - min(spans) > 0.05


def test_reference_pose_projection_preserves_cell_identity_and_is_bounded() -> None:
    source = json.loads(ORGANISMS.read_text(encoding="utf-8")); manifest = _manifest()
    for family_id in range(5):
        record = next(record for record in source["offspring"] if record["family_id"] == family_id)
        arrays = _load_arrays(ORGANISMS.parent / record["arrays"]["path"])
        for clip in manifest["programs"][family_id]["clips"]:
            for facing in clip["facings"]:
                for frame in (facing["frames"][0], facing["frames"][len(facing["frames"]) // 2], facing["frames"][-1]):
                    points = _pose_points(arrays, record, frame, facing["rotation_degrees"])
                    assert points.shape == arrays["position_xy"].shape
                    assert np.isfinite(points).all()
                    assert np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)) < 48.0
                    assert len({tuple(np.round(point, 5)) for point in points}) == len(points)


def test_action_events_have_semantically_named_peaks() -> None:
    clips = _manifest()["programs"][0]["clips"]
    events = {clip["motion"]: {event["name"] for event in clip["events"]} for clip in clips}
    assert events["attack"] == {"strike"}
    assert events["cast"] == {"release"}
    assert events["hit"] == {"impact"}
    assert events["death"] == {"expired"}
    assert events["locomote"] == {"left_plant", "right_plant"}
