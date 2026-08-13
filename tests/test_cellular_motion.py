from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from forge.cellular_motion import replay_bank, validate_bank
from forge.cellular_motion.compiler import _channels, _pose_points
from forge.cellular_motion.contract import DRIVER_NAMES, FACING_NAMES, MOTION_NAMES, MOTION_SPECS
from forge.cellular_organism.compiler import _load_arrays


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "outputs/cellular_motion_v1/cellular_motion_manifest.json"
ORGANISMS = ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(BANK.read_text(encoding="utf-8"))


def test_cellular_motion_bank_validates_and_exact_replays() -> None:
    validation = validate_bank(BANK); replay = replay_bank(BANK)
    assert validation["passed"] is True
    assert (validation["identity_count"], validation["clip_count"], validation["frame_count"]) == (45, 520, 4720)
    assert replay["exact_replay"] is True and replay["artifact_count"] == 2 and replay["artifact_bytes"] == 2_731_184


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
