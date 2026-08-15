from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from forge.creature_stage_motion_audit import (
    MotionAuditValidationError,
    assert_valid_motion_audit,
)
from forge.creature_stage_motion_audit.validation import FAMILIES, MORPHOTYPES, MOTIONS


def _document() -> dict:
    clips = []
    summaries = {}
    for family_id, family in enumerate(FAMILIES):
        family_hashes = []
        family_maximum = 0.5 + family_id * 0.1
        for morphotype_id, morphotype in enumerate(MORPHOTYPES[family_id]):
            seed = 0x6D0F0000 + family_id * 0x100 + morphotype_id
            for motion in MOTIONS:
                pose_hash = hashlib.sha256(
                    f"{family_id}:{morphotype_id}:{motion}".encode()
                ).hexdigest()
                family_hashes.append(pose_hash)
                clips.append(
                    {
                        "family": family,
                        "family_id": family_id,
                        "morphotype": morphotype,
                        "morphotype_id": morphotype_id,
                        "seed": seed,
                        "motion": motion,
                        "frames": 72,
                        "cell_count": 120 + family_id,
                        "appendage_cell_count": 32,
                        "maximum_displacement": family_maximum,
                        "maximum_appendage_displacement": 0.4,
                        "maximum_core_displacement": 0.3,
                        "spread_delta": 0.0,
                        "maximum_replay_delta": 0.0,
                        "pose_sha256": pose_hash,
                        "deterministic": True,
                        "finite": True,
                        "vertical_lock": True,
                        "organs_preserved": True,
                        "motion_retained": True,
                    }
                )
        summaries[str(family_id)] = {
            "family": family,
            "chassis_count": 4,
            "clip_count": 52,
            "unique_pose_signatures": len(set(family_hashes)),
            "minimum_meaningful_displacement": 0.3,
            "maximum_displacement": family_maximum,
        }
    identity_material = "|".join(
        f"{clip['family_id']}:{clip['morphotype_id']}:{clip['motion']}:{clip['pose_sha256']}"
        for clip in clips
    )
    return {
        "format": "nullvector-creature-stage-motion-audit-v1",
        "passed": True,
        "fixed_hz": 30,
        "frames_per_clip": 72,
        "family_count": 5,
        "chassis_count": 20,
        "motion_count": 13,
        "clip_count": 260,
        "clip_identity_sha256": hashlib.sha256(identity_material.encode()).hexdigest(),
        "motions": MOTIONS,
        "unique_pose_signatures": 260,
        "maximum_displacement": 0.9,
        "contracts": {
            "morphology": "coordinate-conditioned-safe-scaffold-v1",
            "motion": "layered-cellular-motion-13x20-v1",
            "orientation": "vertical-locked-2.5d-v1",
            "replay": "twin-body-exact-projection-v1",
        },
        "family_summary": summaries,
        "failures": [],
        "clips": clips,
    }


def _write(path: Path, document: dict, *, rehash: bool = False) -> Path:
    if rehash:
        identity_material = "|".join(
            f"{clip['family_id']}:{clip['morphotype_id']}:{clip['motion']}:{clip['pose_sha256']}"
            for clip in document["clips"]
        )
        document["clip_identity_sha256"] = hashlib.sha256(identity_material.encode()).hexdigest()
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def test_complete_motion_matrix_is_valid(tmp_path: Path) -> None:
    report = assert_valid_motion_audit(_write(tmp_path / "audit.json", _document()))
    assert report["passed"] is True
    assert report["clip_count"] == 260
    assert report["unique_pose_signatures"] == 260
    assert report["minimum_meaningful_displacement"] == pytest.approx(0.3)


def test_stale_clip_identity_hash_is_rejected(tmp_path: Path) -> None:
    document = _document()
    document["clips"][0]["pose_sha256"] = "0" * 64
    with pytest.raises(MotionAuditValidationError, match="clip identity SHA-256"):
        assert_valid_motion_audit(_write(tmp_path / "stale.json", document))


def test_rehashed_reordered_matrix_is_rejected(tmp_path: Path) -> None:
    document = _document()
    document["clips"][0], document["clips"][1] = (
        document["clips"][1],
        document["clips"][0],
    )
    with pytest.raises(MotionAuditValidationError, match="matrix"):
        assert_valid_motion_audit(_write(tmp_path / "reordered.json", document, rehash=True))


def test_rehashed_collapsed_motion_is_rejected(tmp_path: Path) -> None:
    document = _document()
    locomote = next(clip for clip in document["clips"] if clip["motion"] == "locomote")
    locomote["maximum_appendage_displacement"] = 0.0
    with pytest.raises(MotionAuditValidationError, match="collapsed motion"):
        assert_valid_motion_audit(_write(tmp_path / "collapsed.json", document, rehash=True))


def test_rehashed_duplicate_pose_is_rejected(tmp_path: Path) -> None:
    document = _document()
    document["clips"][1]["pose_sha256"] = document["clips"][0]["pose_sha256"]
    with pytest.raises(MotionAuditValidationError, match="duplicate pose"):
        assert_valid_motion_audit(_write(tmp_path / "duplicate-pose.json", document, rehash=True))


def test_rehashed_summary_forgery_is_rejected(tmp_path: Path) -> None:
    document = _document()
    document["family_summary"]["2"]["minimum_meaningful_displacement"] = 0.7
    with pytest.raises(MotionAuditValidationError, match="family minimum"):
        assert_valid_motion_audit(_write(tmp_path / "summary.json", document, rehash=True))


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "duplicate-key.json", deepcopy(_document()))
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace('{\n  "format"', '{\n  "format": "forged",\n  "format"', 1),
        encoding="utf-8",
    )
    with pytest.raises(MotionAuditValidationError, match="duplicate JSON key"):
        assert_valid_motion_audit(path)
