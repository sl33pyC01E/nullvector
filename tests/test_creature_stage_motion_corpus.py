from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import struct

import pytest

from forge.creature_stage_motion_corpus import (
    MotionCorpusValidationError,
    assert_valid_motion_corpus,
    load_clip_deltas,
)
from forge.creature_stage_motion_corpus.validation import (
    FAMILIES,
    GAME_ROOT,
    MORPHOTYPES,
    MOTIONS,
    MOTION_SPECS,
    SOURCE_PATHS,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cell_identity(cells: list[dict]) -> str:
    material = "|".join(
        f"{cell['grid'][0]},{cell['grid'][1]},{cell['tissue']},{cell['organ']},"
        f"{cell['appendage']},{cell['side']},{cell['initial_health_q']}"
        for cell in cells
    )
    return _sha(material.encode())


def _corpus_identity(document: dict) -> str:
    parts = []
    for chassis in document["chassis"]:
        chassis_id = chassis["chassis_id"]
        parts.append(f"chassis:{chassis_id}:{chassis['cell_identity_sha256']}")
        first = chassis_id * len(MOTIONS)
        for clip in document["clips"][first : first + len(MOTIONS)]:
            parts.append(f"clip:{clip['clip_id']}:{clip['trajectory_sha256']}")
    return _sha("|".join(parts).encode())


def _write_manifest(root: Path, document: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def _build_fixture(root: Path) -> Path:
    root.mkdir()
    cells = []
    for cell_id in range(70):
        cells.append(
            {
                "grid": [cell_id % 10 - 5, cell_id // 10 - 3],
                "tissue": "skin",
                "organ": "none",
                "appendage": -1 if cell_id == 0 else 0,
                "side": 0 if cell_id == 0 else (1 if cell_id % 2 else -1),
                "initial_health_q": 255,
            }
        )
    cell_hash = _cell_identity(cells)
    chassis = []
    for chassis_id in range(20):
        family_id, morphotype_id = divmod(chassis_id, 4)
        chassis.append(
            {
                "chassis_id": chassis_id,
                "family": FAMILIES[family_id],
                "family_id": family_id,
                "morphotype": MORPHOTYPES[family_id][morphotype_id],
                "morphotype_id": morphotype_id,
                "seed": 0x6D0F0000 + family_id * 0x100 + morphotype_id,
                "generation": 0,
                "genes": {
                    "width": 1.0,
                    "height": 1.0,
                    "asymmetry": 0.0,
                    "symmetry": 1.0,
                    "repair": 1.0,
                    "metabolism": 1.0,
                    "fertility": 1.0,
                    "bond_strength": 1.0,
                },
                "cell_count": 70,
                "cell_identity_sha256": cell_hash,
                "cells": deepcopy(cells),
            }
        )

    frame = b"".join(struct.pack("<HH", 32896, 32768) for _ in range(70))
    trajectory = frame * 72
    trajectory_sha = _sha(trajectory)
    binary = trajectory * 260
    (root / "motion_frames.u16le").write_bytes(binary)
    clips = []
    for clip_id in range(260):
        chassis_id, motion_id = divmod(clip_id, len(MOTIONS))
        family_id, morphotype_id = divmod(chassis_id, 4)
        motion = MOTIONS[motion_id]
        clips.append(
            {
                "clip_id": clip_id,
                "chassis_id": chassis_id,
                "family_id": family_id,
                "morphotype_id": morphotype_id,
                "motion": motion,
                "motion_id": motion_id,
                "frames": 72,
                "cell_count": 70,
                "frame_stride_bytes": 280,
                "byte_offset": clip_id * len(trajectory),
                "byte_length": len(trajectory),
                "trajectory_sha256": trajectory_sha,
                "controls": {
                    "move": [1.0, 0.0] if motion == "locomote" else [0.0, 0.0],
                    "aim": [0.8, -0.6],
                    "attack": 1.0 if motion == "attack" else 0.0,
                    "utility": 1.0 if motion == "cast" else 0.0,
                    "external_event": "impact" if motion == "hit" else ("terminal" if motion == "death" else "none"),
                },
            }
        )
    source_hashes = {
        relative: _sha((GAME_ROOT / relative).read_bytes()) for relative in SOURCE_PATHS
    }
    source_combined = _sha(
        "|".join(f"{relative}:{source_hashes[relative]}" for relative in SOURCE_PATHS).encode()
    )
    document = {
        "format": "nullvector-creature-stage-motion-corpus-v1",
        "passed": True,
        "fixed_hz": 30,
        "frames_per_clip": 72,
        "family_count": 5,
        "chassis_count": 20,
        "motion_count": 13,
        "clip_count": 260,
        "total_frames": 18720,
        "total_cell_samples": 260 * 72 * 70,
        "motion_order": MOTIONS,
        "motion_specs": MOTION_SPECS,
        "quantization": {
            "format": "position-delta-u16le-biased-v1",
            "components": ["x", "y"],
            "scale": 256,
            "bias": 32768,
            "clipped_values": 0,
            "minimum_encoded": 32768,
            "maximum_encoded": 32896,
        },
        "control_contract": {
            "move": "unit-disk-vec2",
            "aim": "normalized-vec2",
            "attack": "unit-scalar",
            "utility": "unit-scalar",
            "external_event": ["none", "impact", "terminal"],
        },
        "contracts": {
            "morphology": "coordinate-conditioned-safe-scaffold-v1",
            "motion": "layered-cellular-motion-13x20-v1",
            "orientation": "vertical-locked-2.5d-v1",
            "binary": "clip-major-frame-major-cell-major-xy-u16le-v1",
        },
        "source": {"files": source_hashes, "combined_sha256": source_combined},
        "corpus_identity_sha256": "",
        "artifacts": {
            "motion_frames": {
                "path": "motion_frames.u16le",
                "bytes": len(binary),
                "sha256": _sha(binary),
            }
        },
        "chassis": chassis,
        "clips": clips,
    }
    document["corpus_identity_sha256"] = _corpus_identity(document)
    _write_manifest(root, document)
    return root


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_fixture(tmp_path_factory.mktemp("motion-corpus") / "corpus")


def _copy(corpus_root: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "corpus"
    shutil.copytree(corpus_root, destination)
    return destination


def test_valid_corpus_and_tensor_loader(corpus_root: Path) -> None:
    report = assert_valid_motion_corpus(corpus_root)
    assert report["passed"] is True
    assert report["clip_count"] == 260
    assert report["total_frames"] == 18720
    tensor = load_clip_deltas(corpus_root, 2)
    assert tensor.shape == (72, 70, 2)
    assert tensor.dtype.name == "float32"
    assert tensor.flags.writeable is False
    assert tensor[0, 0].tolist() == pytest.approx([0.5, 0.0])


def test_binary_tamper_is_rejected(corpus_root: Path, tmp_path: Path) -> None:
    root = _copy(corpus_root, tmp_path)
    binary = bytearray((root / "motion_frames.u16le").read_bytes())
    binary[17] ^= 1
    (root / "motion_frames.u16le").write_bytes(binary)
    with pytest.raises(MotionCorpusValidationError, match="binary SHA-256"):
        assert_valid_motion_corpus(root)


def test_rehashed_collapsed_trajectory_is_rejected(corpus_root: Path, tmp_path: Path) -> None:
    root = _copy(corpus_root, tmp_path)
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    binary = bytearray((root / "motion_frames.u16le").read_bytes())
    first_length = document["clips"][0]["byte_length"]
    binary[:first_length] = struct.pack("<H", 32768) * (first_length // 2)
    (root / "motion_frames.u16le").write_bytes(binary)
    document["artifacts"]["motion_frames"]["sha256"] = _sha(binary)
    document["clips"][0]["trajectory_sha256"] = _sha(binary[:first_length])
    document["corpus_identity_sha256"] = _corpus_identity(document)
    _write_manifest(root, document)
    with pytest.raises(MotionCorpusValidationError, match="motion collapsed"):
        assert_valid_motion_corpus(root)


def test_rehashed_offset_gap_is_rejected(corpus_root: Path, tmp_path: Path) -> None:
    root = _copy(corpus_root, tmp_path)
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    document["clips"][4]["byte_offset"] += 4
    _write_manifest(root, document)
    with pytest.raises(MotionCorpusValidationError, match="clip layout"):
        assert_valid_motion_corpus(root)


def test_cell_identity_forgery_is_rejected(corpus_root: Path, tmp_path: Path) -> None:
    root = _copy(corpus_root, tmp_path)
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    document["chassis"][3]["cells"][0]["tissue"] = "armor"
    _write_manifest(root, document)
    with pytest.raises(MotionCorpusValidationError, match="cell identity"):
        assert_valid_motion_corpus(root)


def test_source_provenance_forgery_is_rejected(corpus_root: Path, tmp_path: Path) -> None:
    root = _copy(corpus_root, tmp_path)
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    document["source"]["files"][SOURCE_PATHS[0]] = "0" * 64
    _write_manifest(root, document)
    with pytest.raises(MotionCorpusValidationError, match="source hashes"):
        assert_valid_motion_corpus(root)


def test_extra_member_is_rejected(corpus_root: Path, tmp_path: Path) -> None:
    root = _copy(corpus_root, tmp_path)
    (root / "untracked.bin").write_bytes(b"forged")
    with pytest.raises(MotionCorpusValidationError, match="unexpected corpus members"):
        assert_valid_motion_corpus(root)


def test_duplicate_json_key_is_rejected(corpus_root: Path, tmp_path: Path) -> None:
    root = _copy(corpus_root, tmp_path)
    path = root / "manifest.json"
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace('{\n  "format"', '{\n  "format": "forged",\n  "format"', 1),
        encoding="utf-8",
    )
    with pytest.raises(MotionCorpusValidationError, match="duplicate JSON key"):
        assert_valid_motion_corpus(root)


def test_clip_loader_rejects_bad_id(corpus_root: Path) -> None:
    with pytest.raises(MotionCorpusValidationError, match="clip_id"):
        load_clip_deltas(corpus_root, 260)
