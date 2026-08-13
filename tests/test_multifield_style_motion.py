from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from forge.morphology import MorphologyGenome, generate_motion_clip, render_specimen
from forge.multifield_style_motion.compiler import build_contract
from forge.multifield_style_motion.family import compile_family_payload
from forge.multifield_style_motion.hashing import (
    canonical_json_bytes,
    deterministic_npz_bytes,
    sha256_bytes,
)
from forge.multifield_style_motion.io import verify_artifact, write_exact
from forge.multifield_style_motion.rendering import build_condition, render_motion_frame
from forge.multifield_style_motion.schema import (
    BANK_SCHEMA,
    FAMILY_SCHEMA,
    validate_schema,
)
from forge.multifield_style_motion.showcase import encode_showcase_mp4, ffmpeg_provenance
from forge.multifield_style_motion.source import (
    EXPECTED_CLIP_COUNT,
    EXPECTED_FRAME_COUNT,
    load_motion_bank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_INDEX = PROJECT_ROOT / "game" / "generated" / "v2" / "asset_index.json"


@pytest.fixture(scope="module")
def motion_bank():
    return load_motion_bank(ASSET_INDEX)


@pytest.fixture(scope="module")
def humanoid_payload(motion_bank):
    return compile_family_payload(motion_bank, "humanoid", build_contract(motion_bank))


def test_full_motion_source_contract_is_bound(motion_bank) -> None:
    assert motion_bank.asset_index_sha256 == hashlib.sha256(ASSET_INDEX.read_bytes()).hexdigest()
    assert len(motion_bank.clips_by_family) == 5
    assert sum(len(clips) for clips in motion_bank.clips_by_family.values()) == EXPECTED_CLIP_COUNT
    assert (
        sum(int(clip["frame_count"]) for clips in motion_bank.clips_by_family.values() for clip in clips)
        == EXPECTED_FRAME_COUNT
    )
    assert all(len(clips) == 104 for clips in motion_bank.clips_by_family.values())


def test_identity_palette_stays_exact_while_frame_authority_moves(motion_bank) -> None:
    source = motion_bank.sources["animalian"]
    specimen = render_specimen(MorphologyGenome.from_dict(dict(source["genome"])))
    condition = build_condition(source, 1)
    identity = "b" * 64
    palette_hash = None
    categorical_hashes: set[str] = set()
    authority_hashes: set[str] = set()
    for motion, facing in (("idle_wiggle", "north"), ("locomote", "east"), ("joy", "southwest")):
        clip = generate_motion_clip(specimen, motion, facing=facing)
        for frame in clip.frames:
            audit = render_motion_frame(
                frame,
                specimen,
                condition,
                identity,
                expected_palette_sha256=palette_hash,
            )
            palette_hash = palette_hash or audit.palette_sha256
            assert audit.palette_sha256 == palette_hash
            assert all(audit.gates.values())
            categorical_hashes.add(audit.categorical_sha256)
            authority_hashes.add(audit.authority_sha256)
    assert len(categorical_hashes) > 3
    assert len(authority_hashes) > 3


def test_family_payload_has_exact_matrix_loop_events_and_schema(
    motion_bank,
    humanoid_payload,
) -> None:
    manifest = humanoid_payload.family_manifest
    validate_schema(manifest, FAMILY_SCHEMA)
    assert manifest["clip_count"] == 104
    assert manifest["frame_count"] == 944
    assert all(manifest["gates"].values())
    assert len({manifest["identity"]["palette_sha256"]}) == 1
    expected = motion_bank.clips_by_family["humanoid"]
    for styled, source in zip(manifest["clips"], expected, strict=True):
        assert styled["source_clip_sha256"] == source["clip_sha256"]
        assert styled["events"] == source["events"]
        assert all(styled["gates"].values())
    with np.load(
        __import__("io").BytesIO(
            humanoid_payload.file_payloads["families/humanoid/frame_index.npz"]
        ),
        allow_pickle=False,
    ) as archive:
        assert archive["categorical_sha256"].shape == (944,)
        assert archive["authority_sha256"].shape == (944,)
        assert archive["presentation_sha256"].shape == (944, 7)
        offsets = archive["clip_offsets"].tolist()
        for ordinal, clip in enumerate(manifest["clips"]):
            if clip["loop"]:
                start, stop = offsets[ordinal], offsets[ordinal + 1]
                assert archive["categorical_sha256"][start] == archive["categorical_sha256"][stop - 1]
                assert archive["authority_sha256"][start] == archive["authority_sha256"][stop - 1]
                assert np.array_equal(
                    archive["presentation_sha256"][start],
                    archive["presentation_sha256"][stop - 1],
                )


def test_deterministic_npz_is_byte_exact_and_has_fixed_metadata() -> None:
    arrays = {
        "z": np.arange(12, dtype=np.uint8).reshape(3, 4),
        "a": np.asarray(["one", "two"]),
    }
    first = deterministic_npz_bytes(arrays)
    second = deterministic_npz_bytes(arrays)
    assert first == second
    with zipfile.ZipFile(__import__("io").BytesIO(first), "r") as archive:
        assert [entry.filename for entry in archive.infolist()] == ["a.npy", "z.npy"]
        assert all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in archive.infolist())


def test_strict_schema_rejects_authority_and_hash_tampering(humanoid_payload) -> None:
    authority_tamper = copy.deepcopy(humanoid_payload.family_manifest)
    authority_tamper["authority"]["presentation_is_derived_only"] = False
    with pytest.raises(ValueError, match="presentation_is_derived_only"):
        validate_schema(authority_tamper, FAMILY_SCHEMA)
    hash_tamper = copy.deepcopy(humanoid_payload.family_manifest)
    hash_tamper["artifacts"]["layers"]["base"]["sha256"] = "0" * 63
    with pytest.raises(ValueError, match="sha256"):
        validate_schema(hash_tamper, FAMILY_SCHEMA)


def test_hash_bound_artifact_rejects_byte_tampering(tmp_path: Path) -> None:
    payload = b"categorical-authority"
    path = tmp_path / "layer.bin"
    write_exact(path, payload)
    record = {"path": "layer.bin", "bytes": len(payload), "sha256": sha256_bytes(payload)}
    assert verify_artifact(tmp_path, record) == path.resolve()
    path.write_bytes(payload[:-1] + b"X")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(tmp_path, record)


def test_ffmpeg_showcase_encoding_is_byte_exact() -> None:
    ffmpeg = ffmpeg_provenance()
    frames = []
    for ordinal in range(4):
        frame = np.zeros((64, 64, 4), dtype=np.uint8)
        frame[..., 3] = 255
        frame[8 + ordinal : 24 + ordinal, 12:28, :3] = (61, 213, 244)
        frames.append(frame)
    first, encoding = encode_showcase_mp4(frames, ffmpeg)
    second, _ = encode_showcase_mp4(frames, ffmpeg)
    assert first == second
    assert b"ftyp" in first[:32]
    assert encoding["codec"] == "libx264"
    assert encoding["threads"] == 1
