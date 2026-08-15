from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from forge.creature_stage_intervention_corpus.validation import (
    FLUID_SCALE,
    FRAMES,
    GAME_ROOT,
    HEAL_FRAME,
    INTERVENTION_FRAME,
    INTERVENTIONS,
    MAX_FLUIDS,
    MORPHOTYPES,
    ORGAN_GROUPS,
    POSITION_BIAS,
    SOURCE_PATHS,
    UNIT_SCALE,
    InterventionCorpusValidationError,
    load_intervention_clip,
    validate_intervention_corpus,
)


FAMILIES = ["humanoid", "animalian", "plantlike", "anomaly", "machine"]
FAMILY_ORGANS = [
    ["brain", "heart", "lung", "gut", "eye"],
    ["brain", "heart", "lung", "gut", "eye"],
    ["meristem", "vascular", "frond", "bulb", "photoreceptor"],
    ["phase_brain", "flux", "orbital", "transmuter", "singularity"],
    ["processor", "coolant_pump", "radiator", "battery", "optic"],
]
TISSUES = ["neural", "circulatory", "respiratory", "digestive", "sensor"]


def _sha(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _unit(value: float) -> int:
    return int(np.floor(np.clip(value, 0.0, 1.0) * UNIT_SCALE + 0.5))


def _cells(family_id: int) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    grids = [(x, y) for y in range(-2, 3) for x in range(-7, 7)]
    for index, grid in enumerate(grids):
        cells.append(
            {
                "grid": [grid[0], grid[1]],
                "tissue": TISSUES[index] if index < 5 else "skin",
                "organ": FAMILY_ORGANS[family_id][index] if index < 5 else "none",
                "appendage": -1 if index < 5 else index % 4,
                "side": 0 if index < 5 else (-1 if grid[0] < 0 else 1),
            }
        )
    return cells


def _cell_identity(cells: list[dict[str, object]]) -> str:
    material = "|".join(
        f"{cell['grid'][0]},{cell['grid'][1]},{cell['tissue']},{cell['organ']},"
        f"{cell['appendage']},{cell['side']}"
        for cell in cells
    )
    return _sha(material.encode())


def _clip_block(intervention: str, target_cell: int) -> np.ndarray:
    cell_count = 70
    frame_stride_words = 10 + cell_count * 4 + MAX_FLUIDS * 6
    block = np.zeros((FRAMES, frame_stride_words), dtype="<u2")
    block[:, :8] = UNIT_SCALE
    cells = block[:, 10 : 10 + cell_count * 4].reshape(FRAMES, cell_count, 4)
    cells[:, :, 0:2] = POSITION_BIAS
    cells[:, :, 2:4] = UNIT_SCALE
    if intervention != "control":
        fluid = block[:, 10 + cell_count * 4 :].reshape(FRAMES, MAX_FLUIDS, 6)
        block[INTERVENTION_FRAME:30, 9] = 1
        fluid[INTERVENTION_FRAME:30, 0, 0:4] = POSITION_BIAS
        fluid[INTERVENTION_FRAME:30, 0, 4] = int(round(0.8 * FLUID_SCALE))
        fluid[INTERVENTION_FRAME + 1 : 30, 0, 4] = int(round(1.25 * FLUID_SCALE))
        fluid[INTERVENTION_FRAME:30, 0, 5] = int(round(4.0 * FLUID_SCALE))
    if intervention in {"wound", "heal"}:
        cells[INTERVENTION_FRAME:, 10, 2] = _unit(0.55)
        if intervention == "heal":
            cells[HEAL_FRAME:, 10, 2] = UNIT_SCALE
    elif intervention == "cut":
        cells[INTERVENTION_FRAME:, 10, 2:4] = 0
        block[INTERVENTION_FRAME:, 0] = _unit(69.0 / 70.0)
    elif intervention.endswith("_ablation"):
        cells[INTERVENTION_FRAME:, target_cell, 2:4] = 0
        block[INTERVENTION_FRAME:, 0] = _unit(69.0 / 70.0)
        summary_index = {
            "neural_ablation": 1,
            "circulation_ablation": 2,
            "respiration_ablation": 3,
            "digestion_ablation": 4,
            "sensory_ablation": 5,
        }[intervention]
        block[INTERVENTION_FRAME:, summary_index] = 0
        if intervention == "neural_ablation":
            block[INTERVENTION_FRAME:, 8] = UNIT_SCALE
    return block


def _build_fixture(root: Path) -> Path:
    root.mkdir()
    binary_path = root / "intervention_frames.u16le"
    chassis_records: list[dict[str, object]] = []
    clip_records: list[dict[str, object]] = []
    identity_material: list[str] = []
    offset = 0
    cell_samples = 0
    binary_hasher = hashlib.sha256()
    with binary_path.open("wb") as binary:
        for chassis_id in range(20):
            family_id, morphotype_id = divmod(chassis_id, 4)
            cells = _cells(family_id)
            identity = _cell_identity(cells)
            chassis_records.append(
                {
                    "chassis_id": chassis_id,
                    "family": FAMILIES[family_id],
                    "family_id": family_id,
                    "morphotype": MORPHOTYPES[family_id][morphotype_id],
                    "morphotype_id": morphotype_id,
                    "seed": 0x710D0000 + family_id * 0x100 + morphotype_id,
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
                    "cell_count": len(cells),
                    "cell_identity_sha256": identity,
                    "cells": cells,
                }
            )
            identity_material.append(f"chassis:{chassis_id}:{identity}")
            for intervention_id, intervention in enumerate(INTERVENTIONS):
                target_cell = max(0, intervention_id - 4)
                block = _clip_block(intervention["name"], target_cell)
                payload = block.tobytes(order="C")
                digest = _sha(payload)
                binary.write(payload)
                binary_hasher.update(payload)
                max_fluid = 0 if intervention_id == 0 else 1
                hit_count = 0 if intervention_id == 0 else 1
                clip_id = len(clip_records)
                clip_records.append(
                    {
                        "clip_id": clip_id,
                        "chassis_id": chassis_id,
                        "family_id": family_id,
                        "morphotype_id": morphotype_id,
                        "intervention": intervention["name"],
                        "intervention_id": intervention_id,
                        "target": intervention["target"],
                        "event_frames": intervention["event_frames"],
                        "hit_count": hit_count,
                        "maximum_fluid_count": max_fluid,
                        "frames": FRAMES,
                        "cell_count": len(cells),
                        "frame_stride_bytes": block.shape[1] * 2,
                        "byte_offset": offset,
                        "byte_length": len(payload),
                        "trajectory_sha256": digest,
                    }
                )
                identity_material.append(f"clip:{clip_id}:{digest}")
                offset += len(payload)
                cell_samples += len(cells) * FRAMES

    source_hashes = {relative: _sha((GAME_ROOT / relative).read_bytes()) for relative in SOURCE_PATHS}
    source_combined = _sha(
        "|".join(f"{relative}:{source_hashes[relative]}" for relative in SOURCE_PATHS).encode()
    )
    manifest = {
        "format": "nullvector-creature-stage-intervention-corpus-v1",
        "passed": True,
        "fixed_hz": 30,
        "frames_per_clip": FRAMES,
        "intervention_frame": INTERVENTION_FRAME,
        "heal_frame": HEAL_FRAME,
        "family_count": 5,
        "chassis_count": 20,
        "intervention_count": 9,
        "clip_count": 180,
        "total_frames": 32400,
        "total_cell_samples": cell_samples,
        "maximum_fluid_slots": MAX_FLUIDS,
        "summary_fields": [
            "integrity", "neural", "circulation", "respiration", "digestion",
            "senses", "energy", "hydration", "dead", "fluid_count",
        ],
        "interventions": INTERVENTIONS,
        "organ_groups": ORGAN_GROUPS,
        "encodings": {
            "position_velocity": {"scale": 256, "bias": POSITION_BIAS},
            "unit": {"scale": UNIT_SCALE},
            "fluid_scalar": {"scale": 1024},
            "clipped_values": 0,
        },
        "contracts": {
            "morphology": "coordinate-conditioned-safe-scaffold-v1",
            "physiology": "cellular-organ-causal-scaffold-v1",
            "orientation": "vertical-locked-2.5d-v1",
            "binary": "clip-frame-summary-cell-fluid-u16le-v1",
            "fluid": "ground-plane-diffuse-puddle-v1",
        },
        "source": {"files": source_hashes, "combined_sha256": source_combined},
        "corpus_identity_sha256": _sha("|".join(identity_material).encode()),
        "artifacts": {
            "intervention_frames": {
                "path": "intervention_frames.u16le",
                "bytes": offset,
                "sha256": binary_hasher.hexdigest(),
            }
        },
        "chassis": chassis_records,
        "clips": clip_records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_fixture(tmp_path_factory.mktemp("intervention-corpus") / "corpus")


@contextmanager
def _restore_file(path: Path) -> Iterator[None]:
    original = path.read_bytes()
    try:
        yield
    finally:
        path.write_bytes(original)


def test_valid_corpus_and_immutable_loader(corpus: Path) -> None:
    report = validate_intervention_corpus(corpus)
    assert report["passed"] and report["clip_count"] == 180
    loaded = load_intervention_clip(corpus, 2)
    assert loaded["position_deltas"].shape == (180, 70, 2)
    assert loaded["fluid"].shape == (180, 160, 6)
    assert all(not array.flags.writeable for array in loaded.values())
    assert loaded["health"][INTERVENTION_FRAME, 10] == pytest.approx(0.55, abs=2e-5)
    assert loaded["health"][HEAL_FRAME, 10] == pytest.approx(1.0)


def test_binary_tamper_is_rejected(corpus: Path) -> None:
    path = corpus / "intervention_frames.u16le"
    with path.open("r+b") as handle:
        handle.seek(4096)
        original = handle.read(1)
        handle.seek(4096)
        handle.write(bytes([original[0] ^ 0x01]))
    try:
        with pytest.raises(InterventionCorpusValidationError, match="SHA-256"):
            validate_intervention_corpus(corpus)
    finally:
        with path.open("r+b") as handle:
            handle.seek(4096)
            handle.write(original)


def test_zero_hit_intervention_is_rejected(corpus: Path) -> None:
    manifest = corpus / "manifest.json"
    with _restore_file(manifest):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["clips"][1]["hit_count"] = 0
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(InterventionCorpusValidationError, match="hit nothing"):
            validate_intervention_corpus(corpus)


def test_offset_gap_is_rejected(corpus: Path) -> None:
    manifest = corpus / "manifest.json"
    with _restore_file(manifest):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["clips"][7]["byte_offset"] += 4
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(InterventionCorpusValidationError, match="clip contract"):
            validate_intervention_corpus(corpus)


def test_source_forgery_is_rejected(corpus: Path) -> None:
    manifest = corpus / "manifest.json"
    with _restore_file(manifest):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["source"]["files"][SOURCE_PATHS[0]] = "0" * 64
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(InterventionCorpusValidationError, match="source hashes"):
            validate_intervention_corpus(corpus)


def test_extra_member_is_rejected(corpus: Path) -> None:
    extra = corpus / "junk.bin"
    extra.write_bytes(b"junk")
    try:
        with pytest.raises(InterventionCorpusValidationError, match="unexpected corpus members"):
            validate_intervention_corpus(corpus)
    finally:
        extra.unlink()


def test_duplicate_json_key_is_rejected(corpus: Path) -> None:
    manifest = corpus / "manifest.json"
    with _restore_file(manifest):
        text = manifest.read_text(encoding="utf-8")
        manifest.write_text(text.replace('{"format":', '{"format":"duplicate","format":', 1), encoding="utf-8")
        with pytest.raises(InterventionCorpusValidationError, match="duplicate JSON key"):
            validate_intervention_corpus(corpus)


def test_loader_rejects_non_integer_clip_id(corpus: Path) -> None:
    with pytest.raises(InterventionCorpusValidationError, match="clip_id"):
        load_intervention_clip(corpus, True)
