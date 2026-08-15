from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from forge.cellular_motion.compiler import _channels
from forge.cellular_organism.compiler import _load_arrays
from forge.neural_cell_motion.contract import FEATURE_CHANNELS, NeuralCellMotionConfig, ORGAN_CHANNELS
from forge.neural_cell_motion.dataset import _build_shard, _selection_plan, _static_features, load_corpus_manifest, validate_corpus
from forge.neural_cell_motion.model import NeuralCellMotionUNet, neural_motion_loss
from forge.neural_cell_motion.supervisor import ACCESS_VIOLATION_CODES, build_corpus_resilient, validate_corpus_resilient


ROOT = Path(__file__).resolve().parents[1]
ANATOMY = ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
MOTION = ROOT / "outputs/cellular_motion_v2/cellular_motion_manifest.json"


def _source():
    anatomy = json.loads(ANATOMY.read_text(encoding="utf-8")); motion = json.loads(MOTION.read_text(encoding="utf-8")); record = anatomy["offspring"][0]
    return record, _load_arrays(ANATOMY.parent / record["arrays"]["path"]), motion["programs"][record["family_id"]]


@pytest.fixture(scope="module")
def smoke_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("neural-motion") / "corpus"
    build_corpus_resilient(output, identities_per_family=1, workers=2, max_attempts=3, timeout_seconds=300)
    return output


def test_static_raster_preserves_every_cell_and_motor_channel() -> None:
    record, arrays, _ = _source(); features = _static_features(arrays, record)
    assert features.shape == (FEATURE_CHANNELS, 48, 48) and features.dtype == np.float32
    assert int(features[0].sum()) == len(arrays["position_xy"])
    assert np.all(features[49:60].sum(0)[features[0] > 0] == 1)
    assert set(_channels(record)) == set(ORGAN_CHANNELS)


def test_shard_has_all_motion_facing_frames_and_recurrent_predecessors() -> None:
    record, arrays, program = _source(); shard, _ = _build_shard(record, arrays, program)
    assert shard["targets"].shape == (944, 4, 48, 48)
    assert shard["indices"].shape == (944, 4) and shard["previous_index"].shape == (944,)
    assert np.isfinite(shard["targets"]).all() and np.max(np.abs(shard["targets"][:, :2])) <= 1
    # Each facing starts at a distinct nine-frame locomotion window and its
    # first recurrent predecessor is the non-duplicate loop endpoint.
    locomotion = np.flatnonzero(shard["indices"][:, 0] == 2)
    for facing in range(8):
        rows = locomotion[shard["indices"][locomotion, 1] == facing]
        assert len(rows) == 9 and shard["previous_index"][rows[0]] == rows[-2]


def test_direction_rotates_motion_delta_without_rotating_rest_chassis() -> None:
    record, arrays, program = _source(); shard, _ = _build_shard(record, arrays, program); locomotion = np.flatnonzero(shard["indices"][:, 0] == 2)
    north = locomotion[(shard["indices"][locomotion, 1] == 0) & (shard["indices"][locomotion, 2] == 2)][0]
    east = locomotion[(shard["indices"][locomotion, 1] == 2) & (shard["indices"][locomotion, 2] == 2)][0]
    north_dx, north_dy = shard["targets"][north, :2]; east_dx, east_dy = shard["targets"][east, :2]
    assert np.allclose(east_dx, -north_dy, atol=1e-3) and np.allclose(east_dy, north_dx, atol=1e-3)
    assert np.array_equal(shard["features"][0], _static_features(arrays, record)[0].astype(np.float16))


def test_production_model_is_substantial_and_shape_exact() -> None:
    model = NeuralCellMotionUNet(); assert 20_000_000 <= model.parameter_count <= 100_000_000
    config = NeuralCellMotionConfig(base_channels=24, channel_multipliers=(1, 2, 3), blocks_per_level=1, condition_dim=96, attention_heads=4, dropout=0)
    smoke = NeuralCellMotionUNet(config); static = torch.zeros(2, 60, 48, 48); static[:, 0, 10:38, 12:36] = 1; previous = torch.zeros(2, 4, 48, 48)
    output = smoke(static, previous, torch.tensor([0, 4]), torch.tensor([0, 12]), torch.tensor([0, 7]), torch.tensor([0.0, 1.0]))
    assert output.shape == previous.shape and torch.isfinite(output).all() and float(output[:, :, :5].abs().max()) == 0


def test_loss_is_differentiable_and_penalizes_temporal_and_spatial_collapse() -> None:
    static = torch.zeros(2, 60, 48, 48); static[:, 0, 8:40, 8:40] = 1; previous = torch.zeros(2, 4, 48, 48); target = torch.zeros_like(previous); target[:, 0, 8:40, 8:40] = .4; target[:, 2, 8:40, 8:40] = .7
    predicted = torch.zeros_like(previous, requires_grad=True); loss, pieces = neural_motion_loss(predicted, target, previous, static); loss.backward()
    assert float(loss) > 0 and float(pieces["displacement"]) > 0 and float(pieces["temporal"]) > 0 and predicted.grad is not None and float(predicted.grad.abs().sum()) > 0


def test_five_family_smoke_corpus_is_strict_and_exact_replayable(smoke_corpus: Path) -> None:
    manifest = load_corpus_manifest(smoke_corpus); assert [record["family_id"] for record in manifest["records"]] == list(range(5)) and all(record["split"] == "smoke" for record in manifest["records"])
    assert manifest["scope"]["selection_mode"] == "balanced_prefix_smoke" and manifest["scope"]["source_family_counts"] == [11, 10, 9, 8, 7]
    replay = validate_corpus_resilient(smoke_corpus, replay=True, workers=2, max_attempts=3, timeout_seconds=300); assert replay["passed"] is True and replay["replay"] is True and replay["sample_count"] == 4720


def test_production_selection_uses_all_real_identities_and_family_local_splits() -> None:
    anatomy = json.loads(ANATOMY.read_text(encoding="utf-8")); selected, totals, production = _selection_plan(anatomy["offspring"], None)
    assert production is True and len(selected) == 45 and totals == [11, 10, 9, 8, 7]
    assert len({record["sample_id"] for record, _ in selected}) == 45
    assert [max(ordinal for record, ordinal in selected if record["family_id"] == family_id) for family_id in range(5)] == [10, 9, 8, 7, 6]
    assert [ordinal for record, ordinal in selected if record["family_id"] == 4] == list(range(7))


def test_resilient_supervisor_has_bounded_worker_and_native_failure_policy(tmp_path: Path) -> None:
    assert {0xC0000005, -1073741819, 0xC0000409, -1073740791} <= ACCESS_VIOLATION_CODES
    with pytest.raises(ValueError, match="worker policy"):
        build_corpus_resilient(tmp_path / "too-many-workers", workers=3)
    with pytest.raises(ValueError, match="timeout"):
        build_corpus_resilient(tmp_path / "unbounded-timeout", timeout_seconds=29)


def test_tampered_shard_fails_closed(tmp_path: Path, smoke_corpus: Path) -> None:
    output = tmp_path / "corpus"; shutil.copytree(smoke_corpus, output); manifest = load_corpus_manifest(output); path = output / manifest["records"][0]["path"]
    payload = bytearray(path.read_bytes()); payload[-20] ^= 0x5A; path.write_bytes(payload)
    with pytest.raises(ValueError, match="artifact drifted"): validate_corpus(output)


def test_manifest_rejects_duplicate_keys_and_unsafe_paths(tmp_path: Path, smoke_corpus: Path) -> None:
    duplicate = tmp_path / "duplicate"; shutil.copytree(smoke_corpus, duplicate)
    manifest_path = duplicate / "neural_cell_motion_corpus.json"; encoded = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(encoded.replace("{\n", '{\n  "format": "shadow",\n', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"): validate_corpus(duplicate)

    traversal = tmp_path / "traversal"; shutil.copytree(smoke_corpus, traversal)
    manifest_path = traversal / "neural_cell_motion_corpus.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["records"][0]["path"] = "shards/../escape.npz"
    from forge.multifield_style_motion.hashing import canonical_json_bytes
    from forge.neural_cell_motion.dataset import sha256_bytes
    manifest["semantic_sha256"] = sha256_bytes(canonical_json_bytes({key: value for key, value in manifest.items() if key != "semantic_sha256"}))
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="schema validation|unsafe"): validate_corpus(traversal)


def test_npz_header_bounds_reject_rehashed_shape_forgery(tmp_path: Path, smoke_corpus: Path) -> None:
    output = tmp_path / "shape-forgery"; shutil.copytree(smoke_corpus, output)
    manifest_path = output / "neural_cell_motion_corpus.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); record = manifest["records"][0]; path = output / record["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["features"] = arrays["features"][:1]
    np.savez_compressed(path, **arrays)
    from forge.multifield_style_motion.hashing import canonical_json_bytes
    from forge.neural_cell_motion.dataset import array_sha256, sha256_bytes, sha256_file
    record["bytes"] = path.stat().st_size; record["sha256"] = sha256_file(path); record["features_sha256"] = array_sha256(arrays["features"])
    manifest["semantic_sha256"] = sha256_bytes(canonical_json_bytes({key: value for key, value in manifest.items() if key != "semantic_sha256"}))
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="NPY header|tensor bound"): validate_corpus(output)
