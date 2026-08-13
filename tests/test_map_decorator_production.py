from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forge.map_decorator_ml.dataset import split_for_identity
from forge.map_decorator_production.contract import (
    OBJECTIVE_BUCKETS,
    PRODUCTION_CONTRACT_SHA256,
    SENTINEL_DIMENSIONS,
    SIZE_PROFILES,
    CorpusConfig,
    production_contract_manifest,
)
from forge.map_decorator_production.corpus import (
    ShardSpec,
    _expected_raw_bytes,
    build_shard,
    estimate_corpus,
    load_shard_array,
    load_shard_arrays,
    production_specs,
    validate_shard,
)
from forge.map_decorator_production.teacher import (
    build_production_sample,
    semantic_teacher_targets,
)
from forge.map_decorator_production.training import ProductionTrainingConfig
from forge.map_decorator_ml.dataset import _teacher_targets
from forge.map_decorator.hashing import json_sha256
from forge.maps import MapConfig, THEMES, generate_map


def _tiny_spec() -> ShardSpec:
    return ShardSpec(
        shard_id="production-test-tiny",
        kind="main",
        ordinal=301,
        theme="anomaly",
        width=32,
        height=32,
        objective_bucket="few",
        objective_count=3,
        spawn_count=4,
        required_splits={"train": 1, "validation": 1},
        global_seed=0xA11CE,
        max_candidates=256,
        replay_every_sample=True,
        npz_path="shards/production-test-tiny/fields.npz",
        sidecar_path="shards/production-test-tiny/shard.json",
        validation_path="validation/production-test-tiny.json",
        estimated_raw_bytes=_expected_raw_bytes(2, 32, 32, 3, 4),
    )


def test_production_contract_exactly_describes_balanced_3072_plus_24() -> None:
    config = CorpusConfig()
    assert config.main_map_count == 3072
    assert config.sentinel_count == 24
    assert len(SIZE_PROFILES) == 8
    assert len(OBJECTIVE_BUCKETS) == 4
    assert SENTINEL_DIMENSIONS == (32, 72, 128, 256)
    assert PRODUCTION_CONTRACT_SHA256 == json_sha256(production_contract_manifest())
    specs = production_specs(config)
    assert len(specs) == 216
    assert len([spec for spec in specs if spec.kind == "main"]) == 192
    assert len([spec for spec in specs if spec.kind == "sentinel"]) == 24
    assert sum(spec.sample_count for spec in specs) == 3096
    assert all(spec.required_splits == {"train": 13, "validation": 3} for spec in specs[:192])
    assert all(spec.required_splits == {"test": 1} for spec in specs[192:])
    strata = {
        (spec.theme, spec.width, spec.height, spec.objective_bucket)
        for spec in specs
        if spec.kind == "main"
    }
    assert len(strata) == len(THEMES) * 8 * 4


def test_estimate_preserves_disk_floor_and_includes_headroom(tmp_path: Path) -> None:
    report = estimate_corpus(output=tmp_path / "planned")
    assert report["safe_to_build"]
    assert report["main_map_count"] == 3072 and report["sentinel_count"] == 24
    assert report["planned_bytes_with_50pct_headroom"] > report["raw_bytes"]
    assert report["disk"]["floor_gb"] == 100.0


def test_semantic_only_teacher_is_exactly_equivalent_to_renderer_teacher() -> None:
    for index, theme in enumerate(THEMES):
        data = generate_map(
            0x7EAC_0000 + index,
            theme,
            MapConfig(width=33 + index, height=34 + index, objective_count=3, spawn_count=5),
        )
        expected, expected_legal, _ = _teacher_targets(data)
        observed, observed_legal, _ = semantic_teacher_targets(data)
        for name in expected:
            np.testing.assert_array_equal(observed[name], expected[name])
        assert observed_legal.masks_sha256 == expected_legal.masks_sha256


def test_production_sample_replays_semantics_features_targets_and_legality() -> None:
    data = generate_map(
        0x5A1A5,
        "garden",
        MapConfig(width=40, height=36, objective_count=6, spawn_count=7),
    )
    replay = generate_map(data.seed, data.theme, data.config)
    sample = build_production_sample(data, feature_seed=0xFEA7, replay_data=replay)
    assert sample.split == split_for_identity(sample.full_map_identity_sha256)
    assert sample.features.shape == (53, 36, 40)
    assert sample.features.dtype == np.float32
    assert sample.replay_sha256 and len(sample.replay_sha256) == 64
    for name in sample.targets:
        yy, xx = np.indices(sample.data.shape)
        assert sample.legal_masks[name][sample.targets[name], yy, xx].all()


def test_atomic_homogeneous_shard_reloads_and_replays_every_sample(tmp_path: Path) -> None:
    spec = _tiny_spec()
    first = build_shard(spec, tmp_path)
    assert first["passed"] and first["sample_count"] == 2
    shard_dir = tmp_path / spec.npz_path
    assert shard_dir.parent.is_dir()
    assert sorted(path.name for path in shard_dir.parent.iterdir()) == ["fields.npz", "shard.json"]
    sidecar, arrays = load_shard_arrays(tmp_path, spec)
    assert sidecar["split_counts"] == spec.required_splits
    assert arrays["features"].shape == (2, 53, 32, 32)
    assert arrays["objectives"].shape == (2, 3, 2)
    streamed = load_shard_array(tmp_path, spec, "features")
    assert isinstance(streamed, np.memmap)
    np.testing.assert_array_equal(streamed, arrays["features"])
    del streamed
    report = validate_shard(spec, tmp_path)
    assert report["passed"] and report["exact_semantic_feature_target_legality_replay"]
    recovered = build_shard(spec, tmp_path)
    assert recovered["recovered_after_atomic_publish"] is True
    assert recovered["npz_sha256"] == first["npz_sha256"]


def test_shard_sidecar_tamper_fails_closed(tmp_path: Path) -> None:
    spec = _tiny_spec()
    build_shard(spec, tmp_path)
    sidecar_path = tmp_path / spec.sidecar_path
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["samples"][0]["split"] = "test"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_shard(spec, tmp_path)
    except ValueError as error:
        assert "record failed exact replay" in str(error)
    else:
        raise AssertionError("Tampered shard sidecar did not fail closed.")


def test_production_training_contract_is_exact_two_epoch_cuda_bf16_segments() -> None:
    config = ProductionTrainingConfig()
    assert config.precision == "bf16"
    assert config.segment_epochs == 2
    assert config.epochs % config.segment_epochs == 0
    assert config.num_workers == 0
    try:
        ProductionTrainingConfig(epochs=11)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Odd epoch schedule did not fail closed.")
    try:
        ProductionTrainingConfig(segment_epochs=1)
    except ValueError as error:
        assert "two epochs" in str(error)
    else:
        raise AssertionError("Non-two-epoch segment did not fail closed.")
