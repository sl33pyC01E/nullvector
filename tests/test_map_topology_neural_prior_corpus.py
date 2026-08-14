from __future__ import annotations

from pathlib import Path

import pytest

from forge.map_topology_neural_prior_corpus.contract import (
    EXPECTED_SAMPLES,
    EXPECTED_SHARDS,
    PRIOR_SOURCE_SHA256,
    authority,
    corpus_source_sha256,
)
from forge.map_topology_neural_prior_corpus.shard import build_shard, validate_shard
from forge.map_topology_neural_prior_corpus.supervisor import _registry


CORPUS = Path("outputs/map_decorator_corpus_v1")


def test_latent_corpus_contract_and_registry_are_exact() -> None:
    dataset, shard_ids = _registry(CORPUS)
    assert len(shard_ids) == EXPECTED_SHARDS
    assert len(dataset.refs) == EXPECTED_SAMPLES
    assert {name: len(dataset.refs_by_split[name]) for name in ("train", "validation", "test")} == {
        "train": 2496,
        "validation": 576,
        "test": 24,
    }
    assert authority()["prior_source_sha256"] == PRIOR_SOURCE_SHA256
    assert len(corpus_source_sha256()) == 64


def test_latent_shard_is_byte_deterministic_and_exactly_replays_source(tmp_path: Path) -> None:
    _, shard_ids = _registry(CORPUS)
    shard_id = shard_ids[0]
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_shard(CORPUS, first, shard_id)
    second_manifest = build_shard(CORPUS, second, shard_id)
    assert first_manifest == second_manifest
    assert (first / "latents.npz").read_bytes() == (second / "latents.npz").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert validate_shard(CORPUS, first, replay_source=True) == first_manifest


def test_latent_shard_tamper_fails_closed(tmp_path: Path) -> None:
    _, shard_ids = _registry(CORPUS)
    output = tmp_path / "shard"
    build_shard(CORPUS, output, shard_ids[1])
    arrays = output / "latents.npz"
    payload = bytearray(arrays.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    arrays.write_bytes(payload)
    with pytest.raises((ValueError, OSError), match="identity|CRC|Bad CRC|canonical|header|magic"):
        validate_shard(CORPUS, output, replay_source=False)

