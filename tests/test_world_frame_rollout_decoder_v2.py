from __future__ import annotations

import inspect

from forge.world_frame_rollout_decoder_v2 import contract, corpus, training


def test_rollout_decoder_binds_promoted_inputs():
    assert contract.NATURAL_CORPUS_SHA256 == "e96b10f80db3e824fdb768dc9e52ac8ff5e7f228cf3b87ba89d1df8d3047662f"
    assert contract.RECURRENT_SHA256 == "1516633d413aa19930dea53d0eb5a526d8528761e4120f4a0e9b70da42489b64"
    assert contract.PARENT_CODEC_SHA256 == "8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"


def test_corpus_uses_multiple_recurrent_horizons_and_immutable_shards():
    source = inspect.getsource(corpus.build_corpus)
    assert contract.HORIZONS == (1, 2, 4, 8, 16, 32)
    assert "np.savez_compressed" in source
    assert "os.replace(staging, output)" in source
    assert "candidate[chosen]" in source


def test_training_mixes_rollout_and_authoritative_domains():
    source = inspect.getsource(training.train)
    assert "authoritative_probability" in source
    assert '"domain": "authoritative" if authoritative else "rollout"' in source
    assert "rollout_test_improves_5pct" in source
    assert "authoritative_test_within_15pct" in source
