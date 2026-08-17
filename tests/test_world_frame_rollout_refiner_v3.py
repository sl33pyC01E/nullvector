from __future__ import annotations

import inspect

from forge.world_frame_rollout_refiner_v3 import cache, contract, training


def test_refiner_binds_rollout_decoder_and_corpus():
    assert contract.DECODER_SHA256 == "68cf898b54948091b225156fa5c357072f81322c0001a4335365c1efb4ca01e2"
    assert contract.ROLLOUT_CORPUS_SHA256 == "97ec406ed32cb3bcb755e98b565af07b05129c1c511f4adc7ff06532788616cf"


def test_cache_is_atomic_and_hash_validated():
    source = inspect.getsource(cache.build_cache)
    assert "np.savez_compressed" in source
    assert "os.replace(staging, output)" in source
    assert "decoder.decode" in source
    assert contract.cache_source_sha256() != contract.source_sha256()


def test_refiner_is_local_residual_and_gated_for_identity():
    source = inspect.getsource(training.train)
    assert "PixelCellRefiner" in source
    assert "identity_probability" in source
    assert "rollout_edge_improves" in source
    assert "identity_drift_below_0_005" in source
