from __future__ import annotations

import inspect

from forge.world_frame_rollout_decoder_v3 import contract, training


def test_v3_binds_promoted_rollout_decoder_and_corpus():
    assert contract.PARENT_SHA256 == "68cf898b54948091b225156fa5c357072f81322c0001a4335365c1efb4ca01e2"
    assert contract.ROLLOUT_CORPUS_SHA256 == "97ec406ed32cb3bcb755e98b565af07b05129c1c511f4adc7ff06532788616cf"


def test_v3_foreground_mask_uses_color_distance_edges_and_dilation():
    source = inspect.getsource(training._foreground)
    assert "median" in source
    assert "distance > 0.025" in source
    assert "edge > 0.02" in source
    assert "max_pool2d" in source


def test_v3_requires_foreground_edge_and_authoritative_retention():
    source = inspect.getsource(training.train)
    assert "foreground_test_improves" in source
    assert "edge_test_improves" in source
    assert "authoritative_test_within_15pct" in source
