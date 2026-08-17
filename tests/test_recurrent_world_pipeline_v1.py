from __future__ import annotations

import inspect

from forge.recurrent_world_pipeline_v1 import contract, release, runtime


def test_pipeline_binds_selected_recurrent_and_decoder():
    assert contract.RECURRENT_SHA256 == "1516633d413aa19930dea53d0eb5a526d8528761e4120f4a0e9b70da42489b64"
    assert contract.DECODER_SHA256 == "03f3e147e1e4007aa01c063cf2cfc8717f169dc4974a7914e78d389a00d0d872"


def test_runtime_carries_latent_actor_perception_and_memory():
    source = inspect.getsource(runtime.RecurrentWorldPipeline.step)
    assert "gated_action" in source
    assert "self.recurrent.actor" in source
    assert "visibility" in source and "memory" in source
    assert "self.decoder.decode" in source


def test_release_benchmarks_full_decode_and_caps_memory():
    source = inspect.getsource(release.build_release)
    assert "frames_per_second" in source
    assert "peak_reserved_bytes" in source
    assert "runtime.step" in source
