from __future__ import annotations

import numpy as np

from forge.playable_neural_runtime_v1 import PlayableNeuralRuntime
from forge.playable_neural_runtime_v1.release import build,validate


def test_promoted_playable_runtime_loads_and_renders_on_cpu() -> None:
    runtime = PlayableNeuralRuntime.from_release(device="cpu")
    assert runtime.component_count >= 13
    frame = np.zeros((256, 256, 3), np.uint8)
    frame[80:176, 104:152] = (30, 190, 220)
    latent = runtime.composite.encode(frame)
    decoded = runtime.composite.decode(latent)
    assert tuple(latent.shape) == (1, 48, 32, 32)
    assert tuple(decoded.shape) == (1, 3, 256, 256)
    assert bool(decoded.isfinite().all())


def test_release_manifest_is_hash_closed(tmp_path) -> None:
    result=build(tmp_path/"release")
    assert result["passed"] and result["components"]>=13
    assert validate(tmp_path/"release")==result
