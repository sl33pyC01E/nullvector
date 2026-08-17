from __future__ import annotations

import inspect

from forge.world_action_natural_v10 import corpus
from forge.world_action_natural_v10.contract import SOURCE_NAMES


def test_natural_corpus_requires_full_episode_continuity() -> None:
    source=inspect.getsource(corpus)
    assert SOURCE_NAMES==tuple(f"natural-world-{letter}" for letter in "abcdef")
    assert "np.diff(raw[\"tick\"])>0" in source
    assert "episode_step" in corpus.RAW_NAMES
    assert "continuous_frames" in source


def test_natural_corpus_preserves_full_cellular_authority() -> None:
    assert corpus.RAW_NAMES==("frame","state","actor_state","actor_field","visibility","memory","control","action","selected","timeline_event","timeline","counterfactual","tick","episode_step")
