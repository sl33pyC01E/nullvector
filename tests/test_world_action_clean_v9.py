from __future__ import annotations

import inspect

from forge.world_action_clean_v9 import corpus
from forge.world_action_clean_v9.contract import CODEC_SHA256, SOURCE_NAMES


def test_clean_corpus_contract_uses_overlay_free_sources_and_adapted_codec() -> None:
    source = inspect.getsource(corpus)
    assert SOURCE_NAMES == tuple(f"clean-world-{letter}" for letter in "abcdef")
    assert CODEC_SHA256 == "8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"
    assert "validate_trajectory(root)" in source
    assert "np.diff(raw[\"tick\"]) > 0" in source
    assert "AdaptedWorldFrameCodec.from_checkpoint" in source


def test_clean_corpus_preserves_cell_and_causal_authority_members() -> None:
    assert corpus.RAW_NAMES == ("frame", "state", "actor_state", "actor_field", "control", "action", "selected", "timeline_event", "timeline", "counterfactual", "tick")
