from __future__ import annotations

import inspect

from forge.recurrent_world_student_v4 import contract, evaluation, training


def test_clean_student_binds_clean_corpus_parent_and_codec() -> None:
    assert contract.PARENT_SHA256 == "6b6123ca21a1115819ea71bb082f95b14983592b14aecbdf7cf07c0819472411"
    assert contract.CODEC_SHA256 == "8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"
    assert "world_action_clean_v9" in inspect.getsource(training)


def test_clean_student_requires_long_horizon_causal_improvement() -> None:
    source=inspect.getsource(evaluation)
    assert "(4,8,16,32)" in source
    assert "all_long_horizons_beat_persistence" in source
    assert "all_long_horizons_beat_parent" in source
    assert "gates[\"all_passed\"]" in source
