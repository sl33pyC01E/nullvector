from __future__ import annotations

from forge.world_action_contiguous_v8 import load, validate


def test_contiguous_corpus_replays_and_has_long_windows():
    result = validate()
    assert result["worlds"] == 6
    sequences, _ = load()
    assert all(len(sequence["latent"]) == 396 for sequence in sequences)
    assert all((sequence["tick"][1:] >= sequence["tick"][:-1]).all() for sequence in sequences)
