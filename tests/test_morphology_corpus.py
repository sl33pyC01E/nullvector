from __future__ import annotations

import numpy as np

from forge.morphology.corpus import (
    CORPUS_FORMAT,
    MINIMUM_CORPUS_COUNT,
    build_morphology_corpus,
    corpus_stratum,
    inspect_corpus,
    split_indices,
    validate_corpus,
)


def test_morphology_corpus_build_split_and_reuse(tmp_path) -> None:
    path = tmp_path / "corpus.npz"
    built = build_morphology_corpus(path, 480, 0xBEEF)
    assert built == path.resolve()
    first_bytes = path.read_bytes()
    assert build_morphology_corpus(path, 480, 0xBEEF) == path.resolve()
    assert path.read_bytes() == first_bytes
    with np.load(path, allow_pickle=False) as payload:
        assert str(payload["format"][0]) == CORPUS_FORMAT
        assert payload["guide"].shape == (480, 8, 48, 48)
        assert payload["guide"].dtype == np.float16
        assert payload["part_owner"].shape == (480, 48, 48)
        strata = list(
            zip(
                payload["morphologies"].tolist(),
                payload["subtypes"].tolist(),
                payload["roles"].tolist(),
                strict=True,
            )
        )
        assert len(set(strata)) == 160
        assert all(strata.count(stratum) == 3 for stratum in set(strata))
        train, validation = split_indices(
            payload["morphologies"], payload["subtypes"], payload["roles"]
        )
        assert not set(train).intersection(validation)
        assert set(train).union(validation) == set(range(480))
        assert len(train) > len(validation) > 0
    report = inspect_corpus(path)
    assert validate_corpus(path, replay_samples=160) == []
    assert report["valid"] is True
    assert report["samples"] == 480
    assert report["subtype_coverage"] == 20
    assert report["role_coverage"] == 8
    assert report["part_owner_coverage"] == list(range(17))
    assert report["guide_storage_dtype"] == "float16"
    assert report["stratum_count_min"] == report["stratum_count_max"] == 3


def test_corpus_strata_are_exact_and_validate_inputs(tmp_path) -> None:
    assert len({corpus_stratum(index) for index in range(160)}) == 160
    assert corpus_stratum(0) == corpus_stratum(160)
    with np.testing.assert_raises(ValueError):
        corpus_stratum(-1)
    with np.testing.assert_raises(ValueError):
        build_morphology_corpus(tmp_path / "too-small.npz", MINIMUM_CORPUS_COUNT - 1, 3)
    with np.testing.assert_raises(ValueError):
        split_indices(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
