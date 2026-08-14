from __future__ import annotations

import json
from pathlib import Path

from forge.sprite_latent_showcase import _select_indices, validate_showcase
from forge.sprite_latent.corpus import SemanticFieldCorpus, stratified_split


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/morphology_32768_4d4f5250.npz"
SHOWCASE = ROOT / "outputs/sprite_latent_production_showcase_v1/showcase_manifest.json"


def test_balanced_selection_is_deterministic_and_validation_only() -> None:
    corpus = SemanticFieldCorpus.load(CORPUS)
    split = stratified_split(corpus)
    first = _select_indices(corpus, split.validation); second = _select_indices(corpus, split.validation)
    assert first.tolist() == second.tolist() and len(first) == 40 and len(set(map(int, first))) == 40
    assert set(map(int, first)) <= set(map(int, split.validation))
    assert [int((corpus.morphologies[first] == family).sum()) for family in range(5)] == [8] * 5


def test_published_showcase_exactly_replays_accepted_ema() -> None:
    validation = validate_showcase(SHOWCASE)
    manifest = json.loads(SHOWCASE.read_text(encoding="utf-8"))
    assert validation["passed"] is True and validation["sample_count"] == 40
    assert all(manifest["gates"].values())
    assert manifest["source"]["checkpoint_epoch"] == 24
    assert manifest["aggregate"]["visible_silhouette_iou"] > 0.95
