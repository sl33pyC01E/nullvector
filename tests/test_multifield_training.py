from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forge.morphology.corpus import build_morphology_corpus
from forge.multifield_data import (
    GuidePolicy,
    MorphologyCorpus,
    MorphologyCorpusDataset,
    augment_scaffold_guides,
    compute_class_weights,
    compute_legal_tuples,
    select_condition_bank,
    stratified_corpus_split,
)
from forge.multifield_diffusion import MultiFieldVocabulary
from forge.multifield_metrics import (
    MultiFieldMetricAccumulator,
    condition_preference_statistics,
    validation_selection_score,
)
from forge.train_multifield import (
    CHECKPOINT_FORMAT,
    MultiFieldTrainConfig,
    atomic_torch_save,
    run_training,
)


@pytest.fixture(scope="module")
def morphology_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("multifield") / "corpus.npz"
    return build_morphology_corpus(path, 320, 0xC0FFEE)


def test_corpus_dataset_split_weights_and_legal_tuples(morphology_corpus: Path) -> None:
    corpus = MorphologyCorpus.load(morphology_corpus)
    split_a = stratified_corpus_split(corpus, validation_fraction=0.10, seed=33)
    split_b = stratified_corpus_split(corpus, validation_fraction=0.10, seed=33)
    assert np.array_equal(split_a.training, split_b.training)
    assert np.array_equal(split_a.validation, split_b.validation)
    assert split_a.fingerprint == split_b.fingerprint
    assert not set(split_a.training).intersection(split_a.validation)
    assert set(split_a.training).union(split_a.validation) == set(range(corpus.count))

    dataset = MorphologyCorpusDataset(corpus, split_a.training)
    sample = dataset[0]
    assert sample["guide"].dtype == torch.float32
    assert sample["guide"].shape == (8, 48, 48)
    assert sample["part"].dtype == torch.int64
    assert sample["genes"].shape == (24,)
    assert not sample["guide"][[0, 1, 5]].any()
    debug = MorphologyCorpusDataset(
        corpus,
        split_a.training,
        guide_policy=GuidePolicy(name="full_debug", thicken_radius=0),
    )[0]
    assert debug["guide"][[0, 1, 5]].any()
    assert corpus.metadata()["corpus_source_sha256"] != "unknown"
    weights = compute_class_weights(corpus, split_a.training)
    assert {name: len(values) for name, values in weights.items()} == {
        "part": 17,
        "material": 10,
        "emission": 4,
    }
    assert all(torch.isfinite(values).all() for values in weights.values())
    legal = compute_legal_tuples(corpus, split_a.training)
    assert legal.ndim == 2 and legal.shape[1] == 3 and len(legal) > 3
    assert [int(legal[:, index].max()) for index in range(3)] == [16, 9, 3]
    bank = select_condition_bank(corpus, split_a.validation, 16, seed=404)
    assert np.array_equal(
        bank, select_condition_bank(corpus, split_a.validation, 16, seed=404)
    )
    assert len(np.unique(corpus.morphologies[bank])) == 5
    assert len(np.unique(corpus.roles[bank])) == 8


def test_scaffold_augmentation_is_explicit_rng_deterministic(
    morphology_corpus: Path,
) -> None:
    corpus = MorphologyCorpus.load(morphology_corpus)
    split = stratified_corpus_split(corpus)
    dataset = MorphologyCorpusDataset(corpus, split.training)
    guide = torch.stack([dataset[0]["guide"], dataset[1]["guide"]])
    policy = GuidePolicy(
        training_channel_dropout=0.4, training_jitter_pixels=2
    )
    first = augment_scaffold_guides(
        guide, policy, generator=torch.Generator().manual_seed(919)
    )
    second = augment_scaffold_guides(
        guide, policy, generator=torch.Generator().manual_seed(919)
    )
    assert torch.equal(first, second)
    assert not first[:, [0, 1, 5]].any()


def test_metric_accumulator_measures_exact_and_illegal_predictions() -> None:
    vocabulary = MultiFieldVocabulary(3, 3, 2)
    legal = np.asarray([[0, 0, 0], [1, 1, 0], [2, 2, 1]], dtype=np.uint8)
    target = (
        torch.tensor([[[0, 1], [2, 1]]]),
        torch.tensor([[[0, 1], [2, 1]]]),
        torch.tensor([[[0, 0], [1, 0]]]),
    )
    exact = MultiFieldMetricAccumulator(vocabulary, legal)
    exact.update(target, target)
    report = exact.report("validation")
    assert report["validation_silhouette_iou"] == 1.0
    assert report["validation_part_macro_iou"] == 1.0
    assert report["validation_joint_tuple_validity"] == 1.0

    illegal = MultiFieldMetricAccumulator(vocabulary, legal)
    prediction = (target[0], torch.zeros_like(target[1]), target[2])
    illegal.update(prediction, target)
    report = illegal.report("validation")
    assert report["validation_silhouette_iou"] == 1.0
    assert report["validation_joint_tuple_validity"] < 1.0

    exact.update_condition_proxy(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.5]))
    report = exact.report("validation")
    assert report["validation_condition_preference_rate"] == 0.5
    score = validation_selection_score(report)
    assert 0.0 < score <= 1.0
    preferred, margin = condition_preference_statistics(
        torch.tensor([1.0, 3.0]),
        torch.tensor([[1.5, 1.2], [2.5, 4.0]]),
    )
    assert preferred.tolist() == [True, False]
    assert torch.allclose(margin, torch.tensor([0.2, -0.5]))


def test_atomic_torch_save_leaves_no_temporary_file(tmp_path: Path) -> None:
    destination = tmp_path / "state.pt"
    atomic_torch_save(destination, {"format": "test", "tensor": torch.arange(3)}, planned_bytes=1024)
    assert torch.equal(torch.load(destination, weights_only=False)["tensor"], torch.arange(3))
    assert list(tmp_path.glob("*.tmp")) == []


def test_cpu_smoke_training_writes_resumable_checkpoint(
    morphology_corpus: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoints"
    config = MultiFieldTrainConfig(
        corpus_path=str(morphology_corpus),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        epochs=1,
        batch_size=2,
        learning_rate=2.0e-4,
        ema_decay=0.9,
        warmup_steps=0,
        width=32,
        diffusion_steps=1,
        validation_fraction=0.10,
        seed=717,
        device="cpu",
        precision="fp32",
        generation_eval_count=1,
        generation_eval_interval=1,
        max_train_batches=1,
        max_validation_batches=1,
        quiet=True,
    )
    summary = run_training(config)
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    assert latest.is_file() and best.is_file()
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    assert checkpoint["format"] == CHECKPOINT_FORMAT
    assert checkpoint["next_epoch"] == 1
    assert checkpoint["global_step"] == 1
    assert checkpoint["corpus"]["file_sha256"] == summary["corpus_sha256"]
    assert checkpoint["split"]["fingerprint"]
    assert checkpoint["legal_tuple_fingerprint"]
    assert checkpoint["rng_state"]["training_generator"].numel() > 0
    assert checkpoint["fixed_validation"]["generation_source_indices"]
    metrics = checkpoint["history"][0]
    expected = {
        "validation_silhouette_iou",
        "validation_part_macro_iou",
        "validation_material_foreground_accuracy",
        "validation_emission_foreground_accuracy",
        "validation_joint_tuple_validity",
        "validation_condition_preference_rate",
        "validation_role_preference_rate",
        "validation_morphology_subtype_preference_rate",
        "validation_genes_preference_rate",
        "generation_silhouette_iou",
        "generation_joint_tuple_validity",
    }
    assert expected.issubset(metrics)
    resumed = run_training(
        MultiFieldTrainConfig.from_dict(checkpoint["config"]),
        resume_checkpoint=checkpoint,
    )
    assert resumed["global_step"] == summary["global_step"]
    assert resumed["best_validation_selection_score"] == summary[
        "best_validation_selection_score"
    ]


def test_epoch_boundary_resume_exactly_matches_uninterrupted_training(
    morphology_corpus: Path, tmp_path: Path
) -> None:
    def make_config(name: str) -> MultiFieldTrainConfig:
        return MultiFieldTrainConfig(
            corpus_path=str(morphology_corpus),
            output_dir=str(tmp_path / name / "output"),
            checkpoint_dir=str(tmp_path / name / "checkpoints"),
            epochs=2,
            batch_size=2,
            warmup_steps=0,
            width=32,
            diffusion_steps=1,
            validation_fraction=0.10,
            seed=0x5151,
            device="cpu",
            precision="fp32",
            generation_eval_count=0,
            max_train_batches=1,
            max_validation_batches=1,
            quiet=True,
        )

    uninterrupted_config = make_config("uninterrupted")
    run_training(uninterrupted_config)
    uninterrupted = torch.load(
        Path(uninterrupted_config.checkpoint_dir) / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )

    split_config = make_config("split")
    first_summary = run_training(split_config, stop_after_epoch=1)
    assert first_summary["epochs_completed"] == 1
    first = torch.load(
        Path(split_config.checkpoint_dir) / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_config = MultiFieldTrainConfig.from_dict(first["config"])
    run_training(resumed_config, resume_checkpoint=first)
    resumed = torch.load(
        Path(split_config.checkpoint_dir) / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert resumed["global_step"] == uninterrupted["global_step"] == 2
    assert resumed["canonical_ema_hash"] == uninterrupted["canonical_ema_hash"]
    assert resumed["history"] == uninterrupted["history"]
    assert torch.equal(
        resumed["rng_state"]["training_generator"],
        uninterrupted["rng_state"]["training_generator"],
    )
