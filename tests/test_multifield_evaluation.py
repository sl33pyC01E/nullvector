from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator

from forge.morphology.constants import (
    EMISSION_LEVEL_NAMES,
    FAMILIES,
    GUIDE_CHANNEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SUBTYPE_NAMES,
)
from forge.morphology.corpus import build_morphology_corpus
from forge.multifield_data import (
    GuidePolicy,
    MorphologyCorpus,
    MorphologyCorpusDataset,
    compute_class_weights,
    compute_legal_tuples,
    legal_tuple_fingerprint,
    select_condition_bank,
    stratified_corpus_split,
)
from forge.multifield_diffusion import MultiFieldSpriteDiffusion, MultiFieldVocabulary
from forge.multifield_eval.benchmark import BENCHMARK_FORMAT, benchmark_checkpoint
from forge.multifield_eval.calibration import calibrate_morphology_corpus
from forge.multifield_eval.checkpoint import (
    CheckpointNotReady,
    CheckpointProvenanceError,
    load_multifield_checkpoint,
    snapshot_published_checkpoint,
)
from forge.multifield_eval.cli import main as evaluation_main
from forge.multifield_eval.conditions import build_condition_grid, validate_grid_coverage
from forge.multifield_eval.pipeline import (
    GENERATION_BANK_FORMAT,
    replay_generation_bank,
    write_generation_bank,
)
from forge.multifield_eval.validation import diversity_report
from forge.multifield_eval.validation import (
    ConditionTemplateBank,
    calibrate_reference_fields,
    validate_generated_fields,
)
from forge.provenance import canonical_state_dict_hash
from forge.train_multifield import CHECKPOINT_FORMAT, training_source_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _validate_schema(payload: dict, filename: str) -> None:
    schema = json.loads(
        (PROJECT_ROOT / "shared" / "schema" / filename).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


@pytest.fixture(scope="module")
def evaluation_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("multifield-evaluation") / "corpus.npz"
    return build_morphology_corpus(destination, 320, 0xE7A1)


@pytest.fixture(scope="module")
def evaluation_checkpoint(
    evaluation_corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    torch.manual_seed(0xE7A1)
    corpus = MorphologyCorpus.load(evaluation_corpus)
    split = stratified_corpus_split(corpus, validation_fraction=0.10, seed=0x5151)
    legal = compute_legal_tuples(corpus, split.training)
    policy = GuidePolicy()
    model = MultiFieldSpriteDiffusion(
        vocabulary=MultiFieldVocabulary(
            len(PART_OWNER_NAMES), len(MATERIAL_NAMES), len(EMISSION_LEVEL_NAMES)
        ),
        morphology_count=len(FAMILIES),
        subtype_count=len(SUBTYPE_NAMES),
        role_count=len(ROLE_NAMES),
        gene_dim=24,
        guide_channels=len(GUIDE_CHANNEL_NAMES),
        steps=1,
        width=32,
        image_size=48,
    )
    ema = model.state_dict()
    fixed = select_condition_bank(corpus, split.validation, 16, seed=0xF17ED)
    config = {
        "corpus_path": str(evaluation_corpus),
        "epochs": 1,
        "width": 32,
        "diffusion_steps": 1,
        "validation_fraction": 0.10,
        "split_seed": 0x5151,
        "seed": 0xE7A1,
        "device": "cpu",
        "precision": "fp32",
        "guide_policy": policy.name,
        "guide_thicken_radius": policy.thicken_radius,
        "guide_channel_dropout": policy.training_channel_dropout,
        "guide_jitter_pixels": policy.training_jitter_pixels,
        "field_part_weight": 1.0,
        "field_material_weight": 0.65,
        "field_emission_weight": 0.45,
        "deterministic": True,
    }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "ema_model": ema,
        "architecture": model.architecture_config(),
        "config": config,
        "corpus": corpus.metadata(),
        "split": split.metadata(),
        "legal_tuples": torch.from_numpy(legal.copy()),
        "legal_tuple_fingerprint": legal_tuple_fingerprint(legal),
        "class_weights": compute_class_weights(corpus, split.training),
        "guide_policy": policy.metadata(),
        "canonical_ema_hash": canonical_state_dict_hash(ema),
        "training_source_hash": training_source_hash(),
        "fixed_validation": {
            "full_mask_seed": 0xF011,
            "generation_seed": 0x6E6,
            "generation_source_indices": list(map(int, fixed)),
        },
        "next_epoch": 1,
        "global_step": 1,
    }
    destination = tmp_path_factory.mktemp("multifield-checkpoint") / "latest.pt"
    torch.save(payload, destination)
    return destination


def test_unpublished_and_corrupt_checkpoints_are_gracefully_incomplete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "latest.pt"
    temporary = tmp_path / ".latest.pt.55.tmp"
    temporary.write_bytes(b"still being written")
    with pytest.raises(CheckpointNotReady, match="unpublished temporary"):
        load_multifield_checkpoint(missing, device="cpu", precision="fp32")
    assert evaluation_main(
        [
            "status",
            "--checkpoint",
            str(missing),
            "--device",
            "cpu",
            "--precision",
            "fp32",
        ]
    ) == 3
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "checkpoint_incomplete"

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointNotReady, match="stable complete checkpoint"):
        load_multifield_checkpoint(corrupt, device="cpu", precision="fp32")


def test_strict_checkpoint_and_condition_grid_contract(
    evaluation_checkpoint: Path, tmp_path: Path,
) -> None:
    bundle = load_multifield_checkpoint(
        evaluation_checkpoint, device="cpu", precision="fp32"
    )
    assert bundle.payload["canonical_ema_hash"] == bundle.provenance()[
        "canonical_ema_hash"
    ]
    assert bundle.training_complete
    snapshot = tmp_path / "snapshot.pt"
    snapshot_report = snapshot_published_checkpoint(evaluation_checkpoint, snapshot)
    assert snapshot_report["status"] == "snapshotted"
    assert load_multifield_checkpoint(
        snapshot, device="cpu", precision="fp32"
    ).payload["canonical_ema_hash"] == bundle.payload["canonical_ema_hash"]
    with pytest.raises(FileExistsError, match="immutable"):
        snapshot_published_checkpoint(evaluation_checkpoint, snapshot)
    stratified = build_condition_grid(bundle, mode="stratified")
    coverage = validate_grid_coverage(stratified, "stratified")
    assert len(stratified) == 40
    assert coverage["covers_all_families"]
    assert coverage["covers_all_subtypes"]
    assert coverage["covers_all_roles"]
    assert coverage["covers_all_family_role_pairs"]
    exhaustive = build_condition_grid(bundle, mode="exhaustive")
    assert len(exhaustive) == 160
    assert [record.sample_seed for record in stratified] == [
        record.sample_seed
        for record in build_condition_grid(bundle, mode="stratified")
    ]
    repeated = build_condition_grid(
        bundle, mode="stratified", samples_per_condition=2, limit=2
    )
    assert repeated[0].sample_seed != repeated[1].sample_seed
    tiny_fields = []
    for index in range(4):
        part = np.zeros((48, 48), dtype=np.uint8)
        part[20:28, 20 + index : 28 + index] = 1
        tiny_fields.append((part, part.copy(), np.zeros_like(part)))
    sampled_diversity = diversity_report(
        tiny_fields,
        stratified[:4],
        max_global_pairs=2,
    )
    assert sampled_diversity["total_possible_pairs"] == 6
    assert sampled_diversity["evaluated_global_pairs"] == 2
    assert sampled_diversity["global_pair_policy"].startswith("deterministic")


def test_provenance_rejects_source_mismatch(
    evaluation_checkpoint: Path, tmp_path: Path
) -> None:
    payload = torch.load(evaluation_checkpoint, map_location="cpu", weights_only=False)
    payload["training_source_hash"] = "0" * 64
    bad = tmp_path / "bad-source.pt"
    torch.save(payload, bad)
    with pytest.raises(CheckpointProvenanceError, match="source hash"):
        load_multifield_checkpoint(bad, device="cpu", precision="fp32")


def test_reference_calibration_is_fully_hard_valid_and_invalid_fields_fail(
    evaluation_checkpoint: Path,
) -> None:
    bundle = load_multifield_checkpoint(
        evaluation_checkpoint, device="cpu", precision="fp32"
    )
    templates = ConditionTemplateBank.build(bundle)
    calibration = calibrate_reference_fields(bundle, templates=templates)
    assert calibration["hard_valid"] == calibration["samples"]
    assert calibration["hard_valid_rate"] == 1.0
    assert calibration["hard_gate_failures"] == {}
    assert calibration["visible_component_counts"] == {
        "1": calibration["samples"]
    }
    assert calibration["owner_presence"]["core"]["rate"] == 1.0
    assert calibration["owner_presence"]["core"]["hard_required"]
    assert not calibration["owner_presence"]["body"]["hard_required"]
    assert not calibration["owner_presence"]["head"]["hard_required"]
    # Exact subtype prediction is intentionally diagnostic: the valid
    # reference set is not rejected even when this lossy classifier misses.
    assert (
        calibration["diagnostic_condition_adherence"]["exact_match"]["subtype"][
            "rate"
        ]
        < 1.0
    )

    source = next(
        int(index)
        for index in bundle.validation_indices
        if bool(
            (
                bundle.corpus.part_owner[int(index)]
                == PART_OWNER_NAMES.index("body")
            ).any()
        )
    )
    sample = MorphologyCorpusDataset(
        bundle.corpus, [source], guide_policy=bundle.guide_policy
    )[0]
    target = (
        bundle.corpus.part_owner[source],
        bundle.corpus.material[source],
        bundle.corpus.emission_level[source],
    )
    record = build_condition_grid(bundle, mode="stratified", limit=1)[0]
    record = type(record)(
        ordinal=0,
        grid_mode="fixed",
        source_index=source,
        variation=0,
        sample_seed=0,
        morphology=int(bundle.corpus.morphologies[source]),
        subtype=int(bundle.corpus.subtypes[source]),
        role=int(bundle.corpus.roles[source]),
    )

    def validate(fields: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
        return validate_generated_fields(
            *fields,
            guide=sample["guide"].numpy(),
            target=target,
            record=record,
            legal_tuples=bundle.legal_tuples,
            templates=templates,
        )

    empty = tuple(np.zeros((48, 48), dtype=np.uint8) for _ in range(3))
    assert not validate(empty)["hard_valid"]
    assert not validate(empty)["accepted"]

    disconnected = tuple(np.array(values, copy=True) for values in target)
    disconnected[0][1, 1] = PART_OWNER_NAMES.index("core")
    disconnected[1][1, 1] = MATERIAL_NAMES.index("organ")
    disconnected[2][1, 1] = EMISSION_LEVEL_NAMES.index("radiant")
    disconnected_report = validate(disconnected)
    assert not disconnected_report["hard_valid"]
    assert disconnected_report["topology"]["visible_component_count"] > 1
    assert not disconnected_report["margins"]["safe"]

    illegal = tuple(np.array(values, copy=True) for values in target)
    foreground_yx = tuple(np.argwhere(illegal[0] != 0)[0])
    illegal[1][foreground_yx] = MATERIAL_NAMES.index("void")
    illegal_report = validate(illegal)
    assert not illegal_report["hard_valid"]
    assert illegal_report["tuples"]["invalid_pixels"] > 0

    coreless = tuple(np.array(values, copy=True) for values in target)
    core_mask = coreless[0] == PART_OWNER_NAMES.index("core")
    body_y, body_x = np.argwhere(
        coreless[0] == PART_OWNER_NAMES.index("body")
    )[0]
    coreless[0][core_mask] = PART_OWNER_NAMES.index("body")
    coreless[1][core_mask] = coreless[1][body_y, body_x]
    coreless[2][core_mask] = coreless[2][body_y, body_x]
    coreless_report = validate(coreless)
    assert not coreless_report["hard_valid"]
    assert not coreless_report["topology"]["owner_presence"]["core"]

    out_of_domain = tuple(np.array(values, copy=True) for values in target)
    out_of_domain[0][0, 0] = np.uint8(255)
    assert not validate(out_of_domain)["hard_valid"]
    wrong_dtype = (target[0].astype(np.float32), target[1], target[2])
    assert not validate(wrong_dtype)["hard_valid"]
    wrong_shape = (target[0][:-1], target[1][:-1], target[2][:-1])
    assert not validate(wrong_shape)["hard_valid"]

    bad_guide = sample["guide"].numpy().copy()
    bad_guide[0, 0, 0] = np.nan
    bad_guide_report = validate_generated_fields(
        *target,
        guide=bad_guide,
        target=target,
        record=record,
        legal_tuples=bundle.legal_tuples,
        templates=templates,
    )
    assert not bad_guide_report["hard_valid"]
    assert not bad_guide_report["hard_gates"]["guide_contract"]
    assert bad_guide_report["hard_gates"]["categorical_domains"]

    duplicate_legal_table = np.concatenate(
        (bundle.legal_tuples, bundle.legal_tuples[:1]), axis=0
    )
    bad_legal_report = validate_generated_fields(
        *target,
        guide=sample["guide"].numpy(),
        target=target,
        record=record,
        legal_tuples=duplicate_legal_table,
        templates=templates,
    )
    assert not bad_legal_report["hard_valid"]
    assert not bad_legal_report["hard_gates"]["legal_table_contract"]

    mismatched_condition = type(record)(
        ordinal=record.ordinal,
        grid_mode=record.grid_mode,
        source_index=record.source_index,
        variation=record.variation,
        sample_seed=record.sample_seed,
        morphology=(record.morphology + 1) % len(FAMILIES),
        subtype=record.subtype,
        role=record.role,
    )
    bad_condition_report = validate_generated_fields(
        *target,
        guide=sample["guide"].numpy(),
        target=target,
        record=mismatched_condition,
        legal_tuples=bundle.legal_tuples,
        templates=templates,
    )
    assert not bad_condition_report["hard_valid"]
    assert not bad_condition_report["hard_gates"]["condition_contract"]

    class DiagnosticRejector:
        def classify(self, *args: object, **kwargs: object) -> dict:
            del args, kwargs
            return {
                "template_format": "test-diagnostic",
                "exact_condition_match": False,
                "all_axes_in_distribution": False,
            }

    diagnostic_report = validate_generated_fields(
        *target,
        guide=sample["guide"].numpy(),
        target=target,
        record=record,
        legal_tuples=bundle.legal_tuples,
        templates=DiagnosticRejector(),  # type: ignore[arg-type]
    )
    assert diagnostic_report["hard_valid"]
    assert diagnostic_report["accepted"]
    assert not diagnostic_report["condition_exact_match"]
    assert not diagnostic_report["condition_in_distribution"]


def test_authoritative_32768_corpus_calibrates_at_100_percent() -> None:
    corpus_path = PROJECT_ROOT / "data" / "morphology_32768_4d4f5250.npz"
    if not corpus_path.is_file():
        pytest.skip("Authoritative 32,768-specimen corpus is not present")
    report = calibrate_morphology_corpus(corpus_path)
    _validate_schema(report, "multifield_reference_calibration.schema.json")
    assert report["samples"] == 2560
    assert report["hard_valid"] == 2560
    assert report["hard_valid_rate"] == 1.0
    assert report["hard_gate_failures"] == {}
    assert set(report["hard_gate_passes"].values()) == {2560}
    assert set(report["hard_gate_passes"]) == {
        "categorical_domains",
        "guide_contract",
        "target_contract",
        "condition_contract",
        "legal_table_contract",
        "nonempty",
        "occupancy",
        "visible_connected",
        "structural_margin",
        "visible_margin",
        "legal_tuples",
        "scaffold_coverage",
        "essential_owners",
    }
    assert report["visible_component_counts"] == {"1": 2560}
    assert report["owner_presence"]["core"]["present"] == 2560
    assert report["owner_presence"]["body"]["present"] == 2543
    assert report["owner_presence"]["head"]["present"] == 2517
    assert report["diagnostic_condition_adherence"]["exact_match"]["subtype"][
        "matched"
    ] == 926


def test_raw_compiled_bank_contact_sheet_and_exact_replay(
    evaluation_checkpoint: Path, tmp_path: Path
) -> None:
    bundle = load_multifield_checkpoint(
        evaluation_checkpoint, device="cpu", precision="fp32"
    )
    destination = tmp_path / "bank"
    manifest = write_generation_bank(
        bundle,
        destination,
        mode="stratified",
        limit=2,
        batch_size=2,
        temperature=0.9,
    )
    assert manifest["format"] == GENERATION_BANK_FORMAT
    _validate_schema(manifest, "multifield_generation_bank.schema.json")
    assert manifest["generation"]["raw_is_authoritative"]
    assert (destination / "raw_contact_sheet.png").is_file()
    assert (destination / "compiled_contact_sheet.png").is_file()
    for sample in manifest["samples"]:
        assert sample["raw_validation"]["tuples"]["valid_fraction"] == 1.0
        assert sample["postprocess_validation"]["valid"]
        assert sample["postprocess"]["changed_fraction"] <= 0.03
        raw_manifest = destination / sample["raw_manifest"]["path"]
        _validate_schema(
            json.loads(raw_manifest.read_text(encoding="utf-8")),
            "multifield_raw_sample.schema.json",
        )
        compiled = destination / sample["compiled_artifacts"]["fields"]["path"]
        assert raw_manifest.stat().st_mtime_ns <= compiled.stat().st_mtime_ns
    replay = replay_generation_bank(
        destination / "generation_manifest.json",
        device="cpu",
        precision="fp32",
    )
    assert replay["status"] == "exact"
    assert replay["exact_samples"] == replay["samples"] == 2
    with pytest.raises(FileExistsError, match="immutable"):
        write_generation_bank(bundle, destination, mode="fixed", limit=1)


def test_checkpoint_benchmark_reports_required_axes(
    evaluation_checkpoint: Path, tmp_path: Path
) -> None:
    bundle = load_multifield_checkpoint(
        evaluation_checkpoint, device="cpu", precision="fp32"
    )
    output = tmp_path / "benchmark.json"
    report = benchmark_checkpoint(
        bundle,
        grid_mode="stratified",
        samples_per_condition=1,
        generation_limit=2,
        generation_batch_size=2,
        full_mask_examples=2,
        full_mask_batch_size=2,
        output_path=output,
    )
    assert report["format"] == BENCHMARK_FORMAT
    _validate_schema(report, "multifield_benchmark.schema.json")
    assert report["full_mask"]["metrics"]["validation_silhouette_iou"] >= 0.0
    assert report["reference_calibration"]["hard_valid_rate"] == 1.0
    generation = report["generation"]
    assert generation["raw_validity"]["mean_tuple_validity"] == 1.0
    assert "pairwise_categorical_hamming" in generation["diversity"]
    assert "per_family" in generation["acceptance"]
    assert "per_role" in generation["acceptance"]
    assert generation["performance"]["mean_seconds_per_sample"] > 0.0
    assert output.is_file()
