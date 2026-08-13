from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.map_decorator import (
    CHANNEL_INDEX,
    build_legal_class_masks,
    encode_features,
    validate_decoration_fields,
)
from forge.map_decorator_ml.artifacts import validate_prediction_pack, write_prediction_pack
from forge.map_decorator_ml.checkpoint import (
    ResumeContractError,
    load_checkpoint,
    save_checkpoint,
    source_sha256,
)
from forge.map_decorator_ml.contract import (
    HEAD_CLASS_COUNTS,
    HEAD_NAMES,
    MODEL_CONTRACT_SHA256,
    ModelConfig,
    model_contract_manifest,
)
from forge.map_decorator_ml.dataset import (
    Crop,
    TeacherRecord,
    assert_no_split_leakage,
    collate_teacher_samples,
    corpus_identity,
    load_teacher_sample,
)
from forge.map_decorator_ml.metrics import decoration_metrics
from forge.map_decorator_ml.legality import (
    legal_masks_to_torch,
    mask_head_logits,
    select_legal_argmax,
)
from forge.map_decorator_ml.model import (
    CategoricalRefinementUNet,
    HeadLogits,
    crop_exact,
    pad_right_bottom,
)
from forge.map_decorator_ml.sampling import SamplerConfig, sample_refinement
from forge.map_decorator_ml.smoke import run_smoke
from forge.map_decorator_ml.training import EMA, TrainingConfig, train_batch
from forge.map_decorator.hashing import json_sha256
from forge.maps import MapConfig, generate_map, load_map_pack, write_map_pack


def _pack(tmp_path: Path, *, width: int = 40, height: int = 40, theme: str = "garden") -> Path:
    data = generate_map(
        0xDEC0_A7E,
        theme,
        MapConfig(width=width, height=height, objective_count=2, spawn_count=4),
    )
    return write_map_pack(data, tmp_path / "packs", preview_scale=2)


def _encoded(data: object, seed: int = 123) -> object:
    return encode_features(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        public_seed=seed,
    )


def _tiny_model() -> CategoricalRefinementUNet:
    return CategoricalRefinementUNet(ModelConfig(base_channels=4, condition_channels=8))


def test_model_contract_has_fixed_four_heads_and_hash() -> None:
    assert HEAD_CLASS_COUNTS == {"variant": 8, "decal": 3, "prop": 3, "emission": 4}
    manifest = model_contract_manifest()
    assert manifest["heads"] == HEAD_CLASS_COUNTS
    assert manifest["feature_channels"] == 53
    assert manifest["object_decoding"].startswith("single categorical choice")
    assert MODEL_CONTRACT_SHA256 == json_sha256(manifest)


def test_padding_and_crop_are_exact_without_resampling() -> None:
    source = torch.arange(2 * 3 * 33 * 47, dtype=torch.float32).reshape(2, 3, 33, 47)
    padded, shape = pad_right_bottom(source, 4)
    assert shape == (33, 47)
    assert padded.shape == (2, 3, 36, 48)
    torch.testing.assert_close(crop_exact(padded, shape), source, rtol=0, atol=0)
    assert torch.count_nonzero(padded[..., 33:, :]) == 0
    assert torch.count_nonzero(padded[..., :, 47:]) == 0


@pytest.mark.parametrize(
    "height,width",
    [(32, 32), (33, 47), (32, 256), (256, 32), (255, 256), (256, 256)],
)
def test_model_preserves_every_rectangular_shape(height: int, width: int) -> None:
    model = _tiny_model().eval()
    features = torch.zeros((1, 53, height, width), dtype=torch.float32)
    labels = {name: torch.zeros((1, height, width), dtype=torch.long) for name in HEAD_NAMES}
    masked = {name: torch.ones((1, height, width), dtype=torch.bool) for name in HEAD_NAMES}
    with torch.inference_mode():
        output = model(
            features,
            labels,
            masked,
            torch.zeros((1,), dtype=torch.long),
            torch.zeros((1, 8), dtype=torch.float32),
            torch.ones((1,), dtype=torch.float32),
        )
    for name, classes in HEAD_CLASS_COUNTS.items():
        assert getattr(output, name).shape == (1, classes, height, width)


def test_teacher_uses_persisted_masks_and_keeps_all_crops_in_one_split(tmp_path: Path) -> None:
    pack = _pack(tmp_path, width=52, height=48, theme="anomaly")
    full = load_teacher_sample(TeacherRecord(pack, 11))
    first = load_teacher_sample(TeacherRecord(pack, 12, Crop(0, 0, 32, 32)))
    second = load_teacher_sample(TeacherRecord(pack, 13, Crop(10, 8, 32, 32)))
    assert {full.split, first.split, second.split} == {full.split}
    assert {full.full_map_identity_sha256, first.full_map_identity_sha256, second.full_map_identity_sha256} == {
        full.full_map_identity_sha256
    }
    assert len({full.sample_identity_sha256, first.sample_identity_sha256, second.sample_identity_sha256}) == 3
    assert_no_split_leakage((full, first, second))
    assert corpus_identity((first, second)) == corpus_identity((first, second))
    with pytest.raises(ValueError, match="duplicate"):
        corpus_identity((full, full))
    data = load_map_pack(pack)
    np.testing.assert_array_equal(
        full.features[CHANNEL_INDEX["protected_backbone"]], data.protected_backbone
    )
    np.testing.assert_array_equal(
        full.features[CHANNEL_INDEX["required_clearance"]], data.required_clearance
    )
    np.testing.assert_array_equal(
        full.features[CHANNEL_INDEX["decoration_forbidden"]], data.decoration_forbidden
    )
    assert not (full.targets["decal"][full.hard_empty] != 0).any()
    assert not (full.targets["prop"][full.hard_empty] != 0).any()
    assert not (full.targets["emission"][full.hard_empty] != 0).any()
    assert not ((full.targets["decal"] != 0) & (full.targets["prop"] != 0)).any()
    for name in HEAD_NAMES:
        yy, xx = np.indices(full.shape)
        assert full.legal_masks[name][full.targets[name], yy, xx].all()


def test_split_leakage_guard_fails_closed_on_forged_label(tmp_path: Path) -> None:
    sample = load_teacher_sample(TeacherRecord(_pack(tmp_path), 1))
    forged = replace(sample, split="validation" if sample.split != "validation" else "test")
    with pytest.raises(ValueError, match="inconsistent"):
        assert_no_split_leakage((forged,))


def test_seeded_refinement_is_exact_legal_and_conditionally_emissive(tmp_path: Path) -> None:
    data = load_map_pack(_pack(tmp_path, theme="archipelago"))
    encoded = _encoded(data, 501)
    model = _tiny_model()
    # Make the adversarial preference explicit: both object heads and emission want nonzero.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.heads["decal"].bias[1] = 9
        model.heads["prop"].bias[1] = 9
        model.heads["emission"].bias[3] = 9
    config = SamplerConfig(steps=3, temperature=0.5)
    first = sample_refinement(model, data, encoded, generation_seed=771, config=config)
    second = sample_refinement(model, data, encoded, generation_seed=771, config=config)
    assert first.report["field_sha256"] == second.report["field_sha256"]
    assert len(first.report["step_field_sha256"]) == config.steps
    for name in HEAD_NAMES:
        np.testing.assert_array_equal(first.arrays()[name], second.arrays()[name])
    hard = data.decoration_forbidden.astype(bool)
    assert not first.decal[hard].any() and not first.prop[hard].any() and not first.emission[hard].any()
    assert not ((first.decal != 0) & (first.prop != 0)).any()
    report = validate_decoration_fields(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        **first.arrays(),
    )
    assert report["passed"], report


def test_every_training_head_is_masked_before_argmax() -> None:
    data = generate_map(0x1E6A1, "rooms", MapConfig(width=34, height=37, spawn_count=4))
    zeros = np.zeros(data.shape, dtype=np.uint8)
    legal = build_legal_class_masks(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        selected_variant=zeros,
        selected_decal=zeros,
        selected_prop=zeros,
    )
    torch_legal = legal_masks_to_torch(legal)
    raw_arrays: dict[str, torch.Tensor] = {}
    for name, classes in HEAD_CLASS_COUNTS.items():
        logits = torch.zeros((1, classes, *data.shape), dtype=torch.float32)
        logits[:, 1:] = 1000.0  # Deliberately prefer illegal non-empty classes.
        raw_arrays[name] = logits
    bounded = mask_head_logits(HeadLogits(**raw_arrays), torch_legal)
    for name in HEAD_NAMES:
        selected = torch.argmax(getattr(bounded, name), dim=1)
        chosen_legal = getattr(torch_legal, name).gather(1, selected[:, None]).squeeze(1)
        assert chosen_legal.all()
        if name in {"decal", "prop", "emission"}:
            assert not selected[0][torch_legal.hard_empty[0]].any()
    coupled = select_legal_argmax(HeadLogits(**raw_arrays), torch_legal)
    assert not ((coupled["decal"] != 0) & (coupled["prop"] != 0)).any()


def test_class_balanced_training_step_and_nonempty_metrics(tmp_path: Path) -> None:
    packs = (
        _pack(tmp_path / "a", theme="arena"),
        _pack(tmp_path / "b", theme="caves"),
    )
    samples = [load_teacher_sample(TeacherRecord(path, 800 + index)) for index, path in enumerate(packs)]
    batch = collate_teacher_samples(samples)
    model = _tiny_model()
    config = TrainingConfig(ema_decay=0.9, seed=19)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ema = EMA(model, config.ema_decay)
    result = train_batch(
        model,
        optimizer,
        ema,
        batch,
        generator=torch.Generator().manual_seed(config.seed),
        config=config,
    )
    assert np.isfinite(list(result["loss"].values())).all()
    assert result["metrics"]["empty_accuracy_in_selection"] is False
    assert 0 <= result["metrics"]["selection_score"] <= 1

    targets = batch["targets"]
    all_empty = {name: torch.zeros_like(targets[name]) for name in HEAD_NAMES}
    metrics = decoration_metrics(all_empty, targets, batch["valid_cells"])
    for name in ("decal", "prop", "emission"):
        if sum(metrics["heads"][name]["target_count"][1:]) > 0:
            assert metrics["heads"][name]["foreground_f1"] == 0.0


def test_checkpoint_resume_contract_hashes_ema_and_rng(tmp_path: Path) -> None:
    model = _tiny_model()
    config = TrainingConfig(ema_decay=0.9, seed=23)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ema = EMA(model, config.ema_decay)
    generator = torch.Generator().manual_seed(config.seed)
    torch.rand((7,), generator=generator)
    path = tmp_path / "checkpoint.pt"
    report = save_checkpoint(
        path,
        model,
        optimizer,
        ema,
        training_config=config,
        corpus_sha256="a" * 64,
        epoch=2,
        global_step=9,
        training_generator=generator,
    )
    expected_next = torch.rand((5,), generator=generator)
    loaded = load_checkpoint(
        path,
        model,
        optimizer,
        ema,
        expected_training_config=config,
        expected_corpus_sha256="a" * 64,
        training_generator=generator,
    )
    resumed_next = torch.rand((5,), generator=generator)
    torch.testing.assert_close(expected_next, resumed_next, rtol=0, atol=0)
    assert loaded["ema_tensor_sha256"] == report["ema_tensor_sha256"]
    assert loaded["source_sha256"] == source_sha256()
    with pytest.raises(ResumeContractError, match="corpus_sha256"):
        load_checkpoint(
            path,
            model,
            optimizer,
            ema,
            expected_training_config=config,
            expected_corpus_sha256="b" * 64,
            training_generator=generator,
        )


def test_prediction_artifact_reloads_and_manifest_tampering_fails(tmp_path: Path) -> None:
    data = load_map_pack(_pack(tmp_path))
    encoded = _encoded(data, 99)
    model = _tiny_model()
    config = TrainingConfig(ema_decay=0.9)
    optimizer = torch.optim.AdamW(model.parameters())
    ema = EMA(model, config.ema_decay)
    generator = torch.Generator().manual_seed(1)
    checkpoint_path = tmp_path / "artifact_model.pt"
    checkpoint = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        ema,
        training_config=config,
        corpus_sha256="c" * 64,
        epoch=0,
        global_step=0,
        training_generator=generator,
    )
    sampler = SamplerConfig(steps=2)
    prediction = sample_refinement(model, data, encoded, generation_seed=55, config=sampler)
    pack = write_prediction_pack(
        tmp_path / "predictions",
        prediction,
        data,
        encoded,
        checkpoint_path=checkpoint_path,
        sampler_config=sampler,
        source_sha256=str(checkpoint["source_sha256"]),
        corpus_sha256="c" * 64,
        ema_tensor_sha256=str(checkpoint["ema_tensor_sha256"]),
    )
    assert validate_prediction_pack(
        pack, data=data, encoded=encoded, checkpoint_path=checkpoint_path
    )["passed"]
    manifest_path = pack / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_tensor_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="failed closed"):
        validate_prediction_pack(pack, data=data, encoded=encoded, checkpoint_path=checkpoint_path)


def test_one_step_cpu_smoke_publishes_atomic_replayable_bundle(tmp_path: Path) -> None:
    packs = tmp_path / "source_packs"
    _pack(packs, theme="rooms")
    output = tmp_path / "smoke"
    args = type(
        "Args",
        (),
        {
            "packs": packs / "packs",
            "output": output,
            "maps": 1,
            "train_steps": 1,
            "refinement_steps": 2,
            "base_channels": 4,
            "condition_channels": 8,
            "seed": 12345,
            "threads": 2,
        },
    )()
    report = run_smoke(args)
    assert report["passed"] and report["device"] == "cpu" and report["cuda_touched"] is False
    assert report["cuda_initialized_after"] is False
    assert report["byte_identical_replay"] is True
    assert output.is_dir() and (output / "smoke_report.json").is_file()
    assert not any(path.name.startswith(f".{output.name}.tmp-") for path in output.parent.iterdir())
    replay_args = type(
        "Args",
        (),
        {**{name: getattr(args, name) for name in (
            "packs", "maps", "train_steps", "refinement_steps", "base_channels",
            "condition_channels", "seed", "threads"
        )}, "output": tmp_path / "smoke_rebuild"},
    )()
    rebuilt = run_smoke(replay_args)
    assert rebuilt["checkpoint"]["model_tensor_sha256"] == report["checkpoint"]["model_tensor_sha256"]
    assert rebuilt["checkpoint"]["ema_tensor_sha256"] == report["checkpoint"]["ema_tensor_sha256"]
    assert rebuilt["prediction_field_sha256"] == report["prediction_field_sha256"]
    assert rebuilt["history"] == report["history"]
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        run_smoke(args)
