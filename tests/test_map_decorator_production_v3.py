from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from forge.map_decorator.hashing import json_sha256
from forge.map_decorator_ml.contract import HEAD_CLASS_COUNTS, HEAD_NAMES
from forge.map_decorator_ml.dataset import TeacherSample, collate_teacher_samples
from forge.map_decorator_ml.legality import TorchLegalMasks
from forge.map_decorator_ml.model import HeadLogits
from forge.map_decorator_production.teacher import build_production_sample
from forge.map_decorator_production_v2.contract import ForegroundPatchConfig
from forge.map_decorator_production_v2.patches import foreground_centered_crop
from forge.map_decorator_production_v2.training import WarmStartEMA
from forge.map_decorator_production_v3.contract import (
    LocatorLossConfig,
    LocatorModelConfig,
    LocatorTrainingConfig,
    V3_CONTRACT_SHA256,
    v3_contract_manifest,
)
from forge.map_decorator_production_v3.decoding import (
    independent_count_quotas,
    select_sparse_locator_argmax,
)
from forge.map_decorator_production_v3.model import SparseLocatorDecoratorV3, SparseLocatorOutput
from forge.map_decorator_production_v3.smoke import run_cpu_smoke, validate_cpu_smoke
from forge.map_decorator_production_v3.training import make_optimizer_v3, train_batch_v3
from forge.maps import MapConfig, generate_map


def _teacher_sample(seed: int = 0x330033, *, size: int = 48) -> TeacherSample:
    data = generate_map(
        seed,
        "garden",
        MapConfig(width=size, height=size, objective_count=3, spawn_count=5),
    )
    production = build_production_sample(
        data,
        feature_seed=seed ^ 0xFEA7,
        replay_data=generate_map(data.seed, data.theme, data.config),
    )
    assert np.count_nonzero(production.targets["decal"]) > 0
    assert np.count_nonzero(production.targets["prop"]) > 0
    return TeacherSample(
        features=production.features,
        targets=production.targets,
        legal_masks=production.legal_masks,
        hard_empty=production.hard_empty,
        global_conditions=production.global_conditions,
        theme_index=2,
        split=production.split,
        full_map_identity_sha256=production.full_map_identity_sha256,
        sample_identity_sha256=production.sample_identity_sha256,
        source_semantic_sha256=production.source_semantic_sha256,
        feature_tensor_sha256=production.feature_tensor_sha256,
        target_fields_sha256=production.target_fields_sha256,
        map_id=data.map_id,
        crop=None,
    )


def _blank_output(*, decal_count: int, prop_count: int, shape: tuple[int, int] = (32, 32)) -> SparseLocatorOutput:
    batch, height, width = 1, *shape
    categorical = HeadLogits(
        **{
            name: torch.zeros((batch, classes, height, width), dtype=torch.float32)
            for name, classes in HEAD_CLASS_COUNTS.items()
        }
    )
    return SparseLocatorOutput(
        categorical=categorical,
        presence_logits={
            "decal": torch.full((batch, height, width), -20.0),
            "prop": torch.full((batch, height, width), -20.0),
        },
        type_logits={
            "decal": torch.zeros((batch, HEAD_CLASS_COUNTS["decal"] - 1, height, width)),
            "prop": torch.zeros((batch, HEAD_CLASS_COUNTS["prop"] - 1, height, width)),
        },
        log1p_counts={
            "decal": torch.tensor([math.log1p(decal_count)]),
            "prop": torch.tensor([math.log1p(prop_count)]),
        },
        maximum_objects_per_head=256,
    )


def _all_legal(output: SparseLocatorOutput) -> TorchLegalMasks:
    batch, _, height, width = output.categorical.variant.shape
    return TorchLegalMasks(
        variant=torch.ones_like(output.categorical.variant, dtype=torch.bool),
        decal=torch.ones_like(output.categorical.decal, dtype=torch.bool),
        prop=torch.ones_like(output.categorical.prop, dtype=torch.bool),
        emission=torch.ones_like(output.categorical.emission, dtype=torch.bool),
        hard_empty=torch.zeros((batch, height, width), dtype=torch.bool),
    )


def test_v3_contract_separates_where_from_how_many_and_self_hashes() -> None:
    manifest = v3_contract_manifest()
    assert V3_CONTRACT_SHA256 == json_sha256(manifest)
    assert manifest["object_factorization"]["count"] == "independent pooled log1p count head"
    assert manifest["localization_supervision"] == {
        "exact_foreground": True,
        "bounded_halo": True,
        "positive_vs_hard-negative_ranking": True,
        "count_decoupled_from_probability_sum": True,
    }
    assert manifest["safety"]["cpu_foundation_only"] is True
    assert manifest["safety"]["production_claim"] is False


def test_locator_model_shapes_and_initial_count_prior_are_exact() -> None:
    sample = _teacher_sample()
    batch = collate_teacher_samples([sample, sample])
    config = LocatorModelConfig(
        base_channels=4,
        condition_channels=8,
        locator_channels=4,
        locator_blocks=1,
        count_hidden_channels=4,
        count_prior=3.0,
    )
    model = SparseLocatorDecoratorV3(config)
    targets = {name: batch["targets"][name] for name in HEAD_NAMES}
    masked = {name: batch["valid_cells"].clone() for name in HEAD_NAMES}
    output = model(
        batch["features"],
        targets,
        masked,
        batch["theme_index"],
        batch["global_conditions"],
        torch.ones((2,), dtype=torch.float32),
    )
    assert output.presence_logits["decal"].shape == (2, 48, 48)
    assert output.type_logits["prop"].shape == (2, 2, 48, 48)
    torch.testing.assert_close(output.log1p_counts["decal"], torch.full((2,), math.log1p(3.0)))
    assert output.maximum_objects_per_head == 256


def test_independent_count_decoder_ignores_probability_sum_and_breaks_ties_stably() -> None:
    output = _blank_output(decal_count=3, prop_count=2)
    legal = _all_legal(output)
    eligible = {
        "decal": legal.decal[:, 1:].any(dim=1),
        "prop": legal.prop[:, 1:].any(dim=1),
    }
    quotas = independent_count_quotas(output, eligible, maximum=256)
    assert quotas["decal"].tolist() == [3]
    assert quotas["prop"].tolist() == [2]
    # Every probability is approximately zero; the independent count still
    # chooses exact quotas. Equal ranking scores use ascending flat indices.
    first = select_sparse_locator_argmax(output, legal)
    second = select_sparse_locator_argmax(output, legal)
    assert torch.equal(first["decal"], second["decal"])
    assert torch.equal(first["prop"], second["prop"])
    assert torch.nonzero(first["decal"].flatten(), as_tuple=False).flatten().tolist() == [0, 1, 2]
    assert torch.nonzero(first["prop"].flatten(), as_tuple=False).flatten().tolist() == [3, 4]
    assert not bool(((first["decal"] != 0) & (first["prop"] != 0)).any())


def test_count_decoder_clamps_to_legal_cells_and_fails_nonfinite() -> None:
    output = _blank_output(decal_count=200, prop_count=200)
    legal = _all_legal(output)
    legal.decal[:, 1:, :, 5:] = False
    legal.prop[:, 1:] = False
    legal.prop[:, 1:, :, -4:] = True
    selected = select_sparse_locator_argmax(output, legal)
    assert int((selected["decal"] != 0).sum()) == 160
    assert int((selected["prop"] != 0).sum()) == 128
    output.log1p_counts["decal"][0] = float("nan")
    with pytest.raises(ValueError, match="finite nonnegative"):
        independent_count_quotas(
            output,
            {"decal": legal.decal[:, 1:].any(dim=1), "prop": legal.prop[:, 1:].any(dim=1)},
            maximum=256,
        )


def test_cpu_training_updates_spatial_ranking_and_independent_count_heads() -> None:
    sample = _teacher_sample()
    patch = ForegroundPatchConfig(batch_size=2, decal_slots=1, prop_slots=1)
    crops = [
        foreground_centered_crop(
            sample,
            focus_head=head,
            epoch=0,
            step=0,
            slot=slot,
            seed=39,
            config=patch,
        )
        for slot, head in enumerate(("decal", "prop"))
    ]
    batch = collate_teacher_samples(crops)
    config = LocatorModelConfig(
        base_channels=4,
        condition_channels=8,
        locator_channels=4,
        locator_blocks=1,
        count_hidden_channels=4,
        count_prior=2.0,
    )
    model = SparseLocatorDecoratorV3(config)
    training = LocatorTrainingConfig(seed=47, ema_decay=0.9, full_mask_stride=1)
    optimizer = make_optimizer_v3(model, training)
    ema = WarmStartEMA(model, training.ema_decay)
    presence_before = {name: model.locators[name].presence.weight.detach().clone() for name in ("decal", "prop")}
    count_before = {name: model.locators[name].count[-1].weight.detach().clone() for name in ("decal", "prop")}
    result = train_batch_v3(
        model,
        optimizer,
        ema,
        batch,
        generator=torch.Generator().manual_seed(training.seed),
        training_config=training,
        loss_config=LocatorLossConfig(halo_radius=2),
    )
    assert np.isfinite(result["gradient_norm"])
    assert result["full_mask_sample_count"] == 2
    for name in ("decal", "prop"):
        assert result["loss"][f"{name}_positive_cells"] > 0
        assert result["loss"][f"{name}_halo_cells"] > 0
        assert result["loss"][f"{name}_hard_negative_cells"] > 0
        assert result["loss"][f"{name}_target_count"] > 0
        assert not torch.equal(presence_before[name], model.locators[name].presence.weight)
        assert not torch.equal(count_before[name], model.locators[name].count[-1].weight)
    assert ema.updates == 1


def test_v3_training_fails_closed_on_cuda_and_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="halo_radius"):
        LocatorLossConfig(halo_radius=0)
    with pytest.raises(ValueError, match="count_prior"):
        LocatorModelConfig(count_prior=300)
    with pytest.raises(ValueError, match="full_mask_stride"):
        LocatorTrainingConfig(full_mask_stride=0)
    sample = _teacher_sample()
    batch = collate_teacher_samples([sample])
    model = SparseLocatorDecoratorV3(
        LocatorModelConfig(base_channels=4, condition_channels=8, locator_channels=4, locator_blocks=1, count_hidden_channels=4)
    )
    training = LocatorTrainingConfig(full_mask_stride=1)
    with pytest.raises(ValueError, match="CPU-only"):
        train_batch_v3(
            model,
            make_optimizer_v3(model, training),
            WarmStartEMA(model, training.ema_decay),
            batch,
            generator=torch.Generator().manual_seed(1),
            training_config=training,
            device="cuda",
        )


def test_cpu_smoke_is_byte_semantic_replayable_and_tamper_checked(tmp_path) -> None:
    first = run_cpu_smoke(tmp_path / "a", steps=2)
    second = run_cpu_smoke(tmp_path / "b", steps=2)
    assert first == second
    assert first["status"] == "passed"
    assert first["runtime"]["device"] == "cpu"
    assert first["runtime"]["cuda_initialized"] is False
    assert first["initial_model_sha256"] != first["final_model_sha256"]
    assert validate_cpu_smoke(tmp_path / "a" / "smoke_report.json", exact_replay=True) == first
    path = tmp_path / "a" / "smoke_report.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = "forged"
    value["smoke_sha256"] = json_sha256({key: item for key, item in value.items() if key != "smoke_sha256"})
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="format/status"):
        validate_cpu_smoke(path)
