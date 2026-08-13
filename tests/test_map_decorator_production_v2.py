from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from forge.map_decorator.hashing import json_sha256
from forge.map_decorator_ml.dataset import TeacherSample, collate_teacher_samples
from forge.map_decorator_ml.training import EMA
from forge.map_decorator_production.teacher import build_production_sample
from forge.map_decorator_production_v2.contract import (
    CALIBRATION_GATES,
    FactoredModelConfig,
    ForegroundPatchConfig,
    V2_CONTRACT_SHA256,
    V2TrainingConfig,
    v2_contract_manifest,
)
from forge.map_decorator_production_v2.model import (
    FactoredDecoratorV2,
    compose_object_logits,
)
from forge.map_decorator_production_v2.patches import (
    ForegroundSampleStat,
    foreground_centered_crop,
    plan_foreground_batches,
)
from forge.map_decorator_production_v2.quality import evaluate_dual_split_gate
from forge.map_decorator_production_v2.training import make_optimizer, train_batch_v2
from forge.maps import MapConfig, generate_map


def _teacher_sample(seed: int = 0x123456, *, size: int = 72) -> TeacherSample:
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
    assert int(np.count_nonzero(production.targets["decal"])) > 0
    assert int(np.count_nonzero(production.targets["prop"])) > 0
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


def _stat(index: int, *, decal: int = 2, prop: int = 2) -> ForegroundSampleStat:
    return ForegroundSampleStat(
        shard_index=index // 16,
        sample_index=index % 16,
        split="train",
        map_id=f"map-{index}",
        sample_identity_sha256=f"{index + 1:064x}",
        full_map_identity_sha256=f"{index + 1000:064x}",
        decal_count=decal,
        prop_count=prop,
    )


def _metrics(*, object_quality: float, object_density: float = 1.0) -> dict[str, object]:
    heads: dict[str, object] = {}
    for name in ("variant", "decal", "prop", "emission"):
        if name == "variant":
            target = [100] * 8
            prediction = [100] * 8
            quality = max(object_quality, 0.9)
        else:
            target = [900, 80, 20] if name != "emission" else [700, 200, 80, 20]
            foreground = sum(target[1:])
            predicted_foreground = round(foreground * object_density)
            prediction = [sum(target) - predicted_foreground, predicted_foreground - 1, 1]
            if name == "emission":
                prediction = [sum(target) - predicted_foreground, predicted_foreground - 2, 1, 1]
            quality = object_quality if name in {"decal", "prop"} else 0.9
        heads[name] = {
            "foreground_macro_iou": quality,
            "foreground_f1": quality,
            "rare_class_recall": quality,
            "target_count": target,
            "prediction_count": prediction,
        }
    return {
        "hard_legality": 1.0,
        "immutable_semantic_changes": 0,
        "source_provenance_failures": 0,
        "heads": heads,
    }


def test_v2_contract_predeclares_factoring_quotas_and_noncollapse_gates() -> None:
    manifest = v2_contract_manifest()
    assert V2_CONTRACT_SHA256 == json_sha256(manifest)
    assert manifest["object_factorization"]["presence"] == "learned binary logit"
    assert manifest["patch_quota"] == ForegroundPatchConfig().to_dict()
    for head in ("decal", "prop"):
        assert CALIBRATION_GATES["heads"][head]["foreground_f1_min"] >= 0.05
        assert CALIBRATION_GATES["heads"][head]["foreground_macro_iou_min"] >= 0.02


def test_factored_object_composition_is_normalized_and_presence_exact() -> None:
    presence = torch.tensor([[[-2.0, 0.0], [2.0, 4.0]]])
    types = torch.tensor(
        [[[[1.0, -1.0], [0.5, 3.0]], [[-1.0, 1.0], [1.5, -2.0]]]]
    )
    logits = compose_object_logits(presence, types)
    probability = logits.exp()
    torch.testing.assert_close(probability.sum(dim=1), torch.ones_like(presence))
    torch.testing.assert_close(probability[:, 1:].sum(dim=1), torch.sigmoid(presence))


def test_foreground_quota_plan_is_exact_duplicate_free_and_replayable() -> None:
    stats = tuple(_stat(index) for index in range(12))
    config = ForegroundPatchConfig(batch_size=4, decal_slots=2, prop_slots=2)
    first = plan_foreground_batches(stats, steps=9, epoch=3, seed=91, config=config)
    second = plan_foreground_batches(stats, steps=9, epoch=3, seed=91, config=config)
    assert first == second
    for batch in first:
        assert [item.focus_head for item in batch].count("decal") == 2
        assert [item.focus_head for item in batch].count("prop") == 2
        identities = [item.stat.sample_identity_sha256 for item in batch]
        assert len(identities) == len(set(identities))
    assert first != plan_foreground_batches(stats, steps=9, epoch=4, seed=91, config=config)


def test_foreground_crop_is_deterministic_and_keeps_required_signal() -> None:
    sample = _teacher_sample()
    config = ForegroundPatchConfig(batch_size=2, decal_slots=1, prop_slots=1)
    first = foreground_centered_crop(
        sample,
        focus_head="prop",
        epoch=1,
        step=2,
        slot=1,
        seed=7,
        config=config,
    )
    second = foreground_centered_crop(
        sample,
        focus_head="prop",
        epoch=1,
        step=2,
        slot=1,
        seed=7,
        config=config,
    )
    assert first.shape == (48, 48)
    assert first.crop == second.crop
    assert first.sample_identity_sha256 == second.sample_identity_sha256
    assert np.array_equal(first.features, second.features)
    assert np.count_nonzero(first.targets["prop"]) > 0
    assert first.full_map_identity_sha256 == sample.full_map_identity_sha256


def test_cpu_factored_training_step_has_presence_type_and_count_gradients() -> None:
    sample = _teacher_sample(size=48)
    config = ForegroundPatchConfig(batch_size=2, decal_slots=1, prop_slots=1)
    crops = [
        foreground_centered_crop(
            sample,
            focus_head=head,
            epoch=0,
            step=0,
            slot=slot,
            seed=17,
            config=config,
        )
        for slot, head in enumerate(("decal", "prop"))
    ]
    batch = collate_teacher_samples(crops)
    model = FactoredDecoratorV2(
        FactoredModelConfig(base_channels=4, condition_channels=8, presence_bias_init=-3.0)
    )
    training = V2TrainingConfig(ema_decay=0.9, seed=23)
    optimizer = make_optimizer(model, training)
    ema = EMA(model, training.ema_decay)
    before = {name: layer.weight.detach().clone() for name, layer in model.presence_heads.items()}
    result = train_batch_v2(
        model,
        optimizer,
        ema,
        batch,
        generator=torch.Generator().manual_seed(training.seed),
        training_config=training,
    )
    assert np.isfinite(list(result["loss"].values())).all()
    assert result["loss"]["decal_positive_cells"] > 0
    assert result["loss"]["prop_positive_cells"] > 0
    assert result["loss"]["decal_target_count"] > 0
    assert result["loss"]["prop_target_count"] > 0
    assert np.isfinite(result["gradient_norm"])
    for name, layer in model.presence_heads.items():
        assert not torch.equal(before[name], layer.weight)


def test_calibration_gate_rejects_empty_collapse_and_density_flooding() -> None:
    collapse = _metrics(object_quality=0.0)
    collapsed = evaluate_dual_split_gate(collapse, collapse, stage="calibration")
    assert not collapsed["passed"]
    assert "decal.foreground_f1" in collapsed["validation"]["failures"]
    flooded_metrics = _metrics(object_quality=0.5, object_density=20.0)
    flooded = evaluate_dual_split_gate(flooded_metrics, flooded_metrics, stage="calibration")
    assert not flooded["passed"]
    assert "prop.foreground_density_ratio" in flooded["test"]["failures"]
    passing_metrics = _metrics(object_quality=0.5, object_density=1.0)
    passing = evaluate_dual_split_gate(passing_metrics, passing_metrics, stage="calibration")
    assert passing["passed"]
