from __future__ import annotations

from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Final
import uuid

import torch

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_ml.contract import HEAD_NAMES, ModelConfig
from ..map_decorator_ml.legality import TorchLegalMasks
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_production.training import CorpusSampleRef
from ..map_decorator_production_v2.contract import ForegroundPatchConfig
from ..map_decorator_production_v2.patches import (
    ForegroundSampleStat,
    foreground_centered_crop,
    plan_foreground_batches,
)
from ..map_decorator_production_v2.quality import evaluate_split_gate
from ..map_decorator_production_v2.training import WarmStartEMA
from ..map_decorator_production_v4.contract import ProposalLocatorConfig
from ..map_decorator_production_v4.decoding import select_proposal_conditioned_argmax
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from ..map_decorator_production_v4.proposal import ProposalAuthority
from ..safety import require_disk_floor
from .checkpoint import inspect_checkpoint, save_checkpoint, tensor_state_sha256, training_source_sha256
from .contract import (
    ResidualCalibrationConfig,
    ResidualLossConfig,
    ResidualTrainingConfig,
    V4_TRAINING_CONTRACT_SHA256,
)
from .dataset import ProposalTeacherSample, collate_proposal_samples, crop_proposals
from .training import make_optimizer, train_batch


CALIBRATION_FORMAT: Final[str] = "nullvector-map-decorator-v4-residual-cuda-calibration/1.0.0"
REPORT_NAME: Final[str] = "calibration_report.json"
CHECKPOINT_NAME: Final[str] = "checkpoint_final.pt"
MAX_REPORT_BYTES: Final[int] = 32 * 1024 * 1024
MODEL_SEED: Final[int] = 0x44D3CA11
CORE = ModelConfig(base_channels=16, condition_channels=32)
LOCATOR = ProposalLocatorConfig(locator_channels=16, locator_blocks=2, count_hidden_channels=16)
TRAINING = ResidualTrainingConfig(ema_decay=0.995, full_mask_stride=2)
LOSS = ResidualLossConfig()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _configure_cuda(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before CUDA startup.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("V4 residual calibration requires CUDA with BF16 support.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def _fresh_model(device: torch.device) -> ProposalConditionedDecoratorV4:
    torch.manual_seed(MODEL_SEED)
    return ProposalConditionedDecoratorV4(CORE, LOCATOR).to(device)


def _ref_for_stat(stat: ForegroundSampleStat) -> CorpusSampleRef:
    return CorpusSampleRef(
        shard_index=stat.shard_index,
        sample_index=stat.sample_index,
        split=stat.split,
        map_id=stat.map_id,
        sample_identity_sha256=stat.sample_identity_sha256,
        full_map_identity_sha256=stat.full_map_identity_sha256,
    )


def _batches(items: Sequence[CorpusSampleRef], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _predict_batch(
    model: ProposalConditionedDecoratorV4,
    batch: dict[str, object],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    features = batch["features"].to(device)  # type: ignore[union-attr]
    targets = {name: batch["targets"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    legal_masks = {name: batch["legal_masks"][name].to(device) for name in HEAD_NAMES}  # type: ignore[index,union-attr]
    valid = batch["valid_cells"].to(device)  # type: ignore[union-attr]
    hard_empty = batch["hard_empty"].to(device)  # type: ignore[union-attr]
    theme = batch["theme_index"].to(device)  # type: ignore[union-attr]
    conditions = batch["global_conditions"].to(device)  # type: ignore[union-attr]
    proposals = {name: batch["proposals"][name].to(device) for name in ("decal", "prop")}  # type: ignore[index,union-attr]
    masked = {name: valid.clone() for name in HEAD_NAMES}
    level = torch.ones((features.shape[0],), dtype=torch.float32, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(features, targets, masked, theme, conditions, level, proposals)
        prediction = select_proposal_conditioned_argmax(
            output,
            TorchLegalMasks(hard_empty=hard_empty, **legal_masks),
        )
    return prediction, targets, valid


def evaluate_model(
    model: ProposalConditionedDecoratorV4,
    authority: ProposalAuthority,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    refs = authority.authority.corpus.epoch_refs(split, 0, TRAINING.seed)
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    sample_ids: list[str] = []
    valid_cells = 0
    was_training = model.training
    model.eval()
    try:
        for group in _batches(refs, batch_size):
            samples = []
            for ref in group:
                teacher, proposals = authority.sample_and_proposals(ref)
                samples.append(ProposalTeacherSample(teacher, proposals))
            batch = collate_proposal_samples(samples)
            observed, truth, valid = _predict_batch(model, batch, device=device)
            for name in HEAD_NAMES:
                predictions[name].append(observed[name][valid].detach().cpu())
                targets[name].append(truth[name][valid].detach().cpu())
            valid_cells += int(valid.sum().item())
            sample_ids.extend(ref.sample_identity_sha256 for ref in group)
    finally:
        model.train(was_training)
    metrics = decoration_metrics(
        {name: torch.cat(values) for name, values in predictions.items()},
        {name: torch.cat(values) for name, values in targets.items()},
        torch.ones((valid_cells,), dtype=torch.bool),
    )
    metrics.update(
        {
            "split": split,
            "sample_count": len(sample_ids),
            "sample_set_sha256": json_sha256(sorted(sample_ids)),
            "full_split": len(sample_ids) == len(authority.authority.corpus.refs_by_split[split]),
            "valid_cell_count": valid_cells,
            "hard_legality": 1.0,
            "immutable_semantic_changes": 0,
            "source_provenance_failures": 0,
        }
    )
    return metrics


def compare_to_baseline(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    tolerance: float,
) -> dict[str, object]:
    checks: dict[str, bool] = {
        "full_split": bool(candidate.get("full_split", False)),
        "sample_set_exact": candidate.get("sample_set_sha256") == baseline.get("sample_set_sha256"),
        "hard_legality": float(candidate.get("hard_legality", -1.0)) == 1.0,
        "immutable_semantic_changes": int(candidate.get("immutable_semantic_changes", -1)) == 0,
        "source_provenance_failures": int(candidate.get("source_provenance_failures", -1)) == 0,
    }
    deltas: dict[str, dict[str, float]] = {}
    for head in HEAD_NAMES:
        deltas[head] = {}
        for metric in ("foreground_macro_iou", "foreground_f1", "rare_class_recall"):
            baseline_value = float(baseline["heads"][head][metric])  # type: ignore[index]
            candidate_value = float(candidate["heads"][head][metric])  # type: ignore[index]
            delta = candidate_value - baseline_value
            deltas[head][metric] = delta
            checks[f"{head}.{metric}"] = math.isfinite(candidate_value) and delta >= -tolerance
    legacy = evaluate_split_gate(candidate, stage="calibration")
    checks["unchanged_calibration_gate"] = bool(legacy["passed"])
    return {"tolerance": tolerance, "deltas": deltas, "legacy_gate": legacy, "checks": checks, "passed": all(checks.values())}


def _evaluate_triplet(
    baseline: ProposalConditionedDecoratorV4,
    raw: ProposalConditionedDecoratorV4,
    ema_model: ProposalConditionedDecoratorV4,
    authority: ProposalAuthority,
    config: ResidualCalibrationConfig,
    device: torch.device,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for split, batch_size in (("validation", config.validation_batch_size), ("test", config.test_batch_size)):
        result[split] = {
            "baseline": evaluate_model(baseline, authority, split, batch_size=batch_size, device=device),
            "raw": evaluate_model(raw, authority, split, batch_size=batch_size, device=device),
            "ema": evaluate_model(ema_model, authority, split, batch_size=batch_size, device=device),
        }
    return result


def _quality_summary(metrics: dict[str, object], config: ResidualCalibrationConfig) -> dict[str, object]:
    candidates: dict[str, object] = {}
    for candidate in ("raw", "ema"):
        split_gates = {
            split: compare_to_baseline(
                metrics[split]["baseline"],  # type: ignore[index]
                metrics[split][candidate],  # type: ignore[index]
                tolerance=config.object_regression_tolerance,
            )
            for split in ("validation", "test")
        }
        candidates[candidate] = {"splits": split_gates, "passed": all(gate["passed"] for gate in split_gates.values())}
    selected = "ema" if candidates["ema"]["passed"] else "raw" if candidates["raw"]["passed"] else "none"  # type: ignore[index]
    return {"candidates": candidates, "selected": selected, "accepted": selected != "none"}


def _ema_model(model: ProposalConditionedDecoratorV4, ema: WarmStartEMA, device: torch.device) -> ProposalConditionedDecoratorV4:
    result = ProposalConditionedDecoratorV4(CORE, LOCATOR).to(device)
    result.load_state_dict(model.state_dict(), strict=True)
    ema.copy_to(result)
    return result


def run_calibration(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    config: ResidualCalibrationConfig = ResidualCalibrationConfig(),
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("V4 residual calibration output is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=1024 * 1024 * 1024)
    device = _configure_cuda(TRAINING.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    baseline = _fresh_model(device).eval()
    model = _fresh_model(device)
    model.core.requires_grad_(False)
    initial_model_sha256 = tensor_state_sha256(model.state_dict())
    optimizer = make_optimizer(model, TRAINING)
    ema = WarmStartEMA(model, TRAINING.ema_decay)
    generator = torch.Generator(device=device).manual_seed(TRAINING.seed)
    patch = ForegroundPatchConfig(
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        decal_slots=2,
        prop_slots=2,
        jitter_radius=min(8, config.patch_size // 4),
    )
    plan = plan_foreground_batches(
        authority.authority.stats["train"],
        steps=config.steps,
        epoch=0,
        seed=TRAINING.seed,
        config=patch,
    )
    history: list[dict[str, object]] = []
    model.train()
    training_started = time.perf_counter()
    for step, planned in enumerate(plan):
        samples: list[ProposalTeacherSample] = []
        for slot, item in enumerate(planned):
            ref = _ref_for_stat(item.stat)
            full_teacher, full_proposals = authority.sample_and_proposals(ref)
            crop = foreground_centered_crop(
                full_teacher,
                focus_head=item.focus_head,
                epoch=0,
                step=step,
                slot=slot,
                seed=TRAINING.seed,
                config=patch,
            )
            samples.append(ProposalTeacherSample(crop, crop_proposals(full_proposals, crop)))
        observed = train_batch(
            model,
            optimizer,
            ema,
            collate_proposal_samples(samples),
            generator=generator,
            training_config=TRAINING,
            loss_config=LOSS,
            device=device,
            autocast_dtype=torch.bfloat16,
        )
        total = float(observed["loss"]["total"])  # type: ignore[index]
        if not math.isfinite(total):
            raise FloatingPointError(f"Non-finite V4 calibration loss at step {step + 1}.")
        history.append(
            {
                "step": step + 1,
                "loss": observed["loss"],
                "gradient_norm": observed["gradient_norm"],
                "full_mask_sample_count": observed["full_mask_sample_count"],
            }
        )
    training_seconds = time.perf_counter() - training_started
    ema_model = _ema_model(model, ema, device).eval()
    evaluation_started = time.perf_counter()
    metrics = _evaluate_triplet(baseline, model.eval(), ema_model, authority, config, device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    quality = _quality_summary(metrics, config)
    elapsed = time.perf_counter() - started
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    checkpoint_path = staging / CHECKPOINT_NAME
    checkpoint_metrics = {"history": history, "evaluation": metrics, "quality": quality}
    sidecar = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        ema,
        generator,
        core_config=CORE,
        locator_config=LOCATOR,
        training_config=TRAINING,
        loss_config=LOSS,
        global_step=config.steps,
        corpus_sha256=authority.authority.corpus.corpus_sha256,
        index_semantic_sha256=authority.authority.index_semantic_sha256,
        metrics=checkpoint_metrics,
    )
    safety_gates = {
        "finite_history": all(math.isfinite(float(item["loss"]["total"])) for item in history),  # type: ignore[index]
        "step_count_exact": len(history) == config.steps and ema.updates == config.steps,
        "model_updated": initial_model_sha256 != tensor_state_sha256(model.state_dict()),
        "categorical_core_frozen": all(not parameter.requires_grad for parameter in model.core.parameters()),
        "all_evaluations_full_split": all(
            bool(metrics[split][candidate]["full_split"])  # type: ignore[index]
            for split in ("validation", "test")
            for candidate in ("baseline", "raw", "ema")
        ),
        "all_evaluations_hard_legal": all(
            float(metrics[split][candidate]["hard_legality"]) == 1.0  # type: ignore[index]
            for split in ("validation", "test")
            for candidate in ("baseline", "raw", "ema")
        ),
        "checkpoint_inspected": inspect_checkpoint(checkpoint_path)["global_step"] == config.steps,
        "disk_floor_preserved": shutil.disk_usage(output.parent).free >= 100 * 1024**3,
        "runtime_integration_disabled": True,
    }
    if not all(safety_gates.values()):
        raise RuntimeError(f"V4 calibration safety gate failed: {safety_gates}")
    report: dict[str, object] = {
        "format": CALIBRATION_FORMAT,
        "status": "passed" if quality["accepted"] else "rejected",
        "accepted": quality["accepted"],
        "training_contract_sha256": V4_TRAINING_CONTRACT_SHA256,
        "source_sha256": training_source_sha256(),
        "authority": {
            "corpus_sha256": authority.authority.corpus.corpus_sha256,
            "corpus_manifest_sha256": authority.authority.corpus.manifest_sha256,
            "index_semantic_sha256": authority.authority.index_semantic_sha256,
            "index_manifest_sha256": authority.authority.index_manifest_sha256,
        },
        "config": {
            "core": CORE.to_dict(),
            "locator": LOCATOR.to_dict(),
            "training": TRAINING.to_dict(),
            "loss": LOSS.to_dict(),
            "calibration": config.to_dict(),
            "patch": patch.to_dict(),
        },
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "precision": "bf16-autocast-float32-loss",
            "elapsed_seconds": elapsed,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "training_steps_per_second": config.steps / training_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "initial_model_sha256": initial_model_sha256,
        "final_model_sha256": tensor_state_sha256(model.state_dict()),
        "final_ema_sha256": tensor_state_sha256(ema.shadow),
        "history": history,
        "evaluation": metrics,
        "quality": quality,
        "checkpoint": {
            "path": CHECKPOINT_NAME,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
            "sidecar_sha256": sidecar["sidecar_sha256"],
        },
        "safety_gates": safety_gates,
        "claim_boundary": {
            "bounded_cuda_calibration": True,
            "production_training": False,
            "godot_integration": False,
            "procedural_baseline_credit_preserved": True,
        },
    }
    report["report_sha256"] = json_sha256(report)
    _atomic_json(staging / REPORT_NAME, report)
    os.replace(staging, output)
    return validate_calibration(output, corpus_root=corpus_root, index_root=index_root, replay_metrics=False)


def _read_report(output: Path) -> dict[str, Any]:
    path = Path(output).resolve() / REPORT_NAME
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("V4 calibration report is missing, unsafe, or oversized.")
    report = json.loads(path.read_text(encoding="utf-8"))
    stored = report.pop("report_sha256", None)
    if stored != json_sha256(report):
        raise ValueError("V4 calibration report self-hash failed.")
    report["report_sha256"] = stored
    return report


def validate_calibration(
    output: Path,
    *,
    corpus_root: Path,
    index_root: Path,
    replay_metrics: bool = True,
) -> dict[str, Any]:
    output = Path(output).resolve()
    report = _read_report(output)
    if report.get("format") != CALIBRATION_FORMAT or report.get("status") not in {"passed", "rejected"}:
        raise ValueError("V4 calibration format or status failed.")
    if report.get("source_sha256") != training_source_sha256() or report.get("training_contract_sha256") != V4_TRAINING_CONTRACT_SHA256:
        raise ValueError("V4 calibration source or contract drifted.")
    config = ResidualCalibrationConfig(**report["config"]["calibration"])
    if report["config"] != {
        "core": CORE.to_dict(),
        "locator": LOCATOR.to_dict(),
        "training": TRAINING.to_dict(),
        "loss": LOSS.to_dict(),
        "calibration": config.to_dict(),
        "patch": ForegroundPatchConfig(
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            decal_slots=2,
            prop_slots=2,
            jitter_radius=min(8, config.patch_size // 4),
        ).to_dict(),
    }:
        raise ValueError("V4 calibration configuration drifted.")
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    expected_authority = {
        "corpus_sha256": authority.authority.corpus.corpus_sha256,
        "corpus_manifest_sha256": authority.authority.corpus.manifest_sha256,
        "index_semantic_sha256": authority.authority.index_semantic_sha256,
        "index_manifest_sha256": authority.authority.index_manifest_sha256,
    }
    if report.get("authority") != expected_authority:
        raise ValueError("V4 calibration authority drifted.")
    checkpoint_path = output / CHECKPOINT_NAME
    if report.get("checkpoint", {}).get("sha256") != file_sha256(checkpoint_path):
        raise ValueError("V4 calibration checkpoint file identity failed.")
    payload = inspect_checkpoint(checkpoint_path)
    if payload.get("global_step") != config.steps or payload.get("metrics") != {
        "history": report["history"],
        "evaluation": report["evaluation"],
        "quality": report["quality"],
    }:
        raise ValueError("V4 calibration checkpoint semantics drifted.")
    derived_quality = _quality_summary(report["evaluation"], config)
    if report.get("quality") != derived_quality or bool(report.get("accepted")) != bool(derived_quality["accepted"]):
        raise ValueError("V4 calibration quality derivation drifted.")
    if report.get("status") != ("passed" if derived_quality["accepted"] else "rejected"):
        raise ValueError("V4 calibration status disagrees with quality gates.")
    if not all(report.get("safety_gates", {}).values()):
        raise ValueError("V4 calibration recorded a failed safety gate.")
    if replay_metrics:
        device = _configure_cuda(TRAINING.seed)
        baseline = _fresh_model(device).eval()
        raw = ProposalConditionedDecoratorV4(CORE, LOCATOR).to(device)
        raw.load_state_dict(payload["model_state"], strict=True)
        ema_model = ProposalConditionedDecoratorV4(CORE, LOCATOR).to(device)
        ema_model.load_state_dict(payload["model_state"], strict=True)
        ema = WarmStartEMA(ema_model, TRAINING.ema_decay)
        ema.load_state_dict(payload["ema_state"])
        ema.copy_to(ema_model)
        replay = _evaluate_triplet(baseline, raw.eval(), ema_model.eval(), authority, config, device)
        if replay != report["evaluation"]:
            raise ValueError("V4 calibration evaluation replay failed.")
    return report
