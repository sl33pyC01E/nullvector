from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Final

import torch

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.dataset import collate_teacher_samples
from ..map_decorator_ml.legality import TorchLegalMasks, select_legal_argmax
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_ml.training import EMA
from ..map_decorator_production.training import CorpusSampleRef, ProductionCorpus
from ..safety import require_disk_floor
from .checkpoint import load_checkpoint, save_checkpoint, source_sha256, tensor_state_sha256
from .contract import (
    DISK_FLOOR_GIB,
    FactoredLossConfig,
    FactoredModelConfig,
    ForegroundPatchConfig,
    V2_CONTRACT_SHA256,
    V2TrainingConfig,
)
from .index import INDEX_MANIFEST_FILE, load_foreground_stats, validate_foreground_index
from .model import FactoredDecoratorV2
from .patches import ForegroundSampleStat, foreground_centered_crop, plan_foreground_batches
from .quality import evaluate_dual_split_gate
from .training import make_optimizer, train_batch_v2


RUN_FORMAT_VERSION: Final[str] = "nullvector-map-decorator-v2-training-run/1.0.0"


@dataclass(frozen=True, slots=True)
class V2RunConfig:
    epochs: int = 12
    segment_epochs: int = 2
    train_steps_per_epoch: int = 256
    validation_batch_size: int = 4
    test_batch_size: int = 1
    model: FactoredModelConfig = FactoredModelConfig()
    training: V2TrainingConfig = V2TrainingConfig()
    loss: FactoredLossConfig = FactoredLossConfig()
    patch: ForegroundPatchConfig = ForegroundPatchConfig()

    def __post_init__(self) -> None:
        if not 2 <= self.epochs <= 128 or self.epochs % self.segment_epochs:
            raise ValueError("epochs must be divisible into exact segment boundaries.")
        if self.segment_epochs != 2:
            raise ValueError("Crash-isolated production segments are exactly two epochs.")
        if not 1 <= self.train_steps_per_epoch <= 100_000:
            raise ValueError("train_steps_per_epoch is outside its bound.")
        if not 1 <= self.validation_batch_size <= 16 or not 1 <= self.test_batch_size <= 8:
            raise ValueError("Evaluation batch sizes are outside their safe bounds.")

    def schedule_dict(self) -> dict[str, int]:
        return {
            "epochs": self.epochs,
            "segment_epochs": self.segment_epochs,
            "train_steps_per_epoch": self.train_steps_per_epoch,
            "validation_batch_size": self.validation_batch_size,
            "test_batch_size": self.test_batch_size,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule": self.schedule_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "loss": self.loss.to_dict(),
            "patch": self.patch.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class V2CorpusAuthority:
    corpus: ProductionCorpus
    index_root: Path
    index_manifest: dict[str, object]
    stats: dict[str, tuple[ForegroundSampleStat, ...]]

    @classmethod
    def load(cls, corpus_root: Path, index_root: Path) -> "V2CorpusAuthority":
        corpus_root = Path(corpus_root).resolve()
        index_root = Path(index_root).resolve()
        validate_foreground_index(corpus_root, index_root)
        corpus = ProductionCorpus(corpus_root)
        manifest_path = index_root / INDEX_MANIFEST_FILE
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["corpus_sha256"] != corpus.corpus_sha256:
            raise ValueError("Foreground index and production corpus SHA differ.")
        stats = {split: load_foreground_stats(index_root, split=split) for split in ("train", "validation", "test")}
        for split, records in stats.items():
            if len(records) != len(corpus.refs_by_split[split]):
                raise ValueError(f"Foreground index split count drifted for {split}.")
            for stat in records:
                ref = CorpusSampleRef(
                    shard_index=stat.shard_index,
                    sample_index=stat.sample_index,
                    split=stat.split,
                    map_id=stat.map_id,
                    sample_identity_sha256=stat.sample_identity_sha256,
                    full_map_identity_sha256=stat.full_map_identity_sha256,
                )
                expected = corpus.refs_by_split[split]
                if ref.sample_identity_sha256 not in {item.sample_identity_sha256 for item in expected}:
                    raise ValueError("Foreground index contains a sample outside the corpus split.")
        return cls(corpus=corpus, index_root=index_root, index_manifest=manifest, stats=stats)

    @property
    def index_semantic_sha256(self) -> str:
        return str(self.index_manifest["foreground_index_sha256"])

    @property
    def index_manifest_sha256(self) -> str:
        return file_sha256(self.index_root / INDEX_MANIFEST_FILE)

    def sample_for_stat(self, stat: ForegroundSampleStat):
        ref = CorpusSampleRef(
            shard_index=stat.shard_index,
            sample_index=stat.sample_index,
            split=stat.split,
            map_id=stat.map_id,
            sample_identity_sha256=stat.sample_identity_sha256,
            full_map_identity_sha256=stat.full_map_identity_sha256,
        )
        sample = self.corpus.sample(ref)
        if sample.sample_identity_sha256 != stat.sample_identity_sha256 or sample.map_id != stat.map_id:
            raise ValueError("Foreground index stat did not resolve to its exact corpus sample.")
        return sample


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=len(encoded) + 1024 * 1024)
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


def _configure_determinism(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before CUDA startup.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def _memory_report(device: torch.device) -> dict[str, object]:
    return {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _copy_ema_model(model: FactoredDecoratorV2, ema: EMA) -> FactoredDecoratorV2:
    target = FactoredDecoratorV2(model.config)
    target.load_state_dict(model.state_dict(), strict=True)
    ema.copy_to(target)
    return target


def _predict(
    model: FactoredDecoratorV2,
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
    masked = {name: valid.clone() for name in HEAD_NAMES}
    level = torch.ones((features.shape[0],), dtype=torch.float32, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(features, targets, masked, theme, conditions, level)
        prediction = select_legal_argmax(
            output.as_head_logits(),
            TorchLegalMasks(hard_empty=hard_empty, **legal_masks),
        )
    return prediction, targets, valid


def _batches(items: Sequence[CorpusSampleRef], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def evaluate_full_split(
    model: FactoredDecoratorV2,
    authority: V2CorpusAuthority,
    split: str,
    *,
    batch_size: int,
    epoch: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    refs = authority.corpus.epoch_refs(split, epoch, seed)
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    sample_ids: list[str] = []
    valid_cells = 0
    was_training = model.training
    model.eval()
    try:
        for group in _batches(refs, batch_size):
            batch = collate_teacher_samples([authority.corpus.sample(ref) for ref in group])
            observed, truth, valid = _predict(model, batch, device=device)
            for name in HEAD_NAMES:
                predictions[name].append(observed[name][valid].detach().cpu())
                targets[name].append(truth[name][valid].detach().cpu())
            valid_cells += int(valid.sum().item())
            sample_ids.extend(ref.sample_identity_sha256 for ref in group)
    finally:
        model.train(was_training)
    joined_prediction = {name: torch.cat(values) for name, values in predictions.items()}
    joined_target = {name: torch.cat(values) for name, values in targets.items()}
    metrics = decoration_metrics(
        joined_prediction,
        joined_target,
        torch.ones((valid_cells,), dtype=torch.bool),
    )
    metrics.update(
        {
            "split": split,
            "sample_count": len(sample_ids),
            "sample_set_sha256": json_sha256(sorted(sample_ids)),
            "full_split": len(sample_ids) == len(authority.corpus.refs_by_split[split]),
            "valid_cell_count": valid_cells,
            "hard_legality": 1.0,
            "immutable_semantic_changes": 0,
            "source_provenance_failures": 0,
        }
    )
    return metrics


def _train_epoch(
    model: FactoredDecoratorV2,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    authority: V2CorpusAuthority,
    *,
    epoch: int,
    steps: int,
    config: V2RunConfig,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, object]:
    plan = plan_foreground_batches(
        authority.stats["train"],
        steps=steps,
        epoch=epoch,
        seed=config.training.seed,
        config=config.patch,
    )
    losses: list[float] = []
    scores: list[float] = []
    head_loss_sums: dict[str, float] = {}
    for step, planned in enumerate(plan):
        crops = []
        for slot, item in enumerate(planned):
            sample = authority.sample_for_stat(item.stat)
            crops.append(
                foreground_centered_crop(
                    sample,
                    focus_head=item.focus_head,
                    epoch=epoch,
                    step=step,
                    slot=slot,
                    seed=config.training.seed,
                    config=config.patch,
                )
            )
        result = train_batch_v2(
            model,
            optimizer,
            ema,
            collate_teacher_samples(crops),
            generator=generator,
            training_config=config.training,
            loss_config=config.loss,
            device=device,
            autocast_dtype=torch.bfloat16,
        )
        loss = float(result["loss"]["total"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}.")
        losses.append(loss)
        scores.append(float(result["metrics"]["selection_score"]))
        for name, value in result["loss"].items():
            if isinstance(value, (float, int)):
                head_loss_sums[name] = head_loss_sums.get(name, 0.0) + float(value)
    return {
        "epoch": epoch + 1,
        "steps": len(losses),
        "samples": len(losses) * config.patch.batch_size,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "selection_score_mean": sum(scores) / len(scores),
        "mean_loss_terms": {name: value / len(losses) for name, value in sorted(head_loss_sums.items())},
    }


def _evaluate(
    model: FactoredDecoratorV2,
    authority: V2CorpusAuthority,
    *,
    epoch: int,
    stage: str,
    config: V2RunConfig,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    held_out = evaluate_full_split(
        model,
        authority,
        "validation",
        batch_size=config.validation_batch_size,
        epoch=epoch,
        device=device,
        seed=config.training.seed,
    )
    sentinel = evaluate_full_split(
        model,
        authority,
        "test",
        batch_size=config.test_batch_size,
        epoch=epoch,
        device=device,
        seed=config.training.seed,
    )
    return held_out, sentinel, evaluate_dual_split_gate(held_out, sentinel, stage=stage)


def calibration_run(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    steps: int = 100,
    config: V2RunConfig = V2RunConfig(),
) -> dict[str, object]:
    if not 1 <= steps <= 1000:
        raise ValueError("Calibration steps must be in [1,1000].")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=2 * 1024**3)
    device = _configure_determinism(config.training.seed)
    torch.cuda.reset_peak_memory_stats(device)
    authority = V2CorpusAuthority.load(corpus_root, index_root)
    model = FactoredDecoratorV2(config.model).to(device)
    optimizer = make_optimizer(model, config.training)
    ema = EMA(model, config.training.ema_decay)
    generator = torch.Generator(device=device).manual_seed(config.training.seed)
    started = time.perf_counter()
    epoch_report = _train_epoch(
        model,
        optimizer,
        ema,
        authority,
        epoch=0,
        steps=steps,
        config=config,
        generator=generator,
        device=device,
    )
    ema_model = _copy_ema_model(model, ema).to(device)
    held_out, sentinel, gate = _evaluate(
        ema_model,
        authority,
        epoch=0,
        stage="calibration",
        config=config,
        device=device,
    )
    report: dict[str, object] = {
        "format_version": RUN_FORMAT_VERSION,
        "kind": "foreground-factored-v2-cuda-bf16-calibration",
        "passed": bool(gate["passed"]),
        "finite_training": True,
        "steps": steps,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "training_source_sha256": source_sha256(),
        "corpus_sha256": authority.corpus.corpus_sha256,
        "corpus_manifest_sha256": authority.corpus.manifest_sha256,
        "index_semantic_sha256": authority.index_semantic_sha256,
        "index_manifest_sha256": authority.index_manifest_sha256,
        "config": config.to_dict(),
        "epoch_report": epoch_report,
        "held_out": held_out,
        "sentinel": sentinel,
        "quality_gate": gate,
        "model_tensor_sha256": tensor_state_sha256(model.state_dict()),
        "ema_tensor_sha256": tensor_state_sha256(ema.shadow),
        "elapsed_seconds": time.perf_counter() - started,
        "steps_per_second": steps / (time.perf_counter() - started),
        "memory": _memory_report(device),
    }
    _atomic_json(output, report)
    return report


def train_segment(
    corpus_root: Path,
    index_root: Path,
    output_root: Path,
    *,
    start_epoch: int,
    stop_epoch: int,
    resume: Path | None,
    config: V2RunConfig = V2RunConfig(),
) -> dict[str, object]:
    if stop_epoch - start_epoch != config.segment_epochs or start_epoch < 0 or stop_epoch > config.epochs:
        raise ValueError("Worker must execute exactly one in-schedule two-epoch segment.")
    segment_dir = Path(output_root).resolve() / "segments" / f"epochs-{start_epoch + 1:03d}-{stop_epoch:03d}"
    if segment_dir.exists():
        raise FileExistsError(segment_dir)
    require_disk_floor(segment_dir.parent, floor_gb=DISK_FLOOR_GIB, planned_bytes=3 * 1024**3)
    device = _configure_determinism(config.training.seed)
    torch.cuda.reset_peak_memory_stats(device)
    authority = V2CorpusAuthority.load(corpus_root, index_root)
    model = FactoredDecoratorV2(config.model).to(device)
    optimizer = make_optimizer(model, config.training)
    ema = EMA(model, config.training.ema_decay)
    generator = torch.Generator(device=device).manual_seed(config.training.seed)
    predecessor_sha: str | None = None
    global_step = 0
    if resume is not None:
        predecessor_sha = file_sha256(resume)
        loaded = load_checkpoint(
            resume,
            model,
            optimizer,
            ema,
            generator,
            expected={
                "v2_contract_sha256": V2_CONTRACT_SHA256,
                "source_sha256": source_sha256(),
                "corpus_sha256": authority.corpus.corpus_sha256,
                "corpus_manifest_sha256": authority.corpus.manifest_sha256,
                "index_semantic_sha256": authority.index_semantic_sha256,
                "index_manifest_sha256": authority.index_manifest_sha256,
                "model_config": config.model.to_dict(),
                "training_config": config.training.to_dict(),
                "loss_config": config.loss.to_dict(),
                "patch_config": config.patch.to_dict(),
                "schedule": config.schedule_dict(),
                "epoch": start_epoch,
            },
        )
        global_step = int(loaded["global_step"])
    elif start_epoch:
        raise ValueError("A nonzero segment requires its exact predecessor checkpoint.")
    staging = segment_dir.parent / f".{segment_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    epoch_reports: list[dict[str, object]] = []
    for epoch in range(start_epoch, stop_epoch):
        epoch_reports.append(
            _train_epoch(
                model,
                optimizer,
                ema,
                authority,
                epoch=epoch,
                steps=config.train_steps_per_epoch,
                config=config,
                generator=generator,
                device=device,
            )
        )
        global_step += config.train_steps_per_epoch
    ema_model = _copy_ema_model(model, ema).to(device)
    held_out, sentinel, gate = _evaluate(
        ema_model,
        authority,
        epoch=stop_epoch,
        stage="production",
        config=config,
        device=device,
    )
    metrics = {"epoch_reports": epoch_reports, "held_out": held_out, "sentinel": sentinel, "quality_gate": gate}
    checkpoint_path = staging / f"checkpoint_epoch_{stop_epoch:03d}.pt"
    checkpoint = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        ema,
        training_config=config.training,
        loss_config=config.loss,
        patch_config=config.patch,
        schedule=config.schedule_dict(),
        corpus_sha256=authority.corpus.corpus_sha256,
        corpus_manifest_sha256=authority.corpus.manifest_sha256,
        index_semantic_sha256=authority.index_semantic_sha256,
        index_manifest_sha256=authority.index_manifest_sha256,
        epoch=stop_epoch,
        global_step=global_step,
        predecessor_checkpoint_sha256=predecessor_sha,
        training_generator=generator,
        metrics=metrics,
    )
    report: dict[str, object] = {
        "format_version": RUN_FORMAT_VERSION,
        "passed": True,
        "production_gate_passed": bool(gate["passed"]),
        "start_epoch": start_epoch,
        "stop_epoch": stop_epoch,
        "global_step": global_step,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
        "training_source_sha256": source_sha256(),
        "corpus_sha256": authority.corpus.corpus_sha256,
        "corpus_manifest_sha256": authority.corpus.manifest_sha256,
        "index_semantic_sha256": authority.index_semantic_sha256,
        "index_manifest_sha256": authority.index_manifest_sha256,
        "config": config.to_dict(),
        "resume_checkpoint": None if resume is None else str(Path(resume).resolve()),
        "predecessor_checkpoint_sha256": predecessor_sha,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint": checkpoint,
        "epoch_reports": epoch_reports,
        "held_out": held_out,
        "sentinel": sentinel,
        "quality_gate": gate,
        "memory": _memory_report(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(staging / "segment_report.json", report)
    os.replace(staging, segment_dir)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foreground-factored neural decorator v2 CUDA worker")
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--corpus", type=Path, required=True)
    calibrate.add_argument("--index", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--steps", type=int, default=100)
    calibrate.add_argument("--base-channels", type=int, default=48)
    segment = sub.add_parser("segment")
    segment.add_argument("--corpus", type=Path, required=True)
    segment.add_argument("--index", type=Path, required=True)
    segment.add_argument("--output-root", type=Path, required=True)
    segment.add_argument("--start-epoch", type=int, required=True)
    segment.add_argument("--stop-epoch", type=int, required=True)
    segment.add_argument("--resume", type=Path)
    segment.add_argument("--epochs", type=int, default=12)
    segment.add_argument("--steps-per-epoch", type=int, default=256)
    segment.add_argument("--base-channels", type=int, default=48)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = V2RunConfig(
        epochs=getattr(args, "epochs", 12),
        train_steps_per_epoch=getattr(args, "steps_per_epoch", 256),
        model=FactoredModelConfig(base_channels=args.base_channels),
    )
    if args.command == "calibrate":
        report = calibration_run(args.corpus, args.index, args.output, steps=args.steps, config=config)
    else:
        report = train_segment(
            args.corpus,
            args.index,
            args.output_root,
            start_epoch=args.start_epoch,
            stop_epoch=args.stop_epoch,
            resume=args.resume,
            config=config,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
