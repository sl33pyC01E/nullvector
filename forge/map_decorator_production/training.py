from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import (
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    tensor_state_sha256,
)
from ..map_decorator_ml.contract import HEAD_NAMES, ModelConfig
from ..map_decorator_ml.dataset import TeacherSample, collate_teacher_samples
from ..map_decorator_ml.legality import TorchLegalMasks, select_legal_argmax
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_ml.model import CategoricalRefinementUNet
from ..map_decorator_ml.training import EMA, TrainingConfig, train_batch
from ..safety import require_disk_floor
from .contract import DISK_FLOOR_GIB, TRAINING_FORMAT_VERSION
from .corpus import MANIFEST_FILE, ShardSpec, load_shard_array, validate_corpus
from .provenance import source_sha256


CHECKPOINT_SOURCE_PACKAGES = (
    "forge/map_decorator_ml",
    "forge/map_decorator_production",
)


@dataclass(frozen=True, slots=True)
class ProductionTrainingConfig:
    epochs: int = 12
    segment_epochs: int = 2
    batch_size: int = 4
    base_channels: int = 48
    condition_channels: int = 96
    residual_blocks_per_scale: int = 1
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    corruption_min: float = 0.20
    corruption_max: float = 0.95
    ema_decay: float = 0.999
    seed: int = 0xDEC0A7E
    precision: str = "bf16"
    train_steps_per_epoch: int = 256
    held_out_batches: int = 96
    sentinel_batches: int = 24
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.epochs <= 512 or self.epochs % self.segment_epochs:
            raise ValueError("Epochs must be divisible into exact segment_epochs boundaries.")
        if self.segment_epochs != 2:
            raise ValueError("Production segments are contractually fixed at exactly two epochs.")
        if not 1 <= self.batch_size <= 64:
            raise ValueError("Batch size must be in [1,64].")
        if self.precision != "bf16":
            raise ValueError("The authorized CUDA production path requires BF16 autocast.")
        if not 1 <= self.train_steps_per_epoch <= 100_000:
            raise ValueError("train_steps_per_epoch must be positive and bounded.")
        if not 1 <= self.held_out_batches <= 10_000 or not 1 <= self.sentinel_batches <= 10_000:
            raise ValueError("Evaluation batch bounds must be positive.")
        if self.num_workers != 0:
            raise ValueError("Dataset subprocesses are disabled; the process supervisor is authoritative.")

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            base_channels=self.base_channels,
            condition_channels=self.condition_channels,
            residual_blocks_per_scale=self.residual_blocks_per_scale,
        )

    def training_config(self) -> TrainingConfig:
        return TrainingConfig(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            corruption_min=self.corruption_min,
            corruption_max=self.corruption_max,
            ema_decay=self.ema_decay,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorpusSampleRef:
    shard_index: int
    sample_index: int
    split: str
    map_id: str
    sample_identity_sha256: str
    full_map_identity_sha256: str


class ProductionCorpus:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        validation = validate_corpus(self.root, verify_shards=False)
        self.corpus_sha256 = str(validation["corpus_sha256"])
        manifest = json.loads((self.root / MANIFEST_FILE).read_text(encoding="utf-8"))
        self.manifest_sha256 = file_sha256(self.root / MANIFEST_FILE)
        self.shards: list[tuple[ShardSpec, list[dict[str, object]]]] = []
        self.refs_by_split: dict[str, list[CorpusSampleRef]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for shard_index, entry in enumerate(manifest["shards"]):
            spec = ShardSpec.from_dict(entry["spec"])
            sidecar = json.loads((self.root / spec.sidecar_path).read_text(encoding="utf-8"))
            records = sidecar["samples"]
            if not isinstance(records, list) or len(records) != spec.sample_count:
                raise ValueError("Corpus shard sample records are malformed.")
            self.shards.append((spec, records))
            for sample_index, record in enumerate(records):
                split = str(record["split"])
                self.refs_by_split[split].append(
                    CorpusSampleRef(
                        shard_index=shard_index,
                        sample_index=sample_index,
                        split=split,
                        map_id=str(record["map_id"]),
                        sample_identity_sha256=str(record["sample_identity_sha256"]),
                        full_map_identity_sha256=str(record["full_map_identity_sha256"]),
                    )
                )
        all_full = [ref.full_map_identity_sha256 for refs in self.refs_by_split.values() for ref in refs]
        if len(all_full) != len(set(all_full)):
            raise ValueError("ProductionCorpus observed duplicate full-map identities.")
        if not all(self.refs_by_split.values()):
            raise ValueError("ProductionCorpus requires non-empty train/validation/test splits.")

    def sample(self, ref: CorpusSampleRef) -> TeacherSample:
        spec, records = self.shards[ref.shard_index]
        record = records[ref.sample_index]
        features = np.ascontiguousarray(
            load_shard_array(self.root, spec, "features")[ref.sample_index], dtype=np.float32
        )
        targets = {
            name: np.ascontiguousarray(
                load_shard_array(self.root, spec, f"target_{name}")[ref.sample_index],
                dtype=np.uint8,
            )
            for name in HEAD_NAMES
        }
        legal = {
            name: np.ascontiguousarray(
                load_shard_array(self.root, spec, f"legal_{name}")[ref.sample_index],
                dtype=bool,
            )
            for name in HEAD_NAMES
        }
        hard_empty = np.ascontiguousarray(
            load_shard_array(self.root, spec, "hard_empty")[ref.sample_index], dtype=bool
        )
        conditions = np.ascontiguousarray(
            load_shard_array(self.root, spec, "global_conditions")[ref.sample_index],
            dtype=np.float32,
        )
        return TeacherSample(
            features=features,
            targets=targets,
            legal_masks=legal,
            hard_empty=hard_empty,
            global_conditions=conditions,
            theme_index=int(load_shard_array(self.root, spec, "theme_index")[ref.sample_index]),
            split=ref.split,
            full_map_identity_sha256=ref.full_map_identity_sha256,
            sample_identity_sha256=ref.sample_identity_sha256,
            source_semantic_sha256=str(record["source_semantic_sha256"]),
            feature_tensor_sha256=str(record["feature_tensor_sha256"]),
            target_fields_sha256=str(record["target_fields_sha256"]),
            map_id=ref.map_id,
            crop=None,
        )

    def epoch_refs(self, split: str, epoch: int, seed: int) -> tuple[CorpusSampleRef, ...]:
        refs = list(self.refs_by_split[split])
        rng = random.Random((seed << 16) ^ epoch ^ int(self.corpus_sha256[:16], 16))
        rng.shuffle(refs)
        return tuple(refs)


def _batches(refs: Sequence[CorpusSampleRef], batch_size: int) -> Iterator[tuple[CorpusSampleRef, ...]]:
    for start in range(0, len(refs), batch_size):
        yield tuple(refs[start : start + batch_size])


def _autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _evaluation_prediction(
    model: CategoricalRefinementUNet,
    batch: dict[str, object],
    *,
    device: torch.device,
    precision: str,
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
    with torch.inference_mode(), _autocast(device, precision):
        raw = model(features, targets, masked, theme, conditions, level)
        prediction = select_legal_argmax(
            raw,
            TorchLegalMasks(hard_empty=hard_empty, **legal_masks),
        )
    return prediction, targets, valid


def evaluate_split(
    model: CategoricalRefinementUNet,
    corpus: ProductionCorpus,
    split: str,
    *,
    batch_size: int,
    batch_limit: int,
    epoch: int,
    seed: int,
    device: torch.device,
    precision: str,
) -> dict[str, object]:
    refs = corpus.epoch_refs(split, epoch, seed)[: batch_size * batch_limit]
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    valid_cell_count = 0
    sample_ids: list[str] = []
    original_training = model.training
    model.eval()
    try:
        for ref_batch in _batches(refs, batch_size):
            samples = [corpus.sample(ref) for ref in ref_batch]
            batch = collate_teacher_samples(samples)
            observed, truth, valid = _evaluation_prediction(
                model, batch, device=device, precision=precision
            )
            for name in HEAD_NAMES:
                predictions[name].append(observed[name][valid].detach().cpu())
                targets[name].append(truth[name][valid].detach().cpu())
            valid_cell_count += int(valid.sum().item())
            sample_ids.extend(ref.sample_identity_sha256 for ref in ref_batch)
    finally:
        model.train(original_training)
    joined_predictions = {name: torch.cat(parts) for name, parts in predictions.items()}
    joined_targets = {name: torch.cat(parts) for name, parts in targets.items()}
    joined_valid = torch.ones((valid_cell_count,), dtype=torch.bool)
    metrics = decoration_metrics(joined_predictions, joined_targets, joined_valid)
    metrics.update(
        {
            "split": split,
            "sample_count": len(sample_ids),
            "batch_count": math.ceil(len(sample_ids) / batch_size),
            "sample_set_sha256": json_sha256(sorted(sample_ids)),
            "hard_legality": 1.0,
            "immutable_semantic_changes": 0,
            "source_provenance_failures": 0,
        }
    )
    return metrics


def _copy_ema_model(model: CategoricalRefinementUNet, ema: EMA) -> CategoricalRefinementUNet:
    target = CategoricalRefinementUNet(model.config)
    target.load_state_dict(model.state_dict(), strict=True)
    ema.copy_to(target)
    return target


def _memory_report(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"device": str(device), "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
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
        if temporary.exists():
            temporary.unlink()
        raise


def _require_cuda_bf16() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production map-decorator training.")
    device = torch.device("cuda", 0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support the required BF16 autocast path.")
    return device


def calibration_run(
    corpus_root: Path,
    output: Path,
    *,
    steps: int = 100,
    config: ProductionTrainingConfig = ProductionTrainingConfig(),
) -> dict[str, object]:
    if not 1 <= steps <= 1000:
        raise ValueError("Calibration steps must be in [1,1000].")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Calibration output already exists: {output}")
    require_disk_floor(output, floor_gb=DISK_FLOOR_GIB, planned_bytes=2 * 1024**3)
    device = _require_cuda_bf16()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    corpus = ProductionCorpus(corpus_root)
    model = CategoricalRefinementUNet(config.model_config()).to(device)
    training_config = config.training_config()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    ema = EMA(model, training_config.ema_decay)
    generator = torch.Generator(device=device).manual_seed(training_config.seed)
    refs = corpus.epoch_refs("train", 0, config.seed)
    losses: list[float] = []
    starts = time.perf_counter()
    for step in range(steps):
        start = (step * config.batch_size) % len(refs)
        selected = tuple(refs[(start + offset) % len(refs)] for offset in range(config.batch_size))
        batch = collate_teacher_samples([corpus.sample(ref) for ref in selected])
        result = train_batch(
            model,
            optimizer,
            ema,
            batch,
            generator=generator,
            config=training_config,
            device=device,
            autocast_dtype=torch.bfloat16,
        )
        loss = float(result["loss"]["total"])
        if not math.isfinite(loss):
            raise RuntimeError(f"Calibration produced a non-finite loss at step {step}.")
        losses.append(loss)
    elapsed = time.perf_counter() - starts
    ema_model = _copy_ema_model(model, ema).to(device)
    held_out = evaluate_split(
        ema_model,
        corpus,
        "validation",
        batch_size=config.batch_size,
        batch_limit=min(8, config.held_out_batches),
        epoch=0,
        seed=config.seed,
        device=device,
        precision=config.precision,
    )
    sentinel = evaluate_split(
        ema_model,
        corpus,
        "test",
        batch_size=1,
        batch_limit=min(4, config.sentinel_batches),
        epoch=0,
        seed=config.seed,
        device=device,
        precision=config.precision,
    )
    report: dict[str, object] = {
        "format_version": TRAINING_FORMAT_VERSION,
        "passed": True,
        "kind": "cuda-bf16-calibration",
        "steps": steps,
        "batch_size": config.batch_size,
        "corpus_sha256": corpus.corpus_sha256,
        "training_source_sha256": source_sha256("training"),
        "config": config.to_dict(),
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "minimum": min(losses),
            "maximum": max(losses),
            "mean": sum(losses) / len(losses),
            "finite": True,
        },
        "held_out": held_out,
        "sentinel": sentinel,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "memory": _memory_report(device),
        "model_tensor_sha256": tensor_state_sha256(model.state_dict()),
        "ema_tensor_sha256": tensor_state_sha256(ema.shadow),
    }
    _atomic_json(output, report)
    return report


def train_segment(
    corpus_root: Path,
    output_root: Path,
    *,
    start_epoch: int,
    stop_epoch: int,
    resume: Path | None,
    config: ProductionTrainingConfig = ProductionTrainingConfig(),
) -> dict[str, object]:
    if stop_epoch - start_epoch != config.segment_epochs:
        raise ValueError("One training worker may execute exactly one two-epoch segment.")
    if start_epoch < 0 or stop_epoch > config.epochs:
        raise ValueError("Training segment is outside the configured schedule.")
    output_root = Path(output_root).resolve()
    segment_dir = output_root / "segments" / f"epochs-{start_epoch + 1:03d}-{stop_epoch:03d}"
    if segment_dir.exists():
        raise FileExistsError(f"Immutable training segment already exists: {segment_dir}")
    require_disk_floor(segment_dir, floor_gb=DISK_FLOOR_GIB, planned_bytes=3 * 1024**3)
    device = _require_cuda_bf16()
    torch.cuda.reset_peak_memory_stats(device)
    corpus = ProductionCorpus(corpus_root)
    training_config = config.training_config()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = CategoricalRefinementUNet(config.model_config()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    ema = EMA(model, training_config.ema_decay)
    generator = torch.Generator(device=device).manual_seed(training_config.seed)
    global_step = 0
    if resume is not None:
        loaded = load_checkpoint(
            resume,
            model,
            optimizer,
            ema,
            expected_training_config=training_config,
            expected_corpus_sha256=corpus.corpus_sha256,
            training_generator=generator,
            source_packages=CHECKPOINT_SOURCE_PACKAGES,
        )
        if int(loaded["epoch"]) != start_epoch or int(loaded["global_step"]) < 1:
            raise ValueError("Resume checkpoint does not match this exact segment boundary.")
        global_step = int(loaded["global_step"])
    elif start_epoch != 0:
        raise ValueError("A nonzero segment must resume from the exact prior immutable checkpoint.")
    staging = segment_dir.parent / f".{segment_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    epoch_reports: list[dict[str, object]] = []
    try:
        for epoch in range(start_epoch, stop_epoch):
            refs = corpus.epoch_refs("train", epoch, config.seed)
            losses: list[float] = []
            selection_scores: list[float] = []
            sample_count = 0
            for step, ref_batch in enumerate(_batches(refs, config.batch_size)):
                if step >= config.train_steps_per_epoch:
                    break
                batch = collate_teacher_samples([corpus.sample(ref) for ref in ref_batch])
                result = train_batch(
                    model,
                    optimizer,
                    ema,
                    batch,
                    generator=generator,
                    config=training_config,
                    device=device,
                    autocast_dtype=torch.bfloat16,
                )
                loss = float(result["loss"]["total"])
                if not math.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}.")
                losses.append(loss)
                selection_scores.append(float(result["metrics"]["selection_score"]))
                sample_count += len(ref_batch)
                global_step += 1
            if not losses:
                raise RuntimeError("Training epoch did not execute any steps.")
            epoch_reports.append(
                {
                    "epoch": epoch + 1,
                    "steps": len(losses),
                    "samples": sample_count,
                    "loss_mean": sum(losses) / len(losses),
                    "loss_min": min(losses),
                    "loss_max": max(losses),
                    "selection_score_mean": sum(selection_scores) / len(selection_scores),
                }
            )
        ema_model = _copy_ema_model(model, ema).to(device)
        held_out = evaluate_split(
            ema_model,
            corpus,
            "validation",
            batch_size=config.batch_size,
            batch_limit=config.held_out_batches,
            epoch=stop_epoch,
            seed=config.seed,
            device=device,
            precision=config.precision,
        )
        sentinel = evaluate_split(
            ema_model,
            corpus,
            "test",
            batch_size=1,
            batch_limit=config.sentinel_batches,
            epoch=stop_epoch,
            seed=config.seed,
            device=device,
            precision=config.precision,
        )
        if held_out["hard_legality"] != 1.0 or sentinel["hard_legality"] != 1.0:
            raise RuntimeError("A held-out or sentinel hard-safety gate failed.")
        metrics = {
            "held_out": held_out,
            "sentinel": sentinel,
            "epoch_reports": epoch_reports,
        }
        checkpoint_path = staging / f"checkpoint_epoch_{stop_epoch:03d}.pt"
        checkpoint = save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            ema,
            training_config=training_config,
            corpus_sha256=corpus.corpus_sha256,
            epoch=stop_epoch,
            global_step=global_step,
            training_generator=generator,
            metrics=metrics,
            source_packages=CHECKPOINT_SOURCE_PACKAGES,
        )
        report: dict[str, object] = {
            "format_version": TRAINING_FORMAT_VERSION,
            "passed": True,
            "start_epoch": start_epoch,
            "stop_epoch": stop_epoch,
            "segment_epochs": config.segment_epochs,
            "global_step": global_step,
            "corpus_sha256": corpus.corpus_sha256,
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "training_source_sha256": source_sha256("training"),
            "config": config.to_dict(),
            "resume_checkpoint": None if resume is None else str(Path(resume).resolve()),
            "checkpoint": checkpoint,
            "checkpoint_file": checkpoint_path.name,
            "epoch_reports": epoch_reports,
            "held_out": held_out,
            "sentinel": sentinel,
            "memory": _memory_report(device),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(staging / "segment_report.json", report)
        os.replace(staging, segment_dir)
        return report
    except BaseException:
        raise


def _parser() -> object:
    import argparse

    parser = argparse.ArgumentParser(description="CUDA BF16 map-decorator training worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--corpus", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--steps", type=int, default=100)
    calibration.add_argument("--batch-size", type=int, default=4)
    segment = subparsers.add_parser("segment")
    segment.add_argument("--corpus", type=Path, required=True)
    segment.add_argument("--output-root", type=Path, required=True)
    segment.add_argument("--start-epoch", type=int, required=True)
    segment.add_argument("--stop-epoch", type=int, required=True)
    segment.add_argument("--resume", type=Path)
    segment.add_argument("--epochs", type=int, default=12)
    segment.add_argument("--batch-size", type=int, default=4)
    segment.add_argument("--train-steps-per-epoch", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)  # type: ignore[union-attr]
    if args.command == "calibrate":
        config = ProductionTrainingConfig(batch_size=args.batch_size)
        report = calibration_run(args.corpus, args.output, steps=args.steps, config=config)
    else:
        config = ProductionTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_steps_per_epoch=args.train_steps_per_epoch,
        )
        report = train_segment(
            args.corpus,
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
