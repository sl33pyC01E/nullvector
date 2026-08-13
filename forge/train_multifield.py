from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from .config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECT_ROOT
from .morphology.constants import (
    EMISSION_LEVEL_NAMES,
    FAMILIES,
    GUIDE_CHANNEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SUBTYPE_NAMES,
)
from .multifield_data import (
    CorpusSplit,
    GuidePolicy,
    MorphologyCorpus,
    MorphologyCorpusDataset,
    augment_scaffold_guides,
    compute_class_weights,
    compute_legal_tuples,
    legal_tuple_fingerprint,
    select_condition_bank,
    stratified_corpus_split,
)
from .multifield_diffusion import (
    MultiFieldSpriteDiffusion,
    MultiFieldVocabulary,
    multifield_diffusion_loss,
    seeded_generators,
)
from .multifield_metrics import (
    MultiFieldMetricAccumulator,
    condition_preference_statistics,
    per_sample_unweighted_nll,
    validation_selection_score,
)
from .provenance import canonical_state_dict_hash
from .safety import require_disk_floor, write_json_atomic


CHECKPOINT_FORMAT = "nullvector-multifield-diffusion-checkpoint-v1"
TRAINING_SOURCE_FILES = (
    "forge/multifield_diffusion.py",
    "forge/multifield_data.py",
    "forge/multifield_metrics.py",
    "forge/train_multifield.py",
    "forge/morphology/constants.py",
    "forge/morphology/corpus.py",
    "forge/morphology/fields.py",
    "forge/morphology/genome.py",
    "forge/morphology/render.py",
    "forge/safety.py",
)


@dataclass(slots=True)
class MultiFieldTrainConfig:
    corpus_path: str = ""
    output_dir: str = str(OUTPUT_DIR / "multifield_training")
    checkpoint_dir: str = str(CHECKPOINT_DIR / "multifield")
    epochs: int = 36
    batch_size: int = 24
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    ema_decay: float = 0.9995
    gradient_clip: float = 1.0
    warmup_steps: int = 500
    minimum_lr_ratio: float = 0.08
    width: int = 96
    diffusion_steps: int = 16
    field_part_weight: float = 1.0
    field_material_weight: float = 0.65
    field_emission_weight: float = 0.45
    validation_fraction: float = 0.08
    split_seed: int = 0x5A17
    seed: int = 0x4E554C4C
    precision: str = "auto"
    device: str = "auto"
    num_workers: int = 0
    guide_policy: str = "scaffold_only"
    guide_thicken_radius: int = 1
    guide_channel_dropout: float = 0.08
    guide_jitter_pixels: int = 1
    generation_eval_count: int = 8
    generation_eval_interval: int = 4
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    quiet: bool = False
    deterministic: bool = True

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MultiFieldTrainConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def field_weights(self) -> tuple[float, float, float]:
        return (
            self.field_part_weight,
            self.field_material_weight,
            self.field_emission_weight,
        )

    def apply_smoke_defaults(self) -> None:
        self.epochs = 1
        self.batch_size = 2
        self.width = 32
        self.diffusion_steps = 2
        self.warmup_steps = 0
        self.ema_decay = 0.9
        self.generation_eval_count = 1
        self.generation_eval_interval = 1
        self.max_train_batches = 2
        self.max_validation_batches = 1
        self.num_workers = 0

    def validate(self) -> None:
        if not self.corpus_path:
            raise ValueError("A morphology --corpus path is required.")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive.")
        if self.warmup_steps < 0 or not 0.0 < self.minimum_lr_ratio <= 1.0:
            raise ValueError("Invalid learning-rate schedule settings.")
        if self.width < 32 or self.width % 32 != 0:
            raise ValueError("width must be at least 32 and divisible by 32.")
        if self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and one half.")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16, or fp16.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        GuidePolicy(
            name=self.guide_policy,
            thicken_radius=self.guide_thicken_radius,
            training_channel_dropout=self.guide_channel_dropout,
            training_jitter_pixels=self.guide_jitter_pixels,
        )
        if self.generation_eval_count < 0 or self.generation_eval_interval <= 0:
            raise ValueError("Invalid generation evaluation settings.")
        if self.max_train_batches is not None and self.max_train_batches <= 0:
            raise ValueError("max_train_batches must be positive when supplied.")
        if self.max_validation_batches is not None and self.max_validation_batches <= 0:
            raise ValueError("max_validation_batches must be positive when supplied.")
        if any(value < 0.0 for value in self.field_weights) or sum(self.field_weights) <= 0:
            raise ValueError("Field weights must be nonnegative with a positive sum.")


def training_source_hash(root: Path = PROJECT_ROOT) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for relative in TRAINING_SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        if device.type == "cuda":
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return "fp32"
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training is only supported on CUDA.")
    return requested


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed & 0x7FFFFFFFFFFFFFFF)


def capture_rng_state(
    shuffle_generator: torch.Generator,
    training_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "shuffle_generator": shuffle_generator.get_state(),
        "training_generator": training_generator.get_state(),
    }


def restore_rng_state(
    state: Mapping[str, Any],
    shuffle_generator: torch.Generator,
    training_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    shuffle_generator.set_state(state["shuffle_generator"])
    training_generator.set_state(state["training_generator"])


@torch.no_grad()
def update_ema(
    ema_model: MultiFieldSpriteDiffusion,
    model: MultiFieldSpriteDiffusion,
    decay: float,
) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        name: values.to(device, non_blocking=device.type == "cuda")
        for name, values in batch.items()
    }


def _model_inputs(batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        batch["morphology"],
        batch["subtype"],
        batch["role"],
        batch["genes"],
    )


def _targets(batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    return batch["part"], batch["material"], batch["emission"]


@torch.inference_mode()
def evaluate_full_mask(
    model: MultiFieldSpriteDiffusion,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    class_weights: Mapping[str, Tensor],
    legal_tuples: np.ndarray,
    fixed_seed: int,
    field_weights: tuple[float, float, float],
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    metrics = MultiFieldMetricAccumulator(model.vocabulary, legal_tuples)
    evaluation_generator = _generator(device, fixed_seed)
    totals = {
        "loss": 0.0,
        "part_loss": 0.0,
        "material_loss": 0.0,
        "emission_loss": 0.0,
    }
    condition_axis_preferred = {
        "morphology_subtype": 0,
        "role": 0,
        "genes": 0,
    }
    condition_axis_margin = {name: 0.0 for name in condition_axis_preferred}
    examples = 0
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(cpu_batch, device)
        target = _targets(batch)
        timestep = torch.full(
            (target[0].shape[0],), model.steps, dtype=torch.long, device=device
        )
        corrupted = model.corrupt(
            *target, timestep, generator=evaluation_generator
        )
        with _autocast_context(device, precision):
            logits = model(
                corrupted[0],
                corrupted[1],
                corrupted[2],
                batch["guide"],
                *_model_inputs(batch),
                timestep,
            )
            result = multifield_diffusion_loss(
                logits,
                *target,
                corrupted[3],
                class_weights=class_weights,
                field_weights=field_weights,
            )
            true_nll = per_sample_unweighted_nll(
                logits,
                target,
                field_weights=field_weights,
                pixel_mask=target[0] != 0,
            )
            counterfactual_losses: list[Tensor] = []
            axis_preferences: dict[str, tuple[Tensor, Tensor]] = {}
            transformations = (
                ("morphology_subtype", 1, 4, 0, lambda genes: genes),
                ("role", 0, 0, 3, lambda genes: genes),
                ("genes", 0, 0, 0, lambda genes: 1.0 - genes),
            )
            for (
                axis,
                morphology_shift,
                subtype_shift,
                role_shift,
                gene_transform,
            ) in transformations:
                counterfactual_logits = model(
                    corrupted[0],
                    corrupted[1],
                    corrupted[2],
                    batch["guide"],
                    (batch["morphology"] + morphology_shift) % model.morphology_count,
                    (batch["subtype"] + subtype_shift) % model.subtype_count,
                    (batch["role"] + role_shift) % model.role_count,
                    gene_transform(batch["genes"]),
                    timestep,
                )
                counterfactual_nll = per_sample_unweighted_nll(
                    counterfactual_logits,
                    target,
                    field_weights=field_weights,
                    pixel_mask=target[0] != 0,
                )
                counterfactual_losses.append(counterfactual_nll)
                axis_preferences[axis] = condition_preference_statistics(
                    true_nll, counterfactual_nll[:, None]
                )
            _, preference_margin = condition_preference_statistics(
                true_nll, torch.stack(counterfactual_losses, dim=1)
            )
        metrics.update_logits(logits, target)
        # Accumulator's preference field accepts arbitrary scores; zero is the
        # preference threshold and the margin is versus the best wrong axis.
        metrics.update_condition_proxy(
            torch.zeros_like(preference_margin), preference_margin
        )
        for axis, (preferred, margin) in axis_preferences.items():
            condition_axis_preferred[axis] += int(preferred.sum().item())
            condition_axis_margin[axis] += float(margin.sum().item())
        batch_size = int(target[0].shape[0])
        examples += batch_size
        totals["loss"] += float(result.loss.item()) * batch_size
        totals["part_loss"] += float(result.part_loss.item()) * batch_size
        totals["material_loss"] += float(result.material_loss.item()) * batch_size
        totals["emission_loss"] += float(result.emission_loss.item()) * batch_size
    if examples == 0:
        raise RuntimeError("Validation loader produced no examples.")
    report = metrics.report("validation")
    report.update(
        {
            "validation_loss": totals["loss"] / examples,
            "validation_part_loss": totals["part_loss"] / examples,
            "validation_material_loss": totals["material_loss"] / examples,
            "validation_emission_loss": totals["emission_loss"] / examples,
            "validation_examples": float(examples),
        }
    )
    for axis in condition_axis_preferred:
        report[f"validation_{axis}_preference_rate"] = (
            condition_axis_preferred[axis] / examples
        )
        report[f"validation_{axis}_nll_margin"] = condition_axis_margin[axis] / examples
    return report


@torch.inference_mode()
def evaluate_generation(
    model: MultiFieldSpriteDiffusion,
    dataset: MorphologyCorpusDataset,
    *,
    device: torch.device,
    precision: str,
    legal_tuples: np.ndarray,
    fixed_seed: int,
    count: int,
    temperature: float = 0.9,
) -> dict[str, float]:
    if count <= 0:
        return {}
    model.eval()
    metrics = MultiFieldMetricAccumulator(model.vocabulary, legal_tuples)
    legal_tensor = torch.as_tensor(legal_tuples, dtype=torch.long, device=device)
    evaluated = min(count, len(dataset))
    batch = _move_batch(
        default_collate([dataset[index] for index in range(evaluated)]), device
    )
    sample_seeds = [
        fixed_seed ^ int(batch["seed"][index].item()) ^ (index * 0x9E3779B1)
        for index in range(evaluated)
    ]
    with _autocast_context(device, precision):
        prediction = model.sample(
            batch["guide"],
            *_model_inputs(batch),
            temperature=temperature,
            generators=seeded_generators(sample_seeds, device),
            legal_tuples=legal_tensor,
        )
    metrics.update(prediction, _targets(batch))
    report = metrics.report("generation")
    report["generation_examples"] = float(evaluated)
    return report


def _lr_multiplier(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
    schedule_steps = max(total_steps - warmup_steps, 1)
    phase = min(max((step - warmup_steps) / schedule_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _checkpoint_planned_bytes(model: MultiFieldSpriteDiffusion) -> int:
    model_bytes = sum(
        value.numel() * value.element_size() for value in model.state_dict().values()
    )
    # Raw + EMA + Adam first/second moments + serialization scratch/headroom.
    return int(model_bytes * 5.5 + 64 * 1024 * 1024)


def atomic_torch_save(
    destination: Path,
    payload: Mapping[str, Any],
    *,
    planned_bytes: int,
) -> None:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(destination.parent, planned_bytes=planned_bytes)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Temporary checkpoint is empty.")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resume_invariants(config: MultiFieldTrainConfig) -> dict[str, Any]:
    names = (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "ema_decay",
        "gradient_clip",
        "warmup_steps",
        "minimum_lr_ratio",
        "width",
        "diffusion_steps",
        "field_part_weight",
        "field_material_weight",
        "field_emission_weight",
        "validation_fraction",
        "split_seed",
        "seed",
        "device",
        "precision",
        "num_workers",
        "max_train_batches",
        "max_validation_batches",
        "generation_eval_count",
        "generation_eval_interval",
        "guide_policy",
        "guide_thicken_radius",
        "guide_channel_dropout",
        "guide_jitter_pixels",
        "deterministic",
    )
    return {name: getattr(config, name) for name in names}


def _build_checkpoint_payload(
    *,
    model: MultiFieldSpriteDiffusion,
    ema_model: MultiFieldSpriteDiffusion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: MultiFieldTrainConfig,
    epoch: int,
    global_step: int,
    best_score: float,
    history: list[dict[str, Any]],
    corpus: MorphologyCorpus,
    split: CorpusSplit,
    legal_tuples: np.ndarray,
    class_weights: Mapping[str, Tensor],
    shuffle_generator: torch.Generator,
    training_generator: torch.Generator,
    source_hash: str,
    fixed_validation: Mapping[str, Any],
    guide_policy: GuidePolicy,
) -> dict[str, Any]:
    ema_state = ema_model.state_dict()
    return {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "ema_model": ema_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.to_dict(),
        "resume_invariants": _resume_invariants(config),
        "architecture": model.architecture_config(),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "global_step": global_step,
        "best_validation_selection_score": best_score,
        "history": history,
        "corpus": corpus.metadata(),
        "split": split.metadata(),
        "legal_tuples": torch.from_numpy(legal_tuples.copy()),
        "legal_tuple_fingerprint": legal_tuple_fingerprint(legal_tuples),
        "class_weights": {
            name: values.detach().cpu() for name, values in class_weights.items()
        },
        "rng_state": capture_rng_state(shuffle_generator, training_generator),
        "fixed_validation": dict(fixed_validation),
        "guide_policy": guide_policy.metadata(),
        "canonical_ema_hash": canonical_state_dict_hash(ema_state),
        "training_source_hash": source_hash,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(next(model.parameters()).device),
            "gpu_name": (
                torch.cuda.get_device_name(next(model.parameters()).device)
                if next(model.parameters()).device.type == "cuda"
                else None
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "training_objective": {
            "name": "aligned-multifield-masked-categorical-denoising",
            "fields": ("part_owner", "material", "emission_level"),
            "shared_corruption_mask": True,
            "mask_schedule": "sine-squared",
            "selection_metric": "validation_selection_score",
            "validation_scope": "full-mask guide-conditioned construction",
            "condition_adherence_proxy": (
                "fraction of specimens whose true-condition NLL is lower than "
                "a deterministic counterfactual-condition NLL"
            ),
        },
    }


def _verify_resume(
    checkpoint: Mapping[str, Any],
    *,
    config: MultiFieldTrainConfig,
    corpus: MorphologyCorpus,
    split: CorpusSplit,
    legal_tuples: np.ndarray,
    source_hash: str,
    allow_source_change: bool,
) -> None:
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Resume checkpoint has an unsupported format.")
    if checkpoint["corpus"]["file_sha256"] != corpus.file_sha256:
        raise ValueError("Resume corpus hash does not match the current corpus file.")
    if checkpoint["split"]["fingerprint"] != split.fingerprint:
        raise ValueError("Resume split fingerprint does not match.")
    if checkpoint["legal_tuple_fingerprint"] != legal_tuple_fingerprint(legal_tuples):
        raise ValueError("Resume train-only legal tuple table does not match.")
    if checkpoint["resume_invariants"] != _resume_invariants(config):
        raise ValueError(
            "Training invariants differ from the checkpoint. Resume with the saved "
            "configuration; choose the final epoch count before the first run."
        )
    if not allow_source_change and checkpoint["training_source_hash"] != source_hash:
        raise ValueError(
            "Training source changed since the checkpoint. Pass --allow-source-change "
            "only after reviewing the reproducibility break."
        )


def _make_model(config: MultiFieldTrainConfig, corpus: MorphologyCorpus) -> MultiFieldSpriteDiffusion:
    return MultiFieldSpriteDiffusion(
        vocabulary=MultiFieldVocabulary(
            len(PART_OWNER_NAMES), len(MATERIAL_NAMES), len(EMISSION_LEVEL_NAMES)
        ),
        morphology_count=len(FAMILIES),
        subtype_count=len(SUBTYPE_NAMES),
        role_count=len(ROLE_NAMES),
        gene_dim=int(corpus.genes.shape[1]),
        guide_channels=len(GUIDE_CHANNEL_NAMES),
        steps=config.diffusion_steps,
        width=config.width,
        image_size=corpus.image_size,
    )


def _train_epoch(
    model: MultiFieldSpriteDiffusion,
    ema_model: MultiFieldSpriteDiffusion,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    training_generator: torch.Generator,
    class_weights: Mapping[str, Tensor],
    config: MultiFieldTrainConfig,
    device: torch.device,
    guide_policy: GuidePolicy,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {"loss": 0.0, "part_accuracy": 0.0, "material_accuracy": 0.0, "emission_accuracy": 0.0}
    examples = 0
    batches = len(loader)
    if config.max_train_batches is not None:
        batches = min(batches, config.max_train_batches)
    progress = tqdm(
        loader,
        total=batches,
        desc=f"multifield epoch {epoch + 1:03d}/{config.epochs}",
        disable=config.quiet,
    )
    for batch_index, cpu_batch in enumerate(progress):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break
        batch = _move_batch(cpu_batch, device)
        target = _targets(batch)
        timestep = torch.randint(
            1,
            model.steps + 1,
            (target[0].shape[0],),
            dtype=torch.long,
            device=device,
            generator=training_generator,
        )
        corrupted = model.corrupt(
            *target, timestep, generator=training_generator
        )
        augmented_guide = augment_scaffold_guides(
            batch["guide"], guide_policy, generator=training_generator
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, config.precision):
            logits = model(
                corrupted[0],
                corrupted[1],
                corrupted[2],
                augmented_guide,
                *_model_inputs(batch),
                timestep,
            )
            result = multifield_diffusion_loss(
                logits,
                *target,
                corrupted[3],
                class_weights=class_weights,
                field_weights=config.field_weights,
            )
        if scaler.is_enabled():
            scaler.scale(result.loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            result.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()
        scheduler.step()
        global_step += 1
        ema_decay = min(
            config.ema_decay,
            (1.0 + global_step) / (10.0 + global_step),
        )
        update_ema(ema_model, model, ema_decay)

        batch_size = int(target[0].shape[0])
        examples += batch_size
        totals["loss"] += float(result.loss.item()) * batch_size
        totals["part_accuracy"] += float(result.part_accuracy.item()) * batch_size
        totals["material_accuracy"] += float(result.material_accuracy.item()) * batch_size
        totals["emission_accuracy"] += float(result.emission_accuracy.item()) * batch_size
        progress.set_postfix(
            loss=f"{totals['loss'] / examples:.4f}",
            part=f"{totals['part_accuracy'] / examples:.3f}",
            grad=f"{float(gradient_norm):.2f}",
        )
    if examples == 0:
        raise RuntimeError("Training loader produced no examples.")
    return (
        {
            "train_loss": totals["loss"] / examples,
            "train_part_masked_accuracy": totals["part_accuracy"] / examples,
            "train_material_masked_accuracy": totals["material_accuracy"] / examples,
            "train_emission_masked_accuracy": totals["emission_accuracy"] / examples,
            "train_examples": float(examples),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        },
        global_step,
    )


def run_training(
    config: MultiFieldTrainConfig,
    *,
    resume_checkpoint: Mapping[str, Any] | None = None,
    allow_source_change: bool = False,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    config.validate()
    device = _resolve_device(config.device)
    config.device = device.type
    config.precision = _resolve_precision(config.precision, device)
    config.validate()

    output_dir = Path(config.output_dir).resolve()
    checkpoint_dir = Path(config.checkpoint_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
    _seed_everything(config.seed)

    corpus = MorphologyCorpus.load(Path(config.corpus_path), verify_hash=True)
    config.corpus_path = str(corpus.path)
    split = stratified_corpus_split(
        corpus,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )
    guide_policy = GuidePolicy(
        name=config.guide_policy,
        thicken_radius=config.guide_thicken_radius,
        training_channel_dropout=config.guide_channel_dropout,
        training_jitter_pixels=config.guide_jitter_pixels,
    )
    training_dataset = MorphologyCorpusDataset(
        corpus, split.training, guide_policy=guide_policy
    )
    validation_dataset = MorphologyCorpusDataset(
        corpus, split.validation, guide_policy=guide_policy
    )
    generation_bank_indices = select_condition_bank(
        corpus,
        split.validation,
        config.generation_eval_count,
        seed=config.seed ^ 0x47454E42414E4B,
    )
    generation_dataset = (
        MorphologyCorpusDataset(
            corpus, generation_bank_indices, guide_policy=guide_policy
        )
        if len(generation_bank_indices)
        else None
    )
    legal_tuples = compute_legal_tuples(corpus, split.training)
    class_weights_cpu = compute_class_weights(corpus, split.training)

    if os.name == "nt" and config.num_workers > 0 and corpus.path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError(
            "Large compressed corpora with num_workers>0 are unsafe on Windows because "
            "spawned workers copy the in-memory arrays. Use --num-workers 0."
        )

    shuffle_generator = torch.Generator(device="cpu").manual_seed(
        config.seed ^ 0x53485546464C45
    )
    training_generator = _generator(device, config.seed ^ 0x545241494E)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
        "persistent_workers": config.num_workers > 0,
    }
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        drop_last=False,
        generator=shuffle_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    model = _make_model(config, corpus).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    steps_per_epoch = len(training_loader)
    if config.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_train_batches)
    total_steps = max(config.epochs * steps_per_epoch, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_multiplier(
            step,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            minimum_ratio=config.minimum_lr_ratio,
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.precision == "fp16"
    )
    class_weights = {
        name: values.to(device) for name, values in class_weights_cpu.items()
    }
    source_hash = training_source_hash()
    fixed_validation = {
        "full_mask_seed": config.seed ^ 0x56414C4944415445,
        "generation_seed": config.seed ^ 0x47454E4552415445,
        "generation_source_indices": list(
            map(int, generation_bank_indices)
        ),
    }

    start_epoch = 0
    global_step = 0
    best_score = -math.inf
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        _verify_resume(
            resume_checkpoint,
            config=config,
            corpus=corpus,
            split=split,
            legal_tuples=legal_tuples,
            source_hash=source_hash,
            allow_source_change=allow_source_change,
        )
        if resume_checkpoint["architecture"] != model.architecture_config():
            raise ValueError("Resume model architecture does not match.")
        model.load_state_dict(resume_checkpoint["model"])
        ema_model.load_state_dict(resume_checkpoint["ema_model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if resume_checkpoint.get("scaler"):
            scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint["next_epoch"])
        global_step = int(resume_checkpoint["global_step"])
        best_score = float(resume_checkpoint["best_validation_selection_score"])
        history = list(resume_checkpoint.get("history", []))
        restore_rng_state(
            resume_checkpoint["rng_state"], shuffle_generator, training_generator
        )

    planned_checkpoint_bytes = _checkpoint_planned_bytes(model)
    require_disk_floor(
        checkpoint_dir,
        planned_bytes=planned_checkpoint_bytes * 2 + 8 * 1024 * 1024,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_epoch = config.epochs
    if stop_after_epoch is not None:
        if stop_after_epoch <= 0:
            raise ValueError("stop_after_epoch must be positive when supplied.")
        final_epoch = min(final_epoch, stop_after_epoch)
    for epoch in range(start_epoch, final_epoch):
        train_metrics, global_step = _train_epoch(
            model,
            ema_model,
            training_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            training_generator=training_generator,
            class_weights=class_weights,
            config=config,
            device=device,
            guide_policy=guide_policy,
            epoch=epoch,
            global_step=global_step,
        )
        validation_metrics = evaluate_full_mask(
            ema_model,
            validation_loader,
            device=device,
            precision=config.precision,
            class_weights=class_weights,
            legal_tuples=legal_tuples,
            fixed_seed=int(fixed_validation["full_mask_seed"]),
            field_weights=config.field_weights,
            max_batches=config.max_validation_batches,
        )
        metrics: dict[str, Any] = {
            "epoch": epoch + 1,
            "global_step": global_step,
            **train_metrics,
            **validation_metrics,
        }
        metrics["validation_selection_score"] = validation_selection_score(metrics)
        generation_due = (
            config.generation_eval_count > 0
            and (
                epoch == start_epoch
                or (epoch + 1) % config.generation_eval_interval == 0
                or epoch + 1 == config.epochs
            )
        )
        if generation_due:
            metrics.update(
                evaluate_generation(
                    ema_model,
                    generation_dataset,
                    device=device,
                    precision=config.precision,
                    legal_tuples=legal_tuples,
                    fixed_seed=int(fixed_validation["generation_seed"]),
                    count=config.generation_eval_count,
                )
            )
        history.append(metrics)
        improved = metrics["validation_selection_score"] > best_score
        if improved:
            best_score = float(metrics["validation_selection_score"])

        payload = _build_checkpoint_payload(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
            history=history,
            corpus=corpus,
            split=split,
            legal_tuples=legal_tuples,
            class_weights=class_weights_cpu,
            shuffle_generator=shuffle_generator,
            training_generator=training_generator,
            source_hash=source_hash,
            fixed_validation=fixed_validation,
            guide_policy=guide_policy,
        )
        atomic_torch_save(
            checkpoint_dir / "latest.pt",
            payload,
            planned_bytes=planned_checkpoint_bytes,
        )
        if improved:
            atomic_torch_save(
                checkpoint_dir / "best.pt",
                payload,
                planned_bytes=planned_checkpoint_bytes,
            )
        write_json_atomic(
            output_dir / "training_history.json",
            {
                "format": CHECKPOINT_FORMAT,
                "config": config.to_dict(),
                "corpus": corpus.metadata(),
                "split": split.metadata(),
                "legal_tuple_count": int(len(legal_tuples)),
                "training_source_hash": source_hash,
                "best_validation_selection_score": best_score,
                "history": history,
            },
        )
        print(json.dumps(metrics, sort_keys=True))

    summary = {
        "device": str(device),
        "precision": config.precision,
        "epochs_completed": max(start_epoch, final_epoch),
        "global_step": global_step,
        "best_validation_selection_score": best_score,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "latest_checkpoint": str((checkpoint_dir / "latest.pt").resolve()),
        "best_checkpoint": str((checkpoint_dir / "best.pt").resolve()),
        "history_path": str((output_dir / "training_history.json").resolve()),
        "corpus_sha256": corpus.file_sha256,
        "training_source_hash": source_hash,
    }
    if device.type == "cuda":
        summary.update(
            {
                "cuda_peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated(device) / 1024**3, 4
                ),
                "cuda_peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved(device) / 1024**3, 4
                ),
            }
        )
    write_json_atomic(output_dir / "training_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the graph-guided aligned morphology field diffusion model."
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--diffusion-steps", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--split-seed", type=lambda value: int(value, 0))
    parser.add_argument("--seed", type=lambda value: int(value, 0))
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--guide-policy", choices=("scaffold_only", "full_debug")
    )
    parser.add_argument("--guide-thicken-radius", type=int)
    parser.add_argument("--guide-channel-dropout", type=float)
    parser.add_argument("--guide-jitter-pixels", type=int)
    parser.add_argument("--generation-eval-count", type=int)
    parser.add_argument("--generation-eval-interval", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument("--allow-source-change", action="store_true")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Operational epoch boundary for chunked runs; does not alter the schedule.",
    )
    parser.add_argument(
        "--allow-nondeterministic",
        dest="deterministic",
        action="store_false",
        default=None,
        help="Permit faster nondeterministic kernels; this breaks exact replay guarantees.",
    )
    return parser.parse_args(argv)


def resolve_config(
    args: argparse.Namespace,
    resume_checkpoint: Mapping[str, Any] | None,
) -> MultiFieldTrainConfig:
    if resume_checkpoint is None:
        config = MultiFieldTrainConfig()
    else:
        if resume_checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Resume checkpoint has an unsupported format.")
        config = MultiFieldTrainConfig.from_dict(resume_checkpoint["config"])
    if args.smoke:
        if resume_checkpoint is not None:
            raise ValueError("--smoke cannot be combined with --resume.")
        config.apply_smoke_defaults()
    overrides = {
        "corpus_path": str(args.corpus.resolve()) if args.corpus else None,
        "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
        "checkpoint_dir": (
            str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "gradient_clip": args.gradient_clip,
        "warmup_steps": args.warmup_steps,
        "width": args.width,
        "diffusion_steps": args.diffusion_steps,
        "validation_fraction": args.validation_fraction,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "precision": args.precision,
        "device": args.device,
        "num_workers": args.num_workers,
        "guide_policy": args.guide_policy,
        "guide_thicken_radius": args.guide_thicken_radius,
        "guide_channel_dropout": args.guide_channel_dropout,
        "guide_jitter_pixels": args.guide_jitter_pixels,
        "generation_eval_count": args.generation_eval_count,
        "generation_eval_interval": args.generation_eval_interval,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "quiet": args.quiet,
        "deterministic": args.deterministic,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume.resolve(), map_location="cpu", weights_only=False
        )
    config = resolve_config(args, resume_checkpoint)
    summary = run_training(
        config,
        resume_checkpoint=resume_checkpoint,
        allow_source_change=args.allow_source_change,
        stop_after_epoch=args.stop_after_epoch,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
