from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import torch

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.dataset import collate_teacher_samples
from ..map_decorator_ml.legality import TorchLegalMasks
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_production_v2.contract import ForegroundPatchConfig
from ..map_decorator_production_v2.patches import (
    foreground_centered_crop,
    plan_foreground_batches,
)
from ..map_decorator_production_v2.runner import V2CorpusAuthority
from ..map_decorator_production_v2.training import WarmStartEMA
from ..safety import require_disk_floor
from .checkpoint import (
    checkpoint_source_sha256,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
    tensor_state_sha256,
)
from .contract import (
    LocatorLossConfig,
    LocatorModelConfig,
    LocatorTrainingConfig,
    V3_CONTRACT_SHA256,
)
from .decoding import select_sparse_locator_argmax
from .model import SparseLocatorDecoratorV3
from .training import make_optimizer_v3, train_batch_v3


PILOT_FORMAT: Final[str] = "nullvector-map-decorator-v3-real-corpus-cpu-pilot/1.0.0"
CHECKPOINT_NAME: Final[str] = "checkpoint.pt"


@dataclass(frozen=True, slots=True)
class RealCorpusPilotConfig:
    steps: int = 4
    eval_samples_per_split: int = 4
    model: LocatorModelConfig = LocatorModelConfig(
        base_channels=8,
        condition_channels=16,
        locator_channels=8,
        locator_blocks=1,
        count_hidden_channels=8,
        count_prior=2.0,
    )
    training: LocatorTrainingConfig = LocatorTrainingConfig(
        learning_rate=5.0e-4,
        ema_decay=0.9,
        seed=0x3300C0DE,
        full_mask_stride=1,
    )
    loss: LocatorLossConfig = LocatorLossConfig(halo_radius=2)
    patch: ForegroundPatchConfig = ForegroundPatchConfig(
        batch_size=2,
        decal_slots=1,
        prop_slots=1,
    )

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not 1 <= self.steps <= 32:
            raise ValueError("Real-corpus pilot steps must be in [1,32].")
        if isinstance(self.eval_samples_per_split, bool) or not 1 <= self.eval_samples_per_split <= 16:
            raise ValueError("Real-corpus pilot evaluation count must be in [1,16].")

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "eval_samples_per_split": self.eval_samples_per_split,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "loss": self.loss.to_dict(),
            "patch": self.patch.to_dict(),
        }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=len(encoded) + 1024 * 1024)
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


def _copy_ema_model(model: SparseLocatorDecoratorV3, ema: WarmStartEMA) -> SparseLocatorDecoratorV3:
    target = SparseLocatorDecoratorV3(model.config)
    target.load_state_dict(model.state_dict(), strict=True)
    ema.copy_to(target)
    return target


def _train(
    model: SparseLocatorDecoratorV3,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    authority: V2CorpusAuthority,
    config: RealCorpusPilotConfig,
    generator: torch.Generator,
) -> list[dict[str, object]]:
    plan = plan_foreground_batches(
        authority.stats["train"],
        steps=config.steps,
        epoch=0,
        seed=config.training.seed,
        config=config.patch,
    )
    history: list[dict[str, object]] = []
    for step, planned in enumerate(plan):
        crops = []
        source_ids: list[str] = []
        focus_heads: list[str] = []
        for slot, item in enumerate(planned):
            sample = authority.sample_for_stat(item.stat)
            source_ids.append(sample.sample_identity_sha256)
            focus_heads.append(item.focus_head)
            crops.append(
                foreground_centered_crop(
                    sample,
                    focus_head=item.focus_head,
                    epoch=0,
                    step=step,
                    slot=slot,
                    seed=config.training.seed,
                    config=config.patch,
                )
            )
        result = train_batch_v3(
            model,
            optimizer,
            ema,
            collate_teacher_samples(crops),
            generator=generator,
            training_config=config.training,
            loss_config=config.loss,
            device="cpu",
        )
        loss = result["loss"]
        if not isinstance(loss, dict):
            raise TypeError("V3 pilot loss payload is malformed.")
        record: dict[str, object] = {
            "step": step + 1,
            "source_sample_sha256": source_ids,
            "focus_heads": focus_heads,
            "total_loss": float(loss["total"]),
            "gradient_norm": float(result["gradient_norm"]),
            "decal_predicted_count": float(loss["decal_predicted_count"]),
            "decal_target_count": float(loss["decal_target_count"]),
            "prop_predicted_count": float(loss["prop_predicted_count"]),
            "prop_target_count": float(loss["prop_target_count"]),
            "full_mask_sample_count": int(result["full_mask_sample_count"]),
        }
        if any(
            not math.isfinite(float(record[key]))
            for key in (
                "total_loss", "gradient_norm", "decal_predicted_count",
                "decal_target_count", "prop_predicted_count", "prop_target_count",
            )
        ):
            raise FloatingPointError("V3 real-corpus pilot produced non-finite evidence.")
        history.append(record)
    return history


def _predict_one(
    model: SparseLocatorDecoratorV3,
    sample: object,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, bool]:
    batch = collate_teacher_samples([sample])
    features = batch["features"]
    targets = {name: batch["targets"][name] for name in HEAD_NAMES}
    legal_masks = {name: batch["legal_masks"][name] for name in HEAD_NAMES}
    valid = batch["valid_cells"]
    hard_empty = batch["hard_empty"]
    masked = {name: valid.clone() for name in HEAD_NAMES}
    with torch.inference_mode():
        output = model(
            features,
            targets,
            masked,
            batch["theme_index"],
            batch["global_conditions"],
            torch.ones((1,), dtype=torch.float32),
        )
        legal = TorchLegalMasks(hard_empty=hard_empty, **legal_masks)
        prediction = select_sparse_locator_argmax(output, legal)
    hard_legal = True
    for name in HEAD_NAMES:
        selected_legal = legal_masks[name].gather(1, prediction[name].unsqueeze(1)).squeeze(1)
        hard_legal = hard_legal and bool(selected_legal[valid].all())
        # Terrain variation remains meaningful on object-empty cells. The
        # authoritative mask only forces object/emission heads to class zero.
        if name != "variant":
            hard_legal = hard_legal and not bool((prediction[name][hard_empty] != 0).any())
    hard_legal = hard_legal and not bool(((prediction["decal"] != 0) & (prediction["prop"] != 0)).any())
    return prediction, targets, valid, hard_legal


def _evaluate(
    model: SparseLocatorDecoratorV3,
    authority: V2CorpusAuthority,
    split: str,
    config: RealCorpusPilotConfig,
) -> dict[str, object]:
    refs = authority.corpus.epoch_refs(split, 0, config.training.seed)[: config.eval_samples_per_split]
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    identities: list[str] = []
    hard_legal = True
    was_training = model.training
    model.eval()
    try:
        for ref in refs:
            sample = authority.corpus.sample(ref)
            observed, truth, valid, sample_legal = _predict_one(model, sample)
            hard_legal = hard_legal and sample_legal
            identities.append(ref.sample_identity_sha256)
            for name in HEAD_NAMES:
                predictions[name].append(observed[name][valid].detach().cpu())
                targets[name].append(truth[name][valid].detach().cpu())
    finally:
        model.train(was_training)
    joined_prediction = {name: torch.cat(values) for name, values in predictions.items()}
    joined_target = {name: torch.cat(values) for name, values in targets.items()}
    valid_cells = torch.ones_like(joined_target[HEAD_NAMES[0]], dtype=torch.bool)
    metrics = decoration_metrics(joined_prediction, joined_target, valid_cells)
    metrics.update(
        {
            "split": split,
            "sample_count": len(identities),
            "sample_identity_sha256": identities,
            "sample_set_sha256": json_sha256(identities),
            "full_split": len(identities) == len(authority.corpus.refs_by_split[split]),
            "hard_legality": 1.0 if hard_legal else 0.0,
            "immutable_semantic_changes": 0,
            "source_provenance_failures": 0,
        }
    )
    return metrics


def _evaluation_pair(
    model: SparseLocatorDecoratorV3,
    authority: V2CorpusAuthority,
    config: RealCorpusPilotConfig,
) -> dict[str, object]:
    return {
        "validation": _evaluate(model, authority, "validation", config),
        "test": _evaluate(model, authority, "test", config),
    }


def _semantic_payload(report: dict[str, object]) -> dict[str, object]:
    return {
        key: report[key]
        for key in (
            "format", "status", "v3_contract_sha256", "checkpoint_source_sha256",
            "authority", "config", "history", "raw_evaluation", "ema_evaluation",
            "model_tensor_sha256", "ema_tensor_sha256", "ema_updates", "gates",
        )
    }


def run_real_corpus_pilot(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    config: RealCorpusPilotConfig = RealCorpusPilotConfig(),
) -> dict[str, object]:
    output = Path(output).resolve()
    report_path = output / "pilot_report.json"
    if report_path.is_file():
        return validate_real_corpus_pilot(report_path, corpus_root=corpus_root, index_root=index_root)
    if output.exists():
        raise FileExistsError(f"Pilot output exists without a complete report: {output}")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=512 * 1024 * 1024)
    if torch.cuda.is_initialized():
        raise RuntimeError("V3 real-corpus CPU pilot refuses a CUDA-initialized process.")
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True)
    previous_rng = torch.get_rng_state()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.training.seed)
        authority = V2CorpusAuthority.load(Path(corpus_root), Path(index_root))
        model = SparseLocatorDecoratorV3(config.model)
        optimizer = make_optimizer_v3(model, config.training)
        ema = WarmStartEMA(model, config.training.ema_decay)
        generator = torch.Generator().manual_seed(config.training.seed)
        history = _train(model, optimizer, ema, authority, config, generator)
        raw_evaluation = _evaluation_pair(model, authority, config)
        ema_model = _copy_ema_model(model, ema)
        ema_evaluation = _evaluation_pair(ema_model, authority, config)
        model_sha = tensor_state_sha256(model.state_dict())
        ema_sha = tensor_state_sha256(ema.shadow)
        raw_safety = all(float(raw_evaluation[split]["hard_legality"]) == 1.0 for split in ("validation", "test"))
        ema_safety = all(float(ema_evaluation[split]["hard_legality"]) == 1.0 for split in ("validation", "test"))
        hard_safety = raw_safety and ema_safety
        metrics = {
            "history": history,
            "raw_evaluation": raw_evaluation,
            "ema_evaluation": ema_evaluation,
            "hard_safety": hard_safety,
            "not_a_quality_claim": True,
        }
        checkpoint_path = staging / CHECKPOINT_NAME
        sidecar = save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            ema,
            training_config=config.training,
            loss_config=config.loss,
            patch_config=config.patch,
            schedule={"epoch": 0, "steps": config.steps, "eval_samples_per_split": config.eval_samples_per_split},
            corpus_sha256=authority.corpus.corpus_sha256,
            corpus_manifest_sha256=authority.corpus.manifest_sha256,
            index_semantic_sha256=authority.index_semantic_sha256,
            index_manifest_sha256=authority.index_manifest_sha256,
            epoch=0,
            global_step=config.steps,
            predecessor_checkpoint_sha256=None,
            training_generator=generator,
            metrics=metrics,
        )
        reload_model = SparseLocatorDecoratorV3(config.model)
        reload_optimizer = make_optimizer_v3(reload_model, config.training)
        reload_ema = WarmStartEMA(reload_model, config.training.ema_decay)
        reload_generator = torch.Generator().manual_seed(0)
        reloaded = load_checkpoint(
            checkpoint_path,
            reload_model,
            reload_optimizer,
            reload_ema,
            reload_generator,
            expected={
                "v3_contract_sha256": V3_CONTRACT_SHA256,
                "corpus_sha256": authority.corpus.corpus_sha256,
                "index_semantic_sha256": authority.index_semantic_sha256,
                "model_config": config.model.to_dict(),
                "training_config": config.training.to_dict(),
                "loss_config": config.loss.to_dict(),
                "patch_config": config.patch.to_dict(),
                "epoch": 0,
                "global_step": config.steps,
            },
        )
        reload_exact = (
            tensor_state_sha256(reload_model.state_dict()) == model_sha
            and tensor_state_sha256(reload_ema.shadow) == ema_sha
            and reloaded["metrics"] == metrics
        )
        report: dict[str, object] = {
            "format": PILOT_FORMAT,
            "status": "passed" if hard_safety and reload_exact else "failed",
            "v3_contract_sha256": V3_CONTRACT_SHA256,
            "checkpoint_source_sha256": checkpoint_source_sha256(),
            "authority": {
                "corpus_root": Path(corpus_root).resolve().name,
                "corpus_sha256": authority.corpus.corpus_sha256,
                "corpus_manifest_sha256": authority.corpus.manifest_sha256,
                "index_root": Path(index_root).resolve().name,
                "index_semantic_sha256": authority.index_semantic_sha256,
                "index_manifest_sha256": authority.index_manifest_sha256,
                "split_counts": {name: len(authority.corpus.refs_by_split[name]) for name in ("train", "validation", "test")},
            },
            "config": config.to_dict(),
            "runtime": {
                "python": os.sys.version.split()[0],
                "torch": torch.__version__,
                "device": "cpu",
                "cuda_initialized": torch.cuda.is_initialized(),
                "threads": torch.get_num_threads(),
            },
            "history": history,
            "raw_evaluation": raw_evaluation,
            "ema_evaluation": ema_evaluation,
            "model_tensor_sha256": model_sha,
            "ema_tensor_sha256": ema_sha,
            "ema_updates": ema.updates,
            "checkpoint": {
                "path": CHECKPOINT_NAME,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": file_sha256(checkpoint_path),
                "sidecar_sha256": sidecar["sidecar_sha256"],
            },
            "gates": {
                "real_corpus_bound": True,
                "foreground_index_bound": True,
                "finite_training": all(math.isfinite(float(item["total_loss"])) for item in history),
                "raw_hard_safety": raw_safety,
                "ema_hard_safety": ema_safety,
                "checkpoint_reload_exact": reload_exact,
                "cpu_only": not torch.cuda.is_initialized(),
                "not_a_quality_claim": True,
            },
        }
        if report["status"] != "passed" or not all(report["gates"].values()):  # type: ignore[union-attr]
            raise RuntimeError("V3 real-corpus pilot failed a publication gate.")
        report["semantic_sha256"] = json_sha256(_semantic_payload(report))
        report["report_sha256"] = json_sha256(report)
        _atomic_json(staging / "pilot_report.json", report)
        os.replace(staging, output)
        return report
    finally:
        torch.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def _read_report(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("V3 pilot report is missing or exceeds its bounded size.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V3 pilot report must be a JSON object.")
    stored = value.pop("report_sha256", None)
    if stored != json_sha256(value):
        raise ValueError("V3 pilot report self-hash failed.")
    value["report_sha256"] = stored
    return value


def validate_real_corpus_pilot(
    report_path: Path,
    *,
    corpus_root: Path,
    index_root: Path,
    exact_replay: bool = False,
) -> dict[str, object]:
    report_path = Path(report_path).resolve()
    value = _read_report(report_path)
    expected_keys = {
        "format", "status", "v3_contract_sha256", "checkpoint_source_sha256",
        "authority", "config", "runtime", "history", "raw_evaluation",
        "ema_evaluation", "model_tensor_sha256", "ema_tensor_sha256", "ema_updates",
        "checkpoint", "gates", "semantic_sha256", "report_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("V3 pilot report members violate the closed contract.")
    if value["format"] != PILOT_FORMAT or value["status"] != "passed":
        raise ValueError("V3 pilot format/status failed.")
    if value["v3_contract_sha256"] != V3_CONTRACT_SHA256 or value["checkpoint_source_sha256"] != checkpoint_source_sha256():
        raise ValueError("V3 pilot source/contract provenance is stale.")
    config_value = value["config"]
    if not isinstance(config_value, dict):
        raise ValueError("V3 pilot configuration is malformed.")
    config = RealCorpusPilotConfig(
        steps=int(config_value["steps"]),
        eval_samples_per_split=int(config_value["eval_samples_per_split"]),
        model=LocatorModelConfig(**config_value["model"]),
        training=LocatorTrainingConfig(**config_value["training"]),
        loss=LocatorLossConfig(**config_value["loss"]),
        patch=ForegroundPatchConfig(**config_value["patch"]),
    )
    if config_value != config.to_dict():
        raise ValueError("V3 pilot configuration is noncanonical.")
    authority = V2CorpusAuthority.load(Path(corpus_root), Path(index_root))
    expected_authority = {
        "corpus_root": Path(corpus_root).resolve().name,
        "corpus_sha256": authority.corpus.corpus_sha256,
        "corpus_manifest_sha256": authority.corpus.manifest_sha256,
        "index_root": Path(index_root).resolve().name,
        "index_semantic_sha256": authority.index_semantic_sha256,
        "index_manifest_sha256": authority.index_manifest_sha256,
        "split_counts": {name: len(authority.corpus.refs_by_split[name]) for name in ("train", "validation", "test")},
    }
    if value["authority"] != expected_authority:
        raise ValueError("V3 pilot corpus/index authority drifted.")
    runtime = value["runtime"]
    if not isinstance(runtime, dict) or runtime.get("device") != "cpu" or runtime.get("cuda_initialized") is not False or runtime.get("threads") != 1:
        raise ValueError("V3 pilot runtime violated its CPU-only contract.")
    history = value["history"]
    if not isinstance(history, list) or len(history) != config.steps:
        raise ValueError("V3 pilot history length differs from its schedule.")
    for step, record in enumerate(history, start=1):
        if not isinstance(record, dict) or record.get("step") != step or record.get("focus_heads") not in (["decal", "prop"], ["prop", "decal"]):
            raise ValueError("V3 pilot history record is malformed.")
        if any(not math.isfinite(float(record[key])) for key in ("total_loss", "gradient_norm", "decal_target_count", "prop_target_count")):
            raise ValueError("V3 pilot history contains non-finite evidence.")
        if float(record["decal_target_count"]) <= 0 or float(record["prop_target_count"]) <= 0:
            raise ValueError("V3 pilot training lost object foreground.")
    for collection_name in ("raw_evaluation", "ema_evaluation"):
        collection = value[collection_name]
        if not isinstance(collection, dict) or set(collection) != {"validation", "test"}:
            raise ValueError("V3 pilot evaluation split registry is malformed.")
        for split in ("validation", "test"):
            metrics = collection[split]
            if metrics.get("split") != split or metrics.get("sample_count") != config.eval_samples_per_split or metrics.get("hard_legality") != 1.0:
                raise ValueError("V3 pilot evaluation evidence failed.")
            if metrics.get("immutable_semantic_changes") != 0 or metrics.get("source_provenance_failures") != 0:
                raise ValueError("V3 pilot evaluation mutated or lost authority.")
    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, dict) or checkpoint.get("path") != CHECKPOINT_NAME:
        raise ValueError("V3 pilot checkpoint artifact record is malformed.")
    checkpoint_path = report_path.parent / CHECKPOINT_NAME
    if checkpoint_path.stat().st_size != checkpoint["bytes"] or file_sha256(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("V3 pilot checkpoint artifact hash failed.")
    sidecar_value = json.loads(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").read_text(encoding="utf-8"))
    if sidecar_value.get("sidecar_sha256") != checkpoint.get("sidecar_sha256"):
        raise ValueError("V3 pilot checkpoint sidecar identity failed.")
    payload = inspect_checkpoint(checkpoint_path)
    for label, expected in {
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "source_sha256": value["checkpoint_source_sha256"],
        "corpus_sha256": authority.corpus.corpus_sha256,
        "index_semantic_sha256": authority.index_semantic_sha256,
        "model_config": config.model.to_dict(),
        "training_config": config.training.to_dict(),
        "loss_config": config.loss.to_dict(),
        "patch_config": config.patch.to_dict(),
        "global_step": config.steps,
        "model_tensor_sha256": value["model_tensor_sha256"],
        "ema_tensor_sha256": value["ema_tensor_sha256"],
    }.items():
        if payload.get(label) != expected:
            raise ValueError(f"V3 pilot checkpoint mismatch for {label}.")
    if payload["metrics"] != {
        "history": value["history"],
        "raw_evaluation": value["raw_evaluation"],
        "ema_evaluation": value["ema_evaluation"],
        "hard_safety": True,
        "not_a_quality_claim": True,
    }:
        raise ValueError("V3 pilot checkpoint metrics do not close the report.")
    gates = value["gates"]
    expected_gates = {
        "real_corpus_bound": value["authority"] == expected_authority,
        "foreground_index_bound": value["authority"] == expected_authority,
        "finite_training": all(math.isfinite(float(item["total_loss"])) for item in history),
        "raw_hard_safety": all(value["raw_evaluation"][split]["hard_legality"] == 1.0 for split in ("validation", "test")),
        "ema_hard_safety": all(value["ema_evaluation"][split]["hard_legality"] == 1.0 for split in ("validation", "test")),
        "checkpoint_reload_exact": True,
        "cpu_only": runtime.get("cuda_initialized") is False and runtime.get("device") == "cpu",
        "not_a_quality_claim": True,
    }
    if gates != expected_gates or not all(expected_gates.values()):
        raise ValueError("V3 pilot publication gates are incomplete or failed.")
    if value["semantic_sha256"] != json_sha256(_semantic_payload(value)):
        raise ValueError("V3 pilot semantic replay identity failed.")
    if exact_replay:
        with tempfile.TemporaryDirectory(prefix="nullvector-v3-real-pilot-replay-") as temporary:
            replay = run_real_corpus_pilot(
                corpus_root,
                index_root,
                Path(temporary) / "pilot",
                config=config,
            )
        if replay["semantic_sha256"] != value["semantic_sha256"]:
            raise ValueError("V3 real-corpus pilot semantic exact replay diverged.")
    return value
