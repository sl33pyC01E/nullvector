from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import torch

from ..map_topology_neural.codec import CategoricalTopologyCodec, build_codec
from ..map_topology_neural.contract import CONTRACT_SHA256
from ..map_topology_neural.corpus import FROZEN_CORPUS_MANIFEST_FILE_SHA256, FROZEN_CORPUS_SHA256
from ..map_topology_neural.hashing import json_sha256
from ..safety import require_disk_floor
from .checkpoint import load_checkpoint, save_checkpoint, tensor_state_sha256
from .contract import (
    CALIBRATION_FORMAT,
    CHECKPOINT_FORMAT,
    MIN_FREE_CUDA_BYTES,
    QUALITY_GATES,
    TopologyCodecCalibrationConfig,
    canonical_json_bytes,
    production_source_manifest,
    production_source_sha256,
    sha256_file,
)
from .dataset import TopologyProductionDataset, ref_registry_sha256
from .metrics import balanced_reconstruction_loss, evaluate_codec


REPORT_NAME: Final[str] = "calibration_report.json"
CHECKPOINT_NAME: Final[str] = "checkpoint_final.pt"
MAX_REPORT_BYTES: Final[int] = 64 * 1024 * 1024
REPORT_KEYS: Final[set[str]] = {
    "format", "status", "source_sha256", "tensor_contract_sha256",
    "corpus_sha256", "corpus_manifest_file_sha256", "dataset_registry_sha256",
    "config", "history", "evaluation", "quality", "initial_model_sha256",
    "final_model_sha256", "final_ema_sha256", "checkpoint", "runtime",
    "safety_gates", "claim_boundary", "report_sha256",
}
SAFETY_GATE_KEYS: Final[set[str]] = {
    "finite_history", "step_count_exact", "model_updated",
    "evaluation_census_exact", "dataset_registry_exact", "checkpoint_inspected",
    "disk_floor_preserved", "representation_only", "production_schedule_disabled",
    "masked_latent_prior_not_started", "godot_integration_disabled",
}
RUNTIME_KEYS: Final[set[str]] = {
    "device", "compute_capability", "precision", "training_seconds",
    "evaluation_seconds", "elapsed_seconds", "peak_allocated_bytes",
    "peak_reserved_bytes",
}
CHECKPOINT_REPORT_KEYS: Final[set[str]] = {
    "format", "file", "bytes", "sha256", "source_sha256", "step",
    "model_state_sha256", "ema_state_sha256", "sidecar_sha256", "path",
}


def _configure_cuda(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before topology CUDA startup.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Topology codec calibration requires CUDA BF16.")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < MIN_FREE_CUDA_BYTES:
        raise RuntimeError("Topology codec calibration requires at least 4 GiB free CUDA memory.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def _fresh_model(config: TopologyCodecCalibrationConfig, device: torch.device) -> CategoricalTopologyCodec:
    return build_codec(config.codec_config(), init_seed=config.seed).to(device)


def _ema_update(shadow: dict[str, torch.Tensor], model: CategoricalTopologyCodec, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            source = value.detach()
            if source.is_floating_point():
                shadow[name].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                shadow[name].copy_(source)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return value


def _model_from_state(
    config: TopologyCodecCalibrationConfig,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> CategoricalTopologyCodec:
    model = _fresh_model(config, device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _quality(history: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    window = max(1, min(16, len(history) // 2))
    first = sum(float(item["loss"]["total"]) for item in history[:window]) / window
    last = sum(float(item["loss"]["total"]) for item in history[-window:]) / window
    improvement = (first - last) / max(abs(first), 1.0e-12)
    checks: dict[str, bool] = {"minimum_loss_improvement": improvement >= QUALITY_GATES["minimum_loss_improvement"]}
    for split in ("validation", "test"):
        metrics = evaluation[split]["ema"]
        checks[f"{split}.codebook_utilization"] = metrics["codebook"]["utilization"] >= QUALITY_GATES["minimum_codebook_utilization"]
        checks[f"{split}.terrain_accuracy"] = metrics["fields"]["terrain"]["accuracy"] >= QUALITY_GATES["minimum_terrain_accuracy"]
        checks[f"{split}.hazard_macro_recall"] = metrics["fields"]["hazard"]["macro_recall"] >= QUALITY_GATES["minimum_hazard_macro_recall"]
        checks[f"{split}.elevation_accuracy"] = metrics["fields"]["elevation"]["accuracy"] >= QUALITY_GATES["minimum_elevation_accuracy"]
        checks[f"{split}.walkability_iou"] = metrics["walkability_iou"] >= QUALITY_GATES["minimum_walkability_iou"]
    return {
        "quality_milestone_reached": all(checks.values()),
        "checks": checks,
        "thresholds": QUALITY_GATES,
        "loss_first_window_mean": first,
        "loss_last_window_mean": last,
        "relative_loss_improvement": improvement,
    }


def _atomic_report(path: Path, report: dict[str, Any]) -> None:
    encoded = canonical_json_bytes(report)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_report_contract(report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or set(report) != REPORT_KEYS:
        raise ValueError("Topology codec calibration report key census drifted.")
    safety = report.get("safety_gates")
    if (
        not isinstance(safety, dict)
        or set(safety) != SAFETY_GATE_KEYS
        or any(type(value) is not bool for value in safety.values())
    ):
        raise ValueError("Topology codec calibration safety gate census drifted.")
    if report.get("status") != ("passed" if all(safety.values()) else "failed"):
        raise ValueError("Topology codec calibration status is not derived from safety gates.")
    quality = report.get("quality")
    if not isinstance(quality, dict) or type(quality.get("quality_milestone_reached")) is not bool:
        raise ValueError("Topology codec calibration quality claim is malformed.")
    expected_boundary = {
        "representation_calibration_only": True,
        "quality_milestone_reached": quality["quality_milestone_reached"],
        "generative_prior_trained": False,
        "compiled_map_bank_published": False,
        "godot_integration": False,
    }
    if report.get("claim_boundary") != expected_boundary:
        raise ValueError("Topology codec calibration claim boundary drifted.")
    if report.get("tensor_contract_sha256") != CONTRACT_SHA256:
        raise ValueError("Topology codec calibration tensor contract drifted.")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_KEYS:
        raise ValueError("Topology codec calibration runtime census drifted.")
    if not isinstance(runtime["device"], str) or not runtime["device"]:
        raise ValueError("Topology codec calibration runtime device is malformed.")
    capability = runtime["compute_capability"]
    if not isinstance(capability, list) or len(capability) != 2 or any(type(value) is not int or value < 0 for value in capability):
        raise ValueError("Topology codec calibration compute capability is malformed.")
    if runtime["precision"] != "bf16-autocast-float32-loss":
        raise ValueError("Topology codec calibration precision drifted.")
    timings = [runtime[name] for name in ("training_seconds", "evaluation_seconds", "elapsed_seconds")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in timings):
        raise ValueError("Topology codec calibration runtime timing is malformed.")
    if runtime["elapsed_seconds"] < runtime["training_seconds"] + runtime["evaluation_seconds"]:
        raise ValueError("Topology codec calibration elapsed time is inconsistent.")
    peaks = [runtime[name] for name in ("peak_allocated_bytes", "peak_reserved_bytes")]
    if any(type(value) is not int or value < 0 for value in peaks) or peaks[1] < peaks[0]:
        raise ValueError("Topology codec calibration memory telemetry is malformed.")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != CHECKPOINT_REPORT_KEYS:
        raise ValueError("Topology codec calibration checkpoint descriptor drifted.")


def run_calibration(
    corpus_root: Path,
    output: Path,
    *,
    config: TopologyCodecCalibrationConfig = TopologyCodecCalibrationConfig(),
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Topology codec calibration output is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=512 * 1024 * 1024)
    dataset = TopologyProductionDataset(Path(corpus_root))
    validation_refs = dataset.evaluation_refs("validation", config.validation_samples)
    test_refs = dataset.evaluation_refs("test", config.test_samples)
    device = _configure_cuda(config.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    baseline_model = _fresh_model(config, device).eval()
    baseline = {
        "validation": evaluate_codec(baseline_model, dataset, validation_refs, device=device),
        "test": evaluate_codec(baseline_model, dataset, test_refs, device=device),
    }
    model = _fresh_model(config, device).train()
    initial_model_sha256 = tensor_state_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x4D4150545241494E)
    ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for step in range(config.steps):
        refs = dataset.training_refs(step, generator, config)
        batch = dataset.collate(refs, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_value = model(batch, update_ema=True)
            losses = balanced_reconstruction_loss(output_value, batch)
        if not bool(torch.isfinite(losses["total"])):
            raise FloatingPointError("Topology codec calibration produced a non-finite loss.")
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("Topology codec calibration produced a non-finite gradient.")
        optimizer.step()
        _ema_update(ema_state, model, config.model_ema_decay)
        history.append(
            {
                "step": step + 1,
                "shape": list(refs[0].shape),
                "batch_size": len(refs),
                "sample_registry_sha256": ref_registry_sha256(refs),
                "loss": {name: float(value.detach()) for name, value in losses.items()},
                "gradient_norm": float(gradient_norm.detach()),
                "codebook_perplexity": float(output_value["perplexity"].detach()),
                "codebook_utilization": float(output_value["utilization"].detach()),
            }
        )
    training_seconds = time.perf_counter() - training_started
    raw_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    ema_cpu = {name: value.detach().cpu() for name, value in ema_state.items()}
    raw_model = _model_from_state(config, raw_state, device)
    ema_model = _model_from_state(config, ema_cpu, device)
    evaluation_started = time.perf_counter()
    evaluation = {
        "validation": {
            "baseline": baseline["validation"],
            "raw": evaluate_codec(raw_model, dataset, validation_refs, device=device),
            "ema": evaluate_codec(ema_model, dataset, validation_refs, device=device),
        },
        "test": {
            "baseline": baseline["test"],
            "raw": evaluate_codec(raw_model, dataset, test_refs, device=device),
            "ema": evaluate_codec(ema_model, dataset, test_refs, device=device),
        },
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    quality = _quality(history, evaluation)
    final_model_sha256 = tensor_state_sha256(raw_state)
    final_ema_sha256 = tensor_state_sha256(ema_cpu)
    checkpoint_payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": production_source_sha256(),
        "source_manifest": production_source_manifest(),
        "tensor_contract_sha256": CONTRACT_SHA256,
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
        "dataset_registry_sha256": dataset.registry_sha256,
        "config": config.to_dict(),
        "step": config.steps,
        "model_state": raw_state,
        "ema_state": ema_cpu,
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "training_generator_state": generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_states": [value.cpu() for value in torch.cuda.get_rng_state_all()],
        "history": history,
        "evaluation": evaluation,
        "model_state_sha256": final_model_sha256,
        "ema_state_sha256": final_ema_sha256,
    }
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    checkpoint_path = staging / CHECKPOINT_NAME
    sidecar = save_checkpoint(checkpoint_path, checkpoint_payload)
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=0)
    safety_gates = {
        "finite_history": all(math.isfinite(float(item["loss"]["total"])) for item in history),
        "step_count_exact": len(history) == config.steps,
        "model_updated": final_model_sha256 != initial_model_sha256,
        "evaluation_census_exact": all(
            evaluation[split][mode]["sample_count"] == (config.validation_samples if split == "validation" else config.test_samples)
            for split in ("validation", "test") for mode in ("baseline", "raw", "ema")
        ),
        "dataset_registry_exact": dataset.registry_sha256 == checkpoint_payload["dataset_registry_sha256"],
        "checkpoint_inspected": True,
        "disk_floor_preserved": True,
        "representation_only": True,
        "production_schedule_disabled": True,
        "masked_latent_prior_not_started": True,
        "godot_integration_disabled": True,
    }
    report: dict[str, Any] = {
        "format": CALIBRATION_FORMAT,
        "status": "passed" if all(safety_gates.values()) else "failed",
        "source_sha256": production_source_sha256(),
        "tensor_contract_sha256": checkpoint_payload["tensor_contract_sha256"],
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
        "dataset_registry_sha256": dataset.registry_sha256,
        "config": config.to_dict(),
        "history": history,
        "evaluation": evaluation,
        "quality": quality,
        "initial_model_sha256": initial_model_sha256,
        "final_model_sha256": final_model_sha256,
        "final_ema_sha256": final_ema_sha256,
        "checkpoint": {**sidecar, "path": CHECKPOINT_NAME},
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "precision": "bf16-autocast-float32-loss",
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "safety_gates": safety_gates,
        "claim_boundary": {
            "representation_calibration_only": True,
            "quality_milestone_reached": quality["quality_milestone_reached"],
            "generative_prior_trained": False,
            "compiled_map_bank_published": False,
            "godot_integration": False,
        },
    }
    report["report_sha256"] = json_sha256(report)
    _validate_report_contract(report)
    _atomic_report(staging / REPORT_NAME, report)
    os.replace(staging, output)
    return validate_calibration(output, corpus_root=corpus_root, replay_metrics=False)


def _read_report(output: Path) -> dict[str, Any]:
    path = Path(output).resolve() / REPORT_NAME
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("Topology codec calibration report is missing or oversized.")
    report = json.loads(path.read_text(encoding="utf-8"))
    stored = report.pop("report_sha256", None)
    if stored != json_sha256(report):
        raise ValueError("Topology codec calibration report self-hash failed.")
    report["report_sha256"] = stored
    return report


def validate_calibration(
    output: Path,
    *,
    corpus_root: Path,
    replay_metrics: bool = True,
) -> dict[str, Any]:
    output = Path(output).resolve()
    report = _read_report(output)
    _validate_report_contract(report)
    if report.get("format") != CALIBRATION_FORMAT or report.get("status") != "passed":
        raise ValueError("Topology codec calibration status/format failed.")
    if report.get("source_sha256") != production_source_sha256():
        raise ValueError("Topology codec calibration source drifted.")
    config = TopologyCodecCalibrationConfig.from_dict(report["config"])
    dataset = TopologyProductionDataset(Path(corpus_root))
    if (
        report.get("corpus_sha256") != FROZEN_CORPUS_SHA256
        or report.get("corpus_manifest_file_sha256") != FROZEN_CORPUS_MANIFEST_FILE_SHA256
        or report.get("dataset_registry_sha256") != dataset.registry_sha256
    ):
        raise ValueError("Topology codec calibration authority drifted.")
    checkpoint_path = output / CHECKPOINT_NAME
    if report.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint_path):
        raise ValueError("Topology codec calibration checkpoint identity failed.")
    payload = load_checkpoint(checkpoint_path)
    sidecar = json.loads(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").read_text(encoding="utf-8"))
    if report["checkpoint"] != {**sidecar, "path": CHECKPOINT_NAME}:
        raise ValueError("Topology codec calibration checkpoint descriptor is not canonical.")
    initial_state = build_codec(config.codec_config(), init_seed=config.seed).state_dict()
    if report.get("initial_model_sha256") != tensor_state_sha256(initial_state):
        raise ValueError("Topology codec calibration initial model identity drifted.")
    if (
        payload["history"] != report["history"]
        or payload["evaluation"] != report["evaluation"]
        or payload["model_state_sha256"] != report["final_model_sha256"]
        or payload["ema_state_sha256"] != report["final_ema_sha256"]
    ):
        raise ValueError("Topology codec calibration checkpoint/report semantics drifted.")
    if report.get("quality") != _quality(report["history"], report["evaluation"]):
        raise ValueError("Topology codec calibration quality derivation drifted.")
    if not all(report.get("safety_gates", {}).values()):
        raise ValueError("Topology codec calibration recorded a failed safety gate.")
    if replay_metrics:
        device = _configure_cuda(config.seed)
        validation_refs = dataset.evaluation_refs("validation", config.validation_samples)
        test_refs = dataset.evaluation_refs("test", config.test_samples)
        baseline = _fresh_model(config, device).eval()
        raw = _model_from_state(config, payload["model_state"], device)
        ema = _model_from_state(config, payload["ema_state"], device)
        replay = {
            "validation": {
                "baseline": evaluate_codec(baseline, dataset, validation_refs, device=device),
                "raw": evaluate_codec(raw, dataset, validation_refs, device=device),
                "ema": evaluate_codec(ema, dataset, validation_refs, device=device),
            },
            "test": {
                "baseline": evaluate_codec(baseline, dataset, test_refs, device=device),
                "raw": evaluate_codec(raw, dataset, test_refs, device=device),
                "ema": evaluate_codec(ema, dataset, test_refs, device=device),
            },
        }
        if replay != report["evaluation"]:
            raise ValueError("Topology codec calibration metric replay failed.")
    return report
