from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Final
import uuid

import torch

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_ml.contract import ModelConfig
from ..map_decorator_production.training import CorpusSampleRef
from ..map_decorator_production_v2.contract import ForegroundPatchConfig
from ..map_decorator_production_v2.patches import foreground_centered_crop, plan_foreground_batches
from ..map_decorator_production_v2.training import WarmStartEMA
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from ..map_decorator_production_v4.contract import ProposalLocatorConfig
from ..map_decorator_production_v4.proposal import ProposalAuthority
from ..map_decorator_production_v4_training.checkpoint import (
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
    tensor_state_sha256,
    training_source_sha256,
)
from ..map_decorator_production_v4_training.dataset import (
    ProposalTeacherSample,
    collate_proposal_samples,
    proposals_for_teacher,
)
from ..map_decorator_production_v4_training.training import make_optimizer, train_batch
from ..map_decorator_production_v4_training.contract import ResidualLossConfig, ResidualTrainingConfig
from ..safety import require_disk_floor
from .contract import (
    CALIBRATION_FORMAT,
    SUPERVISOR_FORMAT,
    CalibrationConfig,
    V4_CALIBRATION_CONTRACT_SHA256,
)
from .evaluation import compare_to_baseline, evaluate_full_split


REPORT_NAME: Final[str] = "calibration_report.json"
SUPERVISOR_REPORT_NAME: Final[str] = "supervisor_report.json"
MAX_REPORT_BYTES: Final[int] = 64 * 1024 * 1024


def calibration_source_manifest(root: Path = PROJECT_ROOT) -> dict[str, object]:
    package = Path(root) / "forge/map_decorator_production_v4_calibration"
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in package.glob("*.py") if item.is_file())
    }
    return {"training_source_sha256": training_source_sha256(root), "calibration_files": files}


def calibration_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return json_sha256(calibration_source_manifest(root))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"V4 calibration JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _read_report(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("V4 calibration report is missing, unsafe, or oversized.")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    stored = value.pop("report_sha256", None)
    if stored != json_sha256(value):
        raise ValueError("V4 calibration report self-hash failed.")
    value["report_sha256"] = stored
    return value


def _configure_cuda(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before CUDA startup.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("V4 calibration requires CUDA with BF16 support.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def _authority_record(authority: ProposalAuthority, corpus_root: Path, index_root: Path) -> dict[str, object]:
    inner = authority.authority
    return {
        "corpus_root": Path(corpus_root).resolve().name,
        "corpus_sha256": inner.corpus.corpus_sha256,
        "corpus_manifest_sha256": inner.corpus.manifest_sha256,
        "index_root": Path(index_root).resolve().name,
        "index_semantic_sha256": inner.index_semantic_sha256,
        "index_manifest_sha256": inner.index_manifest_sha256,
        "split_counts": {name: len(inner.corpus.refs_by_split[name]) for name in ("train", "validation", "test")},
    }


def _evaluate_pair(
    model: ProposalConditionedDecoratorV4,
    authority: ProposalAuthority,
    config: CalibrationConfig,
    device: torch.device,
) -> dict[str, object]:
    return {
        "validation": evaluate_full_split(
            model,
            authority,
            "validation",
            batch_size=config.validation_batch_size,
            device=device,
            seed=config.training.seed,
        ),
        "test": evaluate_full_split(
            model,
            authority,
            "test",
            batch_size=config.test_batch_size,
            device=device,
            seed=config.training.seed,
        ),
    }


def _copy_ema(
    model: ProposalConditionedDecoratorV4,
    ema: WarmStartEMA,
    device: torch.device,
) -> ProposalConditionedDecoratorV4:
    target = ProposalConditionedDecoratorV4(model.core_config, model.locator_config)
    target.load_state_dict(model.state_dict(), strict=True)
    ema.copy_to(target)
    return target.to(device)


def _ref_from_stat(stat) -> CorpusSampleRef:
    return CorpusSampleRef(
        shard_index=stat.shard_index,
        sample_index=stat.sample_index,
        split=stat.split,
        map_id=stat.map_id,
        sample_identity_sha256=stat.sample_identity_sha256,
        full_map_identity_sha256=stat.full_map_identity_sha256,
    )


def _train(
    model: ProposalConditionedDecoratorV4,
    optimizer: torch.optim.Optimizer,
    ema: WarmStartEMA,
    authority: ProposalAuthority,
    config: CalibrationConfig,
    generator: torch.Generator,
    device: torch.device,
) -> list[dict[str, object]]:
    plan = plan_foreground_batches(
        authority.authority.stats["train"],
        steps=config.steps,
        epoch=0,
        seed=config.training.seed,
        config=config.patch,
    )
    history: list[dict[str, object]] = []
    for step, planned in enumerate(plan):
        samples: list[ProposalTeacherSample] = []
        source_ids: list[str] = []
        focus_heads: list[str] = []
        for slot, item in enumerate(planned):
            full = authority.authority.sample_for_stat(item.stat)
            crop = foreground_centered_crop(
                full,
                focus_head=item.focus_head,
                epoch=0,
                step=step,
                slot=slot,
                seed=config.training.seed,
                config=config.patch,
            )
            proposal = proposals_for_teacher(authority, _ref_from_stat(item.stat), crop)
            samples.append(ProposalTeacherSample(crop, proposal))
            source_ids.append(full.sample_identity_sha256)
            focus_heads.append(item.focus_head)
        result = train_batch(
            model,
            optimizer,
            ema,
            collate_proposal_samples(samples),
            generator=generator,
            training_config=config.training,
            loss_config=config.loss,
            device=device,
            autocast_dtype=torch.bfloat16,
        )
        loss = result["loss"]
        record = {
            "step": step + 1,
            "source_sample_sha256": source_ids,
            "focus_heads": focus_heads,
            "total_loss": float(loss["total"]),
            "gradient_norm": float(result["gradient_norm"]),
            "decal_presence_loss": float(loss["decal_presence"]),
            "prop_presence_loss": float(loss["prop_presence"]),
            "decal_count_loss": float(loss["decal_count"]),
            "prop_count_loss": float(loss["prop_count"]),
            "full_mask_sample_count": int(result["full_mask_sample_count"]),
        }
        if any(not math.isfinite(float(record[key])) for key in (
            "total_loss", "gradient_norm", "decal_presence_loss", "prop_presence_loss",
            "decal_count_loss", "prop_count_loss",
        )):
            raise FloatingPointError("V4 calibration produced non-finite training evidence.")
        history.append(record)
    return history


def _config_from_dict(value: object) -> CalibrationConfig:
    if not isinstance(value, dict):
        raise ValueError("V4 calibration configuration is malformed.")
    copy = dict(value)
    if copy.pop("precision", None) != "bf16":
        raise ValueError("V4 calibration precision drifted.")
    config = CalibrationConfig(
        steps=int(copy["steps"]),
        validation_batch_size=int(copy["validation_batch_size"]),
        test_batch_size=int(copy["test_batch_size"]),
        core=ModelConfig(**copy["core"]),
        locator=ProposalLocatorConfig(**copy["locator"]),
        training=ResidualTrainingConfig(**copy["training"]),
        loss=ResidualLossConfig(**copy["loss"]),
        patch=ForegroundPatchConfig(**copy["patch"]),
    )
    if config.to_dict() != value:
        raise ValueError("V4 calibration configuration is noncanonical.")
    return config


def run_worker(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    config: CalibrationConfig = CalibrationConfig(),
) -> dict[str, object]:
    output = Path(output).resolve()
    if (output / REPORT_NAME).is_file():
        return validate_calibration(output, corpus_root=corpus_root, index_root=index_root)
    if output.exists():
        raise FileExistsError("V4 calibration output exists without a complete report.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=2 * 1024**3)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    device = _configure_cuda(config.training.seed)
    torch.cuda.reset_peak_memory_stats(device)
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    model = ProposalConditionedDecoratorV4(config.core, config.locator).to(device)
    initial_model_sha = tensor_state_sha256(model.state_dict())
    started = time.perf_counter()
    baseline_started = time.perf_counter()
    baseline = _evaluate_pair(model, authority, config, device)
    baseline_seconds = time.perf_counter() - baseline_started
    optimizer = make_optimizer(model, config.training)
    ema = WarmStartEMA(model, config.training.ema_decay)
    generator = torch.Generator(device=device).manual_seed(config.training.seed)
    training_started = time.perf_counter()
    history = _train(model, optimizer, ema, authority, config, generator, device)
    training_seconds = time.perf_counter() - training_started
    evaluation_started = time.perf_counter()
    raw = _evaluate_pair(model, authority, config, device)
    ema_model = _copy_ema(model, ema, device)
    ema_evaluation = _evaluate_pair(ema_model, authority, config, device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    raw_comparison = compare_to_baseline(baseline, raw)
    ema_comparison = compare_to_baseline(baseline, ema_evaluation)
    quality_passed = bool(raw_comparison["passed"] and ema_comparison["passed"])
    model_sha = tensor_state_sha256(model.state_dict())
    ema_sha = tensor_state_sha256(ema.shadow)
    metrics = {
        "history": history,
        "baseline_evaluation": baseline,
        "raw_evaluation": raw,
        "ema_evaluation": ema_evaluation,
        "raw_comparison": raw_comparison,
        "ema_comparison": ema_comparison,
        "quality_passed": quality_passed,
    }
    checkpoint_name = f"checkpoint_step_{config.steps:04d}.pt"
    checkpoint_path = staging / checkpoint_name
    sidecar = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        ema,
        generator,
        core_config=config.core,
        locator_config=config.locator,
        training_config=config.training,
        loss_config=config.loss,
        global_step=config.steps,
        corpus_sha256=authority.authority.corpus.corpus_sha256,
        index_semantic_sha256=authority.authority.index_semantic_sha256,
        metrics=metrics,
    )
    reload_model = ProposalConditionedDecoratorV4(config.core, config.locator).to(device)
    reload_optimizer = make_optimizer(reload_model, config.training)
    reload_ema = WarmStartEMA(reload_model, config.training.ema_decay)
    reload_generator = torch.Generator(device=device).manual_seed(0)
    loaded = load_checkpoint(
        checkpoint_path,
        reload_model,
        reload_optimizer,
        reload_ema,
        reload_generator,
        expected_step=config.steps,
        expected_corpus_sha256=authority.authority.corpus.corpus_sha256,
        expected_index_semantic_sha256=authority.authority.index_semantic_sha256,
        expected_training_config=config.training,
        expected_loss_config=config.loss,
    )
    reload_exact = (
        tensor_state_sha256(reload_model.state_dict()) == model_sha
        and tensor_state_sha256(reload_ema.shadow) == ema_sha
        and loaded["metrics"] == metrics
    )
    safety = all(
        collection[split]["hard_legality"] == 1.0
        and collection[split]["full_split"] is True
        for collection in (baseline, raw, ema_evaluation)
        for split in ("validation", "test")
    )
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "format": CALIBRATION_FORMAT,
        "status": "quality_passed" if quality_passed else "quality_failed",
        "calibration_contract_sha256": V4_CALIBRATION_CONTRACT_SHA256,
        "calibration_source_sha256": calibration_source_sha256(),
        "training_source_sha256": training_source_sha256(),
        "authority": _authority_record(authority, Path(corpus_root), Path(index_root)),
        "config": config.to_dict(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "precision": "bf16",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "elapsed_seconds": elapsed,
            "baseline_seconds": baseline_seconds,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "steps_per_second": config.steps / training_seconds,
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "initial_model_sha256": initial_model_sha,
        "model_tensor_sha256": model_sha,
        "ema_tensor_sha256": ema_sha,
        **metrics,
        "checkpoint": {
            "path": checkpoint_name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
            "sidecar_sha256": sidecar["sidecar_sha256"],
        },
        "gates": {
            "full_baseline_raw_and_ema_splits": safety,
            "finite_training": all(math.isfinite(float(item["total_loss"])) for item in history),
            "checkpoint_reload_exact": reload_exact,
            "raw_hard_safety": bool(raw_comparison["hard_safety"]),
            "ema_hard_safety": bool(ema_comparison["hard_safety"]),
            "cuda_bf16": str(device) == "cuda:0" and torch.cuda.is_bf16_supported(),
            "quality_separate_from_safety": True,
            "no_runtime_integration": True,
        },
    }
    if not all(report["gates"].values()):  # type: ignore[union-attr]
        raise RuntimeError(f"V4 calibration hard publication gate failed: {report['gates']}")
    report["semantic_sha256"] = json_sha256({key: value for key, value in report.items() if key != "runtime"})
    report["report_sha256"] = json_sha256(report)
    _atomic_json(staging / REPORT_NAME, report)
    os.replace(staging, output)
    return validate_calibration(output, corpus_root=corpus_root, index_root=index_root)


def validate_calibration(output: Path, *, corpus_root: Path, index_root: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    value = _read_report(output / REPORT_NAME)
    if value.get("format") != CALIBRATION_FORMAT or value.get("calibration_contract_sha256") != V4_CALIBRATION_CONTRACT_SHA256:
        raise ValueError("V4 calibration format or contract drifted.")
    if value.get("calibration_source_sha256") != calibration_source_sha256() or value.get("training_source_sha256") != training_source_sha256():
        raise ValueError("V4 calibration source provenance drifted.")
    config = _config_from_dict(value.get("config"))
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    if value.get("authority") != _authority_record(authority, Path(corpus_root), Path(index_root)):
        raise ValueError("V4 calibration authority drifted.")
    if len(value.get("history", [])) != config.steps:
        raise ValueError("V4 calibration history length drifted.")
    for index, record in enumerate(value["history"], start=1):
        if record.get("step") != index or not math.isfinite(float(record.get("total_loss", math.nan))):
            raise ValueError("V4 calibration history is malformed.")
    runtime = value.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("device") != "cuda:0"
        or runtime.get("precision") != "bf16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise ValueError("V4 calibration runtime contract failed.")
    for name in ("elapsed_seconds", "baseline_seconds", "training_seconds", "evaluation_seconds", "steps_per_second"):
        if not math.isfinite(float(runtime.get(name, math.nan))) or float(runtime[name]) <= 0:
            raise ValueError("V4 calibration runtime timing evidence is malformed.")
    expected_raw = compare_to_baseline(value["baseline_evaluation"], value["raw_evaluation"])
    expected_ema = compare_to_baseline(value["baseline_evaluation"], value["ema_evaluation"])
    if value.get("raw_comparison") != expected_raw or value.get("ema_comparison") != expected_ema:
        raise ValueError("V4 calibration baseline comparison replay failed.")
    expected_quality = bool(expected_raw["passed"] and expected_ema["passed"])
    if value.get("quality_passed") is not expected_quality or value.get("status") != ("quality_passed" if expected_quality else "quality_failed"):
        raise ValueError("V4 calibration quality status is inconsistent.")
    checkpoint = value.get("checkpoint")
    expected_name = f"checkpoint_step_{config.steps:04d}.pt"
    if not isinstance(checkpoint, dict) or checkpoint.get("path") != expected_name:
        raise ValueError("V4 calibration checkpoint record is malformed.")
    checkpoint_path = output / expected_name
    if checkpoint_path.stat().st_size != checkpoint.get("bytes") or file_sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError("V4 calibration checkpoint artifact identity failed.")
    payload = inspect_checkpoint(checkpoint_path)
    if (
        payload.get("global_step") != config.steps
        or payload.get("corpus_sha256") != authority.authority.corpus.corpus_sha256
        or payload.get("index_semantic_sha256") != authority.authority.index_semantic_sha256
        or payload.get("core_config") != config.core.to_dict()
        or payload.get("locator_config") != config.locator.to_dict()
        or payload.get("training_config") != config.training.to_dict()
        or payload.get("loss_config") != config.loss.to_dict()
    ):
        raise ValueError("V4 calibration checkpoint authority or configuration drifted.")
    metrics = {
        "history": value["history"],
        "baseline_evaluation": value["baseline_evaluation"],
        "raw_evaluation": value["raw_evaluation"],
        "ema_evaluation": value["ema_evaluation"],
        "raw_comparison": value["raw_comparison"],
        "ema_comparison": value["ema_comparison"],
        "quality_passed": value["quality_passed"],
    }
    if payload.get("metrics") != metrics:
        raise ValueError("V4 calibration report and checkpoint metrics differ.")
    if payload.get("model_tensor_sha256") != value.get("model_tensor_sha256") or payload.get("ema_tensor_sha256") != value.get("ema_tensor_sha256"):
        raise ValueError("V4 calibration checkpoint tensor identity drifted.")
    sidecar = json.loads(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").read_text(encoding="utf-8"))
    if sidecar.get("sidecar_sha256") != checkpoint.get("sidecar_sha256"):
        raise ValueError("V4 calibration checkpoint sidecar identity drifted.")
    semantic = json_sha256({key: item for key, item in value.items() if key not in {"runtime", "report_sha256", "semantic_sha256"}})
    if semantic != value.get("semantic_sha256"):
        raise ValueError("V4 calibration semantic hash failed.")
    expected_gates = {
        "full_baseline_raw_and_ema_splits": all(
            collection[split].get("hard_legality") == 1.0 and collection[split].get("full_split") is True
            for collection in (value["baseline_evaluation"], value["raw_evaluation"], value["ema_evaluation"])
            for split in ("validation", "test")
        ),
        "finite_training": all(math.isfinite(float(item["total_loss"])) for item in value["history"]),
        "checkpoint_reload_exact": True,
        "raw_hard_safety": bool(expected_raw["hard_safety"]),
        "ema_hard_safety": bool(expected_ema["hard_safety"]),
        "cuda_bf16": True,
        "quality_separate_from_safety": True,
        "no_runtime_integration": True,
    }
    if value.get("gates") != expected_gates or not all(expected_gates.values()):
        raise ValueError("V4 calibration recorded a failed hard gate.")
    return value


def supervise(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    config: CalibrationConfig = CalibrationConfig(),
    max_attempts: int = 3,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("V4 supervised calibration output is immutable.")
    if not 1 <= max_attempts <= 3 or not 60 <= timeout_seconds <= 7200:
        raise ValueError("V4 calibration supervisor bounds are invalid.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=3 * 1024**3)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, object]] = []
    successful: Path | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_root = staging / f"attempt_{attempt:02d}"
        attempt_root.mkdir()
        worker_output = attempt_root / "calibration"
        command = [
            sys.executable, "-m", "forge.map_decorator_production_v4_calibration", "worker",
            "--corpus", str(Path(corpus_root).resolve()), "--index", str(Path(index_root).resolve()),
            "--output", str(worker_output), "--steps", str(config.steps),
            "--validation-batch-size", str(config.validation_batch_size),
            "--test-batch-size", str(config.test_batch_size),
            "--base-channels", str(config.core.base_channels),
        ]
        environment = dict(os.environ)
        environment.update({
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, env=environment)
            returncode = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            returncode, stdout, stderr = -1, exc.stdout or "", exc.stderr or "timeout"
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        (attempt_root / "stdout.txt").write_text(stdout, encoding="utf-8")
        (attempt_root / "stderr.txt").write_text(stderr, encoding="utf-8")
        record = {
            "attempt": attempt,
            "returncode": returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "windows_access_violation": returncode in {3221225477, -1073741819},
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        }
        attempts.append(record)
        if returncode == 0:
            validate_calibration(worker_output, corpus_root=corpus_root, index_root=index_root)
            successful = worker_output
            break
    if successful is None:
        failure = {"format": SUPERVISOR_FORMAT, "status": "failed", "attempts": attempts}
        failure["report_sha256"] = json_sha256(failure)
        _atomic_json(staging / SUPERVISOR_REPORT_NAME, failure)
        raise RuntimeError("V4 calibration exhausted all isolated attempts.")
    final = staging / "calibration"
    os.replace(successful, final)
    supervisor: dict[str, object] = {
        "format": SUPERVISOR_FORMAT,
        "status": "passed",
        "calibration_contract_sha256": V4_CALIBRATION_CONTRACT_SHA256,
        "calibration_source_sha256": calibration_source_sha256(),
        "attempts": attempts,
        "successful_attempt": attempts[-1]["attempt"],
        "calibration_report_sha256": file_sha256(final / REPORT_NAME),
    }
    supervisor["report_sha256"] = json_sha256(supervisor)
    _atomic_json(staging / SUPERVISOR_REPORT_NAME, supervisor)
    os.replace(staging, output)
    return validate_supervised(output, corpus_root=corpus_root, index_root=index_root)


def validate_supervised(output: Path, *, corpus_root: Path, index_root: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    supervisor = _read_report(output / SUPERVISOR_REPORT_NAME)
    if supervisor.get("format") != SUPERVISOR_FORMAT or supervisor.get("status") != "passed":
        raise ValueError("V4 calibration supervisor report failed.")
    if supervisor.get("calibration_source_sha256") != calibration_source_sha256():
        raise ValueError("V4 calibration supervisor source drifted.")
    calibration = validate_calibration(output / "calibration", corpus_root=corpus_root, index_root=index_root)
    if supervisor.get("calibration_report_sha256") != file_sha256(output / "calibration" / REPORT_NAME):
        raise ValueError("V4 supervisor did not bind its calibration report.")
    return {"supervisor": supervisor, "calibration": calibration}
