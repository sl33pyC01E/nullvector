from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..cellular_motion.contract import MOTION_SPECS
from ..config import PROJECT_ROOT
from ..multifield_style_motion.hashing import canonical_json_bytes
from ..safety import require_disk_floor
from .contract import DEFAULT_CORPUS, MODEL_FORMAT, NeuralCellMotionConfig, model_source_sha256
from .dataset import _read_canonical_json, load_corpus_manifest, sha256_bytes, sha256_file
from .model import NeuralCellMotionUNet, neural_motion_loss
from .training import _atomic_bytes, _atomic_torch, _state_sha256


PRODUCTION_FORMAT = "nullvector-neural-cell-motion-production-v1"
CHECKPOINT_FORMAT = "nullvector-neural-cell-motion-production-checkpoint-v1"
TELEMETRY_FORMAT = "nullvector-neural-cell-motion-production-telemetry-v1"
CONTRACT_NAME = "production_training_contract.json"
TELEMETRY_NAME = "production_training_telemetry.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/neural_cell_motion/production_v1"
SEED = 0x4E434D4F54494F4E
ACCESS_VIOLATION_CODES = {0xC0000005, -1073741819, 0xC0000409, -1073740791}
MAX_CHECKPOINT_BYTES = 2 * 1024**3
ATTEMPT_KEYS = {
    "end_step", "attempt", "returncode", "seconds", "stdout_tail", "stderr_tail",
    "access_violation", "timed_out", "artifact_valid", "validation_error",
}


def _semantic(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _telemetry_payload(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": TELEMETRY_FORMAT,
        "source_sha256": model_source_sha256(),
        "attempts": attempts,
    }
    payload["semantic_sha256"] = _semantic(payload)
    return payload


def _validate_attempts(attempts: Any) -> list[dict[str, Any]]:
    if not isinstance(attempts, list):
        raise ValueError("Neural motion telemetry attempts drifted.")
    seen: set[tuple[int, int]] = set()
    successful: set[int] = set()
    prior_end_step = 0
    for row in attempts:
        if not isinstance(row, dict) or set(row) != ATTEMPT_KEYS:
            raise ValueError("Neural motion telemetry row drifted.")
        end_step, attempt = row["end_step"], row["attempt"]
        if type(end_step) is not int or end_step <= 0 or type(attempt) is not int or not 1 <= attempt <= 3:
            raise ValueError("Neural motion telemetry coordinate drifted.")
        if end_step < prior_end_step or (end_step, attempt) in seen or end_step in successful:
            raise ValueError("Neural motion telemetry ordering drifted.")
        prior = [item for item in seen if item[0] == end_step]
        if attempt != len(prior) + 1:
            raise ValueError("Neural motion telemetry attempt sequence drifted.")
        if type(row["returncode"]) is not int or not isinstance(row["seconds"], (int, float)) or isinstance(row["seconds"], bool) or not math.isfinite(float(row["seconds"])) or row["seconds"] < 0:
            raise ValueError("Neural motion telemetry runtime drifted.")
        if not all(isinstance(row[key], str) and len(row[key]) <= (4000 if key != "stdout_tail" else 2000) for key in ("stdout_tail", "stderr_tail", "validation_error")):
            raise ValueError("Neural motion telemetry diagnostic drifted.")
        if not all(type(row[key]) is bool for key in ("access_violation", "timed_out", "artifact_valid")):
            raise ValueError("Neural motion telemetry flags drifted.")
        if row["access_violation"] != (row["returncode"] in ACCESS_VIOLATION_CODES):
            raise ValueError("Neural motion access-violation telemetry drifted.")
        if row["timed_out"] and row["returncode"] != -1:
            raise ValueError("Neural motion timeout telemetry drifted.")
        if row["artifact_valid"] and (row["returncode"] != 0 or row["validation_error"]):
            raise ValueError("Neural motion successful-attempt telemetry drifted.")
        if not row["artifact_valid"] and row["returncode"] == 0 and not row["validation_error"]:
            raise ValueError("Neural motion failed-attempt evidence drifted.")
        seen.add((end_step, attempt))
        if row["artifact_valid"]:
            successful.add(end_step)
    return attempts


def _config_from_dict(payload: dict[str, Any]) -> NeuralCellMotionConfig:
    return NeuralCellMotionConfig(**{**payload, "channel_multipliers": tuple(payload["channel_multipliers"])})


def _replay_authority(corpus: Path) -> dict[str, Any]:
    path = corpus.parent / f"{corpus.name}_validation_telemetry.json"
    report = _read_canonical_json(path, maximum_bytes=16 * 1024 * 1024)
    required = {"format", "status", "source_sha256", "corpus_semantic_sha256", "replay", "identity_count", "validated_identity_count", "attempt_count", "retry_count", "native_failure_count", "timeout_count", "events", "semantic_sha256"}
    manifest = load_corpus_manifest(corpus)
    if set(report) != required or report["format"] != "nullvector-neural-cell-motion-resilient-validation-v1" or report["status"] != "passed" or report["source_sha256"] != manifest["source_sha256"] or report["corpus_semantic_sha256"] != manifest["semantic_sha256"] or report["replay"] is not True or report["identity_count"] != 45 or report["validated_identity_count"] != 45 or report["semantic_sha256"] != _semantic({key: value for key, value in report.items() if key != "semantic_sha256"}):
        raise ValueError("Neural motion production requires the exact 45-identity replay authority.")
    successful = [event["sample_id"] for event in report["events"] if event.get("passed")]
    expected = [record["sample_id"] for record in manifest["records"]]
    if sorted(successful) != sorted(expected) or len(successful) != len(set(successful)):
        raise ValueError("Neural motion replay authority coverage drifted.")
    return report


def prepare_production(
    output: Path = DEFAULT_OUTPUT, *, corpus: Path = DEFAULT_CORPUS,
    total_steps: int = 12_000, segment_steps: int = 500, batch_size: int = 10,
    max_attempts: int = 3,
) -> dict[str, Any]:
    output, corpus = Path(output).resolve(), Path(corpus).resolve()
    if type(total_steps) is not int or not 100 <= total_steps <= 100_000 or type(segment_steps) is not int or not 100 <= segment_steps <= 2_000 or total_steps % segment_steps:
        raise ValueError("Neural motion production schedule drifted.")
    if type(batch_size) is not int or not 5 <= batch_size <= 40 or batch_size % 5 or type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("Neural motion production batch/retry policy drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=12 * 1024**3)
    manifest = load_corpus_manifest(corpus); replay = _replay_authority(corpus)
    if manifest["scope"]["production_complete"] is not True or manifest["scope"]["split_counts"] != {"train": 35, "validation": 5, "test": 5, "smoke": 0}:
        raise ValueError("Neural motion production corpus split drifted.")
    contract: dict[str, Any] = {
        "format": PRODUCTION_FORMAT, "source_sha256": model_source_sha256(), "seed": SEED,
        "corpus": {"path": corpus.relative_to(PROJECT_ROOT).as_posix(), "manifest_sha256": sha256_file(corpus / "neural_cell_motion_corpus.json"), "semantic_sha256": manifest["semantic_sha256"], "replay_telemetry_sha256": sha256_file(corpus.parent / f"{corpus.name}_validation_telemetry.json"), "replay_semantic_sha256": replay["semantic_sha256"]},
        "model": NeuralCellMotionConfig().to_dict(), "total_steps": total_steps, "segment_steps": segment_steps, "batch_size": batch_size,
        "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-5, "gradient_clip": 1.0, "warmup_steps": min(500, total_steps // 5)},
        "ema_decay": .9995, "precision": "bf16-autocast-float32-loss", "family_balanced": True,
        "cache_shards": 12, "minimum_free_vram_bytes": 14 * 1024**3,
        "supervisor": {"max_attempts_per_segment": max_attempts, "segment_timeout_seconds": 1800},
    }
    contract["semantic_sha256"] = _semantic(contract)
    output.mkdir(parents=True, exist_ok=True); path = output / CONTRACT_NAME
    if path.exists():
        if _read_canonical_json(path) != contract: raise ValueError("Neural motion production contract changed during resume.")
    else: _atomic_bytes(path, canonical_json_bytes(contract))
    return contract


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


class MotionBatchSampler:
    def __init__(self, corpus: Path, *, batch_size: int, seed: int = SEED, cache_shards: int = 12) -> None:
        self.corpus = Path(corpus).resolve(); self.manifest = load_corpus_manifest(self.corpus); self.batch_size = batch_size; self.seed = seed; self.cache_shards = cache_shards
        if batch_size % 5 or not 5 <= batch_size <= 40 or not 5 <= cache_shards <= 20: raise ValueError("Neural motion sampler policy drifted.")
        self.by_family = {family: [record for record in self.manifest["records"] if record["split"] == "train" and record["family_id"] == family] for family in range(5)}
        if [len(self.by_family[family]) for family in range(5)] != [9, 8, 7, 6, 5]: raise ValueError("Neural motion train family census drifted.")
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def _shard(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        sample_id = record["sample_id"]
        if sample_id in self.cache:
            self.cache.move_to_end(sample_id); return self.cache[sample_id]
        with np.load(self.corpus / record["path"], allow_pickle=False) as archive:
            shard = {name: archive[name] for name in ("features", "targets", "indices", "previous_index")}
        self.cache[sample_id] = shard
        if len(self.cache) > self.cache_shards: self.cache.popitem(last=False)
        return shard

    def coordinates(self, step: int) -> list[tuple[int, int, int]]:
        if type(step) is not int or step < 0: raise ValueError("Neural motion sampler step drifted.")
        result: list[tuple[int, int, int]] = []
        for slot in range(self.batch_size):
            family = slot % 5; token = _mix64(self.seed ^ (step * 0xD1342543DE82EF95) ^ (slot * 0xA24BAED4963EE407))
            identity = token % len(self.by_family[family]); frame = _mix64(token ^ 0xC6BC279692B5CC83) % 944; result.append((family, int(identity), int(frame)))
        return result

    def batch(self, step: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        static_rows: list[np.ndarray] = []; previous_rows: list[np.ndarray] = []; target_rows: list[np.ndarray] = []; families: list[int] = []; motions: list[int] = []; facings: list[int] = []; phases: list[float] = []
        for family, identity, frame in self.coordinates(step):
            record = self.by_family[family][identity]; shard = self._shard(record); metadata = shard["indices"][frame]; previous = int(shard["previous_index"][frame])
            static_rows.append(shard["features"]); previous_rows.append(shard["targets"][previous]); target_rows.append(shard["targets"][frame]); motion, facing, frame_index = map(int, metadata[:3]); frames = MOTION_SPECS[list(MOTION_SPECS)[motion]][0]
            families.append(family); motions.append(motion); facings.append(facing); phases.append(frame_index / max(1, frames - 1))
        tensors = (np.stack(static_rows), np.stack(previous_rows), np.stack(target_rows), np.asarray(families, dtype=np.int64), np.asarray(motions, dtype=np.int64), np.asarray(facings, dtype=np.int64), np.asarray(phases, dtype=np.float32))
        return tuple(torch.from_numpy(value.astype(np.float32) if index < 3 else value).to(device, non_blocking=True) for index, value in enumerate(tensors))  # type: ignore[return-value]


def checkpoint_name(step: int) -> str:
    return f"motion_segment_{step:07d}.pt"


def _load_checkpoint(path: Path, contract: dict[str, Any], *, expected_step: int | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES: raise ValueError("Neural motion checkpoint missing or oversized.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "source_sha256", "contract_semantic_sha256", "step", "model_state", "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256", "cpu_rng_state", "cuda_rng_state", "history", "runtime"}
    if not isinstance(payload, dict) or set(payload) != required or payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != model_source_sha256() or payload["contract_semantic_sha256"] != contract["semantic_sha256"]:
        raise ValueError("Neural motion checkpoint provenance drifted.")
    if type(payload["step"]) is not int or not 1 <= payload["step"] <= contract["total_steps"] or expected_step is not None and payload["step"] != expected_step or not isinstance(payload["history"], list) or len(payload["history"]) != payload["step"]:
        raise ValueError("Neural motion checkpoint census drifted.")
    model = NeuralCellMotionUNet(_config_from_dict(contract["model"])); model.load_state_dict(payload["model_state"], strict=True)
    if _state_sha256(model.state_dict()) != payload["model_state_sha256"]: raise ValueError("Neural motion model state drifted.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_sha256(model.state_dict()) != payload["ema_state_sha256"]: raise ValueError("Neural motion EMA state drifted.")
    if not isinstance(payload["cpu_rng_state"], Tensor) or payload["cpu_rng_state"].dtype != torch.uint8 or not isinstance(payload["cuda_rng_state"], Tensor) or payload["cuda_rng_state"].dtype != torch.uint8:
        raise ValueError("Neural motion checkpoint RNG drifted.")
    for index, row in enumerate(payload["history"], 1):
        if set(row) != {"step", "loss", "displacement", "activation", "emission", "coherence", "temporal", "outside", "gradient_norm", "lr"} or row["step"] != index or any(not math.isfinite(float(value)) for key, value in row.items() if key != "step"):
            raise ValueError("Neural motion checkpoint history drifted.")
    return payload


def _checkpoint_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": True,
        "step": payload["step"],
        "model_state_sha256": payload["model_state_sha256"],
        "ema_state_sha256": payload["ema_state_sha256"],
        "runtime": payload["runtime"],
    }


def train_segment(output: Path, *, end_step: int) -> dict[str, Any]:
    output = Path(output).resolve(); contract = _read_canonical_json(output / CONTRACT_NAME)
    if contract.get("source_sha256") != model_source_sha256() or contract.get("semantic_sha256") != _semantic({key: value for key, value in contract.items() if key != "semantic_sha256"}): raise ValueError("Neural motion training authority drifted.")
    segment_steps = contract["segment_steps"]
    if type(end_step) is not int or end_step % segment_steps or not segment_steps <= end_step <= contract["total_steps"]: raise ValueError("Neural motion segment endpoint drifted.")
    destination = output / checkpoint_name(end_step)
    if destination.exists(): return _checkpoint_summary(_load_checkpoint(destination, contract, expected_step=end_step))
    previous_step = end_step - segment_steps; previous_path = output / checkpoint_name(previous_step) if previous_step else None
    if previous_path is not None and not previous_path.exists(): raise FileNotFoundError("Previous neural motion segment is missing.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(0)[0] < contract["minimum_free_vram_bytes"]:
        raise RuntimeError("Neural motion training requires deterministic CUDA BF16 and its declared free-VRAM window.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(contract["seed"]); torch.cuda.manual_seed_all(contract["seed"]); np.random.seed(contract["seed"] & 0xFFFFFFFF)
    device = torch.device("cuda", 0); torch.cuda.reset_peak_memory_stats(device); config = _config_from_dict(contract["model"]); model = NeuralCellMotionUNet(config).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=contract["optimizer"]["lr"], weight_decay=contract["optimizer"]["weight_decay"]); ema = {name: value.detach().clone() for name, value in model.state_dict().items()}; history: list[dict[str, Any]] = []; start_step = 0
    if previous_path is not None:
        previous = _load_checkpoint(previous_path, contract, expected_step=previous_step); start_step = previous["step"]; model.load_state_dict(previous["model_state"], strict=True); optimizer.load_state_dict(previous["optimizer_state"]); ema = {name: value.to(device) for name, value in previous["ema_state"].items()}; torch.set_rng_state(previous["cpu_rng_state"]); torch.cuda.set_rng_state(previous["cuda_rng_state"], device); history = list(previous["history"])
    corpus = PROJECT_ROOT / contract["corpus"]["path"]; sampler = MotionBatchSampler(corpus, batch_size=contract["batch_size"], seed=contract["seed"], cache_shards=contract["cache_shards"]); started = time.perf_counter(); model.train()
    for step in range(start_step, end_step):
        static, previous_state, target, family, motion, facing, phase = sampler.batch(step, device)
        warmup = contract["optimizer"]["warmup_steps"]; lr_scale = min(1.0, (step + 1) / max(1, warmup)); lr = contract["optimizer"]["lr"] * lr_scale
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): predicted = model(static, previous_state, family, motion, facing, phase)
        loss, pieces = neural_motion_loss(predicted.float(), target.float(), previous_state.float(), static.float()); loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["optimizer"]["gradient_clip"])
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)): raise FloatingPointError("Neural motion training became non-finite.")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point: ema[name].lerp_(value.detach(), 1 - contract["ema_decay"])
                else: ema[name].copy_(value)
        history.append({"step": step + 1, **{name: round(float(value), 8) for name, value in pieces.items()}, "gradient_norm": round(float(gradient), 8), "lr": round(float(lr), 10)})
    elapsed = time.perf_counter() - started; model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": model_source_sha256(), "contract_semantic_sha256": contract["semantic_sha256"], "step": end_step, "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(), "model_state_sha256": _state_sha256(model_state), "ema_state_sha256": _state_sha256(ema_state), "cpu_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(device), "history": history, "runtime": {"segment_seconds": round(elapsed, 6), "updates_per_second": round(segment_steps / elapsed, 6), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "device": torch.cuda.get_device_name(device), "torch": str(torch.__version__)}}
    _atomic_torch(destination, payload); return _checkpoint_summary(_load_checkpoint(destination, contract, expected_step=end_step))


def _telemetry(output: Path) -> dict[str, Any]:
    path = output / TELEMETRY_NAME
    if not path.exists(): return _telemetry_payload([])
    payload = _read_canonical_json(path)
    required = {"format", "source_sha256", "attempts", "semantic_sha256"}
    if set(payload) != required or payload["format"] != TELEMETRY_FORMAT or payload["source_sha256"] != model_source_sha256() or payload["semantic_sha256"] != _semantic({key: value for key, value in payload.items() if key != "semantic_sha256"}):
        raise ValueError("Neural motion telemetry drifted.")
    _validate_attempts(payload["attempts"])
    return payload


def run_supervisor(output: Path = DEFAULT_OUTPUT, **schedule: Any) -> dict[str, Any]:
    from .evaluation_report import evaluation_name, validate_evaluation_report
    output = Path(output).resolve(); contract = prepare_production(output, **schedule); telemetry = _telemetry(output); attempts: list[dict[str, Any]] = telemetry["attempts"]
    for end_step in range(contract["segment_steps"], contract["total_steps"] + 1, contract["segment_steps"]):
        checkpoint = output / checkpoint_name(end_step)
        report_path = output / evaluation_name(end_step)
        if checkpoint.exists() and report_path.exists(): _load_checkpoint(checkpoint, contract, expected_step=end_step); validate_evaluation_report(output, step=end_step); continue
        prior = [row for row in attempts if row["end_step"] == end_step]; first_attempt = max((row["attempt"] for row in prior), default=0) + 1
        for attempt in range(first_attempt, contract["supervisor"]["max_attempts_per_segment"] + 1):
            require_disk_floor(output, floor_gb=100, planned_bytes=1024**3); env = os.environ.copy(); env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}); started = time.perf_counter(); timed_out = False
            try:
                process = subprocess.run([sys.executable, "-m", "forge.neural_cell_motion", "segment", "--output", str(output), "--end-step", str(end_step)], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=contract["supervisor"]["segment_timeout_seconds"]); returncode, stdout, stderr = process.returncode, process.stdout[-2000:], process.stderr[-4000:]
            except subprocess.TimeoutExpired as error:
                timed_out = True; returncode = -1; stdout = error.stdout[-2000:] if isinstance(error.stdout, str) else ""; stderr = error.stderr[-4000:] if isinstance(error.stderr, str) else ""
            valid, validation_error = False, ""
            if returncode == 0:
                try:
                    _load_checkpoint(checkpoint, contract, expected_step=end_step)
                    evaluation_process = subprocess.run([sys.executable, "-m", "forge.neural_cell_motion", "evaluate-production", "--output", str(output), "--step", str(end_step)], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=contract["supervisor"]["segment_timeout_seconds"])
                    stdout = (stdout + "\n" + evaluation_process.stdout)[-2000:]; stderr = (stderr + "\n" + evaluation_process.stderr)[-4000:]
                    if evaluation_process.returncode != 0: returncode = evaluation_process.returncode
                    else: validate_evaluation_report(output, step=end_step); valid = True
                except Exception as error: validation_error = f"{type(error).__name__}: {error}"[:4000]
            attempts.append({"end_step": end_step, "attempt": attempt, "returncode": returncode, "seconds": round(time.perf_counter() - started, 6), "stdout_tail": stdout, "stderr_tail": stderr, "access_violation": returncode in ACCESS_VIOLATION_CODES, "timed_out": timed_out, "artifact_valid": valid, "validation_error": validation_error}); _validate_attempts(attempts); _atomic_bytes(output / TELEMETRY_NAME, canonical_json_bytes(_telemetry_payload(attempts)))
            if valid: break
        else: raise RuntimeError(f"Neural motion segment {end_step} exhausted bounded retries.")
    final = _load_checkpoint(output / checkpoint_name(contract["total_steps"]), contract, expected_step=contract["total_steps"]); evaluation = validate_evaluation_report(output, step=contract["total_steps"])
    return {"passed": True, "promotion_eligible": evaluation["promotion_eligible"], "step": final["step"], "model_state_sha256": final["model_state_sha256"], "ema_state_sha256": final["ema_state_sha256"], "evaluation_semantic_sha256": evaluation["semantic_sha256"], "attempt_count": len(attempts), "retry_count": len(attempts) - contract["total_steps"] // contract["segment_steps"]}


def sampler_report(corpus: Path = DEFAULT_CORPUS, *, batch_size: int = 10, steps: int = 100) -> dict[str, Any]:
    sampler = MotionBatchSampler(corpus, batch_size=batch_size); counts = [0] * 5; identities: set[tuple[int, int]] = set(); frames: set[int] = set()
    for step in range(steps):
        for family, identity, frame in sampler.coordinates(step): counts[family] += 1; identities.add((family, identity)); frames.add(frame)
    return {"passed": counts == [steps * batch_size // 5] * 5, "family_counts": counts, "identity_coordinates": len(identities), "frame_count": len(frames), "steps": steps, "batch_size": batch_size}
