from __future__ import annotations

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

from ..cellular_nca.contract import CellularNCAConfig, canonical_json_bytes, source_sha256 as parent_source_sha256
from ..cellular_nca.corpus import load_corpus
from ..cellular_nca.model import OrganismCellularAutomaton
from ..cellular_nca.teacher import cellular_loss, teacher_step
from ..cellular_nca.training import load_final_checkpoint as load_parent_checkpoint
from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, PARENT_OUTPUT, SEGMENT_TIMEOUT_SECONDS, TELEMETRY_FORMAT, TELEMETRY_NAME, TRAINING_FORMAT, TRAINING_NAME, causal_source_sha256, read_canonical_json, sha256_file
from .curriculum import PRE_ROLL_CHOICES, SYSTEMS, causal_contrast_loss, make_targeted_pairs


SEED = 0x43415553414C4E43
MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024


def _valid_generator_state(value: object) -> bool:
    """Accept bounded Torch RNG encodings without assuming one backend layout.

    Torch 2.6 uses a compact 16-byte Philox state for CUDA generators while
    CPU generators use a much larger state.  Shape, dtype, and a conservative
    size ceiling are the durable contract; a 1,000-byte lower bound was an
    accidental CPU-only assumption.
    """
    return (
        isinstance(value, Tensor)
        and value.dtype == torch.uint8
        and value.ndim == 1
        and 16 <= value.numel() <= 100_000
    )


def checkpoint_name(step: int) -> str:
    return f"causal_segment_{step:07d}.pt"


def _state_hash(state: dict[str, Tensor]) -> str:
    return tensor_state_sha256({name: value.detach().cpu() for name, value in state.items()})


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    try: torch.save(payload, temporary); os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _parent_authority(parent_output: Path) -> tuple[OrganismCellularAutomaton, dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_output = Path(parent_output).resolve(); model, checkpoint, training = load_parent_checkpoint(parent_output)
    manifest_path = parent_output / "cellular_nca_manifest.json"; manifest = read_canonical_json(manifest_path)
    if manifest.get("format") != "nullvector-organism-neural-cellular-automaton-v1" or manifest.get("source_sha256") != parent_source_sha256() or manifest.get("provenance", {}).get("checkpoint_sha256") != sha256_file(parent_output / manifest["provenance"]["checkpoint_path"]):
        raise ValueError("Parent cellular NCA authority drifted.")
    return model, checkpoint, training, manifest


def prepare_training(output: Path, *, parent_output: Path = PARENT_OUTPUT, total_steps: int = 512, segment_steps: int = 128, batch_size: int = 8, max_attempts: int = 3) -> dict[str, Any]:
    output = Path(output).resolve(); parent_output = Path(parent_output).resolve()
    if type(total_steps) is not int or not 64 <= total_steps <= 4096 or type(segment_steps) is not int or not 64 <= segment_steps <= total_steps or total_steps % segment_steps or type(batch_size) is not int or not 4 <= batch_size <= 16 or type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("Causal fine-tune schedule drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024**3); output.mkdir(parents=True, exist_ok=True)
    if (output / "cellular_nca_causal_manifest.json").exists(): raise FileExistsError("Finalized causal NCA output is immutable.")
    _, parent_checkpoint, parent_training, parent_manifest = _parent_authority(parent_output)
    relative_parent = parent_output.relative_to(Path(__file__).resolve().parents[2]).as_posix()
    contract = {
        "format": TRAINING_FORMAT, "source_sha256": causal_source_sha256(), "seed": SEED,
        "parent": {"path": relative_parent, "source_sha256": parent_source_sha256(), "manifest_sha256": sha256_file(parent_output / "cellular_nca_manifest.json"), "checkpoint_sha256": parent_manifest["provenance"]["checkpoint_sha256"], "ema_state_sha256": parent_checkpoint["ema_state_sha256"]},
        "model": parent_training["model"], "total_steps": total_steps, "segment_steps": segment_steps, "batch_size": batch_size,
        "optimizer": {"name": "AdamW", "lr": 5e-5, "weight_decay": 2e-5, "gradient_clip": .75}, "ema_decay": .9995,
        "curriculum": {"systems": [name for name, _, _ in SYSTEMS], "pre_roll_steps": list(PRE_ROLL_CHOICES), "teacher_dt": .1, "rollout_steps": 2, "counterfactual_weight": .18},
        "supervisor": {"max_attempts_per_segment": max_attempts, "segment_timeout_seconds": SEGMENT_TIMEOUT_SECONDS},
        "precision": "bf16-autocast-float32-loss",
    }
    path = output / TRAINING_NAME
    if path.exists():
        current = read_canonical_json(path)
        if current != contract: raise ValueError("Causal training contract changed during resume.")
    else: _atomic_bytes(path, canonical_json_bytes(contract))
    return contract


def _load_checkpoint(path: Path, contract: dict[str, Any], *, expected_step: int | None = None, require_current_source: bool = True) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES: raise ValueError("Causal checkpoint missing or oversized.")
    payload = torch.load(path, map_location="cpu", weights_only=True); required = {"format", "source_sha256", "contract", "step", "model_state", "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256", "cuda_generator_state", "history", "runtime"}
    expected_source = causal_source_sha256() if require_current_source else contract.get("source_sha256")
    if not isinstance(payload, dict) or set(payload) != required or payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != expected_source or payload["contract"] != contract: raise ValueError("Causal checkpoint provenance drifted.")
    if type(payload["step"]) is not int or not 1 <= payload["step"] <= contract["total_steps"] or (expected_step is not None and payload["step"] != expected_step) or not isinstance(payload["history"], list) or len(payload["history"]) != payload["step"]: raise ValueError("Causal checkpoint census drifted.")
    history_keys = {"step", "loss", "base_loss", "contrast_loss", "contrast_direction", "contrast_magnitude", "gradient_norm", "health_mae", "oxygen_mae", "energy_mae", "neural_mae"}
    for index, row in enumerate(payload["history"], 1):
        if not isinstance(row, dict) or set(row) != history_keys or row["step"] != index or any(type(row[key]) not in (int, float) or not math.isfinite(float(row[key])) for key in history_keys - {"step"}):
            raise ValueError("Causal checkpoint history drifted.")
    model = OrganismCellularAutomaton(CellularNCAConfig(**contract["model"])); model.load_state_dict(payload["model_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["model_state_sha256"]: raise ValueError("Causal model state hash drifted.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["ema_state_sha256"]: raise ValueError("Causal EMA state hash drifted.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract["optimizer"]["lr"], weight_decay=contract["optimizer"]["weight_decay"])
    try: optimizer.load_state_dict(payload["optimizer_state"])
    except (KeyError, TypeError, ValueError) as error: raise ValueError("Causal optimizer state drifted.") from error
    runtime = payload["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"segment_seconds", "updates_per_second", "peak_allocated_bytes", "peak_reserved_bytes", "device", "torch"} or any(type(runtime[key]) not in (int, float) or not math.isfinite(float(runtime[key])) or float(runtime[key]) < 0 for key in ("segment_seconds", "updates_per_second", "peak_allocated_bytes", "peak_reserved_bytes")) or not all(isinstance(runtime[key], str) and runtime[key] for key in ("device", "torch")):
        raise ValueError("Causal checkpoint runtime drifted.")
    if not _valid_generator_state(payload["cuda_generator_state"]): raise ValueError("Causal checkpoint RNG drifted.")
    return payload


def rebind_checkpoint(source_output: Path, output: Path, *, end_step: int) -> dict[str, Any]:
    """Additively recover a valid checkpoint after validator-only source drift."""
    source_output = Path(source_output).resolve(); output = Path(output).resolve()
    old_contract = read_canonical_json(source_output / TRAINING_NAME)
    if old_contract.get("format") != TRAINING_FORMAT or old_contract.get("source_sha256") == causal_source_sha256():
        raise ValueError("Rebind requires a stale causal training authority.")
    old_path = source_output / checkpoint_name(end_step)
    old_payload = _load_checkpoint(old_path, old_contract, expected_step=end_step, require_current_source=False)
    project_root = Path(__file__).resolve().parents[2]
    parent_output = project_root / old_contract["parent"]["path"]
    new_contract = prepare_training(
        output,
        parent_output=parent_output,
        total_steps=old_contract["total_steps"],
        segment_steps=old_contract["segment_steps"],
        batch_size=old_contract["batch_size"],
        max_attempts=old_contract["supervisor"]["max_attempts_per_segment"],
    )
    comparable_old = dict(old_contract); comparable_new = dict(new_contract)
    comparable_old.pop("source_sha256", None); comparable_new.pop("source_sha256", None)
    if comparable_old != comparable_new:
        raise ValueError("Rebind refused a semantic training-contract change.")
    rebound = dict(old_payload); rebound["source_sha256"] = causal_source_sha256(); rebound["contract"] = new_contract
    destination = output / checkpoint_name(end_step)
    if destination.exists(): raise FileExistsError(destination)
    _atomic_torch(destination, rebound)
    verified = _load_checkpoint(destination, new_contract, expected_step=end_step)
    report = {
        "format": "nullvector-organism-neural-cellular-automaton-causal-rebind-v1",
        "source_sha256": causal_source_sha256(),
        "from": {"path": source_output.relative_to(project_root).as_posix(), "source_sha256": old_contract["source_sha256"], "checkpoint_sha256": sha256_file(old_path)},
        "to": {"path": output.relative_to(project_root).as_posix(), "checkpoint_sha256": sha256_file(destination)},
        "step": end_step,
        "model_state_sha256": verified["model_state_sha256"],
        "ema_state_sha256": verified["ema_state_sha256"],
        "semantic_contract_unchanged": True,
    }
    _atomic_bytes(output / "causal_checkpoint_rebind.json", canonical_json_bytes(report))
    return report


def _validate_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"format", "source_sha256", "attempts"} or payload.get("format") != TELEMETRY_FORMAT or payload.get("source_sha256") != causal_source_sha256() or not isinstance(payload.get("attempts"), list):
        raise ValueError("Causal telemetry authority drifted.")
    seen: set[tuple[int, int]] = set()
    required = {"end_step", "attempt", "returncode", "seconds", "stdout_tail", "stderr_tail", "windows_access_violation", "timed_out", "artifact_valid", "recovered_checkpoint", "validation_error"}
    for row in payload["attempts"]:
        if not isinstance(row, dict) or set(row) != required or type(row["end_step"]) is not int or type(row["attempt"]) is not int or row["attempt"] < 0 or type(row["returncode"]) is not int or type(row["seconds"]) not in (int, float) or not math.isfinite(float(row["seconds"])) or row["seconds"] < 0 or not all(type(row[key]) is bool for key in ("windows_access_violation", "timed_out", "artifact_valid", "recovered_checkpoint")) or not all(isinstance(row[key], str) and len(row[key]) <= 4000 for key in ("stdout_tail", "stderr_tail", "validation_error")):
            raise ValueError("Causal telemetry record drifted.")
        identity = (row["end_step"], row["attempt"])
        if identity in seen: raise ValueError("Causal telemetry contains a duplicate attempt.")
        seen.add(identity)
        if row["artifact_valid"] and (row["returncode"] != 0 or row["timed_out"]): raise ValueError("Causal telemetry success is inconsistent.")
        if row["recovered_checkpoint"] != (row["attempt"] == 0) or row["recovered_checkpoint"] and not row["artifact_valid"]:
            raise ValueError("Causal recovery telemetry is inconsistent.")
    return payload


def _load_telemetry(output: Path) -> dict[str, Any]:
    path = Path(output) / TELEMETRY_NAME
    if not path.exists(): return {"format": TELEMETRY_FORMAT, "source_sha256": causal_source_sha256(), "attempts": []}
    return _validate_telemetry(read_canonical_json(path))


def train_segment(output: Path, *, end_step: int) -> dict[str, Any]:
    output = Path(output).resolve(); contract = read_canonical_json(output / TRAINING_NAME)
    if contract.get("source_sha256") != causal_source_sha256(): raise ValueError("Causal training authority drifted.")
    segment_steps = contract["segment_steps"]
    if type(end_step) is not int or end_step % segment_steps or not segment_steps <= end_step <= contract["total_steps"]: raise ValueError("Causal segment endpoint drifted.")
    destination = output / checkpoint_name(end_step)
    if destination.exists(): return _load_checkpoint(destination, contract, expected_step=end_step)
    previous_step = end_step - segment_steps; previous_path = output / checkpoint_name(previous_step) if previous_step else None
    if previous_path is not None and not previous_path.exists(): raise FileNotFoundError("Previous causal segment is missing.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(0)[0] < 9 * 1024**3: raise RuntimeError("Causal NCA requires deterministic CUDA BF16 and 9 GiB free VRAM.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED & 0xffffffff)
    device = torch.device("cuda", 0); torch.cuda.reset_peak_memory_stats(device); parent_output = Path(__file__).resolve().parents[2] / contract["parent"]["path"]
    parent_model, parent_checkpoint, _, _ = _parent_authority(parent_output); corpus = load_corpus(parent_output); arrays = corpus["arrays"]
    tensors = {name: torch.from_numpy(arrays[name]).to(device) for name in ("static", "initial_state", "live_bonds")}
    config = CellularNCAConfig(**contract["model"]); model = OrganismCellularAutomaton(config).to(device); model.load_state_dict(parent_checkpoint["ema_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract["optimizer"]["lr"], weight_decay=contract["optimizer"]["weight_decay"]); generator = torch.Generator(device=device).manual_seed(SEED ^ 0x545241494E)
    history: list[dict[str, Any]] = []; ema = {name: value.detach().clone() for name, value in model.state_dict().items()}; start_step = 0
    if previous_path is not None:
        previous = _load_checkpoint(previous_path, contract, expected_step=previous_step); start_step = previous["step"]; model.load_state_dict(previous["model_state"], strict=True); optimizer.load_state_dict(previous["optimizer_state"]); ema = {name: value.to(device) for name, value in previous["ema_state"].items()}; generator.set_state(previous["cuda_generator_state"]); history = list(previous["history"])
    batch_size = contract["batch_size"]; started = time.perf_counter(); model.train()
    for step in range(start_step, end_step):
        indices = torch.randint(0, len(tensors["static"]), (batch_size,), device=device, generator=generator); static = tensors["static"][indices]; initial = tensors["initial_state"][indices]; bonds = tensors["live_bonds"][indices]
        system_ids = (torch.arange(batch_size, device=device) + step) % len(SYSTEMS); pre_rolls = torch.tensor(PRE_ROLL_CHOICES, device=device, dtype=torch.long)[(torch.arange(batch_size, device=device) + step // len(SYSTEMS)) % len(PRE_ROLL_CHOICES)]
        control, damaged = make_targeted_pairs(static, initial, bonds, system_ids, pre_rolls, dt=contract["curriculum"]["teacher_dt"])
        pair_static = torch.cat((static, static)); pair_bonds = torch.cat((bonds, bonds)); pair_state = torch.cat((control, damaged))
        target_one = teacher_step(pair_static, pair_state, pair_bonds, contract["curriculum"]["teacher_dt"]); target_two = teacher_step(pair_static, target_one, pair_bonds, contract["curriculum"]["teacher_dt"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predicted_one = model(pair_static, pair_state, pair_bonds); predicted_two = model(pair_static, predicted_one, pair_bonds)
        base_one, _ = cellular_loss(predicted_one, target_one, pair_static, pair_state); base_two, pieces = cellular_loss(predicted_two, target_two, pair_static, predicted_one); base_loss = .45 * base_one + .55 * base_two
        contrast, contrast_pieces = causal_contrast_loss(predicted_two[:batch_size], predicted_two[batch_size:], target_two[:batch_size], target_two[batch_size:], static, system_ids)
        loss = base_loss + contract["curriculum"]["counterfactual_weight"] * contrast
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["optimizer"]["gradient_clip"])
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)): raise FloatingPointError("Causal NCA became non-finite.")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point: ema[name].lerp_(value.detach(), 1 - contract["ema_decay"])
                else: ema[name].copy_(value)
        history.append({"step": step + 1, "loss": round(float(loss), 8), "base_loss": round(float(base_loss), 8), "contrast_loss": round(float(contrast), 8), "contrast_direction": round(float(contrast_pieces["direction"]), 8), "contrast_magnitude": round(float(contrast_pieces["magnitude"]), 8), "gradient_norm": round(float(gradient), 8), "health_mae": round(float(pieces["channel_0"]), 8), "oxygen_mae": round(float(pieces["channel_4"]), 8), "energy_mae": round(float(pieces["channel_3"]), 8), "neural_mae": round(float(pieces["channel_8"]), 8)})
    elapsed = time.perf_counter() - started; model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": causal_source_sha256(), "contract": contract, "step": end_step, "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(), "model_state_sha256": _state_hash(model_state), "ema_state_sha256": _state_hash(ema_state), "cuda_generator_state": generator.get_state().cpu(), "history": history, "runtime": {"segment_seconds": round(elapsed, 6), "updates_per_second": round(segment_steps / elapsed, 6), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "device": torch.cuda.get_device_name(device), "torch": str(torch.__version__)}}
    _atomic_torch(destination, payload); return _load_checkpoint(destination, contract, expected_step=end_step)


def run_supervisor(output: Path, *, parent_output: Path = PARENT_OUTPUT, total_steps: int = 512, segment_steps: int = 128, batch_size: int = 8, max_attempts: int = 3) -> dict[str, Any]:
    output = Path(output).resolve(); contract = prepare_training(output, parent_output=parent_output, total_steps=total_steps, segment_steps=segment_steps, batch_size=batch_size, max_attempts=max_attempts); telemetry_payload = _load_telemetry(output); telemetry: list[dict[str, Any]] = telemetry_payload["attempts"]
    for end_step in range(segment_steps, total_steps + 1, segment_steps):
        if (output / checkpoint_name(end_step)).exists():
            _load_checkpoint(output / checkpoint_name(end_step), contract, expected_step=end_step)
            if not any(row["end_step"] == end_step and row["artifact_valid"] for row in telemetry):
                telemetry.append({"end_step": end_step, "attempt": 0, "returncode": 0, "seconds": 0.0, "stdout_tail": "", "stderr_tail": "", "windows_access_violation": False, "timed_out": False, "artifact_valid": True, "recovered_checkpoint": True, "validation_error": ""})
                telemetry_payload = _validate_telemetry({"format": TELEMETRY_FORMAT, "source_sha256": causal_source_sha256(), "attempts": telemetry}); _atomic_bytes(output / TELEMETRY_NAME, canonical_json_bytes(telemetry_payload))
            continue
        prior = [row for row in telemetry if row["end_step"] == end_step]
        if any(row["artifact_valid"] for row in prior): raise ValueError("Causal telemetry claims a missing successful checkpoint.")
        first_attempt = max((row["attempt"] for row in prior), default=0) + 1
        for attempt in range(first_attempt, max_attempts + 1):
            require_disk_floor(output, floor_gb=100, planned_bytes=512 * 1024**2); env = os.environ.copy(); env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
            started = time.perf_counter(); timed_out = False
            try:
                process = subprocess.run([sys.executable, "-m", "forge.cellular_nca_causal", "segment", "--output", str(output), "--end-step", str(end_step)], cwd=str(Path(__file__).resolve().parents[2]), env=env, capture_output=True, text=True, timeout=contract["supervisor"]["segment_timeout_seconds"])
                returncode, stdout_tail, stderr_tail = process.returncode, process.stdout[-2000:], process.stderr[-4000:]
            except subprocess.TimeoutExpired as error:
                timed_out = True; returncode = -1; stdout_tail = (error.stdout or "")[-2000:] if isinstance(error.stdout, str) else ""; stderr_tail = (error.stderr or "")[-4000:] if isinstance(error.stderr, str) else ""
            artifact_valid = False; validation_error = ""
            if returncode == 0:
                try: _load_checkpoint(output / checkpoint_name(end_step), contract, expected_step=end_step); artifact_valid = True
                except Exception as error:
                    validation_error = f"{type(error).__name__}: {error}"[:4000]
                    invalid = output / checkpoint_name(end_step)
                    if invalid.exists(): os.replace(invalid, output / f"{invalid.name}.invalid-attempt-{attempt}")
            telemetry.append({"end_step": end_step, "attempt": attempt, "returncode": returncode, "seconds": round(time.perf_counter() - started, 6), "stdout_tail": stdout_tail, "stderr_tail": stderr_tail, "windows_access_violation": returncode in (-1073741819, 3221225477), "timed_out": timed_out, "artifact_valid": artifact_valid, "recovered_checkpoint": False, "validation_error": validation_error}); telemetry_payload = _validate_telemetry({"format": TELEMETRY_FORMAT, "source_sha256": causal_source_sha256(), "attempts": telemetry}); _atomic_bytes(output / TELEMETRY_NAME, canonical_json_bytes(telemetry_payload))
            if artifact_valid: break
        else: raise RuntimeError(f"Causal NCA segment {end_step} exhausted bounded retries.")
    checkpoint = _load_checkpoint(output / checkpoint_name(total_steps), contract, expected_step=total_steps); successful_segments = total_steps // segment_steps; retry_count = len(telemetry) - successful_segments
    if retry_count < 0 or sum(row["artifact_valid"] for row in telemetry) != successful_segments: raise ValueError("Causal telemetry does not close the segment census.")
    return {"passed": True, "step": total_steps, "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"], "attempt_count": len(telemetry), "retry_count": retry_count}


def load_final_checkpoint(output: Path) -> tuple[OrganismCellularAutomaton, dict[str, Any], dict[str, Any]]:
    output = Path(output).resolve(); contract = read_canonical_json(output / TRAINING_NAME)
    if contract.get("source_sha256") != causal_source_sha256(): raise ValueError("Causal final contract drifted.")
    checkpoint = _load_checkpoint(output / checkpoint_name(contract["total_steps"]), contract, expected_step=contract["total_steps"]); model = OrganismCellularAutomaton(CellularNCAConfig(**contract["model"])); model.load_state_dict(checkpoint["ema_state"], strict=True); model.eval(); return model, checkpoint, contract
