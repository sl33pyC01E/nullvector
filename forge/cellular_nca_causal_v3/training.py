from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

from ..cellular_nca.contract import CellularNCAConfig
from ..cellular_nca.corpus import load_corpus
from ..cellular_nca.model import OrganismCellularAutomaton
from ..cellular_nca_causal.curriculum import PRE_ROLL_CHOICES, SYSTEMS, make_targeted_pairs
from ..cellular_nca_causal.training import load_final_checkpoint as load_v2_checkpoint
from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, DEFAULT_OUTPUT, PARENT_OUTPUT, TELEMETRY_FORMAT, TRAINING_FORMAT, canonical, sha256_file, source_sha256
from .curriculum import ROLLOUT_STEPS, long_horizon_loss


SEED = 0x43415553414C5633
CONTRACT_NAME = "training_contract.json"
TELEMETRY_NAME = "training_telemetry.json"


def checkpoint_name(step: int) -> str:
    return f"causal_v3_segment_{step:07d}.pt"


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return tensor_state_sha256({name: value.detach().cpu() for name, value in state.items()})


def prepare(
    output: Path = DEFAULT_OUTPUT,
    *,
    total_steps: int = 256,
    segment_steps: int = 64,
    batch_size: int = 4,
    max_attempts: int = 3,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if total_steps % segment_steps or not 64 <= segment_steps <= total_steps <= 2048:
        raise ValueError("V3 schedule drifted")
    if not 2 <= batch_size <= 6 or not 1 <= max_attempts <= 3:
        raise ValueError("V3 batch/retry contract drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024**3)
    output.mkdir(parents=True, exist_ok=True)
    parent_model, parent_checkpoint, parent_contract = load_v2_checkpoint(PARENT_OUTPUT)
    del parent_model
    contract = {
        "format": TRAINING_FORMAT,
        "source_sha256": source_sha256(),
        "seed": SEED,
        "parent": {
            "path": PARENT_OUTPUT.relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            "checkpoint_sha256": sha256_file(PARENT_OUTPUT / "causal_segment_0000512.pt"),
            "model_state_sha256": parent_checkpoint["model_state_sha256"],
            "ema_state_sha256": parent_checkpoint["ema_state_sha256"],
        },
        "corpus_parent": parent_contract["parent"],
        "model": parent_contract["model"],
        "total_steps": total_steps,
        "segment_steps": segment_steps,
        "batch_size": batch_size,
        "rollout_steps": ROLLOUT_STEPS,
        "optimizer": {"lr": 2.5e-5, "weight_decay": 2e-5, "gradient_clip": .6},
        "ema_decay": .999,
        "max_attempts": max_attempts,
    }
    path = output / CONTRACT_NAME
    if path.exists():
        if json.loads(path.read_bytes()) != contract:
            raise ValueError("V3 training contract changed during resume")
    else:
        _atomic_bytes(path, canonical(contract))
    return contract


def _load(path: Path, contract: dict[str, Any], expected_step: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format", "source_sha256", "contract", "step", "model_state", "ema_state",
        "optimizer_state", "generator_state", "model_state_sha256", "ema_state_sha256",
        "history", "runtime",
    }
    if set(payload) != required or payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source_sha256() or payload["contract"] != contract or payload["step"] != expected_step:
        raise ValueError("V3 checkpoint provenance drifted")
    model = OrganismCellularAutomaton(CellularNCAConfig(**contract["model"]))
    model.load_state_dict(payload["model_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("V3 model hash drifted")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["ema_state_sha256"]:
        raise ValueError("V3 EMA hash drifted")
    if not isinstance(payload["generator_state"], torch.Tensor) or payload["generator_state"].dtype != torch.uint8:
        raise ValueError("V3 RNG state drifted")
    return payload


def train_segment(output: Path, end_step: int) -> Path:
    output = Path(output).resolve(); contract = json.loads((output / CONTRACT_NAME).read_bytes())
    segment_steps = int(contract["segment_steps"])
    if end_step % segment_steps or not segment_steps <= end_step <= contract["total_steps"]:
        raise ValueError("V3 segment endpoint drifted")
    destination = output / checkpoint_name(end_step)
    if destination.exists():
        _load(destination, contract, end_step); return destination
    previous_step = end_step - segment_steps
    previous_path = output / checkpoint_name(previous_step) if previous_step else None
    if previous_path is not None and not previous_path.exists():
        raise FileNotFoundError(previous_path)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 8 * 1024**3:
        raise RuntimeError("V3 requires deterministic CUDA with 8 GiB free")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED); np.random.seed(SEED & 0xFFFFFFFF)
    device = torch.device("cuda:0"); torch.cuda.set_per_process_memory_fraction(.65, 0); torch.cuda.reset_peak_memory_stats(device)
    parent_model, parent_checkpoint, parent_contract = load_v2_checkpoint(PARENT_OUTPUT)
    corpus_root = Path(__file__).resolve().parents[2] / parent_contract["parent"]["path"]
    arrays = load_corpus(corpus_root)["arrays"]
    tensors = {name: torch.from_numpy(arrays[name]).to(device) for name in ("static", "initial_state", "live_bonds")}
    model = OrganismCellularAutomaton(CellularNCAConfig(**contract["model"])).to(device)
    model.load_state_dict(parent_checkpoint["ema_state"], strict=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract["optimizer"]["lr"], weight_decay=contract["optimizer"]["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(SEED ^ 0x545241494E)
    history: list[dict[str, Any]] = []
    start_step = 0
    if previous_path is not None:
        previous = _load(previous_path, contract, previous_step)
        model.load_state_dict(previous["model_state"], strict=True); ema.load_state_dict(previous["ema_state"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state"]); generator.set_state(previous["generator_state"])
        history = list(previous["history"]); start_step = previous_step
    model.train(); started = time.perf_counter()
    for step in range(start_step, end_step):
        indices = torch.randint(0, len(tensors["static"]), (contract["batch_size"],), device=device, generator=generator)
        static = tensors["static"][indices]; initial = tensors["initial_state"][indices]; bonds = tensors["live_bonds"][indices]
        system_ids = (torch.arange(len(static), device=device) + step) % len(SYSTEMS)
        pre_rolls = torch.tensor(PRE_ROLL_CHOICES, device=device)[(torch.arange(len(static), device=device) + step // len(SYSTEMS)) % len(PRE_ROLL_CHOICES)]
        control, damaged = make_targeted_pairs(static, initial, bonds, system_ids, pre_rolls, dt=.1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, pieces = long_horizon_loss(model, static, control, damaged, bonds, system_ids, dt=.1)
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["optimizer"]["gradient_clip"])
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)):
            raise FloatingPointError("V3 training became non-finite")
        optimizer.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()), contract["ema_decay"])
            torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - contract["ema_decay"])
        history.append({
            "step": step + 1, "loss": round(float(loss), 8), "base": round(float(pieces["base"]), 8),
            "contrast": round(float(pieces["contrast"]), 8), "direction": round(float(pieces["direction"]), 8),
            "magnitude": round(float(pieces["magnitude"]), 8), "gradient_norm": round(float(gradient), 8),
            "health_mae": round(float(pieces["health"]), 8), "oxygen_mae": round(float(pieces["oxygen"]), 8),
            "energy_mae": round(float(pieces["energy"]), 8), "neural_mae": round(float(pieces["neural"]), 8),
        })
    seconds = time.perf_counter() - started
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    ema_state = {name: value.detach().cpu() for name, value in ema.state_dict().items()}
    payload = {
        "format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "contract": contract,
        "step": end_step, "model_state": model_state, "ema_state": ema_state,
        "optimizer_state": optimizer.state_dict(), "generator_state": generator.get_state().cpu(),
        "model_state_sha256": _state_hash(model_state), "ema_state_sha256": _state_hash(ema_state),
        "history": history,
        "runtime": {"seconds": round(seconds, 6), "updates_per_second": round(segment_steps / seconds, 6), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "device": torch.cuda.get_device_name(device)},
    }
    temporary = output / f".{destination.name}.tmp-{os.getpid()}"
    torch.save(payload, temporary); os.replace(temporary, destination)
    _load(destination, contract, end_step)
    return destination


def run_supervisor(output: Path = DEFAULT_OUTPUT, **kwargs: Any) -> dict[str, Any]:
    output = Path(output).resolve(); contract = prepare(output, **kwargs)
    telemetry_path = output / TELEMETRY_NAME
    telemetry = json.loads(telemetry_path.read_bytes()) if telemetry_path.exists() else {"format": TELEMETRY_FORMAT, "source_sha256": source_sha256(), "attempts": []}
    for end_step in range(contract["segment_steps"], contract["total_steps"] + 1, contract["segment_steps"]):
        destination = output / checkpoint_name(end_step)
        if destination.exists():
            _load(destination, contract, end_step); continue
        prior = [row for row in telemetry["attempts"] if row["end_step"] == end_step]
        for attempt in range(len(prior) + 1, contract["max_attempts"] + 1):
            env = os.environ.copy(); env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})
            started = time.perf_counter()
            process = subprocess.run([sys.executable, "-m", "forge.cellular_nca_causal_v3", "segment", "--output", str(output), "--end-step", str(end_step)], cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True, text=True)
            valid = False; error = ""
            if process.returncode == 0:
                try: _load(destination, contract, end_step); valid = True
                except Exception as exception: error = f"{type(exception).__name__}: {exception}"
            telemetry["attempts"].append({"end_step": end_step, "attempt": attempt, "returncode": process.returncode, "seconds": round(time.perf_counter() - started, 6), "valid": valid, "error": error, "stderr_tail": process.stderr[-2000:]})
            _atomic_bytes(telemetry_path, canonical(telemetry))
            if valid: break
        else: raise RuntimeError(f"V3 segment {end_step} exhausted retries")
    final = _load(output / checkpoint_name(contract["total_steps"]), contract, contract["total_steps"])
    return {"passed": True, "step": final["step"], "ema_state_sha256": final["ema_state_sha256"], "attempts": len(telemetry["attempts"])}


def load_final(output: Path = DEFAULT_OUTPUT):
    contract = json.loads((Path(output) / CONTRACT_NAME).read_bytes())
    payload = _load(Path(output) / checkpoint_name(contract["total_steps"]), contract, contract["total_steps"])
    model = OrganismCellularAutomaton(CellularNCAConfig(**contract["model"])); model.load_state_dict(payload["ema_state"]); model.eval()
    return model, payload, contract
