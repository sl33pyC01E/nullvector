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

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, CellularNCAConfig, canonical_json_bytes, sha256_file, source_sha256
from .corpus import CORPUS_MANIFEST_NAME, build_corpus, load_corpus
from .model import OrganismCellularAutomaton, parameter_count
from .teacher import cellular_loss, make_scenarios, teacher_step


SEED = 0x4E554C4C43414E43
TRAINING_NAME = "training_contract.json"
TELEMETRY_NAME = "training_telemetry.json"
MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024


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


def _state_hash(state: dict[str, Tensor]) -> str:
    return tensor_state_sha256({name: value.detach().cpu() for name, value in state.items()})


def checkpoint_name(step: int) -> str:
    return f"segment_{step:07d}.pt"


def prepare_training(output: Path, *, total_steps: int, segment_steps: int, batch_size: int, config: CellularNCAConfig) -> dict[str, Any]:
    output = Path(output).resolve()
    if type(total_steps) is not int or not 64 <= total_steps <= 100_000 or type(segment_steps) is not int or not 32 <= segment_steps <= total_steps or total_steps % segment_steps or type(batch_size) is not int or not 4 <= batch_size <= 128:
        raise ValueError("Cellular NCA training schedule drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=3 * 1024**3); output.mkdir(parents=True, exist_ok=True)
    if (output / "cellular_nca_manifest.json").exists(): raise FileExistsError("Finalized cellular NCA output is immutable.")
    if not (output / CORPUS_MANIFEST_NAME).exists(): build_corpus(output)
    corpus = load_corpus(output); corpus_manifest_sha256 = sha256_file(output / CORPUS_MANIFEST_NAME)
    contract = {
        "format": "nullvector-organism-neural-cellular-automaton-training-v1", "source_sha256": source_sha256(),
        "seed": SEED, "total_steps": total_steps, "segment_steps": segment_steps, "batch_size": batch_size,
        "model": config.to_dict(), "parameter_count": parameter_count(OrganismCellularAutomaton(config)),
        "corpus_semantic_sha256": corpus["manifest"]["arrays_semantic_sha256"], "corpus_manifest_sha256": corpus_manifest_sha256,
        "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-4, "gradient_clip": 1.0},
        "precision": "bf16-autocast-float32-loss", "teacher_dt": .1, "rollout_steps": 2, "ema_decay": config.ema_decay,
    }
    path = output / TRAINING_NAME
    if path.exists():
        encoded = path.read_bytes(); current = json.loads(encoded)
        if encoded != canonical_json_bytes(current) or current != contract: raise ValueError("Cellular NCA training contract changed during resume.")
    else: _atomic_bytes(path, canonical_json_bytes(contract))
    return contract


def _load_checkpoint(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES: raise ValueError("Cellular NCA checkpoint missing or oversized.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "source_sha256", "contract", "step", "model_state", "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256", "cuda_generator_state", "history", "runtime"}
    if not isinstance(payload, dict) or set(payload) != required or payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source_sha256() or payload["contract"] != contract: raise ValueError("Cellular NCA checkpoint provenance drifted.")
    if type(payload["step"]) is not int or not 1 <= payload["step"] <= contract["total_steps"] or not isinstance(payload["history"], list) or len(payload["history"]) != payload["step"]: raise ValueError("Cellular NCA checkpoint census drifted.")
    model = OrganismCellularAutomaton(CellularNCAConfig(**contract["model"])); model.load_state_dict(payload["model_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["model_state_sha256"]: raise ValueError("Cellular NCA model hash drifted.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["ema_state_sha256"]: raise ValueError("Cellular NCA EMA hash drifted.")
    if not isinstance(payload["cuda_generator_state"], Tensor) or payload["cuda_generator_state"].dtype != torch.uint8 or any(row.get("step") != index or not math.isfinite(float(row.get("loss", math.nan))) for index, row in enumerate(payload["history"], 1)): raise ValueError("Cellular NCA checkpoint RNG/history drifted.")
    return payload


def train_segment(output: Path, *, end_step: int) -> dict[str, Any]:
    output = Path(output).resolve(); encoded = (output / TRAINING_NAME).read_bytes(); contract = json.loads(encoded)
    if encoded != canonical_json_bytes(contract) or contract.get("source_sha256") != source_sha256(): raise ValueError("Cellular NCA training authority drifted.")
    segment_steps = contract["segment_steps"]
    if type(end_step) is not int or end_step % segment_steps or not segment_steps <= end_step <= contract["total_steps"]: raise ValueError("Cellular NCA segment endpoint drifted.")
    destination = output / checkpoint_name(end_step)
    if destination.exists(): return _load_checkpoint(destination, contract)
    previous_step = end_step - segment_steps; previous_path = output / checkpoint_name(previous_step) if previous_step else None
    if previous_path is not None and not previous_path.exists(): raise FileNotFoundError("Previous cellular NCA segment is missing.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(0)[0] < 8 * 1024**3: raise RuntimeError("Cellular NCA requires deterministic CUDA BF16 and 8 GiB free VRAM.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED & 0xffffffff)
    device = torch.device("cuda", 0); torch.cuda.reset_peak_memory_stats(device); corpus = load_corpus(output)
    if corpus["manifest"]["arrays_semantic_sha256"] != contract["corpus_semantic_sha256"] or sha256_file(output / CORPUS_MANIFEST_NAME) != contract["corpus_manifest_sha256"]: raise ValueError("Cellular NCA corpus drifted during training.")
    tensors = {name: torch.from_numpy(value).to(device) for name, value in corpus["arrays"].items() if name in ("static", "initial_state", "live_bonds")}
    config = CellularNCAConfig(**contract["model"]); model = OrganismCellularAutomaton(config).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=contract["optimizer"]["lr"], weight_decay=contract["optimizer"]["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(SEED ^ 0x545241494E); history: list[dict[str, Any]] = []; ema = {name: value.detach().clone() for name, value in model.state_dict().items()}; start_step = 0
    if previous_path is not None:
        previous = _load_checkpoint(previous_path, contract); start_step = previous["step"]; model.load_state_dict(previous["model_state"], strict=True); optimizer.load_state_dict(previous["optimizer_state"]); ema = {name: value.to(device) for name, value in previous["ema_state"].items()}; generator.set_state(previous["cuda_generator_state"]); history = list(previous["history"])
    batch_size = contract["batch_size"]; started = time.perf_counter(); model.train()
    for step in range(start_step, end_step):
        indices = torch.randint(0, 45, (batch_size,), device=device, generator=generator); static = tensors["static"][indices]; initial = tensors["initial_state"][indices]; base_bonds = tensors["live_bonds"][indices]
        state, live_bonds = make_scenarios(static, initial, base_bonds, generator); target_one = teacher_step(static, state, live_bonds, contract["teacher_dt"]); target_two = teacher_step(static, target_one, live_bonds, contract["teacher_dt"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predicted_one = model(static, state, live_bonds); predicted_two = model(static, predicted_one, live_bonds)
        loss_one, pieces_one = cellular_loss(predicted_one, target_one, static, state); loss_two, pieces_two = cellular_loss(predicted_two, target_two, static, predicted_one); loss = .58 * loss_one + .42 * loss_two
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["optimizer"]["gradient_clip"])
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)): raise FloatingPointError("Cellular NCA became non-finite.")
        optimizer.step(); ema_decay = min(config.ema_decay, (step + 2) / (step + 11))
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point: ema[name].lerp_(value.detach(), 1 - ema_decay)
                else: ema[name].copy_(value)
        history.append({"step": step + 1, "loss": round(float(loss), 8), "one_step": round(float(loss_one), 8), "two_step": round(float(loss_two), 8), "gradient_norm": round(float(gradient), 8), "health_mae": round(float(pieces_two["channel_0"]), 8), "fluid_mae": round(float(pieces_two["channel_1"]), 8), "neural_mae": round(float(pieces_two["channel_8"]), 8), "surface_mae": round(float(pieces_two["channel_9"]), 8)})
    elapsed = time.perf_counter() - started; model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "contract": contract, "step": end_step, "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(), "model_state_sha256": _state_hash(model_state), "ema_state_sha256": _state_hash(ema_state), "cuda_generator_state": generator.get_state().cpu(), "history": history, "runtime": {"segment_seconds": round(elapsed, 6), "updates_per_second": round(segment_steps / elapsed, 6), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "device": torch.cuda.get_device_name(device), "torch": str(torch.__version__)}}
    _atomic_torch(destination, payload); return _load_checkpoint(destination, contract)


def run_supervisor(output: Path, *, total_steps: int, segment_steps: int, batch_size: int, config: CellularNCAConfig, max_attempts: int = 3) -> dict[str, Any]:
    output = Path(output).resolve(); contract = prepare_training(output, total_steps=total_steps, segment_steps=segment_steps, batch_size=batch_size, config=config); telemetry: list[dict[str, Any]] = []
    for end_step in range(segment_steps, total_steps + 1, segment_steps):
        if (output / checkpoint_name(end_step)).exists(): _load_checkpoint(output / checkpoint_name(end_step), contract); continue
        for attempt in range(1, max_attempts + 1):
            require_disk_floor(output, floor_gb=100, planned_bytes=1024**3)
            env = os.environ.copy(); env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
            started = time.perf_counter(); process = subprocess.run([sys.executable, "-m", "forge.cellular_nca", "segment", "--output", str(output), "--end-step", str(end_step)], cwd=str(Path(__file__).resolve().parents[2]), env=env, capture_output=True, text=True)
            record = {"end_step": end_step, "attempt": attempt, "returncode": process.returncode, "seconds": round(time.perf_counter() - started, 6), "stdout_tail": process.stdout[-2000:], "stderr_tail": process.stderr[-4000:], "windows_access_violation": process.returncode in (-1073741819, 3221225477)}; telemetry.append(record); _atomic_bytes(output / TELEMETRY_NAME, canonical_json_bytes({"format": "nullvector-cellular-nca-training-telemetry-v1", "source_sha256": source_sha256(), "attempts": telemetry}))
            if process.returncode == 0:
                _load_checkpoint(output / checkpoint_name(end_step), contract); break
        else: raise RuntimeError(f"Cellular NCA segment {end_step} exhausted bounded retries.")
    checkpoint = _load_checkpoint(output / checkpoint_name(total_steps), contract)
    return {"passed": True, "step": total_steps, "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"], "attempt_count": len(telemetry), "retry_count": len(telemetry) - total_steps // segment_steps}


def load_final_checkpoint(output: Path) -> tuple[OrganismCellularAutomaton, dict[str, Any], dict[str, Any]]:
    output = Path(output).resolve(); contract = json.loads((output / TRAINING_NAME).read_text(encoding="utf-8")); checkpoint = _load_checkpoint(output / checkpoint_name(contract["total_steps"]), contract)
    model = OrganismCellularAutomaton(CellularNCAConfig(**contract["model"])); model.load_state_dict(checkpoint["ema_state"], strict=True); model.eval()
    return model, checkpoint, contract
