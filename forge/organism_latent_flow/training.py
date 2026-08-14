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

import numpy as np
import torch
from torch import Tensor

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, OrganismFlowConfig, canonical_json_bytes, source_manifest, source_sha256
from .corpus import build_latent_corpus, load_latent_corpus, save_latent_corpus
from .model import HierarchicalOrganismFlow, flow_matching_loss


SEED: Final[int] = 0x4F5247464C4F57
CORPUS_NAME: Final[str] = "latent_corpus.pt"
TRAINING_NAME: Final[str] = "training_contract.json"
TELEMETRY_NAME: Final[str] = "training_telemetry.json"
MAX_CHECKPOINT_BYTES: Final[int] = 1024 * 1024 * 1024


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_hash(state: dict[str, Tensor]) -> str:
    return tensor_state_sha256({name: value.detach().cpu() for name, value in state.items()})


def checkpoint_name(step: int) -> str:
    return f"segment_{step:07d}.pt"


def _contract(total_steps: int, segment_steps: int, batch_size: int, config: OrganismFlowConfig, corpus_semantic_sha256: str) -> dict[str, Any]:
    return {
        "format": "nullvector-organism-flow-training-contract/1.0.0",
        "source_sha256": source_sha256(), "source_manifest": source_manifest(),
        "corpus_semantic_sha256": corpus_semantic_sha256, "config": config.to_dict(),
        "seed": SEED, "total_steps": total_steps, "segment_steps": segment_steps,
        "batch_size": batch_size, "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-4, "gradient_clip": 1.0},
        "precision": "bf16-autocast-float32-loss", "integration": {"method": "midpoint-time-euler", "steps": 32, "guidance": 1.6},
    }


def prepare_training(output: Path, *, total_steps: int, segment_steps: int, batch_size: int, config: OrganismFlowConfig) -> dict[str, Any]:
    output = Path(output).resolve()
    if type(total_steps) is not int or not 64 <= total_steps <= 100_000 or type(segment_steps) is not int or not 32 <= segment_steps <= total_steps or total_steps % segment_steps:
        raise ValueError("Organism flow segment schedule drifted.")
    if type(batch_size) is not int or not 8 <= batch_size <= 256:
        raise ValueError("Organism flow batch size drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=4 * 1024**3)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "organism_flow_manifest.json").exists():
        raise FileExistsError("Finalized organism flow output is immutable.")
    corpus_path = output / CORPUS_NAME
    if not corpus_path.exists():
        corpus = build_latent_corpus()
        temporary = output / f".{CORPUS_NAME}.tmp-{os.getpid()}"
        save_latent_corpus(temporary, corpus)
        os.replace(temporary, corpus_path)
    corpus = load_latent_corpus(corpus_path)
    contract = _contract(total_steps, segment_steps, batch_size, config, corpus["semantic"]["semantic_sha256"])
    contract_path = output / TRAINING_NAME
    if contract_path.exists():
        encoded = contract_path.read_bytes()
        existing = json.loads(encoded)
        if encoded != canonical_json_bytes(existing) or existing != contract:
            raise ValueError("Organism flow training contract changed during resume.")
    else:
        _atomic_bytes(contract_path, canonical_json_bytes(contract))
    return contract


def _load_checkpoint(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Organism flow checkpoint missing or oversized.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "source_sha256", "source_manifest", "contract", "step", "model_state", "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256", "cuda_generator_state", "history", "runtime"}
    if not isinstance(payload, dict) or set(payload) != required or payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source_sha256() or payload["source_manifest"] != source_manifest() or payload["contract"] != contract:
        raise ValueError("Organism flow checkpoint provenance drifted.")
    if type(payload["step"]) is not int or not 1 <= payload["step"] <= contract["total_steps"] or not isinstance(payload["history"], list) or len(payload["history"]) != payload["step"]:
        raise ValueError("Organism flow checkpoint step/history drifted.")
    config = OrganismFlowConfig(**contract["config"])
    model = HierarchicalOrganismFlow(config)
    model.load_state_dict(payload["model_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("Organism flow model-state hash failed.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["ema_state_sha256"]:
        raise ValueError("Organism flow EMA-state hash failed.")
    if not isinstance(payload["cuda_generator_state"], Tensor) or payload["cuda_generator_state"].dtype != torch.uint8:
        raise ValueError("Organism flow RNG state drifted.")
    if any(not isinstance(row, dict) or row.get("step") != index or not math.isfinite(float(row.get("loss", math.nan))) for index, row in enumerate(payload["history"], 1)):
        raise ValueError("Organism flow history drifted.")
    return payload


def train_segment(output: Path, *, end_step: int) -> dict[str, Any]:
    output = Path(output).resolve()
    contract_path = output / TRAINING_NAME
    encoded = contract_path.read_bytes(); contract = json.loads(encoded)
    if encoded != canonical_json_bytes(contract) or contract.get("source_sha256") != source_sha256() or contract.get("source_manifest") != source_manifest():
        raise ValueError("Organism flow training authority drifted.")
    segment_steps = contract["segment_steps"]
    if type(end_step) is not int or end_step % segment_steps or not segment_steps <= end_step <= contract["total_steps"]:
        raise ValueError("Organism flow segment endpoint drifted.")
    destination = output / checkpoint_name(end_step)
    if destination.exists():
        return _load_checkpoint(destination, contract)
    previous_step = end_step - segment_steps
    previous_path = output / checkpoint_name(previous_step) if previous_step else None
    if previous_path is not None and not previous_path.exists():
        raise FileNotFoundError("Previous organism flow segment is missing.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Organism flow segment requires deterministic CUDA BF16.")
    if torch.cuda.mem_get_info(0)[0] < 8 * 1024**3:
        raise RuntimeError("Organism flow segment requires 8 GiB free VRAM.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED & 0xffffffff)
    device = torch.device("cuda", 0); torch.cuda.reset_peak_memory_stats(device)
    corpus = load_latent_corpus(output / CORPUS_NAME)
    if corpus["semantic"]["semantic_sha256"] != contract["corpus_semantic_sha256"]:
        raise ValueError("Organism flow corpus identity drifted.")
    tensors = {key: value.to(device) for key, value in corpus["tensors"].items()}
    config = OrganismFlowConfig(**contract["config"])
    model = HierarchicalOrganismFlow(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract["optimizer"]["lr"], weight_decay=contract["optimizer"]["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(SEED ^ 0x545241494E)
    history: list[dict[str, Any]] = []
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    start_step = 0
    if previous_path is not None:
        previous = _load_checkpoint(previous_path, contract); start_step = previous["step"]
        model.load_state_dict(previous["model_state"], strict=True); optimizer.load_state_dict(previous["optimizer_state"])
        ema = {name: value.to(device) for name, value in previous["ema_state"].items()}
        # CUDA generators deliberately serialize their state as a CPU
        # ByteTensor; set_state performs the device transfer internally.
        generator.set_state(previous["cuda_generator_state"]); history = list(previous["history"])
    if start_step != previous_step:
        raise ValueError("Organism flow resume step drifted.")
    batch_size = contract["batch_size"]; started = time.perf_counter(); model.train()
    for step in range(start_step, end_step):
        indices = torch.randint(0, 45, (batch_size,), device=device, generator=generator)
        coarse_posterior = tensors["coarse_mean"][indices] + config.posterior_noise * torch.exp(.5 * tensors["coarse_log_variance"][indices]) * torch.randn((batch_size, 32, 12, 12), device=device, generator=generator)
        fine_posterior = tensors["fine_mean"][indices] + config.posterior_noise * torch.exp(.5 * tensors["fine_log_variance"][indices]) * torch.randn((batch_size, 16, 24, 24), device=device, generator=generator)
        target_coarse = (coarse_posterior - tensors["coarse_center"]) / tensors["coarse_scale"]
        target_fine = (fine_posterior - tensors["fine_center"]) / tensors["fine_scale"]
        noise_coarse = torch.randn(target_coarse.shape, device=device, generator=generator)
        noise_fine = torch.randn(target_fine.shape, device=device, generator=generator)
        time_value = .01 + .98 * torch.rand((batch_size,), device=device, generator=generator)
        keep = torch.rand((batch_size,), device=device, generator=generator) >= config.condition_dropout
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, pieces = flow_matching_loss(model, target_coarse, target_fine, tensors["condition"][indices], time_value, noise_coarse, noise_fine, keep)
        loss.float().backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["optimizer"]["gradient_clip"])
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)):
            raise FloatingPointError("Organism flow training became non-finite.")
        optimizer.step()
        with torch.no_grad():
            # Warm the EMA toward the trained network quickly, then asymptote
            # to the long-horizon decay.  A fixed .999 decay leaves short
            # calibrations dominated by random initialization and obscures
            # whether topology is actually learning.
            ema_decay = min(config.ema_decay, (step + 2) / (step + 11))
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    ema[name].lerp_(value.detach(), 1 - ema_decay)
                else:
                    ema[name].copy_(value)
        history.append({"step": step + 1, **{name: round(float(value), 8) for name, value in pieces.items()}, "gradient_norm": round(float(gradient), 8)})
    elapsed = time.perf_counter() - started
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    payload = {
        "format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "source_manifest": source_manifest(), "contract": contract,
        "step": end_step, "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": _state_hash(model_state), "ema_state_sha256": _state_hash(ema_state),
        "cuda_generator_state": generator.get_state().cpu(), "history": history,
        "runtime": {"segment_start": start_step + 1, "segment_end": end_step, "seconds": elapsed, "device": torch.cuda.get_device_name(device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
    }
    require_disk_floor(output, floor_gb=100, planned_bytes=MAX_CHECKPOINT_BYTES)
    _atomic_torch(destination, payload)
    return _load_checkpoint(destination, contract)


def run_supervisor(output: Path, *, total_steps: int = 4096, segment_steps: int = 512, batch_size: int = 90, max_attempts: int = 3) -> dict[str, Any]:
    output = Path(output).resolve(); config = OrganismFlowConfig()
    contract = prepare_training(output, total_steps=total_steps, segment_steps=segment_steps, batch_size=batch_size, config=config)
    telemetry: list[dict[str, Any]] = []
    for end_step in range(segment_steps, total_steps + 1, segment_steps):
        checkpoint = output / checkpoint_name(end_step)
        if checkpoint.exists():
            _load_checkpoint(checkpoint, contract); telemetry.append({"end_step": end_step, "attempt": 0, "status": "reused"}); continue
        passed = False
        for attempt in range(1, max_attempts + 1):
            environment = dict(os.environ); environment.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
            command = [sys.executable, "-m", "forge.organism_latent_flow", "segment", "--output", str(output), "--end-step", str(end_step)]
            started = time.time(); result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True, text=True)
            record = {"end_step": end_step, "attempt": attempt, "status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode, "seconds": time.time() - started, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-4000:]}
            telemetry.append(record); _atomic_bytes(output / TELEMETRY_NAME, canonical_json_bytes({"format": "nullvector-organism-flow-telemetry/1.0.0", "records": telemetry}))
            if result.returncode == 0:
                _load_checkpoint(checkpoint, contract); passed = True; break
        if not passed:
            raise RuntimeError(f"Organism flow segment {end_step} failed after {max_attempts} attempts.")
    return {"contract": contract, "final_checkpoint": str(output / checkpoint_name(total_steps)), "telemetry": telemetry}


def load_final_checkpoint(output: Path) -> tuple[HierarchicalOrganismFlow, dict[str, Any], dict[str, Any]]:
    output = Path(output).resolve(); encoded = (output / TRAINING_NAME).read_bytes(); contract = json.loads(encoded)
    if encoded != canonical_json_bytes(contract):
        raise ValueError("Organism flow training contract is not canonical.")
    payload = _load_checkpoint(output / checkpoint_name(contract["total_steps"]), contract)
    model = HierarchicalOrganismFlow(OrganismFlowConfig(**contract["config"])); model.load_state_dict(payload["ema_state"], strict=True); model.eval()
    return model, payload, contract
