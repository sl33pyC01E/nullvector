from __future__ import annotations

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
import torch.nn.functional as F

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..organism_latent_flow.artifacts import MANIFEST_NAME as FLOW_MANIFEST_NAME
from ..organism_latent_flow.contract import canonical_json_bytes as flow_canonical_json_bytes
from ..organism_latent_flow.corpus import load_latent_corpus
from ..organism_latent_flow.training import CORPUS_NAME as FLOW_CORPUS_NAME
from ..organism_latent_flow.training import load_final_checkpoint as load_flow_checkpoint
from ..organism_raster_vae_v2.smoke import CHECKPOINT_NAME as VAE_CHECKPOINT_NAME
from ..organism_raster_vae_v2.smoke import _load_checkpoint as load_vae_checkpoint
from ..organism_latent_flow.contract import VAE_OUTPUT
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FLOW_CHECKPOINT_SHA256, FLOW_MANIFEST_SHA256, FLOW_OUTPUT, FLOW_SOURCE_SHA256, OrganismRefinerConfig, canonical_json_bytes, sha256_file, source_manifest, source_sha256
from .model import HierarchicalLatentRefiner, latent_refinement_loss


SEED: Final[int] = 0x4F5247524546494E
TRAINING_NAME: Final[str] = "training_contract.json"
TELEMETRY_NAME: Final[str] = "training_telemetry.json"
MAX_CHECKPOINT_BYTES: Final[int] = 768 * 1024 * 1024


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    try: torch.save(payload, temporary); os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _state_hash(state: dict[str, Tensor]) -> str: return tensor_state_sha256({name: value.detach().cpu() for name, value in state.items()})
def checkpoint_name(step: int) -> str: return f"segment_{step:07d}.pt"


def _flow_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = (FLOW_OUTPUT / FLOW_MANIFEST_NAME).read_bytes(); manifest = json.loads(encoded)
    if encoded != flow_canonical_json_bytes(manifest) or manifest.get("source_sha256") != FLOW_SOURCE_SHA256 or manifest.get("manifest_sha256") != FLOW_MANIFEST_SHA256 or manifest.get("artifacts", {}).get("checkpoint", {}).get("sha256") != FLOW_CHECKPOINT_SHA256:
        raise ValueError("Frozen organism flow authority changed.")
    _, checkpoint, contract = load_flow_checkpoint(FLOW_OUTPUT)
    return checkpoint, contract


def _contract(total_steps: int, segment_steps: int, batch_size: int, config: OrganismRefinerConfig, corpus_sha: str) -> dict[str, Any]:
    return {"format": "nullvector-organism-refiner-training-contract/1.0.0", "source_sha256": source_sha256(), "source_manifest": source_manifest(), "flow_source_sha256": FLOW_SOURCE_SHA256, "flow_manifest_sha256": FLOW_MANIFEST_SHA256, "flow_checkpoint_sha256": FLOW_CHECKPOINT_SHA256, "corpus_semantic_sha256": corpus_sha, "config": config.to_dict(), "seed": SEED, "total_steps": total_steps, "segment_steps": segment_steps, "batch_size": batch_size, "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-4, "gradient_clip": 1.0}, "precision": "bf16-autocast-float32-loss", "refinement_schedule": [.45, .25, .12]}


def prepare_training(output: Path, *, total_steps: int, segment_steps: int, batch_size: int, config: OrganismRefinerConfig) -> dict[str, Any]:
    output = Path(output).resolve()
    if type(total_steps) is not int or not 64 <= total_steps <= 100_000 or type(segment_steps) is not int or not 32 <= segment_steps <= total_steps or total_steps % segment_steps or type(batch_size) is not int or not 8 <= batch_size <= 192:
        raise ValueError("Organism refiner schedule drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=3 * 1024**3); output.mkdir(parents=True, exist_ok=True)
    if (output / "organism_refiner_manifest.json").exists(): raise FileExistsError("Finalized organism refiner output is immutable.")
    _flow_authority(); corpus = load_latent_corpus(FLOW_OUTPUT / FLOW_CORPUS_NAME)
    contract = _contract(total_steps, segment_steps, batch_size, config, corpus["semantic"]["semantic_sha256"]); path = output / TRAINING_NAME
    if path.exists():
        encoded = path.read_bytes(); existing = json.loads(encoded)
        if encoded != canonical_json_bytes(existing) or existing != contract: raise ValueError("Organism refiner training contract changed during resume.")
    else: _atomic_bytes(path, canonical_json_bytes(contract))
    return contract


def _load_checkpoint(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES: raise ValueError("Organism refiner checkpoint missing or oversized.")
    payload = torch.load(path, map_location="cpu", weights_only=True); required = {"format", "source_sha256", "source_manifest", "contract", "step", "model_state", "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256", "cuda_generator_state", "history", "runtime"}
    if not isinstance(payload, dict) or set(payload) != required or payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source_sha256() or payload["source_manifest"] != source_manifest() or payload["contract"] != contract: raise ValueError("Organism refiner checkpoint provenance drifted.")
    if type(payload["step"]) is not int or not 1 <= payload["step"] <= contract["total_steps"] or not isinstance(payload["history"], list) or len(payload["history"]) != payload["step"]: raise ValueError("Organism refiner history census drifted.")
    model = HierarchicalLatentRefiner(OrganismRefinerConfig(**contract["config"])); model.load_state_dict(payload["model_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["model_state_sha256"]: raise ValueError("Organism refiner model-state hash failed.")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_hash(model.state_dict()) != payload["ema_state_sha256"]: raise ValueError("Organism refiner EMA-state hash failed.")
    if not isinstance(payload["cuda_generator_state"], Tensor) or payload["cuda_generator_state"].dtype != torch.uint8 or any(not isinstance(row, dict) or row.get("step") != index or not math.isfinite(float(row.get("loss", math.nan))) for index, row in enumerate(payload["history"], 1)): raise ValueError("Organism refiner RNG/history drifted.")
    return payload


def _family_table(family: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    members = torch.zeros((5, 11), dtype=torch.long, device=device); counts = torch.zeros(5, dtype=torch.long, device=device)
    for value in range(5):
        indices = torch.nonzero(family == value).flatten(); members[value, :len(indices)] = indices; counts[value] = len(indices)
    return members, counts


def _decoder_auxiliary(vae: torch.nn.Module, predicted_coarse: Tensor, predicted_fine: Tensor, clean_coarse: Tensor, clean_fine: Tensor, condition: Tensor) -> Tensor:
    with torch.no_grad(): target = vae.decode(clean_coarse, clean_fine, condition)
    predicted = vae.decode(predicted_coarse, predicted_fine, condition)
    target_alpha = target.occupancy_logits.sigmoid(); predicted_alpha = predicted.occupancy_logits.sigmoid()
    occupancy = F.l1_loss(predicted_alpha.float(), target_alpha.float()); rgba = F.l1_loss(predicted.rgba.float(), target.rgba.float()); physiology = F.l1_loss(predicted.physiology.float(), target.physiology.float())
    target_roles = target.system_role_logits.argmax(2)
    role = -predicted.system_role_logits.float().log_softmax(2).gather(2, target_roles[:, :, None]).mean()
    edge = F.l1_loss(predicted_alpha[:, :, 1:] - predicted_alpha[:, :, :-1], target_alpha[:, :, 1:] - target_alpha[:, :, :-1]) + F.l1_loss(predicted_alpha[:, :, :, 1:] - predicted_alpha[:, :, :, :-1], target_alpha[:, :, :, 1:] - target_alpha[:, :, :, :-1])
    return occupancy + .7 * rgba + .35 * physiology + .08 * role + .25 * edge


def train_segment(output: Path, *, end_step: int) -> dict[str, Any]:
    output = Path(output).resolve(); encoded = (output / TRAINING_NAME).read_bytes(); contract = json.loads(encoded)
    if encoded != canonical_json_bytes(contract) or contract.get("source_sha256") != source_sha256() or contract.get("source_manifest") != source_manifest(): raise ValueError("Organism refiner training authority drifted.")
    segment_steps = contract["segment_steps"]
    if type(end_step) is not int or end_step % segment_steps or not segment_steps <= end_step <= contract["total_steps"]: raise ValueError("Organism refiner segment endpoint drifted.")
    destination = output / checkpoint_name(end_step)
    if destination.exists(): return _load_checkpoint(destination, contract)
    previous_step = end_step - segment_steps; previous_path = output / checkpoint_name(previous_step) if previous_step else None
    if previous_path is not None and not previous_path.exists(): raise FileNotFoundError("Previous organism refiner segment is missing.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(0)[0] < 10 * 1024**3: raise RuntimeError("Organism refiner requires deterministic CUDA BF16 and 10 GiB free VRAM.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED & 0xffffffff)
    device = torch.device("cuda", 0); torch.cuda.reset_peak_memory_stats(device); corpus = load_latent_corpus(FLOW_OUTPUT / FLOW_CORPUS_NAME); tensors = {key: value.to(device) for key, value in corpus["tensors"].items()}
    if corpus["semantic"]["semantic_sha256"] != contract["corpus_semantic_sha256"]: raise ValueError("Organism refiner corpus drifted.")
    config = OrganismRefinerConfig(**contract["config"]); model = HierarchicalLatentRefiner(config).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=contract["optimizer"]["lr"], weight_decay=contract["optimizer"]["weight_decay"])
    vae, _ = load_vae_checkpoint(VAE_OUTPUT / VAE_CHECKPOINT_NAME); vae.requires_grad_(False).to(device).eval(); generator = torch.Generator(device=device).manual_seed(SEED ^ 0x545241494E); history: list[dict[str, Any]] = []; ema = {name: value.detach().clone() for name, value in model.state_dict().items()}; start_step = 0
    if previous_path is not None:
        previous = _load_checkpoint(previous_path, contract); start_step = previous["step"]; model.load_state_dict(previous["model_state"], strict=True); optimizer.load_state_dict(previous["optimizer_state"]); ema = {name: value.to(device) for name, value in previous["ema_state"].items()}; generator.set_state(previous["cuda_generator_state"]); history = list(previous["history"])
    family_members, family_counts = _family_table(tensors["family"], device); batch_size = contract["batch_size"]; started = time.perf_counter(); model.train()
    for step in range(start_step, end_step):
        left = torch.randint(0, 45, (batch_size,), device=device, generator=generator); family = tensors["family"][left]; rank = torch.randint(0, 11, (batch_size,), device=device, generator=generator) % family_counts[family]; right = family_members[family, rank]
        interpolate = torch.rand((batch_size,), device=device, generator=generator) < config.interpolation_probability; right = torch.where(interpolate, right, left); alpha = torch.where(interpolate, .05 + .9 * torch.rand((batch_size,), device=device, generator=generator), torch.zeros(batch_size, device=device))
        alpha_field = alpha[:, None, None, None]; alpha_condition = alpha[:, None]
        coarse_means = (tensors["coarse_mean"] - tensors["coarse_center"]) / tensors["coarse_scale"]; fine_means = (tensors["fine_mean"] - tensors["fine_center"]) / tensors["fine_scale"]
        clean_coarse = coarse_means[left] * (1 - alpha_field) + coarse_means[right] * alpha_field; clean_fine = fine_means[left] * (1 - alpha_field) + fine_means[right] * alpha_field; condition = tensors["condition"][left] * (1 - alpha_condition) + tensors["condition"][right] * alpha_condition
        log_sigma = math.log(config.corruption_min) + (math.log(config.corruption_max) - math.log(config.corruption_min)) * torch.rand((batch_size,), device=device, generator=generator); sigma = log_sigma.exp()
        raw_coarse = torch.randn(clean_coarse.shape, device=device, generator=generator); raw_fine = torch.randn(clean_fine.shape, device=device, generator=generator); noise_coarse = .72 * raw_coarse + .28 * F.avg_pool2d(raw_coarse, 3, 1, 1); noise_fine = .72 * raw_fine + .28 * F.avg_pool2d(raw_fine, 3, 1, 1)
        corrupted_coarse = clean_coarse + sigma[:, None, None, None] * noise_coarse; corrupted_fine = clean_fine + sigma[:, None, None, None] * noise_fine; optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): coarse_delta, fine_delta = model(corrupted_coarse, corrupted_fine, sigma, condition)
        latent_loss, pieces, predicted_coarse, predicted_fine = latent_refinement_loss(coarse_delta, fine_delta, corrupted_coarse, corrupted_fine, clean_coarse, clean_fine)
        count = min(config.auxiliary_batch, batch_size); indices = slice(0, count); scale_coarse, center_coarse = tensors["coarse_scale"], tensors["coarse_center"]; scale_fine, center_fine = tensors["fine_scale"], tensors["fine_center"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): auxiliary = _decoder_auxiliary(vae, predicted_coarse[indices] * scale_coarse + center_coarse, predicted_fine[indices] * scale_fine + center_fine, clean_coarse[indices] * scale_coarse + center_coarse, clean_fine[indices] * scale_fine + center_fine, condition[indices])
        loss = latent_loss + .35 * auxiliary.float(); loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["optimizer"]["gradient_clip"])
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)): raise FloatingPointError("Organism refiner became non-finite.")
        optimizer.step(); ema_decay = min(config.ema_decay, (step + 2) / (step + 11))
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point: ema[name].lerp_(value.detach(), 1 - ema_decay)
                else: ema[name].copy_(value)
        history.append({"step": step + 1, "loss": round(float(loss), 8), **{name: round(float(value), 8) for name, value in pieces.items()}, "decoder_auxiliary": round(float(auxiliary), 8), "gradient_norm": round(float(gradient), 8)})
    elapsed = time.perf_counter() - started; model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "source_manifest": source_manifest(), "contract": contract, "step": end_step, "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(), "model_state_sha256": _state_hash(model_state), "ema_state_sha256": _state_hash(ema_state), "cuda_generator_state": generator.get_state().cpu(), "history": history, "runtime": {"segment_start": start_step + 1, "segment_end": end_step, "seconds": elapsed, "device": torch.cuda.get_device_name(device), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}}
    require_disk_floor(output, floor_gb=100, planned_bytes=MAX_CHECKPOINT_BYTES); _atomic_torch(destination, payload); return _load_checkpoint(destination, contract)


def run_supervisor(output: Path, *, total_steps: int = 4096, segment_steps: int = 512, batch_size: int = 64, max_attempts: int = 3) -> dict[str, Any]:
    output = Path(output).resolve(); contract = prepare_training(output, total_steps=total_steps, segment_steps=segment_steps, batch_size=batch_size, config=OrganismRefinerConfig()); telemetry: list[dict[str, Any]] = []
    for end_step in range(segment_steps, total_steps + 1, segment_steps):
        checkpoint = output / checkpoint_name(end_step)
        if checkpoint.exists(): _load_checkpoint(checkpoint, contract); telemetry.append({"end_step": end_step, "attempt": 0, "status": "reused"}); continue
        passed = False
        for attempt in range(1, max_attempts + 1):
            environment = dict(os.environ); environment.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}); command = [sys.executable, "-m", "forge.organism_latent_refiner", "segment", "--output", str(output), "--end-step", str(end_step)]; started = time.time(); result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True, text=True)
            record = {"end_step": end_step, "attempt": attempt, "status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode, "seconds": time.time() - started, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-4000:]}; telemetry.append(record); _atomic_bytes(output / TELEMETRY_NAME, canonical_json_bytes({"format": "nullvector-organism-refiner-telemetry/1.0.0", "records": telemetry}))
            if result.returncode == 0: _load_checkpoint(checkpoint, contract); passed = True; break
        if not passed: raise RuntimeError(f"Organism refiner segment {end_step} failed after {max_attempts} attempts.")
    return {"contract": contract, "final_checkpoint": str(output / checkpoint_name(total_steps)), "telemetry": telemetry}


def load_final_checkpoint(output: Path) -> tuple[HierarchicalLatentRefiner, dict[str, Any], dict[str, Any]]:
    output = Path(output).resolve(); encoded = (output / TRAINING_NAME).read_bytes(); contract = json.loads(encoded)
    if encoded != canonical_json_bytes(contract): raise ValueError("Organism refiner contract is not canonical.")
    payload = _load_checkpoint(output / checkpoint_name(contract["total_steps"]), contract); model = HierarchicalLatentRefiner(OrganismRefinerConfig(**contract["config"])); model.load_state_dict(payload["ema_state"], strict=True); model.eval(); return model, payload, contract
