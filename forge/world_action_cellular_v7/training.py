from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..action_teacher_v1.contract import ACTIONS
from ..world_action_sparse_v5.contract import CHECKPOINT_FORMAT as V5_CHECKPOINT_FORMAT, ModelConfig as V5ModelConfig, source_sha256 as v5_source_sha256
from ..world_action_sparse_v5.model import SparseActionDiT
from ..world_action_sparse_v5.training import TARGETED_INDICES, counterfactual_control, latent_edit_mask
from .checkpoint import RecoveryCheckpointStore, load_recovery_checkpoint
from .contract import CHECKPOINT_FORMAT, REPORT_FORMAT, ModelConfig, TrainingConfig, canonical, config_dict, source_sha256
from .corpus import load_encoded_corpus
from .model import CellularTemporalActionDiT, load_v5_latent_editor


ACTOR_STATE_CHANGED_WEIGHT = 8.0
ACTOR_FIELD_CHANGED_WEIGHT = 24.0
ACTOR_FIELD_GATE_WEIGHT = 0.20


def _process_rss_bytes() -> int:
    """Return resident process memory without a third-party dependency."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = (
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            )
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError("unable to read cellular action process memory")
        return int(counters.working_set_size)
    statm = Path("/proc/self/statm")
    if statm.is_file():
        return int(statm.read_text().split()[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    return 0


class ResourceGuard:
    """Enforce bounded host/GPU usage and report actual peaks."""

    def __init__(self, training: TrainingConfig, device: torch.device):
        if not 1 <= training.cpu_threads <= 24:
            raise ValueError("cellular action cpu thread limit is invalid")
        if not 0.1 <= training.cuda_memory_fraction <= 0.95:
            raise ValueError("cellular action CUDA memory fraction is invalid")
        if not 1.0 <= training.max_process_memory_gib <= 128.0:
            raise ValueError("cellular action process memory limit is invalid")
        if not 0.1 <= training.target_duty_cycle <= 1.0:
            raise ValueError("cellular action duty cycle is invalid")
        if training.validation_batch_size < 1:
            raise ValueError("cellular action validation batch size is invalid")
        self.training = training
        self.device = device
        self.maximum_rss_bytes = 0
        self.maximum_cuda_allocated_bytes = 0
        self.maximum_cuda_reserved_bytes = 0
        self.below_normal_priority_applied = False
        torch.set_num_threads(training.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(training.cuda_memory_fraction, torch.cuda.current_device())
        if os.name == "nt":
            # BELOW_NORMAL_PRIORITY_CLASS. Failure is harmless and reported by
            # the absent effect rather than treated as a training failure.
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
            kernel32.SetPriorityClass.restype = ctypes.c_int
            self.below_normal_priority_applied = bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000))
        self.sample()

    @property
    def memory_limit_bytes(self) -> int:
        return int(self.training.max_process_memory_gib * 1024**3)

    def sample(self) -> None:
        rss = _process_rss_bytes()
        self.maximum_rss_bytes = max(self.maximum_rss_bytes, rss)
        if rss > self.memory_limit_bytes:
            raise MemoryError(f"cellular action process memory limit exceeded: {rss} > {self.memory_limit_bytes}")
        if self.device.type == "cuda":
            self.maximum_cuda_allocated_bytes = max(self.maximum_cuda_allocated_bytes, torch.cuda.max_memory_allocated(self.device))
            self.maximum_cuda_reserved_bytes = max(self.maximum_cuda_reserved_bytes, torch.cuda.max_memory_reserved(self.device))

    def throttle(self, active_seconds: float) -> None:
        self.sample()
        if self.training.target_duty_cycle < 1.0 and active_seconds > 0:
            time.sleep(active_seconds * (1.0 / self.training.target_duty_cycle - 1.0))

    def report(self) -> dict:
        self.sample()
        return {
            "cpu_threads": self.training.cpu_threads,
            "cuda_memory_fraction": self.training.cuda_memory_fraction,
            "max_process_memory_gib": self.training.max_process_memory_gib,
            "target_duty_cycle": self.training.target_duty_cycle,
            "below_normal_priority_applied": self.below_normal_priority_applied,
            "maximum_rss_bytes": self.maximum_rss_bytes,
            "maximum_cuda_allocated_bytes": self.maximum_cuda_allocated_bytes,
            "maximum_cuda_reserved_bytes": self.maximum_cuda_reserved_bytes,
        }


def _state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0" + value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _group(episodes):
    return {name: np.concatenate([episode[name] for episode in episodes]) for name in episodes[0]}


def _split(episodes, manifest, train_sessions, validation_sessions, test_sessions):
    sets = tuple(set(group) for group in (train_sessions, validation_sessions, test_sessions))
    if any(not group for group in sets) or sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("cellular action world splits overlap or are empty")
    available = {record["session_id"]: episode for record, episode in zip(manifest["shards"], episodes)}
    requested = set().union(*sets)
    if requested != set(available):
        raise ValueError("cellular action split must account for every encoded world exactly once")
    return tuple(tuple(available[name] for name in group) for group in (train_sessions, validation_sessions, test_sessions))


def _masked_mae(predicted, target, mask):
    error = (predicted - target).abs()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(1)
    expanded = mask.expand_as(error)
    return float((error * expanded).sum() / expanded.sum().clamp_min(1))


def selection_score(metrics: dict) -> float:
    ratios = (
        metrics["latent_mae"] / max(metrics["latent_persistence_mae"], 1e-8),
        metrics["changed_latent_mae"] / max(metrics["changed_latent_persistence_mae"], 1e-8),
        metrics["actor_state_mae"] / max(metrics["actor_state_persistence_mae"], 1e-8),
        metrics["changed_actor_field_mae"] / max(metrics["changed_actor_field_persistence_mae"], 1e-8),
    )
    penalty = max(0.0, -metrics["correct_action_advantage"]) * 3 + max(0.0, -metrics["targeted_control_advantage"]) * 3
    return float(sum(ratios) / len(ratios) + penalty)


def _predict(model, group, latent_mean, latent_std, actor_mean, actor_std, device, *, batch_size=24):
    outputs = {name: [] for name in ("latent", "actor_state", "actor_field", "gate", "wrong_action", "wrong_control")}
    wrong_actions = (group["action"].astype(np.int64) + 7) % len(ACTIONS)
    wrong_controls = counterfactual_control(group["control"])
    for start in range(0, len(group["current"]), batch_size):
        stop = start + batch_size
        current = torch.from_numpy(group["current"][start:stop]).to(device)
        previous = torch.from_numpy(group["previous"][start:stop]).to(device)
        current_n = (current - latent_mean) / latent_std
        previous_n = (previous - latent_mean) / latent_std
        action = torch.from_numpy(group["action"][start:stop].astype(np.int64)).to(device)
        previous_action = torch.from_numpy(group["previous_action"][start:stop].astype(np.int64)).to(device)
        control = torch.from_numpy(group["control"][start:stop]).to(device)
        previous_control = torch.from_numpy(group["previous_control"][start:stop]).to(device)
        state = torch.from_numpy(group["state"][start:stop]).to(device)
        actor = torch.from_numpy(group["actor_state"][start:stop]).to(device)
        actor_n = (actor - actor_mean) / actor_std
        field = torch.from_numpy(group["actor_field"][start:stop]).float().to(device)
        time = torch.zeros(len(current), device=device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latent_n, next_actor_n, next_field, gate, *_ = model.edit(current_n, previous_n, time, action, control, state, actor_n, field, previous_action, previous_control)
            wrong_action_n = model.edit(current_n, previous_n, time, torch.from_numpy(wrong_actions[start:stop]).to(device), control, state, actor_n, field, previous_action, previous_control)[0]
            wrong_control_n = model.edit(current_n, previous_n, time, action, torch.from_numpy(wrong_controls[start:stop]).to(device), state, actor_n, field, previous_action, previous_control)[0]
        outputs["latent"].append((latent_n * latent_std + latent_mean).float().cpu())
        outputs["actor_state"].append((next_actor_n * actor_std + actor_mean).float().cpu())
        outputs["actor_field"].append(next_field.float().cpu())
        outputs["gate"].append(gate.float().cpu())
        outputs["wrong_action"].append((wrong_action_n * latent_std + latent_mean).float().cpu())
        outputs["wrong_control"].append((wrong_control_n * latent_std + latent_mean).float().cpu())
    return {name: torch.cat(value) for name, value in outputs.items()}


def evaluate(model, episodes, latent_mean, latent_std, actor_mean, actor_std, device, *, batch_size=24):
    group = _group(episodes)
    output = _predict(model, group, latent_mean, latent_std, actor_mean, actor_std, device, batch_size=batch_size)
    current = torch.from_numpy(group["current"])
    target = torch.from_numpy(group["target"])
    actor = torch.from_numpy(group["actor_state"])
    target_actor = torch.from_numpy(group["target_actor_state"])
    field = torch.from_numpy(group["actor_field"]).float()
    target_field = torch.from_numpy(group["target_actor_field"]).float()
    latent_mask = torch.from_numpy(latent_edit_mask(group["current_frame"], group["target_frame"]).astype(np.float32)) > 0.5
    field_mask = (target_field - field).abs().amax(1) > 1e-3
    targeted = torch.from_numpy(np.isin(group["action"].astype(np.int64), TARGETED_INDICES))
    metrics = {
        "pairs": len(current),
        "latent_mae": float(F.l1_loss(output["latent"], target)),
        "latent_persistence_mae": float(F.l1_loss(current, target)),
        "changed_latent_mae": _masked_mae(output["latent"], target, latent_mask),
        "changed_latent_persistence_mae": _masked_mae(current, target, latent_mask),
        "actor_state_mae": float(F.l1_loss(output["actor_state"], target_actor)),
        "actor_state_persistence_mae": float(F.l1_loss(actor, target_actor)),
        "actor_field_mae": float(F.l1_loss(output["actor_field"], target_field)),
        "actor_field_persistence_mae": float(F.l1_loss(field, target_field)),
        "changed_actor_field_mae": _masked_mae(output["actor_field"], target_field, field_mask),
        "changed_actor_field_persistence_mae": _masked_mae(field, target_field, field_mask),
        "correct_action_advantage": float(F.l1_loss(output["wrong_action"], target) - F.l1_loss(output["latent"], target)),
        "targeted_control_advantage": float(F.l1_loss(output["wrong_control"][targeted], target[targeted]) - F.l1_loss(output["latent"][targeted], target[targeted])) if bool(targeted.any()) else 0.0,
        "edit_gate_mean": float(output["gate"].mean()),
    }
    metrics.update({
        "latent_improvement": 1 - metrics["latent_mae"] / max(metrics["latent_persistence_mae"], 1e-8),
        "actor_state_improvement": 1 - metrics["actor_state_mae"] / max(metrics["actor_state_persistence_mae"], 1e-8),
        "actor_field_improvement": 1 - metrics["actor_field_mae"] / max(metrics["actor_field_persistence_mae"], 1e-8),
    })
    return metrics


def _checkpoint_payload(*, corpus_sha, model_config, training_config, warm, step, model, ema, optimizer, rng, actor_mean, actor_std, history, validation_history, best_score, best_step, best_ema):
    return {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source_sha256(),
        "corpus_sha256": corpus_sha,
        "model_config": config_dict(model_config),
        "training_config": config_dict(training_config),
        "warm_start": warm,
        "step": step,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "ema_state": {name: value.detach().cpu() for name, value in ema.items()},
        "optimizer_state": optimizer.state_dict(),
        "rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "actor_mean": actor_mean.tolist(),
        "actor_std": actor_std.tolist(),
        "latent_mean": warm["latent_mean"],
        "latent_std": warm["latent_std"],
        "history": history,
        "validation_history": validation_history,
        "best_score": best_score,
        "best_step": best_step,
        "best_ema_state": best_ema,
        "status": "training",
    }


def train(output: Path, corpus: Path, warm_start: Path, *, train_sessions, validation_sessions, test_sessions, training=TrainingConfig(), resume: Path | None = None, stop_after_step: int | None = None):
    output = Path(output)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("cellular action resume output missing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resources = ResourceGuard(training, device)
    episodes, manifest = load_encoded_corpus(corpus)
    resources.sample()
    train_episodes, validation_episodes, test_episodes = _split(episodes, manifest, train_sessions, validation_sessions, test_sessions)
    corpus_sha = manifest["manifest_sha256"]
    warm_payload = torch.load(Path(warm_start), map_location="cpu", weights_only=True)
    if warm_payload.get("format") != V5_CHECKPOINT_FORMAT or warm_payload.get("source_sha256") != v5_source_sha256():
        raise ValueError("cellular action v5 warm-start provenance drifted")
    parent_config = V5ModelConfig(**warm_payload["model_config"])
    model_config = ModelConfig(width=parent_config.width, layers=parent_config.layers, heads=parent_config.heads, patch=parent_config.patch, spatial_channels=parent_config.spatial_channels, gate_bias=parent_config.gate_bias)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(training.seed)
    np.random.seed(training.seed & 0xFFFFFFFF)
    random.seed(training.seed)
    parent = SparseActionDiT(parent_config); parent.load_state_dict(warm_payload["ema_state"])
    model = CellularTemporalActionDiT(model_config); load_v5_latent_editor(model, parent); model.to(device)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=2e-3, fused=device.type == "cuda")
    train_group = _group(train_episodes)
    actor_mean_array = train_group["actor_state"].mean(0).astype(np.float32)
    actor_std_array = (train_group["actor_state"].std(0) + 1e-4).astype(np.float32)
    train_field_change = np.abs(train_group["target_actor_field"].astype(np.float32) - train_group["actor_field"].astype(np.float32)) > 1e-3
    field_change_frequency = train_field_change.mean((0, 2, 3), dtype=np.float64)
    field_gate_pos_weight = np.clip((1.0 - field_change_frequency) / np.maximum(field_change_frequency, 1e-8), 1.0, 128.0).astype(np.float32)
    rng = np.random.default_rng(training.seed)
    history = []; validation_history = []; best_score = float("inf"); best_step = 0; best_ema = None; start_step = 0
    if resume is not None:
        recovery = load_recovery_checkpoint(resume, corpus_sha256=corpus_sha)
        if recovery["model_config"] != config_dict(model_config) or recovery["training_config"] != config_dict(training):
            raise ValueError("cellular action resume configuration drifted")
        model.load_state_dict(recovery["model_state"]); model.to(device)
        ema = {name: value.to(device) for name, value in recovery["ema_state"].items()}
        optimizer.load_state_dict(recovery["optimizer_state"])
        rng.bit_generator.state = recovery["rng_state"]
        torch.set_rng_state(recovery["torch_rng_state"])
        if device.type == "cuda" and recovery["cuda_rng_state"]: torch.cuda.set_rng_state_all(recovery["cuda_rng_state"])
        actor_mean_array = np.asarray(recovery["actor_mean"], np.float32); actor_std_array = np.asarray(recovery["actor_std"], np.float32)
        history = list(recovery["history"]); validation_history = list(recovery["validation_history"]); best_score = float(recovery["best_score"]); best_step = int(recovery["best_step"]); best_ema = recovery["best_ema_state"]; start_step = int(recovery["step"])
    latent_mean = torch.as_tensor(warm_payload["latent_mean"], device=device).view(1, -1, 1, 1)
    latent_std = torch.as_tensor(warm_payload["latent_std"], device=device).view(1, -1, 1, 1)
    actor_mean = torch.from_numpy(actor_mean_array).to(device).view(1, -1)
    actor_std = torch.from_numpy(actor_std_array).to(device).view(1, -1)
    edit_masks = latent_edit_mask(train_group["current_frame"], train_group["target_frame"]).astype(np.float32)
    store = RecoveryCheckpointStore(output / "checkpoints")
    validation_model = CellularTemporalActionDiT(model_config).to(device).eval()
    warm_record = {"path": str(warm_start), "source_sha256": warm_payload["source_sha256"], "ema_sha256": warm_payload["ema_sha256"], "latent_mean": warm_payload["latent_mean"], "latent_std": warm_payload["latent_std"]}
    segment_end = training.steps if stop_after_step is None else min(training.steps, int(stop_after_step))
    if segment_end <= start_step:
        raise ValueError("cellular action segment end must advance the checkpoint")
    for step in range(start_step + 1, segment_end + 1):
        step_started = time.perf_counter()
        indices = rng.integers(0, len(train_group["current"]), training.batch_size)
        current_raw = torch.from_numpy(train_group["current"][indices]).to(device); previous_raw = torch.from_numpy(train_group["previous"][indices]).to(device); target_raw = torch.from_numpy(train_group["target"][indices]).to(device)
        current = (current_raw - latent_mean) / latent_std; previous = (previous_raw - latent_mean) / latent_std; target = (target_raw - latent_mean) / latent_std
        noisy = current + torch.randn_like(current) * training.input_noise
        action = torch.from_numpy(train_group["action"][indices].astype(np.int64)).to(device); previous_action = torch.from_numpy(train_group["previous_action"][indices].astype(np.int64)).to(device)
        control = torch.from_numpy(train_group["control"][indices]).to(device); previous_control = torch.from_numpy(train_group["previous_control"][indices]).to(device); state = torch.from_numpy(train_group["state"][indices]).to(device)
        actor_raw = torch.from_numpy(train_group["actor_state"][indices]).to(device); target_actor_raw = torch.from_numpy(train_group["target_actor_state"][indices]).to(device); actor = (actor_raw - actor_mean) / actor_std; target_actor = (target_actor_raw - actor_mean) / actor_std
        field = torch.from_numpy(train_group["actor_field"][indices]).float().to(device); target_field = torch.from_numpy(train_group["target_actor_field"][indices]).float().to(device)
        mask = torch.from_numpy(edit_masks[indices]).to(device); field_mask = ((target_field - field).abs() > 1e-3).float(); actor_change = ((target_actor - actor).abs() > 1e-3).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predicted, predicted_actor, predicted_field, gate, _, gate_logits, field_gate, field_gate_logits = model.edit(noisy, previous, torch.zeros(len(indices), device=device), action, control, state, actor, field, previous_action, previous_control)
            latent_error = F.smooth_l1_loss(predicted, target, reduction="none"); latent_loss = (latent_error * (1 + training.changed_weight * mask)).mean()
            actor_error = F.smooth_l1_loss(predicted_actor, target_actor, reduction="none"); actor_loss = (actor_error * (1 + ACTOR_STATE_CHANGED_WEIGHT * actor_change)).mean()
            field_error = F.smooth_l1_loss(predicted_field, target_field, reduction="none"); field_loss = (field_error * (1 + ACTOR_FIELD_CHANGED_WEIGHT * field_mask)).mean()
            gate_bce = F.binary_cross_entropy_with_logits(gate_logits.float(), mask, pos_weight=torch.tensor(training.gate_positive_weight, device=device)); intersection = (gate * mask).sum((1, 2, 3)); gate_dice = 1 - ((2 * intersection + 1) / (gate.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)).mean(); leakage = (gate * (1 - mask)).mean()
            field_pos_weight = torch.from_numpy(field_gate_pos_weight).to(device).view(1, -1, 1, 1)
            field_gate_bce = F.binary_cross_entropy_with_logits(field_gate_logits.float(), field_mask, pos_weight=field_pos_weight)
            field_intersection = (field_gate * field_mask).sum((2, 3)); field_gate_dice = 1 - ((2 * field_intersection + 1) / (field_gate.sum((2, 3)) + field_mask.sum((2, 3)) + 1)).mean()
            field_gate_loss = field_gate_bce + field_gate_dice
            contrast_count = min(training.contrastive_batch, len(indices)); wrong_action = (action[:contrast_count] + 7) % len(ACTIONS); wrong_control = torch.from_numpy(counterfactual_control(train_group["control"][indices[:contrast_count]])).to(device); correct_error = (predicted[:contrast_count] - target[:contrast_count]).abs().mean((1, 2, 3)); wrong_a = model.edit(noisy[:contrast_count], previous[:contrast_count], torch.zeros(contrast_count, device=device), wrong_action, control[:contrast_count], state[:contrast_count], actor[:contrast_count], field[:contrast_count], previous_action[:contrast_count], previous_control[:contrast_count])[0]; wrong_c = model.edit(noisy[:contrast_count], previous[:contrast_count], torch.zeros(contrast_count, device=device), action[:contrast_count], wrong_control, state[:contrast_count], actor[:contrast_count], field[:contrast_count], previous_action[:contrast_count], previous_control[:contrast_count])[0]; targeted = torch.isin(action[:contrast_count], torch.as_tensor(TARGETED_INDICES, device=device)); contrastive = F.relu(training.contrastive_margin + correct_error - (wrong_a - target[:contrast_count]).abs().mean((1, 2, 3))).mean(); contrastive = contrastive + (F.relu(training.contrastive_margin + correct_error[targeted] - (wrong_c[targeted] - target[:contrast_count][targeted]).abs().mean((1, 2, 3))).mean() if bool(targeted.any()) else correct_error.new_zeros(()))
            loss = latent_loss + training.actor_state_weight * actor_loss + training.actor_field_weight * field_loss + ACTOR_FIELD_GATE_WEIGHT * field_gate_loss + training.gate_weight * (gate_bce + gate_dice) + training.leakage_weight * leakage + training.contrastive_weight * contrastive
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items(): ema[name].lerp_(value.detach(), 1 - training.ema_decay)
        if step == 1 or step % 250 == 0 or step == training.steps:
            row = {"step": step, "loss": round(float(loss), 6), "latent": round(float(latent_loss), 6), "actor_state": round(float(actor_loss), 6), "actor_field": round(float(field_loss), 6), "actor_field_gate": round(float(field_gate_loss), 6), "gate": round(float(gate_bce + gate_dice), 6), "leakage": round(float(leakage), 6), "contrastive": round(float(contrastive), 6)}; history.append(row); print(json.dumps(row), flush=True)
        validate_now = step % training.validate_every == 0 or step == training.steps
        checkpoint_now = step % training.checkpoint_every == 0 or validate_now or step == segment_end
        if validate_now:
            validation_model.load_state_dict(ema); metrics = evaluate(validation_model, validation_episodes, latent_mean, latent_std, actor_mean, actor_std, device, batch_size=training.validation_batch_size); score = selection_score(metrics); validation_history.append({"step": step, "selection": score, **metrics}); print(json.dumps({"validation_step": step, "selection": round(score, 6), "latent_improvement": round(metrics["latent_improvement"], 6), "actor_state_improvement": round(metrics["actor_state_improvement"], 6), "actor_field_improvement": round(metrics["actor_field_improvement"], 6)}, sort_keys=True), flush=True)
            if score < best_score:
                best_score = score; best_step = step; best_ema = {name: value.detach().cpu().to(torch.bfloat16) if value.is_floating_point() else value.detach().cpu() for name, value in ema.items()}
            model.train()
        if checkpoint_now:
            payload = _checkpoint_payload(corpus_sha=corpus_sha, model_config=model_config, training_config=training, warm=warm_record, step=step, model=model, ema=ema, optimizer=optimizer, rng=rng, actor_mean=actor_mean_array, actor_std=actor_std_array, history=history, validation_history=validation_history, best_score=best_score, best_step=best_step, best_ema=best_ema)
            store.save(payload, step=step, milestone=validate_now and (step % training.milestone_every == 0 or step == best_step))
        resources.throttle(time.perf_counter() - step_started)
    if segment_end < training.steps:
        return {
            "format": REPORT_FORMAT,
            "status": "segment_complete",
            "source_sha256": source_sha256(),
            "corpus_sha256": corpus_sha,
            "step": segment_end,
            "total_steps": training.steps,
            "resource_usage": resources.report(),
            "latest_history": history[-1] if history else None,
            "latest_validation": validation_history[-1] if validation_history else None,
        }
    if best_ema is None:
        raise RuntimeError("cellular action training selected no checkpoint")
    validation_model.load_state_dict(best_ema); validation_model.to(device).eval(); validation_metrics = evaluate(validation_model, validation_episodes, latent_mean, latent_std, actor_mean, actor_std, device, batch_size=training.validation_batch_size); test_metrics = evaluate(validation_model, test_episodes, latent_mean, latent_std, actor_mean, actor_std, device, batch_size=training.validation_batch_size)
    gates = {"latent_beats_persistence": test_metrics["latent_improvement"] > 0, "actor_state_beats_persistence": test_metrics["actor_state_improvement"] > 0, "actor_field_beats_persistence": test_metrics["actor_field_improvement"] > 0, "correct_beats_wrong_action": test_metrics["correct_action_advantage"] > 0, "targeted_control_beats_wrong_control": test_metrics["targeted_control_advantage"] > 0}; gates["all_passed"] = all(gates.values())
    report = {"format": REPORT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "warm_start": warm_record, "model_config": config_dict(model_config), "training_config": config_dict(training), "resource_usage": resources.report(), "device": str(device), "parameters": sum(value.numel() for value in model.parameters()), "best_step": best_step, "best_validation_score": best_score, "splits": {"train": list(train_sessions), "validation": list(validation_sessions), "test": list(test_sessions)}, "validation": validation_metrics, "test": test_metrics, "gates": gates, "history": history, "validation_history": validation_history}
    final = _checkpoint_payload(corpus_sha=corpus_sha, model_config=model_config, training_config=training, warm=warm_record, step=training.steps, model=model, ema=ema, optimizer=optimizer, rng=rng, actor_mean=actor_mean_array, actor_std=actor_std_array, history=history, validation_history=validation_history, best_score=best_score, best_step=best_step, best_ema=best_ema); final.update({"status": "evaluated", "runtime_ema_state": best_ema, "runtime_ema_sha256": _state_hash(best_ema), "report": report})
    torch.save(final, output / "evaluated.pt"); (output / "report.json").write_bytes(canonical(report)); return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--train-sessions", nargs="+", required=True)
    parser.add_argument("--validation-sessions", nargs="+", required=True)
    parser.add_argument("--test-sessions", nargs="+", required=True)
    parser.add_argument("--total-steps", type=int, default=12000)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cuda-memory-fraction", type=float, default=.85)
    parser.add_argument("--max-process-memory-gib", type=float, default=32.0)
    parser.add_argument("--target-duty-cycle", type=float, default=.90)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    training = TrainingConfig(
        steps=args.total_steps,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        cpu_threads=args.cpu_threads,
        cuda_memory_fraction=args.cuda_memory_fraction,
        max_process_memory_gib=args.max_process_memory_gib,
        target_duty_cycle=args.target_duty_cycle,
    )
    report = train(
        args.output,
        args.corpus,
        args.warm_start,
        train_sessions=args.train_sessions,
        validation_sessions=args.validation_sessions,
        test_sessions=args.test_sessions,
        training=training,
        resume=args.resume,
        stop_after_step=args.stop_after_step,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
