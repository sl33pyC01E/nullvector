from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..safety import require_disk_floor
from ..world_action_cellular_v7.corpus import load_encoded_corpus
from .contract import CHECKPOINT_FORMAT, DEFAULT_CORPUS, DEFAULT_OUTPUT, ModelConfig, REPORT_FORMAT, TrainingPlan, canonical, source_sha256, state_sha256
from .model import RecurrentWorldStudent


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary); os.replace(temporary, path)


def _normalizers(episodes):
    latent = np.concatenate([np.concatenate((episode["current"], episode["target"])) for episode in episodes])
    actor = np.concatenate([np.concatenate((episode["actor_state"], episode["target_actor_state"])) for episode in episodes])
    return latent.mean((0, 2, 3)).astype(np.float32), (latent.std((0, 2, 3)) + 1e-4).astype(np.float32), actor.mean(0).astype(np.float32), (actor.std(0) + 1e-4).astype(np.float32)


def _sample(episodes, rng, batch: int, sequence: int):
    rows = []
    for _ in range(batch):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        start = int(rng.integers(0, len(episode["current"]) - sequence + 1))
        rows.append({name: value[start:start + sequence] for name, value in episode.items()})
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def _tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype, device=device)


def train(output: Path = DEFAULT_OUTPUT, *, corpus: Path = DEFAULT_CORPUS, plan: TrainingPlan = TrainingPlan()) -> dict:
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=768 * 1024**2)
    if not torch.cuda.is_available(): raise RuntimeError("RSSM production requires CUDA")
    torch.set_num_threads(2); torch.manual_seed(plan.seed); np.random.seed(plan.seed & 0xffffffff)
    device = torch.device("cuda:0"); torch.cuda.set_per_process_memory_fraction(.55, 0)
    episodes, corpus_manifest = load_encoded_corpus(corpus)
    train_episodes, validation_episode, test_episode = episodes[:4], episodes[4], episodes[5]
    latent_mean, latent_std, actor_mean, actor_std = _normalizers(train_episodes)
    lm = _tensor(latent_mean, device)[None, :, None, None]; ls = _tensor(latent_std, device)[None, :, None, None]
    am = _tensor(actor_mean, device)[None]; ast = _tensor(actor_std, device)[None]
    config = ModelConfig(); model = RecurrentWorldStudent(config).to(device); ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-4, fused=True)
    rng = np.random.default_rng(plan.seed); history = []; start = 0
    for step in range(plan.segment_updates, plan.total_updates + 1, plan.segment_updates):
        path = output / f"rssm_{step:07d}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True); start = step
    if start:
        model.load_state_dict(payload["model_state"]); ema.load_state_dict(payload["ema_state"]); optimizer.load_state_dict(payload["optimizer_state"]); rng.bit_generator.state = payload["rng_state"]; history = list(payload["history"])
    for end in range(start + plan.segment_updates, plan.total_updates + 1, plan.segment_updates):
        began = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); model.train()
        for update in range(end - plan.segment_updates + 1, end + 1):
            batch = _sample(train_episodes, rng, plan.batch_size, plan.sequence)
            previous = (_tensor(batch["previous"][:, 0], device) - lm) / ls
            current = (_tensor(batch["current"][:, 0], device) - lm) / ls
            actor = (_tensor(batch["actor_state"][:, 0], device) - am) / ast
            hidden = None; total = current.new_zeros(()); latent_value = actor_value = gate_value = current.new_zeros(())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for offset in range(plan.sequence):
                    target = (_tensor(batch["target"][:, offset], device) - lm) / ls
                    target_actor = (_tensor(batch["target_actor_state"][:, offset], device) - am) / ast
                    action = _tensor(batch["action"][:, offset], device, torch.long)
                    control = _tensor(batch["control"][:, offset], device)
                    state = _tensor(batch["state"][:, offset], device)
                    result = model(current, previous, action, control, state, actor, hidden)
                    change = (target - current).abs().mean(1, keepdim=True)
                    changed = (change > .035).float()
                    latent_error = F.smooth_l1_loss(result.latent, target, reduction="none")
                    latent_loss = (latent_error * (1 + 5 * changed)).mean()
                    actor_loss = F.smooth_l1_loss(result.actor_state, target_actor)
                    with torch.autocast("cuda", enabled=False):
                        gate_loss = F.binary_cross_entropy(result.edit_gate.float().clamp(1e-5, 1 - 1e-5), changed.float())
                    persistence = ((result.latent - current).abs() * (1 - changed)).mean()
                    step_loss = latent_loss + .25 * actor_loss + .15 * gate_loss + .5 * persistence
                    total = total + step_loss / plan.sequence
                    latent_value += latent_loss.detach() / plan.sequence; actor_value += actor_loss.detach() / plan.sequence; gate_value += gate_loss.detach() / plan.sequence
                    previous, current, actor, hidden = current, result.latent, result.actor_state, result.hidden
            if not bool(torch.isfinite(total)): raise FloatingPointError("RSSM loss became non-finite")
            total.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1)); optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(list(ema.parameters()), plan.ema_decay)
                torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
            if update == 1 or update % 25 == 0:
                history.append({"update": update, "loss": round(float(total), 7), "latent": round(float(latent_value), 7), "actor": round(float(actor_value), 7), "gate": round(float(gate_value), 7), "gradient": round(gradient, 7)})
        model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}; ema_state = {name: value.detach().cpu() for name, value in ema.state_dict().items()}
        payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_manifest["manifest_sha256"], "model_config": config.__dict__ if hasattr(config, "__dict__") else {field: getattr(config, field) for field in config.__slots__}, "plan": plan.to_dict(), "update": end, "model_state": model_state, "ema_state": ema_state, "model_state_sha256": state_sha256(model_state), "ema_state_sha256": state_sha256(ema_state), "optimizer_state": optimizer.state_dict(), "rng_state": rng.bit_generator.state, "history": history, "normalization": {"latent_mean": latent_mean.tolist(), "latent_std": latent_std.tolist(), "actor_mean": actor_mean.tolist(), "actor_std": actor_std.tolist()}, "runtime": {"segment_seconds": round(time.perf_counter() - began, 6), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}}
        _atomic_torch(output / f"rssm_{end:07d}.pt", payload)
        print(json.dumps({"update": end, "loss": history[-1]["loss"], **payload["runtime"]}), flush=True)
    return {"passed": True, "update": plan.total_updates, "checkpoint": str(output / f"rssm_{plan.total_updates:07d}.pt"), "ema_state_sha256": payload["ema_state_sha256"]}


@torch.inference_mode()
def evaluate(output: Path = DEFAULT_OUTPUT, *, corpus: Path = DEFAULT_CORPUS) -> dict:
    output = Path(output).resolve(); checkpoints = sorted(output.glob("rssm_*.pt"))
    if not checkpoints: raise FileNotFoundError("RSSM checkpoint missing")
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256(): raise ValueError("RSSM checkpoint provenance drifted")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); config = ModelConfig(**payload["model_config"]); model = RecurrentWorldStudent(config); model.load_state_dict(payload["ema_state"]); model.to(device).eval()
    episodes, manifest = load_encoded_corpus(corpus); episode = episodes[5]; norm = payload["normalization"]
    lm = _tensor(norm["latent_mean"], device)[None, :, None, None]; ls = _tensor(norm["latent_std"], device)[None, :, None, None]; am = _tensor(norm["actor_mean"], device)[None]; ast = _tensor(norm["actor_std"], device)[None]
    predicted = []; predicted_actor = []; wrong = []; hidden = None
    for index in range(len(episode["current"])):
        previous = (_tensor(episode["previous"][index:index+1], device) - lm) / ls; current = (_tensor(episode["current"][index:index+1], device) - lm) / ls; actor = (_tensor(episode["actor_state"][index:index+1], device) - am) / ast; action = _tensor(episode["action"][index:index+1], device, torch.long); control = _tensor(episode["control"][index:index+1], device); state = _tensor(episode["state"][index:index+1], device)
        result = model(current, previous, action, control, state, actor, hidden); wrong_result = model(current, previous, (action + 7) % config.action_count, control, state, actor, hidden)
        hidden = result.hidden; predicted.append((result.latent * ls + lm).cpu()); predicted_actor.append((result.actor_state * ast + am).cpu()); wrong.append((wrong_result.latent * ls + lm).cpu())
    predicted = torch.cat(predicted); predicted_actor = torch.cat(predicted_actor); wrong = torch.cat(wrong); current = torch.from_numpy(episode["current"]); target = torch.from_numpy(episode["target"]); actor = torch.from_numpy(episode["actor_state"]); target_actor = torch.from_numpy(episode["target_actor_state"])
    latent_mae = float(F.l1_loss(predicted, target)); latent_persistence = float(F.l1_loss(current, target)); actor_mae = float(F.l1_loss(predicted_actor, target_actor)); actor_persistence = float(F.l1_loss(actor, target_actor)); wrong_mae = float(F.l1_loss(wrong, target))
    metrics = {"pairs": len(current), "latent_mae": latent_mae, "latent_persistence_mae": latent_persistence, "latent_improvement": 1 - latent_mae / latent_persistence, "actor_mae": actor_mae, "actor_persistence_mae": actor_persistence, "actor_improvement": 1 - actor_mae / actor_persistence, "correct_action_advantage": wrong_mae - latent_mae}
    gates = {"latent_beats_persistence": metrics["latent_improvement"] > 0, "actor_beats_persistence": metrics["actor_improvement"] > 0, "correct_beats_wrong_action": metrics["correct_action_advantage"] > 0}; gates["all_passed"] = all(gates.values())
    report = {"format": REPORT_FORMAT, "status": "ready" if gates["all_passed"] else "experimental", "source_sha256": source_sha256(), "corpus_sha256": manifest["manifest_sha256"], "checkpoint": {"path": checkpoints[-1].name, "bytes": checkpoints[-1].stat().st_size, "sha256": _file_hash(checkpoints[-1]), "ema_state_sha256": payload["ema_state_sha256"]}, "parameters": model.parameter_count, "metrics": metrics, "gates": gates, "limitations": ["V1 evaluates recurrent teacher-forced transitions; free-rollout and decoded-frame gates are the next promotion boundary."]}
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest(); (output / "report.json").write_bytes(canonical(report)); return report
