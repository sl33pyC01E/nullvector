from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..safety import require_disk_floor
from ..world_action_contiguous_v8 import load
from ..world_latent_dit.contract import ModelConfig
from .contract import ACTION_CHECKPOINT, ACTION_SHA256, ACTOR_CHECKPOINT, ACTOR_FILE_SHA256, CHECKPOINT_FORMAT, CORPUS, DEFAULT_OUTPUT, REPORT_FORMAT, TrainingPlan, canonical, file_sha256, source_sha256, state_sha256
from .model import RecurrentWorldStudent


def _atomic(path, payload):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _parents(model):
    if file_sha256(ACTION_CHECKPOINT) != ACTION_SHA256 or file_sha256(ACTOR_CHECKPOINT) != ACTOR_FILE_SHA256:
        raise ValueError("recurrent V3 parent drifted")
    action = torch.load(ACTION_CHECKPOINT, map_location="cpu", weights_only=True)
    actor = torch.load(ACTOR_CHECKPOINT, map_location="cpu", weights_only=True)
    model.action.load_state_dict(action["state"], strict=True)
    model.actor.load_state_dict(actor["model_state"], strict=True)
    return action, actor


def _normalizers(action, actor, device):
    lm = torch.tensor(action["normalization"]["mean"], device=device)[None, :, None, None]
    ls = torch.tensor(action["normalization"]["std"], device=device)[None, :, None, None]
    am = torch.tensor(actor["normalization"]["mean"], device=device)[None]
    ass = torch.tensor(actor["normalization"]["std"], device=device)[None]
    return lm, ls, am, ass


def _batch(sequences, rng, count, steps):
    rows = []
    for _ in range(count):
        sequence = sequences[int(rng.integers(0, len(sequences)))]
        start = int(rng.integers(1, len(sequence["latent"]) - steps))
        rows.append((sequence, start))
    return rows


def _tensors(rows, offset, device):
    def stack(name, index):
        return torch.from_numpy(np.stack([sequence[name][start + index] for sequence, start in rows])).to(device)
    return {
        "previous": stack("latent", offset - 1), "current": stack("latent", offset), "target": stack("latent", offset + 1),
        "previous_actor": stack("actor_state", offset - 1), "actor": stack("actor_state", offset), "target_actor": stack("actor_state", offset + 1),
        "action": stack("action", offset + 1).long(), "control": stack("control", offset + 1), "state": stack("state", offset),
    }


def _one_step_metrics(model, sequence, norms, device):
    lm, ls, am, ass = norms
    previous = torch.from_numpy(sequence["latent"][:-2]).to(device)
    current = torch.from_numpy(sequence["latent"][1:-1]).to(device)
    target = torch.from_numpy(sequence["latent"][2:]).to(device)
    actor = torch.from_numpy(sequence["actor_state"][1:-1]).to(device)
    action = torch.from_numpy(sequence["action"][2:].astype(np.int64)).to(device)
    control = torch.from_numpy(sequence["control"][2:]).to(device)
    state = torch.from_numpy(sequence["state"][1:-1]).to(device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(current), 8):
            cn = (current[start:start + 8] - lm) / ls
            pn = (previous[start:start + 8] - lm) / ls
            delta = model.action(cn, pn, action[start:start + 8], control[start:start + 8], state[start:start + 8], actor[start:start + 8])
            gate = delta.abs().mean(1, keepdim=True) >= 0.18
            predictions.append(((cn + gate * delta) * ls + lm).cpu())
    prediction = torch.cat(predictions)
    mae = float(F.l1_loss(prediction, target.cpu()))
    persistence = float(F.l1_loss(current.cpu(), target.cpu()))
    return {"mae": mae, "persistence_mae": persistence, "improvement": 1 - mae / persistence}


def train(output: Path = DEFAULT_OUTPUT, *, plan: TrainingPlan = TrainingPlan()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if not torch.cuda.is_available():
        raise RuntimeError("recurrent world V3 training requires CUDA")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.48, 0)
    torch.manual_seed(plan.seed)
    rng = np.random.default_rng(plan.seed)
    device = torch.device("cuda:0")
    sequences, manifest = load(CORPUS)
    model = RecurrentWorldStudent(ModelConfig())
    action_parent, actor_parent = _parents(model)
    model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-3, fused=True)
    norms = _normalizers(action_parent, actor_parent, device)
    lm, ls, am, ass = norms
    history = []
    start_update = 0
    latest = output / "latest.pt"
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("corpus_sha256") != manifest["manifest_sha256"]:
            raise ValueError("recurrent world V3 resume drifted")
        model.load_state_dict(payload["model_state"]);ema.load_state_dict(payload["ema_state"]);optimizer.load_state_dict(payload["optimizer_state"]);rng.bit_generator.state = payload["rng_state"];history = list(payload["history"]);start_update = payload["update"]
    for end in range(start_update + plan.segment_updates, plan.total_updates + 1, plan.segment_updates):
        began = time.perf_counter();torch.cuda.reset_peak_memory_stats(device);model.train()
        for update in range(end - plan.segment_updates + 1, end + 1):
            rows = _batch(sequences[:4], rng, plan.batch_size, plan.rollout_steps)
            optimizer.zero_grad(set_to_none=True);latent_total = actor_total = 0.0
            previous = current = previous_actor = actor = None
            for offset in range(plan.rollout_steps):
                values = _tensors(rows, offset, device)
                if offset == 0:
                    previous, current = values["previous"], values["current"]
                    previous_actor, actor = values["previous_actor"], values["actor"]
                target, target_actor = values["target"], values["target_actor"]
                cn, pn = (current - lm) / ls, (previous - lm) / ls
                an, pan, tan = (actor - am) / ass, (previous_actor - am) / ass, (target_actor - am) / ass
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    delta = model.action(cn, pn, values["action"], values["control"], values["state"], actor)
                    magnitude = ((target - current) / ls).abs().mean(1, keepdim=True)
                    latent_loss = (F.smooth_l1_loss(delta, (target - current) / ls, reduction="none") * (1 + 5 * torch.clamp(magnitude / 0.35, 0, 2))).mean()
                    actor_result = model.actor(an, pan, values["action"], values["control"], values["state"])
                    changed = (tan - an).abs() > 0.025
                    actor_loss = (F.smooth_l1_loss(actor_result.state, tan, reduction="none") * (1 + 6 * changed)).mean()
                    loss = (latent_loss + plan.actor_weight * actor_loss) / plan.rollout_steps
                loss.backward()
                latent_total += float(latent_loss);actor_total += float(actor_loss)
                with torch.no_grad():
                    gate = delta.abs().mean(1, keepdim=True) >= 0.18
                    next_latent = (cn + gate * delta) * ls + lm
                    actor_keep = actor_result.gate >= 0.7
                    next_actor = (an + 0.9 * actor_keep * (actor_result.state - an)) * ass + am
                previous, current = current.detach(), next_latent.detach()
                previous_actor, actor = actor.detach(), next_actor.detach()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1));optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(list(ema.parameters()), plan.ema_decay);torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
            if update == 1 or update % 25 == 0:
                history.append({"update": update, "latent": round(latent_total / plan.rollout_steps, 7), "actor": round(actor_total / plan.rollout_steps, 7), "gradient": round(gradient, 7)})
        raw_validation = _one_step_metrics(model.eval(), sequences[4], norms, device)
        ema_validation = _one_step_metrics(ema.eval(), sequences[4], norms, device)
        model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()};ema_state = {name: value.detach().cpu() for name, value in ema.state_dict().items()}
        payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": manifest["manifest_sha256"], "parents": {"action": ACTION_SHA256, "actor": ACTOR_FILE_SHA256}, "model_config": action_parent["model_config"], "plan": plan.to_dict(), "update": end, "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(), "rng_state": rng.bit_generator.state, "history": history, "normalization": {"latent_mean": action_parent["normalization"]["mean"], "latent_std": action_parent["normalization"]["std"], "actor_mean": actor_parent["normalization"]["mean"], "actor_std": actor_parent["normalization"]["std"]}, "validation": {"raw": raw_validation, "ema": ema_validation}, "runtime": {"segment_seconds": round(time.perf_counter() - began, 6), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}}
        _atomic(latest, payload);_atomic(output / f"milestone_{end:07d}.pt", payload)
        print(json.dumps({"update": end, "raw": raw_validation["improvement"], "ema": ema_validation["improvement"], **payload["runtime"]}), flush=True)
    variant = "raw" if payload["validation"]["raw"]["improvement"] >= payload["validation"]["ema"]["improvement"] else "ema"
    chosen = model if variant == "raw" else ema
    test_metrics = _one_step_metrics(chosen.eval(), sequences[5], norms, device)
    state = {name: value.detach().cpu() for name, value in chosen.state_dict().items()}
    report = {"format": REPORT_FORMAT, "status": "ready" if test_metrics["improvement"] > 0 else "experimental", "source_sha256": source_sha256(), "corpus_sha256": manifest["manifest_sha256"], "parameters": chosen.parameter_count, "updates": plan.total_updates, "rollout_steps": plan.rollout_steps, "selection": {"variant": variant, "validation": payload["validation"][variant]}, "test": test_metrics, "runtime": payload["runtime"]}
    release = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": manifest["manifest_sha256"], "model_config": action_parent["model_config"], "state": state, "state_sha256": state_sha256(state), "normalization": payload["normalization"], "report": report}
    _atomic(output / "runtime.pt", release);report["checkpoint"] = {"path": "runtime.pt", "sha256": file_sha256(output / "runtime.pt"), "state_sha256": release["state_sha256"]};(output / "report.json").write_bytes(canonical(report));return report
