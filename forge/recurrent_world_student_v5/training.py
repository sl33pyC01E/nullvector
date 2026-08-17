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
from ..world_action_natural_v10 import load
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT, CORPUS, DEFAULT_OUTPUT, PARENT, PARENT_SHA256, TrainingPlan, file_sha256, source_sha256, state_sha256
from .model import PerceptionRecurrentWorldStudent


def _atomic(path, payload):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _normalizers(parent, device):
    values = parent["normalization"]
    return (
        torch.tensor(values["latent_mean"], device=device)[None, :, None, None],
        torch.tensor(values["latent_std"], device=device)[None, :, None, None],
        torch.tensor(values["actor_mean"], device=device)[None],
        torch.tensor(values["actor_std"], device=device)[None],
    )


def _sample_batch(sequences, rng, count, steps, device):
    rows = []
    for _ in range(count):
        sequence = sequences[int(rng.integers(0, len(sequences)))]
        start = int(rng.integers(1, len(sequence["latent"]) - steps))
        rows.append((sequence, start))

    def gather(name, begin, length):
        array = np.stack([sequence[name][start + begin:start + begin + length] for sequence, start in rows])
        return torch.from_numpy(array).to(device, non_blocking=True)

    # One host-to-device transfer per tensor for the entire recurrent rollout.
    return {
        "latent": gather("latent", -1, steps + 2),
        "actor": gather("actor_state", -1, steps + 2),
        "action": gather("action", 1, steps).long(),
        "control": gather("control", 1, steps),
        "state": gather("state", 0, steps),
        "visibility": gather("visibility", 0, steps),
        "memory": gather("memory", 0, steps),
    }


@torch.inference_mode()
def _metrics(model, sequence, norms, device, perception="normal"):
    lm, ls, _, _ = norms
    total_mae = total_persistence = 0.0
    samples = 0
    for start in range(1, len(sequence["latent"]) - 1, 32):
        stop = min(start + 32, len(sequence["latent"]) - 1)
        previous = torch.from_numpy(sequence["latent"][start - 1:stop - 1]).to(device)
        current = torch.from_numpy(sequence["latent"][start:stop]).to(device)
        target = torch.from_numpy(sequence["latent"][start + 1:stop + 1]).to(device)
        actor = torch.from_numpy(sequence["actor_state"][start:stop]).to(device)
        action = torch.from_numpy(sequence["action"][start + 1:stop + 1].astype(np.int64)).to(device)
        control = torch.from_numpy(sequence["control"][start + 1:stop + 1]).to(device)
        state = torch.from_numpy(sequence["state"][start:stop]).to(device)
        visibility = torch.from_numpy(sequence["visibility"][start:stop]).to(device)
        memory = torch.from_numpy(sequence["memory"][start:stop]).to(device)
        if perception == "zero":
            visibility = torch.zeros_like(visibility); memory = torch.zeros_like(memory)
        elif perception == "shuffle":
            visibility = visibility.flip(0); memory = memory.flip(0)
        cn, pn = (current - lm) / ls, (previous - lm) / ls
        with torch.autocast("cuda", dtype=torch.bfloat16):
            delta = model.action(cn, pn, action, control, state, actor, visibility, memory)
        prediction = (cn + delta) * ls + lm
        total_mae += float(F.l1_loss(prediction, target, reduction="sum"))
        total_persistence += float(F.l1_loss(current, target, reduction="sum"))
        samples += target.numel()
    mae = total_mae / samples; persistence = total_persistence / samples
    return {"mae": mae, "persistence_mae": persistence, "improvement": 1 - mae / persistence}


def train(output: Path = DEFAULT_OUTPUT, *, plan: TrainingPlan = TrainingPlan()):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if not torch.cuda.is_available():
        raise RuntimeError("perception recurrent V5 training requires CUDA")
    if file_sha256(PARENT) != PARENT_SHA256:
        raise ValueError("perception recurrent V5 parent drifted")
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.45, 0)
    torch.manual_seed(plan.seed); rng = np.random.default_rng(plan.seed); device = torch.device("cuda:0")
    sequences, manifest = load(CORPUS); parent = torch.load(PARENT, map_location="cpu", weights_only=True)
    model = PerceptionRecurrentWorldStudent(ModelConfig(**parent["model_config"])); model.load_parent(parent["state"]); model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-3, fused=True)
    norms = _normalizers(parent, device); lm, ls, am, ass = norms
    history = []; start_update = 0; latest = output / "latest.pt"
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("corpus_sha256") != manifest["manifest_sha256"] or payload.get("parent_sha256") != PARENT_SHA256:
            raise ValueError("perception recurrent V5 resume drifted")
        model.load_state_dict(payload["model_state"]); ema.load_state_dict(payload["ema_state"]); optimizer.load_state_dict(payload["optimizer_state"]); rng.bit_generator.state = payload["rng_state"]; history = list(payload["history"]); start_update = payload["update"]
    if plan.total_updates % plan.segment_updates:
        raise ValueError("total_updates must be divisible by segment_updates")
    payload = None
    for end in range(start_update + plan.segment_updates, plan.total_updates + 1, plan.segment_updates):
        began = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); model.train()
        for update in range(end - plan.segment_updates + 1, end + 1):
            batch = _sample_batch(sequences[:4], rng, plan.batch_size, plan.rollout_steps, device)
            optimizer.zero_grad(set_to_none=True); latent_total = actor_total = 0.0
            previous = batch["latent"][:, 0]; current = batch["latent"][:, 1]
            previous_actor = batch["actor"][:, 0]; actor = batch["actor"][:, 1]
            for offset in range(plan.rollout_steps):
                target = batch["latent"][:, offset + 2]; target_actor = batch["actor"][:, offset + 2]
                visibility = batch["visibility"][:, offset]; memory = batch["memory"][:, offset]
                if plan.perception_dropout and float(rng.random()) < plan.perception_dropout:
                    visibility = torch.zeros_like(visibility); memory = torch.zeros_like(memory)
                cn, pn = (current - lm) / ls, (previous - lm) / ls
                an, pan, tan = (actor - am) / ass, (previous_actor - am) / ass, (target_actor - am) / ass
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    delta = model.action(cn, pn, batch["action"][:, offset], batch["control"][:, offset], batch["state"][:, offset], actor, visibility, memory)
                    magnitude = ((target - current) / ls).abs().mean(1, keepdim=True)
                    latent_loss = (F.smooth_l1_loss(delta, (target-current)/ls, reduction="none") * (1 + 5 * torch.clamp(magnitude/.35, 0, 2))).mean()
                    actor_result = model.actor(an, pan, batch["action"][:, offset], batch["control"][:, offset], batch["state"][:, offset], visibility, memory)
                    changed = (tan - an).abs() > .025
                    actor_loss = (F.smooth_l1_loss(actor_result.state, tan, reduction="none") * (1 + 6 * changed)).mean()
                    loss = (latent_loss + plan.actor_weight * actor_loss) / plan.rollout_steps
                loss.backward(); latent_total += float(latent_loss); actor_total += float(actor_loss)
                with torch.no_grad():
                    next_latent = (cn + delta) * ls + lm
                    next_actor = (an + .9 * (actor_result.gate >= .7) * (actor_result.state - an)) * ass + am
                previous, current = current.detach(), next_latent.detach(); previous_actor, actor = actor.detach(), next_actor.detach()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1)); optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(list(ema.parameters()), plan.ema_decay); torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1-plan.ema_decay)
            if update == 1 or update % 10 == 0:
                history.append({"update": update, "latent": round(latent_total/plan.rollout_steps, 7), "actor": round(actor_total/plan.rollout_steps, 7), "gradient": round(gradient, 7)})
        validation = {"normal": _metrics(ema.eval(), sequences[4], norms, device), "zero": _metrics(ema.eval(), sequences[4], norms, device, "zero"), "shuffle": _metrics(ema.eval(), sequences[4], norms, device, "shuffle")}
        validation["perception_preference"] = {"zero": validation["zero"]["mae"]-validation["normal"]["mae"], "shuffle": validation["shuffle"]["mae"]-validation["normal"]["mae"]}
        model_state = {name:value.detach().cpu() for name,value in model.state_dict().items()}; ema_state = {name:value.detach().cpu() for name,value in ema.state_dict().items()}
        payload = {"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parent_sha256":PARENT_SHA256,"model_config":parent["model_config"],"plan":plan.to_dict(),"update":end,"model_state":model_state,"ema_state":ema_state,"optimizer_state":optimizer.state_dict(),"rng_state":rng.bit_generator.state,"history":history,"normalization":parent["normalization"],"validation":validation,"runtime":{"segment_seconds":round(time.perf_counter()-began,6),"updates_per_second":round(plan.segment_updates/(time.perf_counter()-began),4),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}}
        _atomic(latest,payload); _atomic(output/f"milestone_{end:07d}.pt",payload); print(json.dumps({"update":end,"validation":validation,"runtime":payload["runtime"]}),flush=True)
    chosen = payload["ema_state"]; test = {"normal":_metrics(ema.eval(),sequences[5],norms,device),"zero":_metrics(ema.eval(),sequences[5],norms,device,"zero"),"shuffle":_metrics(ema.eval(),sequences[5],norms,device,"shuffle")}
    gates = {"beats_persistence":test["normal"]["improvement"]>0,"uses_visibility_memory":test["normal"]["mae"]<min(test["zero"]["mae"],test["shuffle"]["mae"]),"under_half_gpu_memory":payload["runtime"]["peak_reserved_bytes"]<12*1024**3}; gates["all_passed"] = all(gates.values())
    release = {"format":CHECKPOINT_FORMAT,"status":"ready" if gates["all_passed"] else "experimental","source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parent_sha256":PARENT_SHA256,"model_config":parent["model_config"],"state":chosen,"state_sha256":state_sha256(chosen),"normalization":parent["normalization"],"test":test,"gates":gates,"plan":plan.to_dict(),"runtime":payload["runtime"]}
    _atomic(output/"runtime.pt",release); return {"status":release["status"],"updates":plan.total_updates,"test":test,"gates":gates,"checkpoint_sha256":file_sha256(output/"runtime.pt"),"runtime":payload["runtime"]}
