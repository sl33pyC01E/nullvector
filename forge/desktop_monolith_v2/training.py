from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..mobile_coordinator_student_v1.training import load_corpus as load_high_corpus
from ..safety import require_disk_floor
from ..world_action_cellular_v7.corpus import load_encoded_corpus
from ..world_latent_dit.contract import ModelConfig as ActionConfig
from .contract import (ACTION_CORPUS, ACTION_PARENT, ACTION_REPORT, CHECKPOINT_FORMAT, DEFAULT_OUTPUT,
                       HIGH_CORPUS, ModelConfig, TrainingConfig, canonical, config_dict, file_sha256, source_sha256)
from .model import DesktopWorldMonolith


def _atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _action_batch(episodes, rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
    rows = []
    for _ in range(size):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        index = int(rng.integers(0, len(episode["current"])))
        rows.append((episode, index))
    return {name: np.stack([episode[name][index] for episode, index in rows]) for name in
            ("previous", "current", "target", "action", "control", "state", "actor_state")}


def _to(value, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _inputs(action: dict[str, np.ndarray], high: dict[str, np.ndarray], indices: np.ndarray,
            device: torch.device, mean: torch.Tensor, std: torch.Tensor):
    current = (_to(action["current"], device) - mean) / std
    previous = (_to(action["previous"], device) - mean) / std
    target = (_to(action["target"], device) - mean) / std
    values = [current, previous, _to(action["action"], device, torch.long), _to(action["control"], device),
              _to(action["state"], device), _to(action["actor_state"], device)]
    for name in ("current", "previous", "global_state", "previous_global", "members", "member_mask", "society", "sequence"):
        tensor = torch.from_numpy(high[name][indices]).to(device)
        values.append(tensor.bool() if name == "member_mask" else tensor.float())
    return tuple(values), target


def _loss(outputs, inputs, target_visual, high, indices, device):
    current = inputs[0]; mask = inputs[11]
    target_names = ("target_macro", "target_macro_global", "target_role", "target_member_action",
                    "target_activity", "target_labor", "target_diplomacy", "target_project",
                    "target_timeline", "target_event", "target_confidence", "target_counter_state",
                    "target_benefit", "target_risk")
    targets = [torch.from_numpy(high[name][indices]).float().to(device) for name in target_names]
    (visual, macro, global_state, role, member_action, activity, labor, diplomacy, project,
     timeline, event, confidence, counter_state, benefit, risk) = outputs
    (tm, tg, tr, ta, tact, tlabor, tdip, tproject, tt, tevent, tconf, tcounter, tb, trisk) = targets
    magnitude = (target_visual - current).detach().abs().mean(1, keepdim=True)
    visual_weight = 1 + 6 * torch.clamp(magnitude / .35, 0, 2)
    losses = {
        "visual": (F.smooth_l1_loss(visual, target_visual - current, reduction="none") * visual_weight).mean(),
        "macro": F.smooth_l1_loss((macro - inputs[6]) * 100, (tm - inputs[6]) * 100),
        "global": F.smooth_l1_loss((global_state - inputs[8]) * 24, (tg - inputs[8]) * 24),
        "role": F.cross_entropy(role[mask], tr.argmax(-1)[mask]),
        "member_action": F.smooth_l1_loss(member_action[mask], ta[mask]),
        "society": (F.cross_entropy(activity, tact.argmax(-1)) + F.cross_entropy(labor, tlabor.argmax(-1)) +
                    F.cross_entropy(diplomacy, tdip.argmax(-1)) + F.cross_entropy(project, tproject.argmax(-1))) / 4,
        "timeline": F.smooth_l1_loss((timeline - inputs[13][:, -1]) * 8, (tt - inputs[13][:, -1]) * 8),
        "event": F.cross_entropy(event, tevent.argmax(-1)),
        "confidence": F.smooth_l1_loss(confidence, tconf),
        "counter_state": F.smooth_l1_loss((counter_state - inputs[13][:, -1, None]) * 5,
                                           (tcounter - inputs[13][:, -1, None]) * 5),
        "counter_value": F.smooth_l1_loss(benefit, tb) + F.smooth_l1_loss(risk, trisk),
        "counter_rank": F.cross_entropy(benefit * 6, tb.argmax(-1)),
    }
    total = (losses["visual"] * 5 + losses["macro"] * 3 + losses["global"] + losses["role"] * .4 +
             losses["member_action"] + losses["society"] * .7 + losses["timeline"] + losses["event"] * .4 +
             losses["confidence"] + losses["counter_state"] + losses["counter_value"] * 2 + losses["counter_rank"])
    return total, {name: float(value.detach()) for name, value in losses.items()}


@torch.inference_mode()
def evaluate(model, action_episode, high, high_indices, device, mean, std):
    count = min(len(action_episode["current"]), len(high_indices), 192)
    metrics = {name: [] for name in ("visual_mae", "persistence_mae", "role", "activity", "labor", "diplomacy", "project", "event", "counter_best", "macro_mae", "macro_persistence")}
    for start in range(0, count, 6):
        end = min(count, start + 6); action = {name: value[start:end] for name, value in action_episode.items() if name in ("previous", "current", "target", "action", "control", "state", "actor_state")}; idx = high_indices[start:end]
        inputs, target = _inputs(action, high, idx, device, mean, std); output = model(*inputs)
        metrics["visual_mae"].append(float((inputs[0] + output[0] - target).abs().mean())); metrics["persistence_mae"].append(float((inputs[0] - target).abs().mean()))
        targets = {name: torch.from_numpy(high[name][idx]).to(device) for name in ("target_macro", "target_role", "target_activity", "target_labor", "target_diplomacy", "target_project", "target_event", "target_benefit")}
        metrics["macro_mae"].append(float((output[1] - targets["target_macro"]).abs().mean())); metrics["macro_persistence"].append(float((inputs[6] - targets["target_macro"]).abs().mean()))
        valid=inputs[11]; metrics["role"].append(float((output[3].argmax(-1)[valid] == targets["target_role"].argmax(-1)[valid]).float().mean()))
        for name, position in (("activity",5),("labor",6),("diplomacy",7),("project",8),("event",10)):
            metrics[name].append(float((output[position].argmax(-1) == targets["target_" + name].argmax(-1)).float().mean()))
        metrics["counter_best"].append(float((output[13].argmax(-1) == targets["target_benefit"].argmax(-1)).float().mean()))
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def train(output: Path = DEFAULT_OUTPUT, *, plan: TrainingConfig = TrainingConfig(), config: ModelConfig = ModelConfig(), device: str = "cuda"):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 << 30)
    if not torch.cuda.is_available() and device == "cuda": raise RuntimeError("desktop monolith training requires CUDA")
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6); torch.manual_seed(plan.seed); rng = np.random.default_rng(plan.seed)
    if target.type == "cuda": torch.cuda.set_per_process_memory_fraction(.62, 0); torch.cuda.reset_peak_memory_stats(target)
    action_report = json.loads(ACTION_REPORT.read_bytes()); parent = torch.load(ACTION_PARENT, map_location="cpu", weights_only=True)
    if file_sha256(ACTION_PARENT) != action_report["checkpoint"]["sha256"]: raise ValueError("desktop Action-DiT parent drifted")
    episodes, action_manifest = load_encoded_corpus(ACTION_CORPUS); high, high_manifest = load_high_corpus(HIGH_CORPUS)
    action_config = ActionConfig(**parent["model_config"]); model = DesktopWorldMonolith(config, action_config); model.load_action_parent(parent["state"]); model.freeze_action_parent(); model.to(target)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=plan.learning_rate, weight_decay=plan.weight_decay, fused=target.type == "cuda")
    mean = _to(parent["normalization"]["mean"], target)[None, :, None, None]; std = _to(parent["normalization"]["std"], target)[None, :, None, None]
    train_episodes, validation = episodes[:4], episodes[4]; high_order = rng.permutation(len(high["current"])); high_train, high_validation = high_order[:-192], high_order[-192:]
    history=[]; began=time.perf_counter()
    for step in range(1, plan.steps + 1):
        action = _action_batch(train_episodes, rng, plan.batch_size); indices = rng.choice(high_train, plan.batch_size, replace=True); inputs, target_visual = _inputs(action, high, indices, target, mean, std)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
            outputs=model(*inputs); loss, parts=_loss(outputs, inputs, target_visual, high, indices, target)
        if not bool(torch.isfinite(loss)): raise FloatingPointError("desktop monolith loss became non-finite")
        loss.backward(); gradient=float(torch.nn.utils.clip_grad_norm_(parameters, 1)); optimizer.step()
        with torch.no_grad():
            for target_parameter, source_parameter in zip(ema.parameters(), model.parameters()): target_parameter.lerp_(source_parameter, 1-plan.ema_decay)
        if step == 1 or step % 100 == 0: history.append({"step":step,"loss":float(loss.detach()),"gradient":gradient,**parts})
        if step % 800 == 0:
            payload={"format":CHECKPOINT_FORMAT,"status":"training","source_sha256":source_sha256(),"step":step,"model_config":config_dict(config),"action_config":config_dict(action_config),"state":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"high_corpus_sha256":high_manifest["manifest_sha256"],"action_corpus_sha256":action_manifest["manifest_sha256"]}; output.mkdir(parents=True,exist_ok=True); _atomic(output/f"milestone_{step:07d}.pt",payload); _atomic(output/"latest.pt",payload)
    metrics=evaluate(model.eval(),validation,high,high_validation,target,mean,std)
    gates={"visual_beats_persistence":metrics["visual_mae"]<metrics["persistence_mae"],"macro_beats_persistence":metrics["macro_mae"]<=metrics["macro_persistence"]*1.1,"role":metrics["role"]>=.85,"society":min(metrics[name] for name in ("activity","labor","diplomacy","project"))>=.80,"event":metrics["event"]>=.80,"counterfactual":metrics["counter_best"]>=.80}
    payload={"format":CHECKPOINT_FORMAT,"status":"desktop_monolith_ready" if all(gates.values()) else "quality_failed","source_sha256":source_sha256(),"step":plan.steps,"model_config":config_dict(config),"action_config":config_dict(action_config),"state":model.state_dict(),"ema":ema.state_dict(),"history":history,"metrics":metrics,"gates":gates,"normalization":parent["normalization"],"high_corpus_sha256":high_manifest["manifest_sha256"],"action_corpus_sha256":action_manifest["manifest_sha256"]}; output.mkdir(parents=True,exist_ok=True); _atomic(output/"runtime.pt",payload)
    report={key:value for key,value in payload.items() if key not in ("state","ema")}; report["parameters"] = sum(value.numel() for value in model.parameters()); report["trainable_parameters"] = sum(value.numel() for value in parameters); report["seconds"] = time.perf_counter()-began; report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(target)) if target.type=="cuda" else 0; report["checkpoint"]={"path":"runtime.pt","bytes":(output/"runtime.pt").stat().st_size,"sha256":file_sha256(output/"runtime.pt")}; report["manifest_sha256"]=hashlib.sha256(canonical(report)).hexdigest(); (output/"report.json").write_bytes(canonical(report)); return report


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--steps",type=int,default=3200); parser.add_argument("--batch-size",type=int,default=6); parser.add_argument("--device",default="cuda"); args=parser.parse_args(argv)
    print(json.dumps(train(args.output,plan=TrainingConfig(steps=args.steps,batch_size=args.batch_size),device=args.device),indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
