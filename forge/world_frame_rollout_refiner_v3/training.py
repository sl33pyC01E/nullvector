from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..safety import require_disk_floor
from ..world_frame_vae_refiner.contract import ModelConfig
from ..world_frame_vae_refiner.model import PixelCellRefiner
from .cache import load_cache
from .contract import CHECKPOINT_FORMAT, DECODER_SHA256, DEFAULT_CACHE, DEFAULT_OUTPUT, REPORT_FORMAT, TrainingPlan, canonical, file_sha256, source_sha256, state_sha256


def _atomic(path, payload):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _edge(value):
    return value[:, :, :, 1:] - value[:, :, :, :-1], value[:, :, 1:] - value[:, :, :-1]


def _loss(prediction, target):
    dx, dy = _edge(prediction)
    tx, ty = _edge(target)
    blur = F.avg_pool2d(prediction, 3, 1, 1)
    target_blur = F.avg_pool2d(target, 3, 1, 1)
    l1 = F.l1_loss(prediction, target)
    edge = F.l1_loss(dx, tx) + F.l1_loss(dy, ty)
    high = F.l1_loss(prediction - blur, target - target_blur)
    laplace = F.l1_loss(dx[:, :, :, 1:] - dx[:, :, :, :-1], tx[:, :, :, 1:] - tx[:, :, :, :-1]) + F.l1_loss(dy[:, :, 1:] - dy[:, :, :-1], ty[:, :, 1:] - ty[:, :, :-1])
    return l1 * 5 + edge * 5 + high * 4 + laplace * 2, (l1, edge, high)


@torch.inference_mode()
def _metrics(model, base, target, device, batch=8):
    refined = []
    for start in range(0, len(base), batch):
        value = torch.from_numpy(base[start:start + batch]).permute(0, 3, 1, 2).float().div_(255).to(device)
        refined.append(model(value).float().cpu())
    base_tensor = torch.from_numpy(base).permute(0, 3, 1, 2).float().div_(255)
    target_tensor = torch.from_numpy(target).permute(0, 3, 1, 2).float().div_(255)
    refined = torch.cat(refined)
    bdx, bdy = _edge(base_tensor)
    rdx, rdy = _edge(refined)
    tdx, tdy = _edge(target_tensor)
    base_mae, refined_mae = float(F.l1_loss(base_tensor, target_tensor)), float(F.l1_loss(refined, target_tensor))
    base_edge = float(F.l1_loss(bdx, tdx) + F.l1_loss(bdy, tdy))
    refined_edge = float(F.l1_loss(rdx, tdx) + F.l1_loss(rdy, tdy))
    mse = float(F.mse_loss(refined, target_tensor))
    return {"base_mae": base_mae, "refined_mae": refined_mae, "mae_improvement": 1 - refined_mae / base_mae, "base_edge_mae": base_edge, "refined_edge_mae": refined_edge, "edge_improvement": 1 - refined_edge / base_edge, "refined_psnr_db": -10 * math.log10(max(mse, 1e-12))}


def train(output: Path = DEFAULT_OUTPUT, *, cache: Path = DEFAULT_CACHE, plan: TrainingPlan = TrainingPlan()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024**3)
    if not torch.cuda.is_available():
        raise RuntimeError("rollout refiner requires CUDA")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.45, 0)
    torch.manual_seed(plan.seed)
    rng = np.random.default_rng(plan.seed)
    device = torch.device("cuda:0")
    rows, manifest = load_cache(cache)
    train_base = np.concatenate([row["base"] for row in rows[:4]])
    train_target = np.concatenate([row["target"] for row in rows[:4]])
    model = PixelCellRefiner(ModelConfig()).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-4, fused=True)
    history = []
    start_update = 0
    latest = output / "latest.pt"
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("cache_sha256") != manifest["manifest_sha256"] or payload.get("plan") != plan.to_dict():
            raise ValueError("rollout refiner resume drifted")
        model.load_state_dict(payload["model_state"])
        ema.load_state_dict(payload["ema_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        rng.bit_generator.state = payload["rng_state"]
        history = list(payload["history"])
        start_update = payload["update"]
    payload = None
    for end in range(start_update + plan.segment_updates, plan.total_updates + 1, plan.segment_updates):
        began = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for update in range(end - plan.segment_updates + 1, end + 1):
            indices = rng.integers(0, len(train_base), plan.batch_size)
            top = int(rng.integers(0, 257 - plan.crop))
            left = int(rng.integers(0, 257 - plan.crop))
            target = torch.from_numpy(train_target[indices, top:top + plan.crop, left:left + plan.crop]).permute(0, 3, 1, 2).float().div_(255).to(device)
            if rng.random() < plan.identity_probability:
                base = target.clone()
                domain = "identity"
            else:
                base = torch.from_numpy(train_base[indices, top:top + plan.crop, left:left + plan.crop]).permute(0, 3, 1, 2).float().div_(255).to(device)
                domain = "rollout"
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(base)
                loss, parts = _loss(prediction, target)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
            optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(list(ema.parameters()), plan.ema_decay)
                torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
            if update == 1 or update % 100 == 0:
                history.append({"update": update, "domain": domain, "loss": round(float(loss), 7), "mae": round(float(parts[0]), 7), "edge": round(float(parts[1]), 7), "high": round(float(parts[2]), 7), "gradient": round(gradient, 7)})
        elapsed = time.perf_counter() - began
        raw = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        smooth = {name: value.detach().cpu() for name, value in ema.state_dict().items()}
        payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "cache_sha256": manifest["manifest_sha256"], "decoder_sha256": DECODER_SHA256, "plan": plan.to_dict(), "model_config": {"width": 64, "blocks": 8, "maximum_delta": 0.24}, "update": end, "model_state": raw, "ema_state": smooth, "optimizer_state": optimizer.state_dict(), "rng_state": rng.bit_generator.state, "history": history, "runtime": {"segment_seconds": round(elapsed, 6), "updates_per_second": round(plan.segment_updates / elapsed, 4), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}}
        _atomic(latest, payload)
        _atomic(output / f"milestone_{end:07d}.pt", payload)
        print(json.dumps({"update": end, "history": history[-1], "runtime": payload["runtime"]}), flush=True)
    raw_validation = _metrics(model.eval(), rows[4]["base"], rows[4]["target"], device)
    ema_validation = _metrics(ema.eval(), rows[4]["base"], rows[4]["target"], device)
    selected_name, selected = ("raw", model) if raw_validation["refined_mae"] <= ema_validation["refined_mae"] else ("ema", ema)
    test = _metrics(selected, rows[5]["base"], rows[5]["target"], device)
    identity = _metrics(selected, rows[5]["target"], rows[5]["target"], device)
    gates = {"rollout_mae_improves": test["mae_improvement"] > 0, "rollout_edge_improves": test["edge_improvement"] > 0, "identity_drift_below_0_005": identity["refined_mae"] < 0.005, "under_half_gpu_memory": payload["runtime"]["peak_reserved_bytes"] < 12 * 1024**3}
    gates["all_passed"] = all(gates.values())
    selected_state = {name: value.detach().cpu() for name, value in selected.state_dict().items()}
    report = {"format": REPORT_FORMAT, "status": "ready" if gates["all_passed"] else "experimental", "source_sha256": source_sha256(), "cache_sha256": manifest["manifest_sha256"], "decoder_sha256": DECODER_SHA256, "updates": plan.total_updates, "selection": {"variant": selected_name, "validation": {"raw": raw_validation, "ema": ema_validation}}, "test": test, "identity_test": identity, "gates": gates, "runtime": payload["runtime"]}
    release = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "cache_sha256": manifest["manifest_sha256"], "decoder_sha256": DECODER_SHA256, "model_config": payload["model_config"], "state": selected_state, "state_sha256": state_sha256(selected_state), "report": report}
    _atomic(output / "runtime.pt", release)
    report["checkpoint_sha256"] = file_sha256(output / "runtime.pt")
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    (output / "report.json").write_bytes(canonical(report))
    return report
