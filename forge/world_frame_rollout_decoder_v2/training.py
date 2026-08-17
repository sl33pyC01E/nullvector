from __future__ import annotations

import copy
from dataclasses import asdict
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
from ..world_action_natural_v10 import load
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from .contract import CHECKPOINT_FORMAT, DEFAULT_CORPUS, DEFAULT_OUTPUT, NATURAL_CORPUS, PARENT_CODEC, PARENT_CODEC_SHA256, REPORT_FORMAT, TrainingPlan, canonical, file_sha256, source_sha256, state_sha256
from .corpus import load_corpus


def _atomic(path, payload):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _decoder_parameters(model):
    return [parameter for module in (model.from_latent, model.decoder, model.out) for parameter in module.parameters()]


def _loss(prediction, target):
    l1 = F.l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    pdx, tdx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1]
    pdy, tdy = prediction[:, :, 1:] - prediction[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]
    edge = F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)
    coarse = F.l1_loss(F.avg_pool2d(prediction, 4), F.avg_pool2d(target, 4))
    return l1 * 5 + mse * 2 + edge * 3 + coarse, (l1, edge)


@torch.inference_mode()
def _metrics(model, candidates, targets, device, batch=4):
    predictions = []
    for start in range(0, len(candidates), batch):
        predictions.append(model.decode(torch.from_numpy(candidates[start:start + batch]).float().to(device)).float().cpu())
    prediction = torch.cat(predictions)
    target = torch.from_numpy(targets).permute(0, 3, 1, 2).float().div_(255)
    mae = float(F.l1_loss(prediction, target))
    mse = float(F.mse_loss(prediction, target))
    edge = float(F.l1_loss(prediction[:, :, :, 1:] - prediction[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1]) + F.l1_loss(prediction[:, :, 1:] - prediction[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]))
    return {"mae": mae, "mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12)), "edge_mae": edge}


def train(output: Path = DEFAULT_OUTPUT, *, corpus: Path = DEFAULT_CORPUS, plan: TrainingPlan = TrainingPlan()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if not torch.cuda.is_available() or file_sha256(PARENT_CODEC) != PARENT_CODEC_SHA256:
        raise RuntimeError("rollout decoder parent or CUDA unavailable")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.45, 0)
    torch.manual_seed(plan.seed)
    rng = np.random.default_rng(plan.seed)
    device = torch.device("cuda:0")
    rollout, corpus_manifest = load_corpus(corpus)
    sequences, natural_manifest = load(NATURAL_CORPUS)
    parent = AdaptedWorldFrameCodec.from_checkpoint(PARENT_CODEC, device="cuda")
    model = copy.deepcopy(parent.model).train().requires_grad_(False)
    parameters = _decoder_parameters(model)
    for parameter in parameters:
        parameter.requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(parameters, lr=plan.learning_rate, weight_decay=1e-4, fused=True)
    train_candidate = np.concatenate([row["candidate"] for row in rollout[:4]])
    train_world = np.concatenate([np.full(len(row["target"]), index, np.int16) for index, row in enumerate(rollout[:4])])
    train_target = np.concatenate([row["target"] for row in rollout[:4]])
    history = []
    latest = output / "latest.pt"
    start_update = 0
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("corpus_sha256") != corpus_manifest["manifest_sha256"] or payload.get("parent_codec_sha256") != PARENT_CODEC_SHA256 or payload.get("plan") != plan.to_dict():
            raise ValueError("rollout decoder resume drifted")
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
        for update in range(end - plan.segment_updates + 1, end + 1):
            authoritative = bool(rng.random() < plan.authoritative_probability)
            if authoritative:
                world = int(rng.integers(0, 4))
                indices = rng.integers(0, len(sequences[world]["latent"]), plan.batch_size)
                latent = torch.from_numpy(sequences[world]["latent"][indices]).to(device)
                target = torch.from_numpy(sequences[world]["frame"][indices]).permute(0, 3, 1, 2).float().div_(255).to(device)
            else:
                indices = rng.integers(0, len(train_candidate), plan.batch_size)
                latent = torch.from_numpy(train_candidate[indices]).float().to(device)
                target = torch.stack([torch.from_numpy(sequences[int(train_world[index])]["frame"][int(train_target[index])]).permute(2, 0, 1) for index in indices]).float().div_(255).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.decode(latent)
                loss, parts = _loss(prediction, target)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1))
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(_decoder_parameters(ema), parameters):
                    ema_parameter.lerp_(parameter, 1 - plan.ema_decay)
            if update == 1 or update % 50 == 0:
                history.append({"update": update, "domain": "authoritative" if authoritative else "rollout", "loss": round(float(loss), 7), "mae": round(float(parts[0]), 7), "edge": round(float(parts[1]), 7), "gradient": round(gradient, 7)})
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        ema_state = {name: value.detach().cpu() for name, value in ema.state_dict().items()}
        elapsed = time.perf_counter() - began
        payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_manifest["manifest_sha256"], "natural_corpus_sha256": natural_manifest["manifest_sha256"], "parent_codec_sha256": PARENT_CODEC_SHA256, "plan": plan.to_dict(), "model_config": asdict(parent.model.config), "update": end, "model_state": state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(), "rng_state": rng.bit_generator.state, "history": history, "runtime": {"segment_seconds": round(elapsed, 6), "updates_per_second": round(plan.segment_updates / elapsed, 4), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}}
        _atomic(latest, payload)
        _atomic(output / f"milestone_{end:07d}.pt", payload)
        print(json.dumps({"update": end, "history": history[-1], "runtime": payload["runtime"]}), flush=True)
    validation = rollout[4]
    validation_targets = sequences[4]["frame"][validation["target"]]
    raw_validation = _metrics(model.eval(), validation["candidate"], validation_targets, device)
    ema_validation = _metrics(ema.eval(), validation["candidate"], validation_targets, device)
    selected_name, selected = ("raw", model) if raw_validation["mae"] <= ema_validation["mae"] else ("ema", ema)
    test = rollout[5]
    test_targets = sequences[5]["frame"][test["target"]]
    parent_test = _metrics(parent.model, test["candidate"], test_targets, device)
    adapted_test = _metrics(selected, test["candidate"], test_targets, device)
    true_test = _metrics(selected, sequences[5]["latent"].astype(np.float16), sequences[5]["frame"], device)
    parent_true_test = _metrics(parent.model, sequences[5]["latent"].astype(np.float16), sequences[5]["frame"], device)
    adapted_test["mae_improvement"] = 1 - adapted_test["mae"] / parent_test["mae"]
    true_test["mae_retention_ratio"] = true_test["mae"] / parent_true_test["mae"]
    gates = {"rollout_test_improves_5pct": adapted_test["mae_improvement"] >= 0.05, "authoritative_test_within_15pct": true_test["mae_retention_ratio"] <= 1.15, "under_half_gpu_memory": payload["runtime"]["peak_reserved_bytes"] < 12 * 1024**3}
    gates["all_passed"] = all(gates.values())
    selected_state = {name: value.detach().cpu() for name, value in selected.state_dict().items()}
    report = {"format": REPORT_FORMAT, "status": "ready" if gates["all_passed"] else "experimental", "source_sha256": source_sha256(), "corpus_sha256": corpus_manifest["manifest_sha256"], "parent_codec_sha256": PARENT_CODEC_SHA256, "updates": plan.total_updates, "selection": {"variant": selected_name, "validation": {"raw": raw_validation, "ema": ema_validation}}, "rollout_test": {"parent": parent_test, "adapted": adapted_test}, "authoritative_test": {"parent": parent_true_test, "adapted": true_test}, "gates": gates, "runtime": payload["runtime"]}
    release = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_manifest["manifest_sha256"], "parent_codec_sha256": PARENT_CODEC_SHA256, "model_config": payload["model_config"], "state": selected_state, "state_sha256": state_sha256(selected_state), "report": report}
    _atomic(output / "runtime.pt", release)
    report["checkpoint_sha256"] = file_sha256(output / "runtime.pt")
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    (output / "report.json").write_bytes(canonical(report))
    return report
