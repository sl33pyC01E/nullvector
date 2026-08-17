from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ..action_teacher_v1 import validate_trajectory
from ..safety import require_disk_floor
from ..world_action_cellular_v7.corpus import load_encoded_corpus
from ..world_frame_vae.contract import ModelConfig
from ..world_frame_vae.model import WorldFrameVAE
from .contract import BASE_CHECKPOINT, BASE_SHA256, CELLULAR_CORPUS, CHECKPOINT_FORMAT, DEFAULT_OUTPUT, ORIGINAL_EPISODES, REPORT_FORMAT, TrainingPlan, canonical, file_sha256, source_sha256, state_sha256


def _atomic_torch(path: Path, payload):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _base(device):
    if file_sha256(BASE_CHECKPOINT) != BASE_SHA256:
        raise ValueError("world-frame VAE parent drifted")
    payload = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    model = WorldFrameVAE(ModelConfig(**payload["model_config"]))
    if state_sha256(payload["ema_state"]) != payload.get("ema_sha256"):
        raise ValueError("world-frame VAE EMA drifted")
    model.load_state_dict(payload["ema_state"], strict=True)
    return model.to(device).eval(), payload


def _original_frames():
    rows = []
    for root in ORIGINAL_EPISODES:
        manifest = validate_trajectory(root)
        with np.load(root / manifest["artifact"]["path"], allow_pickle=False) as archive:
            rows.append(archive["frame"].copy())
    frames = np.concatenate(rows)
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)


def _cellular_frames(episodes):
    return [torch.from_numpy(np.concatenate((episode["current_frame"], episode["target_frame"]))).permute(0, 3, 1, 2).float().div_(255) for episode in episodes]


@torch.inference_mode()
def _encode(model, frames, device, batch=4):
    rows = []
    for start in range(0, len(frames), batch):
        rows.append(model.encode(frames[start:start + batch].to(device))[0].float().cpu())
    return torch.cat(rows)


def _decoder_parameters(model):
    modules = (model.from_latent, model.decoder, model.out)
    return [parameter for module in modules for parameter in module.parameters()]


def _loss(prediction, target):
    l1 = F.l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    pdx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    tdx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pdy = prediction[:, :, 1:] - prediction[:, :, :-1]
    tdy = target[:, :, 1:] - target[:, :, :-1]
    edge = F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)
    coarse = F.l1_loss(F.avg_pool2d(prediction, 4), F.avg_pool2d(target, 4))
    return l1 * 5 + mse * 2 + edge * 3 + coarse, (l1, mse, edge)


@torch.inference_mode()
def _metrics(model, latent, target, device, batch=4):
    prediction = []
    for start in range(0, len(target), batch):
        prediction.append(model.decode(latent[start:start + batch].to(device)).float().cpu())
    prediction = torch.cat(prediction)
    mae = float(F.l1_loss(prediction, target))
    mse = float(F.mse_loss(prediction, target))
    edge = float(F.l1_loss(prediction[:, :, :, 1:] - prediction[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1]) + F.l1_loss(prediction[:, :, 1:] - prediction[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]))
    return {"mae": mae, "mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12)), "edge_mae": edge}, prediction


def _contact(path, target, base, adapted):
    count = min(8, len(target))
    sheet = Image.new("RGB", (count * 256, 3 * 256))
    for index in range(count):
        for row, values in enumerate((target, base, adapted)):
            image = np.clip(values[index].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
            sheet.paste(Image.fromarray(image), (index * 256, row * 256))
    sheet.save(path)


def train(output: Path = DEFAULT_OUTPUT, *, plan: TrainingPlan = TrainingPlan()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    if not torch.cuda.is_available():
        raise RuntimeError("decoder adaptation requires CUDA")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.45, 0)
    torch.manual_seed(plan.seed)
    rng = np.random.default_rng(plan.seed)
    device = torch.device("cuda:0")
    episodes, corpus_manifest = load_encoded_corpus(CELLULAR_CORPUS)
    original = _original_frames()
    base, base_payload = _base(device)
    cellular = _cellular_frames(episodes)
    # The encoder is immutable. Pre-encoding once both enforces and accelerates that contract.
    original_latent = _encode(base, original, device)
    cellular_latent = [_encode(base, frames, device) for frames in cellular]
    base_state = {name: value.detach().cpu().clone() for name, value in base.state_dict().items()}
    model = copy.deepcopy(base)
    model.train()
    model.requires_grad_(False)
    parameters = _decoder_parameters(model)
    for parameter in parameters:
        parameter.requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(parameters, lr=plan.learning_rate, weight_decay=1e-4, fused=True)
    history = []
    start = 0
    latest = output / "latest.pt"
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256() or payload.get("corpus_sha256") != corpus_manifest["manifest_sha256"] or payload.get("base_sha256") != BASE_SHA256:
            raise ValueError("decoder adaptation resume drifted")
        model.load_state_dict(payload["model_state"])
        ema.load_state_dict(payload["ema_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        rng.bit_generator.state = payload["rng_state"]
        history = list(payload["history"])
        start = int(payload["update"])
    train_frames = torch.cat(cellular[:4])
    train_latent = torch.cat(cellular_latent[:4])
    original_train = original[:-60]
    original_train_latent = original_latent[:-60]
    for end in range(start + plan.segment_updates, plan.total_updates + 1, plan.segment_updates):
        began = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for update in range(end - plan.segment_updates + 1, end + 1):
            use_original = bool(rng.random() < plan.original_probability)
            source_frames, source_latent = (original_train, original_train_latent) if use_original else (train_frames, train_latent)
            indices = torch.from_numpy(rng.integers(0, len(source_frames), plan.batch_size))
            target = source_frames[indices].to(device)
            latent = source_latent[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.decode(latent)
                loss, parts = _loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("decoder adaptation loss became non-finite")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1))
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(_decoder_parameters(ema), parameters):
                    ema_parameter.lerp_(parameter, 1 - plan.ema_decay)
            if update == 1 or update % 50 == 0:
                history.append({"update": update, "domain": "original" if use_original else "cellular", "loss": round(float(loss), 7), "mae": round(float(parts[0]), 7), "edge": round(float(parts[2]), 7), "gradient": round(gradient, 7)})
        # Assert the entire encoder half remains byte-identical to the promoted parent.
        current = model.state_dict()
        frozen_names = [name for name in current if not name.startswith(("from_latent.", "decoder.", "out."))]
        if any(not torch.equal(current[name].cpu(), base_state[name]) for name in frozen_names):
            raise RuntimeError("frozen world encoder changed during decoder adaptation")
        model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        ema_state = {name: value.detach().cpu() for name, value in ema.state_dict().items()}
        payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_manifest["manifest_sha256"], "base_sha256": BASE_SHA256, "model_config": base_payload["model_config"], "plan": plan.to_dict(), "update": end, "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(), "rng_state": rng.bit_generator.state, "history": history, "runtime": {"segment_seconds": round(time.perf_counter() - began, 6), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}}
        _atomic_torch(latest, payload)
        _atomic_torch(output / f"milestone_{end:07d}.pt", payload)
        print(json.dumps({"update": end, "loss": history[-1]["loss"], **payload["runtime"]}), flush=True)
    validation_target = cellular[4]
    validation_latent = cellular_latent[4]
    raw_metrics, _ = _metrics(model.eval(), validation_latent, validation_target, device)
    ema_metrics, _ = _metrics(ema.eval(), validation_latent, validation_target, device)
    selected_name = "raw" if raw_metrics["mae"] <= ema_metrics["mae"] else "ema"
    selected = model if selected_name == "raw" else ema
    selected_state = {name: value.detach().cpu() for name, value in selected.state_dict().items()}
    base_test_metrics, base_test = _metrics(base, cellular_latent[5], cellular[5], device)
    test_metrics, adapted_test = _metrics(selected, cellular_latent[5], cellular[5], device)
    original_base_metrics, _ = _metrics(base, original_latent[-60:], original[-60:], device)
    original_metrics, _ = _metrics(selected, original_latent[-60:], original[-60:], device)
    test_metrics["mae_improvement"] = 1 - test_metrics["mae"] / base_test_metrics["mae"]
    original_metrics["mae_retention_ratio"] = original_metrics["mae"] / original_base_metrics["mae"]
    gates = {"cellular_mae_improves_20pct": test_metrics["mae_improvement"] >= 0.20, "original_mae_within_35pct": original_metrics["mae_retention_ratio"] <= 1.35, "encoder_exact": True}
    gates["all_passed"] = all(gates.values())
    report = {"format": REPORT_FORMAT, "status": "ready" if gates["all_passed"] else "experimental", "source_sha256": source_sha256(), "corpus_sha256": corpus_manifest["manifest_sha256"], "base_sha256": BASE_SHA256, "parameters": sum(parameter.numel() for parameter in selected.parameters()), "trainable_parameters": sum(parameter.numel() for parameter in parameters), "updates": plan.total_updates, "selection": {"variant": selected_name, "validation": {"raw": raw_metrics, "ema": ema_metrics}}, "cellular_test": {"base": base_test_metrics, "adapted": test_metrics}, "original_test": {"base": original_base_metrics, "adapted": original_metrics}, "gates": gates, "runtime": payload["runtime"]}
    release = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_manifest["manifest_sha256"], "base_sha256": BASE_SHA256, "model_config": base_payload["model_config"], "state": selected_state, "state_sha256": state_sha256(selected_state), "report": report}
    _atomic_torch(output / "runtime.pt", release)
    report["checkpoint"] = {"path": "runtime.pt", "sha256": file_sha256(output / "runtime.pt"), "state_sha256": release["state_sha256"]}
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    (output / "report.json").write_bytes(canonical(report))
    _contact(output / "cellular_test_contact_sheet.png", cellular[5], base_test, adapted_test)
    return report
