from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from ..whole_viewport_latent_v1.data import load_corpus, rows, split_episodes
from ..whole_viewport_latent_v1.decoder import load_decoder
from .contract import CHECKPOINT_FORMAT, DEFAULT_CORPUS, DEFAULT_OUTPUT, DEFAULT_VAE, FORMAT, ModelConfig, TrainingConfig, canonical, config_dict, source_sha256
from .model import MobileViewportDecoder


def _atomic(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    torch.save(payload, temporary); os.replace(temporary, path)


def _edge(prediction, target):
    return F.l1_loss(prediction[:, :, :, 1:] - prediction[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1]) + F.l1_loss(prediction[:, :, 1:] - prediction[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def _laplacian(value):
    kernel = value.new_tensor(((0, 1, 0), (1, -4, 1), (0, 1, 0))).reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return F.conv2d(value, kernel, padding=1, groups=3)


def _multiscale(prediction, target):
    return sum(F.l1_loss(F.avg_pool2d(prediction, scale), F.avg_pool2d(target, scale)) for scale in (2, 4, 8)) / 3


@torch.inference_mode()
def evaluate(model, teacher, data, indices, device, batch_size=12):
    model.eval(); totals = {"rgb_mae": 0.0, "edge_mae": 0.0, "teacher_rgb_mae": 0.0}; count = 0
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        target = torch.as_tensor(data["frame"][chosen], device=device).permute(0, 3, 1, 2).float() / 255
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latent = teacher.encode(target)[0]; prediction = model(latent); teacher_frame = teacher.decode(latent)
        n = len(chosen); totals["rgb_mae"] += float(F.l1_loss(prediction.float(), target)) * n
        totals["edge_mae"] += float(_edge(prediction.float(), target)) * n
        totals["teacher_rgb_mae"] += float(F.l1_loss(teacher_frame.float(), target)) * n; count += n
    return {name: value / count for name, value in totals.items()}


@torch.inference_mode()
def _contact(model, teacher, data, indices, device, path: Path):
    chosen = np.asarray(indices[:6]); target = torch.as_tensor(data["frame"][chosen], device=device).permute(0, 3, 1, 2).float() / 255
    latent = teacher.encode(target)[0]; prediction = model(latent).float().clamp(0, 1); teacher_frame = teacher.decode(latent).float().clamp(0, 1)
    sheet = Image.new("RGB", (256 * 3, 280 * len(chosen)), (4, 11, 15)); draw = ImageDraw.Draw(sheet)
    for row in range(len(chosen)):
        y = row * 280; draw.text((8, y + 8), "TARGET", fill=(95, 255, 222)); draw.text((264, y + 8), "MOBILE", fill=(255, 105, 196)); draw.text((520, y + 8), "TEACHER VAE", fill=(220, 225, 230))
        for column, tensor in enumerate((target, prediction, teacher_frame)):
            image = tensor[row].mul(255).byte().permute(1, 2, 0).cpu().numpy(); sheet.paste(Image.fromarray(image), (column * 256, y + 24))
    sheet.save(path, optimize=True)


def train(*, corpus: Path = DEFAULT_CORPUS, vae_release: Path = DEFAULT_VAE, output: Path = DEFAULT_OUTPUT, model_config: ModelConfig = ModelConfig(), training: TrainingConfig = TrainingConfig()):
    corpus, output = Path(corpus), Path(output)
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available(): raise RuntimeError("mobile viewport decoder distillation requires CUDA")
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(0.35, 0)
    torch.manual_seed(training.seed); np.random.seed(training.seed & 0xffffffff); random.seed(training.seed)
    device = torch.device("cuda"); episodes, manifests = load_corpus(corpus); train_episodes, validation_episodes, holdout = split_episodes(episodes)
    train_data, validation_data = rows(train_episodes), rows(validation_episodes); train_indices = np.arange(len(train_data["frame"])); validation_indices = np.arange(len(validation_data["frame"]))
    teacher, teacher_provenance = load_decoder(device, vae_release); teacher.eval().requires_grad_(False)
    model = MobileViewportDecoder(model_config).to(device); ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay, fused=True); scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(training.seed); work = output.parent / f".{output.name}.work"; latest = work / "latest.pt"; history = []; best = None; best_state = None; step = 0
    identity = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "model_config": config_dict(model_config), "training_config": config_dict(training), "corpus_manifests": [item["manifest_sha256"] for item in manifests], "holdout_indices": holdout, "teacher": teacher_provenance}
    if latest.exists():
        payload = torch.load(latest, map_location=device, weights_only=False)
        if any(payload.get(name) != value for name, value in identity.items()): raise ValueError("mobile decoder resume provenance drifted")
        model.load_state_dict(payload["model"]); ema.load_state_dict(payload["ema"]); optimizer.load_state_dict(payload["optimizer"]); scaler.load_state_dict(payload["scaler"]); rng.bit_generator.state = payload["rng_state"]
        step, history, best, best_state = payload["step"], payload["history"], payload["best"], payload["best_state"]
    began = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); model.train()
    for step in range(step + 1, training.steps + 1):
        chosen = rng.choice(train_indices, training.batch_size, replace=False); target = torch.as_tensor(train_data["frame"][chosen], device=device).permute(0, 3, 1, 2).float() / 255
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): latent = teacher.encode(target)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(latent); rgb = F.l1_loss(prediction.float(), target); edge = _edge(prediction.float(), target); laplacian = F.l1_loss(_laplacian(prediction.float()), _laplacian(target)); multiscale = _multiscale(prediction.float(), target)
            loss = rgb + training.edge_weight * edge + training.laplacian_weight * laplacian + training.multiscale_weight * multiscale
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1)); scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            decay = min(training.ema_decay, step / (step + 10))
            for smooth, current in zip(ema.state_dict().values(), model.state_dict().values()):
                if smooth.is_floating_point(): smooth.lerp_(current.detach(), 1 - decay)
                else: smooth.copy_(current)
        if step == 1 or step % 25 == 0: history.append({"step": step, "loss": float(loss), "rgb": float(rgb), "edge": float(edge), "laplacian": float(laplacian), "multiscale": float(multiscale), "gradient": gradient})
        if step % training.validation_every == 0 or step == training.steps:
            sample = validation_indices[::max(1, len(validation_indices) // 192)][:192]; metrics = evaluate(ema, teacher, validation_data, sample, device, training.batch_size); score = metrics["rgb_mae"] + metrics["edge_mae"]
            if best is None or score < best: best = score; best_state = {name: value.detach().cpu().clone() for name, value in ema.state_dict().items()}
            print(json.dumps({"step": step, **metrics, "score": score, "best": best}), flush=True)
        if step % training.checkpoint_every == 0 or step == training.steps: _atomic({**identity, "step": step, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng_state": rng.bit_generator.state, "history": history, "best": best, "best_state": best_state}, latest)
    model.load_state_dict(best_state); full = evaluate(model, teacher, validation_data, validation_indices, device, training.batch_size); runtime = {"seconds": time.perf_counter() - began, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; staging.mkdir(parents=True); artifact = staging / "model.pt"; state = {name: value.detach().cpu() for name, value in model.state_dict().items()}; torch.save({**identity, "state": state, "validation": full, "runtime": runtime}, artifact)
    _contact(model, teacher, validation_data, validation_indices[::max(1, len(validation_indices) // 6)], device, staging / "contact.png")
    parameters = sum(parameter.numel() for parameter in model.parameters()); gates = {"rgb_mae_under_0_025": full["rgb_mae"] < .025, "edge_mae_under_0_040": full["edge_mae"] < .040, "within_25_percent_teacher_mae": full["rgb_mae"] < full["teacher_rgb_mae"] * 1.25, "under_2m_parameters": parameters < 2_000_000, "under_8gib_training_vram": runtime["peak_reserved_bytes"] < 8 * 1024**3}; gates["all_passed"] = all(gates.values())
    manifest = {"format": FORMAT, "status": "accepted" if gates["all_passed"] else "rejected", "source_sha256": source_sha256(), "model_config": config_dict(model_config), "training_config": config_dict(training), "teacher": teacher_provenance, "frames": sum(len(item["frame"]) for item in episodes), "episodes": len(episodes), "parameters": parameters, "validation": full, "runtime": runtime, "gates": gates, "artifact": {"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}, "contact": "contact.png"}; manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest(); (staging / "manifest.json").write_bytes(canonical(manifest)); os.replace(staging, output); return manifest
