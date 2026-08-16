from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from ..action_teacher_v1.contract import ACTIONS
from ..world_frame_vae import WorldFrameVAERuntime
from ..world_frame_vae_refiner import RefinedWorldFrameVAERuntime
from ..world_latent_dit.contract import ModelConfig as BackboneConfig
from ..world_latent_dit.model import ActionDiT
from .contract import CHECKPOINT_FORMAT, REPORT_FORMAT, ModelConfig, TrainingConfig, canonical, config_dict, source_sha256
from .data import encode_episodes
from .runtime import WorldActionStepRuntime


def _hash(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0" + value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _concat(episodes, name):
    return np.concatenate([episode[name] for episode in episodes])


def _contact(path: Path, current, target, predicted, wrong, actions):
    count = min(8, len(current))
    sheet = Image.new("RGB", (256 * count, 256 * 4), (3, 7, 10))
    draw = ImageDraw.Draw(sheet)
    for index in range(count):
        for row, value in enumerate((current[index], target[index], predicted[index], wrong[index])):
            sheet.paste(Image.fromarray(value), (index * 256, row * 256))
        draw.text((index * 256 + 4, 4), ACTIONS[int(actions[index])], fill=(72, 245, 255))
    sheet.save(path)


def _decode(vae, latent, device):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(latent), 8):
            decoded = vae.model.decode(latent[start : start + 8].to(device))
            if hasattr(vae, "refiner"):
                decoded = vae.refiner(decoded)
            rows.append(decoded.float().cpu())
    return torch.cat(rows)


def _per_action(action, target, current, predicted, wrong):
    rows = []
    for index, name in enumerate(ACTIONS):
        mask = torch.from_numpy(action == index)
        if not bool(mask.any()):
            continue
        baseline = float(F.l1_loss(current[mask], target[mask]))
        model = float(F.l1_loss(predicted[mask], target[mask]))
        ablated = float(F.l1_loss(wrong[mask], target[mask]))
        rows.append({"action": name, "count": int(mask.sum()), "model_latent_mae": model, "persistence_latent_mae": baseline, "wrong_action_latent_mae": ablated, "improvement": 1 - model / baseline if baseline else 0.0})
    return rows


def train(output: Path, episodes, vae_checkpoint: Path, refiner_checkpoint: Path, *, resume_checkpoint: Path | None = None, config=ModelConfig(), training=TrainingConfig()):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(training.seed)
    np.random.seed(training.seed & 0xFFFFFFFF)
    random.seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = RefinedWorldFrameVAERuntime.from_checkpoints(vae_checkpoint, refiner_checkpoint, device=str(device))
    vae.model.requires_grad_(False)
    vae.refiner.requires_grad_(False)
    encoded, sources, corpus_sha = encode_episodes(episodes, vae)
    train_episodes, heldout = encoded[:-1], encoded[-1]
    current_raw = _concat(train_episodes, "current")
    target_raw = _concat(train_episodes, "target")
    control = _concat(train_episodes, "control")
    action = _concat(train_episodes, "action")
    state = _concat(train_episodes, "state")
    target_frame = _concat(train_episodes, "target_frame")
    mean = np.mean(np.concatenate((current_raw, target_raw)), axis=(0, 2, 3))
    std = np.std(np.concatenate((current_raw, target_raw)), axis=(0, 2, 3)) + 1e-4
    current = (current_raw - mean[None, :, None, None]) / std[None, :, None, None]
    target = (target_raw - mean[None, :, None, None]) / std[None, :, None, None]
    counts = np.bincount(action.astype(np.int64), minlength=len(ACTIONS)).clip(1)
    action_weight = np.sqrt(counts.max() / counts).astype(np.float32)
    action_weight /= action_weight.mean()
    backbone = BackboneConfig(width=config.width, layers=config.layers, heads=config.heads, patch=config.patch)
    model = ActionDiT(backbone).to(device)
    initialization = {"kind": "random"}
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("format") != CHECKPOINT_FORMAT or resume.get("model_config") != config_dict(config):
            raise ValueError("causal action transfer checkpoint contract drifted")
        if resume.get("corpus_sha256") != corpus_sha or _hash(resume["ema_state"]) != resume.get("ema_sha256"):
            raise ValueError("causal action transfer checkpoint provenance drifted")
        model.load_state_dict(resume["ema_state"])
        initialization = {"kind": "validated-ema-transfer", "checkpoint_sha256": hashlib.sha256(resume_path.read_bytes()).hexdigest(), "source_sha256": resume["source_sha256"], "ema_sha256": resume["ema_sha256"]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=2e-3, fused=device.type == "cuda")
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rng = np.random.default_rng(training.seed)
    history = []
    mean_tensor = torch.from_numpy(mean).to(device).view(1, -1, 1, 1)
    std_tensor = torch.from_numpy(std).to(device).view(1, -1, 1, 1)
    for step in range(1, training.steps + 1):
        indices = rng.integers(0, len(current), training.batch_size)
        x0 = torch.from_numpy(current[indices]).to(device)
        x1 = torch.from_numpy(target[indices]).to(device)
        desired = x1 - x0
        noisy = x0 + torch.randn_like(x0) * training.input_noise
        act = torch.from_numpy(action[indices].astype(np.int64)).to(device)
        ctl = torch.from_numpy(control[indices]).to(device)
        st = torch.from_numpy(state[indices]).to(device)
        sample_weight = torch.from_numpy(action_weight[action[indices].astype(np.int64)]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(noisy, torch.zeros(len(x0), device=device), act, ctl, st)
            magnitude = desired.detach().abs().mean(1, keepdim=True)
            spatial_weight = 1 + training.changed_weight * torch.clamp(magnitude / 0.25, 0, 2)
            residual = (F.smooth_l1_loss(prediction, desired, reduction="none") * spatial_weight * sample_weight[:, None, None, None]).mean()
            edge = F.l1_loss(prediction[:, :, :, 1:] - prediction[:, :, :, :-1], desired[:, :, :, 1:] - desired[:, :, :, :-1]) + F.l1_loss(prediction[:, :, 1:] - prediction[:, :, :-1], desired[:, :, 1:] - desired[:, :, :-1])
            pixel_count = min(training.pixel_batch, len(indices))
            predicted_latent = (x0[:pixel_count] + prediction[:pixel_count]) * std_tensor + mean_tensor
            decoded = vae.refiner(vae.model.decode(predicted_latent))
            target_pixel = torch.from_numpy(target_frame[indices[:pixel_count]]).to(device).permute(0, 3, 1, 2).float().div_(255)
            pixel = F.l1_loss(decoded, target_pixel)
            loss = residual + training.edge_weight * edge + training.pixel_weight * pixel
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].lerp_(value.detach(), 1 - training.ema_decay)
        if step == 1 or step % 250 == 0 or step == training.steps:
            row = {"step": step, "loss": round(float(loss), 6), "residual": round(float(residual), 6), "edge": round(float(edge), 6), "pixel": round(float(pixel), 6)}
            history.append(row)
            print(json.dumps(row), flush=True)
    ema_cpu = {name: value.detach().cpu() for name, value in ema.items()}
    recovery = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "model_config": config_dict(config), "training_config": config_dict(training), "initialization": initialization, "latent_mean": mean.tolist(), "latent_std": std.tolist(), "ema_state": ema_cpu, "ema_sha256": _hash(ema_cpu), "history": history, "status": "trained_pending_evaluation"}
    torch.save(recovery, output / "trained_pending_evaluation.pt")
    model.load_state_dict(ema_cpu)
    model.to(device).eval()
    runtime = WorldActionStepRuntime(model, device, {}, mean_tensor, std_tensor)
    held_current = torch.from_numpy(heldout["current"])
    held_target = torch.from_numpy(heldout["target"])
    action_values = heldout["action"].astype(np.int64)
    wrong_actions = (action_values + 7) % len(ACTIONS)
    predicted, wrong, zero_control = [], [], []
    for start in range(0, len(held_current), 16):
        stop = start + 16
        value = held_current[start:stop]
        state_value = heldout["state"][start:stop]
        control_value = heldout["control"][start:stop]
        predicted.append(runtime.predict_latent(value, action=action_values[start:stop], control=control_value, state=state_value).cpu())
        wrong.append(runtime.predict_latent(value, action=wrong_actions[start:stop], control=control_value, state=state_value).cpu())
        zero_control.append(runtime.predict_latent(value, action=action_values[start:stop], control=np.zeros_like(control_value), state=state_value).cpu())
    predicted = torch.cat(predicted)
    wrong = torch.cat(wrong)
    zero_control = torch.cat(zero_control)
    persistence_latent = float(F.l1_loss(held_current, held_target))
    model_latent = float(F.l1_loss(predicted, held_target))
    wrong_latent = float(F.l1_loss(wrong, held_target))
    zero_control_latent = float(F.l1_loss(zero_control, held_target))
    decoded = _decode(vae, predicted, device)
    wrong_decoded = _decode(vae, wrong, device)
    target_frame = torch.from_numpy(heldout["target_frame"]).permute(0, 3, 1, 2).float().div_(255)
    current_frame = torch.from_numpy(heldout["current_frame"]).permute(0, 3, 1, 2).float().div_(255)
    model_rgb = float(F.l1_loss(decoded, target_frame))
    persistence_rgb = float(F.l1_loss(current_frame, target_frame))
    wrong_rgb = float(F.l1_loss(wrong_decoded, target_frame))
    predicted_frame = np.clip(decoded.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
    wrong_frame = np.clip(wrong_decoded.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
    per_action = _per_action(action_values, held_target, held_current, predicted, wrong)
    report = {
        "format": REPORT_FORMAT,
        "source_sha256": source_sha256(),
        "corpus_sha256": corpus_sha,
        "vae_source_sha256": vae.report["source_sha256"],
        "refined_decoder": True,
        "sources": list(sources),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "steps": training.steps,
        "initialization": initialization,
        "alignment": "current[t] + command/control[t+1] -> post-command frame[t+1]",
        "train_pairs": len(current),
        "heldout_pairs": len(held_current),
        "heldout_model_latent_mae": model_latent,
        "heldout_persistence_latent_mae": persistence_latent,
        "heldout_wrong_action_latent_mae": wrong_latent,
        "heldout_zero_control_latent_mae": zero_control_latent,
        "heldout_model_rgb_mae": model_rgb,
        "heldout_persistence_rgb_mae": persistence_rgb,
        "heldout_wrong_action_rgb_mae": wrong_rgb,
        "latent_improvement": 1 - model_latent / persistence_latent,
        "rgb_improvement": 1 - model_rgb / persistence_rgb,
        "correct_action_advantage": wrong_latent - model_latent,
        "correct_control_advantage": zero_control_latent - model_latent,
        "per_action": per_action,
        "history": history,
    }
    payload = {**recovery, "status": "evaluated", "report": report}
    torch.save(payload, output / "checkpoint.pt")
    runtime_payload = {**payload, "ema_state": {name: value.to(torch.bfloat16) if value.is_floating_point() else value for name, value in ema_cpu.items()}, "runtime_precision": "bfloat16"}
    torch.save(runtime_payload, output / "runtime.pt")
    (output / "report.json").write_bytes(canonical(report))
    _contact(output / "heldout_contact_sheet.png", heldout["current_frame"], heldout["target_frame"], predicted_frame, wrong_frame, action_values)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=7500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--pixel-weight", type=float, default=0.65)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.episodes, args.vae, args.refiner, resume_checkpoint=args.resume, training=TrainingConfig(steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate, pixel_weight=args.pixel_weight)), indent=2))


if __name__ == "__main__":
    main()
