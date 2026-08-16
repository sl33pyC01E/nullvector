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
from ..world_frame_vae_refiner import RefinedWorldFrameVAERuntime
from .contract import CHECKPOINT_FORMAT, REPORT_FORMAT, ModelConfig, TrainingConfig, canonical, config_dict, source_sha256
from .data import encode_spatial_episodes
from .model import SpatialActionDiT
from .runtime import SpatialWorldActionRuntime


def _state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0" + value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _concat(episodes, name):
    return np.concatenate([episode[name] for episode in episodes])


def _latent_change_mask(episodes) -> np.ndarray:
    rows = []
    for episode in episodes:
        delta = np.abs(episode["target_frame"].astype(np.int16) - episode["current_frame"].astype(np.int16)).mean(3).astype(np.float32) / 255
        tensor = torch.from_numpy(delta[:, None])
        pooled = F.adaptive_avg_pool2d(tensor, (32, 32))
        rows.append(F.max_pool2d(pooled, 3, 1, 1).numpy().astype(np.float16))
    return np.concatenate(rows)


def _decode(refined, latent, device):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(latent), 8):
            decoded = refined.refiner(refined.model.decode(latent[start : start + 8].to(device)))
            rows.append(decoded.float().cpu())
    return torch.cat(rows)


def _masked_rgb_mae(predicted, target, mask):
    error = (predicted - target).abs().mean(1)
    weight = mask.float()
    return float((error * weight).sum() / weight.sum().clamp_min(1))


def _per_action(actions, target, current, predicted, wrong_action, wrong_control):
    rows = []
    for index, name in enumerate(ACTIONS):
        mask = torch.from_numpy(actions == index)
        if not bool(mask.any()):
            continue
        persistence = float(F.l1_loss(current[mask], target[mask]))
        correct = float(F.l1_loss(predicted[mask], target[mask]))
        wrong_a = float(F.l1_loss(wrong_action[mask], target[mask]))
        wrong_c = float(F.l1_loss(wrong_control[mask], target[mask]))
        rows.append({"action": name, "count": int(mask.sum()), "model_latent_mae": correct, "persistence_latent_mae": persistence, "wrong_action_latent_mae": wrong_a, "wrong_control_latent_mae": wrong_c, "improvement": 1 - correct / persistence if persistence else 0.0})
    return rows


def _contact(path, current, target, predicted, wrong_action, wrong_control, actions):
    action_indices = np.flatnonzero(actions != 0)
    indices = action_indices[:8] if len(action_indices) >= 8 else np.arange(min(8, len(actions)))
    sheet = Image.new("RGB", (256 * len(indices), 256 * 5), (3, 7, 10))
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate(indices):
        for row, value in enumerate((current[index], target[index], predicted[index], wrong_action[index], wrong_control[index])):
            sheet.paste(Image.fromarray(value), (column * 256, row * 256))
        draw.text((column * 256 + 4, 4), ACTIONS[int(actions[index])], fill=(72, 245, 255))
    sheet.save(path)


def train(output: Path, episodes, vae_checkpoint: Path, refiner_checkpoint: Path, *, config=ModelConfig(), training=TrainingConfig()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(training.seed)
    np.random.seed(training.seed & 0xFFFFFFFF)
    random.seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refined = RefinedWorldFrameVAERuntime.from_checkpoints(vae_checkpoint, refiner_checkpoint, device=str(device))
    refined.model.requires_grad_(False)
    refined.refiner.requires_grad_(False)
    encoded, sources, corpus_sha = encode_spatial_episodes(episodes, refined)
    train_episodes, heldout = encoded[:-1], encoded[-1]
    current_raw = _concat(train_episodes, "current")
    target_raw = _concat(train_episodes, "target")
    control = _concat(train_episodes, "control")
    action = _concat(train_episodes, "action").astype(np.int64)
    state = _concat(train_episodes, "state")
    current_frame = _concat(train_episodes, "current_frame")
    target_frame = _concat(train_episodes, "target_frame")
    change_mask = _latent_change_mask(train_episodes)
    mean = np.mean(np.concatenate((current_raw, target_raw)), axis=(0, 2, 3))
    std = np.std(np.concatenate((current_raw, target_raw)), axis=(0, 2, 3)) + 1e-4
    current = (current_raw - mean[None, :, None, None]) / std[None, :, None, None]
    target = (target_raw - mean[None, :, None, None]) / std[None, :, None, None]
    counts = np.bincount(action, minlength=len(ACTIONS)).clip(1)
    action_weight = np.sqrt(counts.max() / counts).astype(np.float32)
    action_weight /= action_weight.mean()
    model = SpatialActionDiT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=2e-3, fused=device.type == "cuda")
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    mean_tensor = torch.from_numpy(mean).to(device).view(1, -1, 1, 1)
    std_tensor = torch.from_numpy(std).to(device).view(1, -1, 1, 1)
    rng = np.random.default_rng(training.seed)
    history = []
    for step in range(1, training.steps + 1):
        indices = rng.integers(0, len(current), training.batch_size)
        x0 = torch.from_numpy(current[indices]).to(device)
        x1 = torch.from_numpy(target[indices]).to(device)
        desired = x1 - x0
        noisy = x0 + torch.randn_like(x0) * training.input_noise
        act = torch.from_numpy(action[indices]).to(device)
        ctl = torch.from_numpy(control[indices]).to(device)
        st = torch.from_numpy(state[indices]).to(device)
        mask = torch.from_numpy(change_mask[indices].astype(np.float32)).to(device)
        sample_weight = torch.from_numpy(action_weight[action[indices]]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(noisy, torch.zeros(len(x0), device=device), act, ctl, st)
            magnitude = desired.detach().abs().mean(1, keepdim=True)
            weight = (1 + training.changed_weight * torch.clamp(magnitude / 0.25, 0, 2) + training.spatial_weight * mask) * sample_weight[:, None, None, None]
            residual = (F.smooth_l1_loss(prediction, desired, reduction="none") * weight).mean()
            edge = F.l1_loss(prediction[:, :, :, 1:] - prediction[:, :, :, :-1], desired[:, :, :, 1:] - desired[:, :, :, :-1]) + F.l1_loss(prediction[:, :, 1:] - prediction[:, :, :-1], desired[:, :, 1:] - desired[:, :, :-1])
            pixel_count = min(training.pixel_batch, len(indices))
            predicted_latent = (x0[:pixel_count] + prediction[:pixel_count]) * std_tensor + mean_tensor
            decoded = refined.refiner(refined.model.decode(predicted_latent))
            target_pixel = torch.from_numpy(target_frame[indices[:pixel_count]]).to(device).permute(0, 3, 1, 2).float().div_(255)
            pixel_delta = torch.from_numpy(np.abs(target_frame[indices[:pixel_count]].astype(np.int16) - current_frame[indices[:pixel_count]].astype(np.int16)).mean(3).astype(np.float32) / 255).to(device)[:, None]
            pixel_weight = 1 + training.pixel_changed_weight * F.max_pool2d(pixel_delta, 9, 1, 4)
            pixel = (F.l1_loss(decoded, target_pixel, reduction="none") * pixel_weight).mean()
            contrast_count = min(training.contrastive_batch, len(indices))
            wrong_action = (act[:contrast_count] + 7) % len(ACTIONS)
            wrong_control = -ctl[:contrast_count]
            wrong = model(noisy[:contrast_count], torch.zeros(contrast_count, device=device), wrong_action, wrong_control, st[:contrast_count])
            correct_error = ((prediction[:contrast_count] - desired[:contrast_count]).abs() * weight[:contrast_count]).mean((1, 2, 3))
            wrong_error = ((wrong - desired[:contrast_count]).abs() * weight[:contrast_count]).mean((1, 2, 3))
            contrastive = F.relu(training.contrastive_margin + correct_error - wrong_error).mean()
            loss = residual + training.edge_weight * edge + training.pixel_weight * pixel + training.contrastive_weight * contrastive
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].lerp_(value.detach(), 1 - training.ema_decay)
        if step == 1 or step % 250 == 0 or step == training.steps:
            row = {"step": step, "loss": round(float(loss), 6), "residual": round(float(residual), 6), "edge": round(float(edge), 6), "pixel": round(float(pixel), 6), "contrastive": round(float(contrastive), 6)}
            history.append(row)
            print(json.dumps(row), flush=True)
    ema_cpu = {name: value.detach().cpu() for name, value in ema.items()}
    recovery = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "model_config": config_dict(config), "training_config": config_dict(training), "latent_mean": mean.tolist(), "latent_std": std.tolist(), "ema_state": ema_cpu, "ema_sha256": _state_hash(ema_cpu), "history": history, "status": "trained_pending_evaluation"}
    torch.save(recovery, output / "trained_pending_evaluation.pt")
    model.load_state_dict(ema_cpu)
    model.to(device).eval()
    runtime = SpatialWorldActionRuntime(model, device, {}, mean_tensor, std_tensor)
    held_current = torch.from_numpy(heldout["current"])
    held_target = torch.from_numpy(heldout["target"])
    held_actions = heldout["action"].astype(np.int64)
    held_control = heldout["control"]
    wrong_actions = (held_actions + 7) % len(ACTIONS)
    wrong_controls = -held_control
    predicted, wrong_a, wrong_c = [], [], []
    for start in range(0, len(held_current), 16):
        stop = start + 16
        value = held_current[start:stop]
        state_value = heldout["state"][start:stop]
        predicted.append(runtime.predict_latent(value, action=held_actions[start:stop], control=held_control[start:stop], state=state_value).cpu())
        wrong_a.append(runtime.predict_latent(value, action=wrong_actions[start:stop], control=held_control[start:stop], state=state_value).cpu())
        wrong_c.append(runtime.predict_latent(value, action=held_actions[start:stop], control=wrong_controls[start:stop], state=state_value).cpu())
    predicted = torch.cat(predicted)
    wrong_a = torch.cat(wrong_a)
    wrong_c = torch.cat(wrong_c)
    model_latent = float(F.l1_loss(predicted, held_target))
    persistence_latent = float(F.l1_loss(held_current, held_target))
    wrong_action_latent = float(F.l1_loss(wrong_a, held_target))
    wrong_control_latent = float(F.l1_loss(wrong_c, held_target))
    predicted_rgb = _decode(refined, predicted, device)
    wrong_action_rgb = _decode(refined, wrong_a, device)
    wrong_control_rgb = _decode(refined, wrong_c, device)
    refined_current_rgb = _decode(refined, held_current, device)
    target_rgb = torch.from_numpy(heldout["target_frame"]).permute(0, 3, 1, 2).float().div_(255)
    raw_current_rgb = torch.from_numpy(heldout["current_frame"]).permute(0, 3, 1, 2).float().div_(255)
    raw_persistence_rgb = float(F.l1_loss(raw_current_rgb, target_rgb))
    refined_persistence_rgb = float(F.l1_loss(refined_current_rgb, target_rgb))
    model_rgb = float(F.l1_loss(predicted_rgb, target_rgb))
    wrong_action_rgb_mae = float(F.l1_loss(wrong_action_rgb, target_rgb))
    wrong_control_rgb_mae = float(F.l1_loss(wrong_control_rgb, target_rgb))
    changed = (target_rgb - raw_current_rgb).abs().mean(1, keepdim=True)
    changed = F.max_pool2d((changed > 0.025).float(), 9, 1, 4).squeeze(1) > 0
    model_changed_rgb = _masked_rgb_mae(predicted_rgb, target_rgb, changed)
    refined_persistence_changed_rgb = _masked_rgb_mae(refined_current_rgb, target_rgb, changed)
    gates = {
        "latent_beats_persistence": model_latent < persistence_latent,
        "correct_beats_wrong_action": model_latent < wrong_action_latent,
        "correct_beats_wrong_control": model_latent < wrong_control_latent,
        "refined_rgb_beats_refined_persistence": model_rgb < refined_persistence_rgb,
        "raw_rgb_beats_raw_persistence": model_rgb < raw_persistence_rgb,
        "changed_rgb_beats_refined_persistence": model_changed_rgb < refined_persistence_changed_rgb,
    }
    gates["all_passed"] = all(gates.values())
    report = {"format": REPORT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "decoder_source_sha256": refined.report["source_sha256"], "sources": list(sources), "parameters": sum(parameter.numel() for parameter in model.parameters()), "device": str(device), "steps": training.steps, "alignment": "centered world[t] + spatial command/control[t+1] -> post-command world[t+1]", "train_pairs": len(current), "heldout_pairs": len(held_current), "heldout_model_latent_mae": model_latent, "heldout_persistence_latent_mae": persistence_latent, "heldout_wrong_action_latent_mae": wrong_action_latent, "heldout_wrong_control_latent_mae": wrong_control_latent, "heldout_model_rgb_mae": model_rgb, "heldout_raw_persistence_rgb_mae": raw_persistence_rgb, "heldout_refined_persistence_rgb_mae": refined_persistence_rgb, "heldout_wrong_action_rgb_mae": wrong_action_rgb_mae, "heldout_wrong_control_rgb_mae": wrong_control_rgb_mae, "heldout_changed_model_rgb_mae": model_changed_rgb, "heldout_changed_refined_persistence_rgb_mae": refined_persistence_changed_rgb, "latent_improvement": 1 - model_latent / persistence_latent, "raw_rgb_improvement": 1 - model_rgb / raw_persistence_rgb, "correct_action_advantage": wrong_action_latent - model_latent, "correct_control_advantage": wrong_control_latent - model_latent, "per_action": _per_action(held_actions, held_target, held_current, predicted, wrong_a, wrong_c), "gates": gates, "history": history}
    payload = {**recovery, "status": "evaluated", "report": report}
    torch.save(payload, output / "checkpoint.pt")
    torch.save({**payload, "ema_state": {name: value.to(torch.bfloat16) if value.is_floating_point() else value for name, value in ema_cpu.items()}, "runtime_precision": "bfloat16"}, output / "runtime.pt")
    (output / "report.json").write_bytes(canonical(report))
    predicted_frame = np.clip(predicted_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
    wrong_action_frame = np.clip(wrong_action_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
    wrong_control_frame = np.clip(wrong_control_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
    _contact(output / "heldout_contact_sheet.png", heldout["current_frame"], heldout["target_frame"], predicted_frame, wrong_action_frame, wrong_control_frame, held_actions)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=7500)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.episodes, args.vae, args.refiner, training=TrainingConfig(steps=args.steps, batch_size=args.batch_size)), indent=2))


if __name__ == "__main__":
    main()
