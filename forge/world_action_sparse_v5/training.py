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
from ..world_action_spatial_v4.data import encode_spatial_episodes
from ..world_frame_vae_refiner import RefinedWorldFrameVAERuntime
from .contract import CHECKPOINT_FORMAT, REPORT_FORMAT, ModelConfig, TrainingConfig, canonical, config_dict, source_sha256
from .model import SparseActionDiT
from .runtime import SparseWorldActionRuntime

TARGETED_ACTIONS = frozenset(("impact", "scrape", "cut", "beam", "projectile"))
TARGETED_INDICES = np.asarray([ACTIONS.index(name) for name in TARGETED_ACTIONS], dtype=np.int64)


def _state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0" + value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _concat(episodes, name):
    return np.concatenate([episode[name] for episode in episodes])


def latent_edit_mask(current_frame: np.ndarray, target_frame: np.ndarray) -> np.ndarray:
    delta = np.abs(target_frame.astype(np.int16) - current_frame.astype(np.int16)).mean(3).astype(np.float32) / 255
    changed = torch.from_numpy((delta > 0.025).astype(np.float32))[:, None]
    pooled = F.adaptive_max_pool2d(changed, (32, 32))
    return F.max_pool2d(pooled, 3, 1, 1).numpy().astype(np.float16)


def counterfactual_control(control: np.ndarray) -> np.ndarray:
    wrong = np.asarray(control, dtype=np.float32).copy()
    x = wrong[:, 2].copy()
    y = wrong[:, 3].copy()
    wrong[:, 2] = -y
    wrong[:, 3] = x
    near = np.linalg.norm(wrong[:, 2:4], axis=1) < 0.08
    wrong[near, 2] = 0.7
    wrong[near, 3] = -0.55
    wrong[:, :2] *= -1
    return wrong


def _decode(refined, latent, device):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(latent), 8):
            rows.append(refined.refiner(refined.model.decode(latent[start : start + 8].to(device))).float().cpu())
    return torch.cat(rows)


def _masked_rgb_mae(predicted, target, mask):
    error = (predicted - target).abs().mean(1)
    weight = mask.float()
    return float((error * weight).sum() / weight.sum().clamp_min(1))


def _masked_latent_mae(predicted, target, mask):
    error = (predicted - target).abs().mean(1)
    weight = mask.float().squeeze(1)
    return float((error * weight).sum() / weight.sum().clamp_min(1))


def _contact(path, current, target, predicted, wrong_action, wrong_control, gate, actions):
    action_indices = np.flatnonzero(actions != 0)
    indices = action_indices[:8] if len(action_indices) >= 8 else np.arange(min(8, len(actions)))
    sheet = Image.new("RGB", (256 * len(indices), 256 * 6), (3, 7, 10))
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate(indices):
        values = (current[index], target[index], predicted[index], wrong_action[index], wrong_control[index])
        for row, value in enumerate(values):
            sheet.paste(Image.fromarray(value), (column * 256, row * 256))
        mask = np.clip(gate[index] * 255, 0, 255).astype(np.uint8)
        mask = np.asarray(Image.fromarray(mask).resize((256, 256), Image.Resampling.NEAREST))
        sheet.paste(Image.fromarray(np.stack((mask, np.zeros_like(mask), 255 - mask), 2)), (column * 256, 5 * 256))
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
    edit_mask = latent_edit_mask(current_frame, target_frame)
    mean = np.mean(np.concatenate((current_raw, target_raw)), axis=(0, 2, 3))
    std = np.std(np.concatenate((current_raw, target_raw)), axis=(0, 2, 3)) + 1e-4
    current = (current_raw - mean[None, :, None, None]) / std[None, :, None, None]
    target = (target_raw - mean[None, :, None, None]) / std[None, :, None, None]
    counts = np.bincount(action, minlength=len(ACTIONS)).clip(1)
    action_weight = np.sqrt(counts.max() / counts).astype(np.float32)
    action_weight /= action_weight.mean()
    model = SparseActionDiT(config).to(device)
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
        mask = torch.from_numpy(edit_mask[indices].astype(np.float32)).to(device)
        sample_weight = torch.from_numpy(action_weight[action[indices]]).to(device)[:, None, None, None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            edited, gate, delta, gate_logits = model.edit(noisy, torch.zeros(len(x0), device=device), act, ctl, st)
            weight = (1 + training.changed_weight * mask) * sample_weight
            residual = (F.smooth_l1_loss(edited, x1, reduction="none") * weight).mean()
            delta_error = F.smooth_l1_loss(delta, desired, reduction="none").mean(1, keepdim=True)
            delta_loss = (delta_error * mask).sum() / (mask.sum().clamp_min(1))
            gate_bce = F.binary_cross_entropy_with_logits(gate_logits.float(), mask, pos_weight=torch.tensor(training.gate_positive_weight, device=device))
            intersection = (gate * mask).sum((1, 2, 3))
            dice = 1 - ((2 * intersection + 1) / (gate.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)).mean()
            gate_loss = gate_bce + dice
            leakage = (gate * (1 - mask)).mean()
            edge = F.l1_loss((edited - x0)[:, :, :, 1:] - (edited - x0)[:, :, :, :-1], desired[:, :, :, 1:] - desired[:, :, :, :-1]) + F.l1_loss((edited - x0)[:, :, 1:] - (edited - x0)[:, :, :-1], desired[:, :, 1:] - desired[:, :, :-1])
            pixel_count = min(training.pixel_batch, len(indices))
            predicted_latent = edited[:pixel_count] * std_tensor + mean_tensor
            decoded = refined.refiner(refined.model.decode(predicted_latent))
            target_pixel = torch.from_numpy(target_frame[indices[:pixel_count]]).to(device).permute(0, 3, 1, 2).float().div_(255)
            pixel_delta = torch.from_numpy(np.abs(target_frame[indices[:pixel_count]].astype(np.int16) - current_frame[indices[:pixel_count]].astype(np.int16)).mean(3).astype(np.float32) / 255).to(device)[:, None]
            pixel_weight = 1 + training.pixel_changed_weight * F.max_pool2d((pixel_delta > 0.025).float(), 9, 1, 4)
            pixel = (F.l1_loss(decoded, target_pixel, reduction="none") * pixel_weight).mean()
            contrast_count = min(training.contrastive_batch, len(indices))
            wrong_action = (act[:contrast_count] + 7) % len(ACTIONS)
            wrong_ctl = torch.from_numpy(counterfactual_control(control[indices[:contrast_count]])).to(device)
            wrong_a = model.edit(noisy[:contrast_count], torch.zeros(contrast_count, device=device), wrong_action, ctl[:contrast_count], st[:contrast_count])[0]
            wrong_c = model.edit(noisy[:contrast_count], torch.zeros(contrast_count, device=device), act[:contrast_count], wrong_ctl, st[:contrast_count])[0]
            correct_error = ((edited[:contrast_count] - x1[:contrast_count]).abs() * weight[:contrast_count]).mean((1, 2, 3))
            wrong_a_error = ((wrong_a - x1[:contrast_count]).abs() * weight[:contrast_count]).mean((1, 2, 3))
            targeted = torch.isin(act[:contrast_count], torch.as_tensor(TARGETED_INDICES, device=device))
            wrong_c_error = ((wrong_c - x1[:contrast_count]).abs() * weight[:contrast_count]).mean((1, 2, 3))
            action_contrast = F.relu(training.contrastive_margin + correct_error - wrong_a_error).mean()
            control_contrast = F.relu(training.contrastive_margin + correct_error[targeted] - wrong_c_error[targeted]).mean() if bool(targeted.any()) else correct_error.new_zeros(())
            contrastive = action_contrast + control_contrast
            loss = residual + training.delta_weight * delta_loss + training.gate_weight * gate_loss + training.leakage_weight * leakage + training.edge_weight * edge + training.pixel_weight * pixel + training.contrastive_weight * contrastive
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].lerp_(value.detach(), 1 - training.ema_decay)
        if step == 1 or step % 250 == 0 or step == training.steps:
            row = {"step": step, "loss": round(float(loss), 6), "residual": round(float(residual), 6), "delta": round(float(delta_loss), 6), "gate": round(float(gate_loss), 6), "gate_mean": round(float(gate.mean()), 6), "leakage": round(float(leakage), 6), "pixel": round(float(pixel), 6), "contrastive": round(float(contrastive), 6)}
            history.append(row)
            print(json.dumps(row), flush=True)
    ema_cpu = {name: value.detach().cpu() for name, value in ema.items()}
    recovery = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "model_config": config_dict(config), "training_config": config_dict(training), "latent_mean": mean.tolist(), "latent_std": std.tolist(), "ema_state": ema_cpu, "ema_sha256": _state_hash(ema_cpu), "history": history, "status": "trained_pending_evaluation"}
    torch.save(recovery, output / "trained_pending_evaluation.pt")
    model.load_state_dict(ema_cpu)
    model.to(device).eval()
    runtime = SparseWorldActionRuntime(model, device, {}, mean_tensor, std_tensor)
    held_current = torch.from_numpy(heldout["current"])
    held_target = torch.from_numpy(heldout["target"])
    held_actions = heldout["action"].astype(np.int64)
    held_control = heldout["control"]
    wrong_actions = (held_actions + 7) % len(ACTIONS)
    wrong_controls = counterfactual_control(held_control)
    predicted, wrong_a, wrong_c, gates_out = [], [], [], []
    for start in range(0, len(held_current), 16):
        stop = start + 16
        value = held_current[start:stop]
        state_value = heldout["state"][start:stop]
        result, gate = runtime.predict_latent(value, action=held_actions[start:stop], control=held_control[start:stop], state=state_value, return_gate=True)
        predicted.append(result.cpu()); gates_out.append(gate.cpu())
        wrong_a.append(runtime.predict_latent(value, action=wrong_actions[start:stop], control=held_control[start:stop], state=state_value).cpu())
        wrong_c.append(runtime.predict_latent(value, action=held_actions[start:stop], control=wrong_controls[start:stop], state=state_value).cpu())
    predicted = torch.cat(predicted); wrong_a = torch.cat(wrong_a); wrong_c = torch.cat(wrong_c); gates_out = torch.cat(gates_out)
    held_mask = torch.from_numpy(latent_edit_mask(heldout["current_frame"], heldout["target_frame"]).astype(np.float32))
    model_latent = float(F.l1_loss(predicted, held_target)); persistence_latent = float(F.l1_loss(held_current, held_target))
    changed_latent = _masked_latent_mae(predicted, held_target, held_mask); changed_persistence_latent = _masked_latent_mae(held_current, held_target, held_mask)
    wrong_action_latent = float(F.l1_loss(wrong_a, held_target)); wrong_control_latent = float(F.l1_loss(wrong_c, held_target))
    targeted_mask = torch.from_numpy(np.isin(held_actions, TARGETED_INDICES))
    targeted_correct = float(F.l1_loss(predicted[targeted_mask], held_target[targeted_mask])); targeted_wrong_control = float(F.l1_loss(wrong_c[targeted_mask], held_target[targeted_mask]))
    predicted_rgb = _decode(refined, predicted, device); wrong_action_rgb = _decode(refined, wrong_a, device); wrong_control_rgb = _decode(refined, wrong_c, device); refined_current_rgb = _decode(refined, held_current, device)
    target_rgb = torch.from_numpy(heldout["target_frame"]).permute(0, 3, 1, 2).float().div_(255); raw_current_rgb = torch.from_numpy(heldout["current_frame"]).permute(0, 3, 1, 2).float().div_(255)
    raw_persistence_rgb = float(F.l1_loss(raw_current_rgb, target_rgb)); refined_persistence_rgb = float(F.l1_loss(refined_current_rgb, target_rgb)); model_rgb = float(F.l1_loss(predicted_rgb, target_rgb))
    changed = F.max_pool2d(((target_rgb - raw_current_rgb).abs().mean(1, keepdim=True) > 0.025).float(), 9, 1, 4).squeeze(1) > 0
    model_changed_rgb = _masked_rgb_mae(predicted_rgb, target_rgb, changed); refined_persistence_changed_rgb = _masked_rgb_mae(refined_current_rgb, target_rgb, changed)
    gate_binary = gates_out > 0.5; mask_binary = held_mask > 0.5
    gate_iou = float((gate_binary & mask_binary).sum() / (gate_binary | mask_binary).sum().clamp_min(1))
    gates = {
        "latent_beats_persistence": model_latent < persistence_latent,
        "changed_latent_beats_persistence": changed_latent < changed_persistence_latent,
        "correct_beats_wrong_action": model_latent < wrong_action_latent,
        "targeted_control_beats_wrong_control": targeted_correct < targeted_wrong_control,
        "refined_rgb_beats_refined_persistence": model_rgb < refined_persistence_rgb,
        "changed_rgb_beats_refined_persistence": model_changed_rgb < refined_persistence_changed_rgb,
        "edit_gate_iou_over_0_20": gate_iou > 0.20,
    }
    gates["all_passed"] = all(gates.values())
    report = {
        "format": REPORT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "decoder_source_sha256": refined.report["source_sha256"], "sources": list(sources), "parameters": sum(parameter.numel() for parameter in model.parameters()), "device": str(device), "steps": training.steps, "alignment": "copy-preserving centered world[t] + spatial command/control[t+1] -> sparse post-command edit", "train_pairs": len(current), "heldout_pairs": len(held_current),
        "heldout_model_latent_mae": model_latent, "heldout_persistence_latent_mae": persistence_latent, "heldout_changed_model_latent_mae": changed_latent, "heldout_changed_persistence_latent_mae": changed_persistence_latent, "heldout_wrong_action_latent_mae": wrong_action_latent, "heldout_wrong_control_latent_mae": wrong_control_latent, "heldout_targeted_model_latent_mae": targeted_correct, "heldout_targeted_wrong_control_latent_mae": targeted_wrong_control,
        "heldout_model_rgb_mae": model_rgb, "heldout_raw_persistence_rgb_mae": raw_persistence_rgb, "heldout_refined_persistence_rgb_mae": refined_persistence_rgb, "heldout_changed_model_rgb_mae": model_changed_rgb, "heldout_changed_refined_persistence_rgb_mae": refined_persistence_changed_rgb, "edit_gate_iou": gate_iou, "edit_gate_mean": float(gates_out.mean()), "target_edit_mask_mean": float(held_mask.mean()), "latent_improvement": 1 - model_latent / persistence_latent, "changed_latent_improvement": 1 - changed_latent / changed_persistence_latent, "correct_action_advantage": wrong_action_latent - model_latent, "targeted_control_advantage": targeted_wrong_control - targeted_correct, "raw_rgb_diagnostic_improvement": 1 - model_rgb / raw_persistence_rgb, "gates": gates, "history": history,
    }
    payload = {**recovery, "status": "evaluated", "report": report}
    torch.save(payload, output / "checkpoint.pt")
    torch.save({**payload, "ema_state": {name: value.to(torch.bfloat16) if value.is_floating_point() else value for name, value in ema_cpu.items()}, "runtime_precision": "bfloat16"}, output / "runtime.pt")
    (output / "report.json").write_bytes(canonical(report))
    predicted_frame = np.clip(predicted_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8); wrong_action_frame = np.clip(wrong_action_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8); wrong_control_frame = np.clip(wrong_control_rgb.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
    _contact(output / "heldout_contact_sheet.png", heldout["current_frame"], heldout["target_frame"], predicted_frame, wrong_action_frame, wrong_control_frame, gates_out[:, 0].numpy(), held_actions)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.episodes, args.vae, args.refiner, training=TrainingConfig(steps=args.steps, batch_size=args.batch_size)), indent=2))


if __name__ == "__main__":
    main()
