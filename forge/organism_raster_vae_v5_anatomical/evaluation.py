from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor
import torch.nn.functional as F

from ..organism_raster_vae_v3.calibration import _canonical, _sha
from ..organism_raster_vae_v3.contract import RasterVAEV3Config
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FORMAT, MAX_APPENDAGES, MAX_JOINTS, MAX_TOKENS
from .dataset import AnatomicalGraphCorpus
from .model import AnatomicalGraphRasterVAE
from .training import _batch, source_sha256


VALIDATION_IDENTITIES = (5, 11, 17, 23, 29)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/consola.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _rgba(tensor: Tensor) -> Image.Image:
    value = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(value * 255).astype(np.uint8), "RGBA")


def _authority(attention: Tensor, owner: Tensor, start: int, stop: int) -> tuple[int, int]:
    target = owner[:, ::2, ::2].reshape(len(attention), -1)
    valid = target >= 0
    predicted = attention[:, :, start:stop].argmax(2) + start
    correct = ((predicted == target) & valid).sum()
    return int(correct), int(valid.sum())


@torch.inference_mode()
def evaluate(checkpoint: Path, destination: Path) -> Path:
    checkpoint = checkpoint.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100, planned_bytes=1024**3)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source_sha256():
        raise ValueError("anatomical evaluation checkpoint drifted")
    corpus = AnatomicalGraphCorpus()
    if payload["corpus_sha256"] != corpus.semantic_sha256:
        raise ValueError("anatomical evaluation corpus drifted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnatomicalGraphRasterVAE(RasterVAEV3Config(**payload["config"])).to(device)
    model.load_state_dict(payload["ema_state"], strict=True)
    model.eval()

    indices = [index for index, (identity, _) in enumerate(corpus.rows) if identity in VALIDATION_IDENTITIES]
    sums = {"alpha_iou": 0.0, "rgba_mae": 0.0, "appendage_recall": 0.0}
    authority = {name: [0, 0] for name in ("appendage", "joint", "organ")}
    per_family: dict[str, dict[str, float]] = {}
    frames: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
    for start in range(0, len(indices), 8):
        chosen = indices[start : start + 8]
        batch = _batch(corpus, chosen, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                batch["living"], batch["family"], batch["traits"], batch["phase"],
                batch["tokens"], batch["token_mask"], stochastic=False,
            )
        prediction = output.rgba.float()
        target_alpha = batch["rgba"][:, 3:] > .5
        predicted_alpha = prediction[:, 3:] > .5
        intersection = (target_alpha & predicted_alpha).flatten(1).sum(1).float()
        union = (target_alpha | predicted_alpha).flatten(1).sum(1).float().clamp_min(1)
        limb = batch["appendage_alpha"] > .5
        recall = (predicted_alpha & limb).flatten(1).sum(1).float() / limb.flatten(1).sum(1).float().clamp_min(1)
        maes = (prediction - batch["rgba"]).abs().flatten(1).mean(1)
        for local, row_index in enumerate(chosen):
            identity, phase = corpus.rows[row_index]
            family = int(batch["family"][local])
            key = str(family)
            record = per_family.setdefault(key, {"count": 0.0, "alpha_iou_sum": 0.0, "rgba_mae_sum": 0.0, "appendage_recall_sum": 0.0})
            record["count"] += 1
            record["alpha_iou_sum"] += float(intersection[local] / union[local])
            record["rgba_mae_sum"] += float(maes[local])
            record["appendage_recall_sum"] += float(recall[local])
            frames[(identity, phase)] = (batch["rgba"][local].cpu(), prediction[local].cpu())
        sums["alpha_iou"] += float((intersection / union).sum())
        sums["rgba_mae"] += float(maes.sum())
        sums["appendage_recall"] += float(recall.sum())
        ranges = {
            "appendage": (0, MAX_APPENDAGES),
            "joint": (MAX_APPENDAGES, MAX_APPENDAGES + MAX_JOINTS),
            "organ": (MAX_APPENDAGES + MAX_JOINTS, MAX_TOKENS),
        }
        for name in authority:
            correct, count = _authority(
                output.attention24, batch[f"{name}_owner"], *ranges[name]
            )
            authority[name][0] += correct
            authority[name][1] += count

    metrics = {key: round(value / len(indices), 9) for key, value in sums.items()}
    for name, (correct, count) in authority.items():
        metrics[f"{name}_owner_accuracy"] = round(correct / max(count, 1), 9)
    family_metrics = {}
    for family, record in per_family.items():
        count = record["count"]
        family_metrics[family] = {
            "alpha_iou": round(record["alpha_iou_sum"] / count, 9),
            "rgba_mae": round(record["rgba_mae_sum"] / count, 9),
            "appendage_recall": round(record["appendage_recall_sum"] / count, 9),
        }

    # Motion diagnostics use alpha-weighted centroid trajectories and exact
    # 16-phase loops for every held-out family.
    motion = {}
    for family, identity in enumerate(VALIDATION_IDENTITIES):
        target_centroids = []
        predicted_centroids = []
        for phase in range(16):
            target, predicted = frames[(identity, phase)]
            yy, xx = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
            for tensor, output_list in ((target, target_centroids), (predicted, predicted_centroids)):
                alpha = tensor[3].clamp_min(0)
                total = alpha.sum().clamp_min(1e-6)
                output_list.append((float((xx * alpha).sum() / total), float((yy * alpha).sum() / total)))
        target_points = np.asarray(target_centroids, dtype=np.float32)
        predicted_points = np.asarray(predicted_centroids, dtype=np.float32)
        target_travel = float(np.linalg.norm(np.diff(np.vstack((target_points, target_points[:1])), axis=0), axis=1).sum())
        predicted_travel = float(np.linalg.norm(np.diff(np.vstack((predicted_points, predicted_points[:1])), axis=0), axis=1).sum())
        motion[str(family)] = {
            "target_centroid_travel": round(target_travel, 7),
            "predicted_centroid_travel": round(predicted_travel, 7),
            "motion_ratio": round(predicted_travel / max(target_travel, 1e-6), 7),
            "centroid_trajectory_mae": round(float(np.abs(predicted_points - target_points).mean()), 7),
        }

    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        width = 5 * 210 + 24
        height = 16 * 210 + 86
        sheet = Image.new("RGBA", (width, height), (2, 7, 12, 255))
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 12), "ANATOMICAL GRAPH VAE // 5 FAMILIES x 16 MOTION PHASES", font=_font(18), fill=(223, 244, 247, 255))
        draw.text((16, 40), "LEFT TARGET / RIGHT NEURAL EMA", font=_font(11), fill=(74, 222, 240, 255))
        for column, identity in enumerate(VALIDATION_IDENTITIES):
            for phase in range(16):
                target, predicted = frames[(identity, phase)]
                x = 16 + column * 210
                y = 70 + phase * 210
                sheet.alpha_composite(_rgba(target), (x, y))
                sheet.alpha_composite(_rgba(predicted), (x + 100, y))
                draw.text((x, y + 98), f"F{column} P{phase:02d}", font=_font(9), fill=(126, 157, 168, 255))
        contact = staging / "heldout_motion_contact.png"
        sheet.save(contact, compress_level=7)

        gif_frames = []
        for phase in range(16):
            frame = Image.new("RGBA", (5 * 210 + 24, 250), (2, 7, 12, 255))
            canvas = ImageDraw.Draw(frame)
            canvas.text((16, 12), f"ANATOMICAL NEURAL LOOP // PHASE {phase:02d}", font=_font(16), fill=(224, 244, 248, 255))
            for column, identity in enumerate(VALIDATION_IDENTITIES):
                target, predicted = frames[(identity, phase)]
                x = 16 + column * 210
                frame.alpha_composite(_rgba(target), (x, 48))
                frame.alpha_composite(_rgba(predicted), (x + 100, 48))
                canvas.text((x, 154), f"F{column} TARGET / EMA", font=_font(9), fill=(89, 212, 229, 255))
            gif_frames.append(frame)
        animation = staging / "heldout_motion_loop.gif"
        gif_frames[0].save(animation, save_all=True, append_images=gif_frames[1:], duration=90, loop=0, disposal=2)

        report = {
            "format": FORMAT,
            "status": "human_review_required",
            "source_sha256": source_sha256(),
            "checkpoint": {"sha256": _sha(checkpoint), "segment": payload["segment"], "global_step": payload["global_step"]},
            "corpus_sha256": corpus.semantic_sha256,
            "heldout": {"identities": list(VALIDATION_IDENTITIES), "phases": 16, "samples": len(indices)},
            "metrics": metrics,
            "per_family": family_metrics,
            "motion": motion,
            "artifacts": {
                "contact": {"path": contact.name, "sha256": _sha(contact), "bytes": contact.stat().st_size},
                "animation": {"path": animation.name, "sha256": _sha(animation), "bytes": animation.stat().st_size},
            },
            "gates": {
                "all_families_present": len(family_metrics) == 5,
                "all_motion_phases_present": len(frames) == 80,
                "appendage_authority_learned": metrics["appendage_owner_accuracy"] > .25,
                "joint_authority_learned": metrics["joint_owner_accuracy"] > .20,
                "organ_authority_learned": metrics["organ_owner_accuracy"] > .20,
                "non_frozen_motion": all(value["motion_ratio"] > .25 for value in motion.values()),
                "production_promotion_allowed": False,
            },
            "claim_boundary": {
                "neural_rasterizer": True,
                "anatomical_authority": True,
                "full_game": False,
                "monolithic_action_world_model": False,
            },
        }
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        report_path = staging / "evaluation.json"
        report_path.write_bytes(_canonical(report))
        staging.replace(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination / "evaluation.json"
