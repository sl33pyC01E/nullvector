from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import BODY_SPEED_SCALE, POSITION_SCALE, VAE_AUTHORITY, canonical_json_bytes, sha256_file, source_sha256
from .dataset import GroundedMotionTeacher
from .raster import living_field_from_cells, load_frozen_vae, neural_raster
from .training import load_model


FORMAT = "nullvector-neural-grounded-cell-motion-evaluation-v1"
EVALUATION_SOURCE_FILES = (
    "forge/creature_stage_neural_grounded/evaluation.py",
    "forge/creature_stage_neural_grounded/raster.py",
)


def evaluation_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-grounded-evaluation-v1\0")
    for relative in EVALUATION_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _font(size: int):
    for path in (Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file(): return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _metrics(predicted: np.ndarray, target: np.ndarray, baseline: np.ndarray,
             mask: np.ndarray, appendage: np.ndarray, contact: np.ndarray,
             body_predicted: np.ndarray, body_target: np.ndarray) -> dict[str, float]:
    active = np.broadcast_to(mask[:, None, :, None], predicted[..., :1].shape)
    pos = np.abs((predicted[..., :2] - target[..., :2]) * POSITION_SCALE)
    base = np.abs((baseline[..., :2] - target[..., :2]) * POSITION_SCALE)
    pos_mae = float(pos[active.repeat(2, axis=3)].mean())
    baseline_mae = float(base[active.repeat(2, axis=3)].mean())
    velocity_mae = float(np.abs((predicted[..., 2:] - target[..., 2:]) * POSITION_SCALE)[active.repeat(2, axis=3)].mean())
    app = np.broadcast_to(appendage[:, None, :, None], active.shape) & active
    contact_active = contact[:, :, :, None] & active
    appendage_mae = float(pos[app.repeat(2, axis=3)].mean())
    contact_mae = float(pos[contact_active.repeat(2, axis=3)].mean()) if contact_active.any() else 0.0
    pred_energy = float(np.sqrt(np.mean(np.square(predicted[..., :2][active.repeat(2, axis=3)] * POSITION_SCALE))))
    target_energy = float(np.sqrt(np.mean(np.square(target[..., :2][active.repeat(2, axis=3)] * POSITION_SCALE))))
    # The authority stores 72 unique phase samples and intentionally omits a
    # duplicated phase-1 endpoint.  Measure error in the 71->0 transition,
    # not raw endpoint distance (fast feet legitimately move across it).
    predicted_seam = predicted[:, 0, :, :2] - predicted[:, -1, :, :2]
    target_seam = target[:, 0, :, :2] - target[:, -1, :, :2]
    seam = float(np.max(np.abs((predicted_seam - target_seam) * POSITION_SCALE)[mask]))
    body_velocity_mae = float(np.abs((body_predicted - body_target) * BODY_SPEED_SCALE).mean())
    predicted_distance = float((body_predicted * BODY_SPEED_SCALE).sum(1).mean())
    target_distance = float((body_target * BODY_SPEED_SCALE).sum(1).mean())
    return {
        "position_mae_px": round(pos_mae, 9), "copy_previous_mae_px": round(baseline_mae, 9),
        "baseline_improvement": round((baseline_mae - pos_mae) / max(baseline_mae, 1e-8), 9),
        "velocity_mae_px": round(velocity_mae, 9), "appendage_mae_px": round(appendage_mae, 9),
        "contact_mae_px": round(contact_mae, 9), "energy_ratio": round(pred_energy / max(target_energy, 1e-8), 9),
        "loop_transition_error_px": round(seam, 9), "body_velocity_mae_px": round(body_velocity_mae, 9),
        "predicted_distance_px": round(predicted_distance, 9), "target_distance_px": round(target_distance, 9),
        "advance_ratio": round(predicted_distance / max(target_distance, 1e-8), 9),
    }


def _render_review(teacher: GroundedMotionTeacher, predictions: np.ndarray,
                   vae_outputs: dict[tuple[int, int], np.ndarray], indices: tuple[int, ...]) -> bytes:
    tile_w, tile_h = 310, 190
    image = Image.new("RGB", (tile_w * 4, 54 + tile_h * len(indices)), (3, 8, 14))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((18, 12), "NEURAL GROUNDED MOTION // CELL AUTHORITY + FROZEN VAE RASTER", font=_font(20), fill=(214, 242, 248, 255))
    phases = (0, 18, 36, 54)
    colors = np.asarray(((238,104,118),(228,225,191),(250,69,108),(240,168,120),(91,145,171),(55,222,245),(224,44,93),(105,207,238),(239,158,49),(247,237,91),(151,86,219),(94,217,78),(176,68,255),(165,188,203),(255,117,49)), dtype=np.uint8)
    for row, identity in enumerate(indices):
        local = predictions[row]
        tissue = teacher.arrays["tissue"][identity]
        mask = teacher.arrays["cell_mask"][identity].astype(bool)
        for column, frame in enumerate(phases):
            x0, y0 = column * tile_w, 54 + row * tile_h
            draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), outline=(27, 68, 80, 255))
            draw.text((x0 + 8, y0 + 7), f"{teacher.manifest['cycles'][identity]['genome_id'].upper()} // F{frame:02d}", font=_font(10), fill=(111, 222, 241, 255))
            points = teacher.arrays["rest_cells"][identity, mask] + local[frame, mask, :2] * POSITION_SCALE
            for point, tissue_id in zip(points, tissue[mask], strict=True):
                px, py = x0 + 73 + float(point[0]) * 3.0, y0 + 97 + float(point[1]) * 3.0
                color = tuple(int(v) for v in colors[int(tissue_id)])
                draw.rectangle((px-1.3, py-1.3, px+1.3, py+1.3), fill=(*color, 230))
            rgba = vae_outputs[(identity, frame)]
            sprite = Image.fromarray(rgba, "RGBA").resize((144, 144), Image.Resampling.NEAREST)
            image.paste(sprite.convert("RGB"), (x0 + 156, y0 + 30))
            draw.text((x0+176, y0+169), "VAE DECODE", font=_font(9), fill=(197, 109, 255, 255))
    from io import BytesIO
    stream = BytesIO(); image.save(stream, format="PNG", compress_level=9); return stream.getvalue()


def evaluate(checkpoint: Path, output: Path, *, device: str = "cuda", ema: bool = True,
             visually_inspected: bool = False) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024**3)
    target_device = torch.device(device)
    teacher = GroundedMotionTeacher()
    model, checkpoint_payload = load_model(checkpoint, ema=ema, device=target_device)
    indices = teacher.split_indices("validation")
    predictions, targets, baselines, contacts, body_predictions, body_targets = [], [], [], [], [], []
    vae_outputs: dict[tuple[int, int], np.ndarray] = {}
    vae, vae_payload = load_frozen_vae(VAE_AUTHORITY, device=target_device)
    with torch.inference_mode():
        for identity in indices:
            first = teacher.sample(identity, 0)
            state = torch.from_numpy(first["state"].copy())[None].to(target_device)
            identity_pred, identity_target, identity_base, identity_contact = [], [], [], []
            body_pred, body_target = [], []
            for frame in range(72):
                row = teacher.sample(identity, frame)
                static = torch.from_numpy(row["static"].copy())[None].to(target_device)
                dynamic = torch.from_numpy(row["dynamic"].copy())[None].to(target_device)
                mask = torch.from_numpy(row["mask"].copy())[None].to(target_device)
                adjacency = torch.from_numpy(row["adjacency"].copy())[None].to(target_device)
                controls = torch.from_numpy(row["controls"].copy())[None].to(target_device)
                result = model(static, state, dynamic, mask, adjacency,
                    torch.tensor([row["family"]], device=target_device), torch.tensor([row["morphotype"]], device=target_device),
                    torch.tensor([2], device=target_device), torch.tensor([row["phase"]], device=target_device), controls)
                pred = result.cells[0].float().cpu().numpy(); target = row["target"]
                identity_pred.append(pred); identity_target.append(target); identity_base.append(state[0].float().cpu().numpy())
                identity_contact.append(row["dynamic"][:, 5] > .5); body_pred.append(float(result.body_velocity[0])); body_target.append(float(row["body_target"]))
                state = result.cells.detach()
                if frame in (0, 18, 36, 54):
                    count = row["cell_count"]
                    cells = teacher.arrays["rest_cells"][identity, :count] + pred[:count, :2] * POSITION_SCALE
                    condition = living_field_from_cells(cells, teacher.arrays["tissue"][identity, :count], teacher.arrays["trait_fields"][identity, :count], teacher.arrays["appendage_owner"][identity, :count], int(row["family"]))
                    decoded = neural_raster(vae, condition)
                    rgba = decoded.rgba[0].float().cpu().clamp(0, 1).numpy()
                    vae_outputs[(identity, frame)] = np.moveaxis(np.rint(rgba * 255).astype(np.uint8), 0, -1)
            predictions.append(np.stack(identity_pred)); targets.append(np.stack(identity_target)); baselines.append(np.stack(identity_base)); contacts.append(np.stack(identity_contact)); body_predictions.append(body_pred); body_targets.append(body_target)
    predicted_np, target_np, baseline_np = np.stack(predictions), np.stack(targets), np.stack(baselines)
    mask = teacher.arrays["cell_mask"][list(indices)].astype(bool)
    appendage = teacher.arrays["appendage_owner"][list(indices)] >= 0
    metrics = _metrics(predicted_np, target_np, baseline_np, mask, appendage, np.stack(contacts), np.asarray(body_predictions), np.asarray(body_targets))
    vae_iou = []
    for identity in indices:
        for frame in (0,18,36,54):
            decoded = vae_outputs[(identity, frame)][...,3] >= 128
            count = int(teacher.arrays["cell_mask"][identity].sum())
            points = np.rint(teacher.arrays["cells_local"][identity, frame, :count] + (24,23)).astype(int)
            target_mask = np.zeros((48,48), bool); valid=(points[:,0]>=0)&(points[:,0]<48)&(points[:,1]>=0)&(points[:,1]<48); target_mask[points[valid,1],points[valid,0]]=True
            vae_iou.append(float((decoded & target_mask).sum() / max(1, (decoded | target_mask).sum())))
    metrics["vae_transfer_silhouette_iou"] = round(float(np.mean(vae_iou)), 9)
    gates = {
        "grafted_matrix_all_five_families": len(indices) == 5,
        "finite": all(math.isfinite(value) for value in metrics.values()),
        "beats_copy_previous": metrics["baseline_improvement"] >= .10,
        "position_accuracy": metrics["position_mae_px"] <= .35,
        "velocity_accuracy": metrics["velocity_mae_px"] <= .20,
        "contact_anchor_accuracy": metrics["contact_mae_px"] <= .35,
        "appendage_accuracy": metrics["appendage_mae_px"] <= .45,
        "motion_energy": .70 <= metrics["energy_ratio"] <= 1.30,
        "loop_closure": metrics["loop_transition_error_px"] <= .35,
        "body_velocity_accuracy": metrics["body_velocity_mae_px"] <= .08,
        "grounded_advance": .70 <= metrics["advance_ratio"] <= 1.30,
        "frozen_vae_transfer": metrics["vae_transfer_silhouette_iou"] >= .45,
    }
    review = _render_review(teacher, predicted_np, vae_outputs, indices)
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    (stage / "heldout_grounded_vae_review.png").write_bytes(review)
    report = {
        "format": FORMAT, "status": "passed" if all(gates.values()) else "failed-quality",
        "source_sha256": source_sha256(), "evaluation_source_sha256": evaluation_source_sha256(),
        "teacher_semantic_sha256": teacher.semantic_sha256,
        "checkpoint": {"path": Path(checkpoint).resolve().relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(checkpoint), "updates": checkpoint_payload["updates"], "ema_state_sha256": checkpoint_payload["ema_state_sha256"]},
        "vae": {"path": VAE_AUTHORITY.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(VAE_AUTHORITY), "model_state_sha256": vae_payload["model_state_sha256"], "actual_neural_decode": True},
        "scope": {"split": "grafted_identity_rollout", "identities": list(indices), "families": 5, "frames": 360, "prediction_fed": True, "weights": "ema" if ema else "model"},
        "metrics": metrics, "gates": gates, "visually_inspected": bool(visually_inspected),
        "promotion_eligible": all(gates.values()) and bool(visually_inspected),
        "artifacts": {"review": {"path": "heldout_grounded_vae_review.png", "bytes": len(review), "sha256": hashlib.sha256(review).hexdigest()}},
    }
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    (stage / "evaluation_manifest.json").write_bytes(canonical_json_bytes(report)); os.replace(stage, output)
    return validate_evaluation(output)


def validate_evaluation(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); path = output / "evaluation_manifest.json"
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 2 * 1024**2:
        raise ValueError("grounded evaluation manifest missing or oversized")
    raw = path.read_bytes(); report = json.loads(raw)
    if raw != canonical_json_bytes(report): raise ValueError("grounded evaluation manifest is not canonical")
    semantic = report.pop("semantic_sha256")
    if semantic != hashlib.sha256(canonical_json_bytes(report)).hexdigest(): raise ValueError("grounded evaluation semantic hash drifted")
    report["semantic_sha256"] = semantic
    if report["format"] != FORMAT or report["source_sha256"] != source_sha256() or report["evaluation_source_sha256"] != evaluation_source_sha256():
        raise ValueError("grounded evaluation source provenance drifted")
    checkpoint = PROJECT_ROOT / report["checkpoint"]["path"]
    if sha256_file(checkpoint) != report["checkpoint"]["sha256"]: raise ValueError("grounded evaluation checkpoint drifted")
    if sha256_file(VAE_AUTHORITY) != report["vae"]["sha256"] or report["vae"]["actual_neural_decode"] is not True:
        raise ValueError("grounded evaluation VAE authority drifted")
    for artifact in report["artifacts"].values():
        artifact_path = output / artifact["path"]
        if artifact_path.stat().st_size != artifact["bytes"] or sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError("grounded evaluation artifact drifted")
    if report["status"] != ("passed" if all(report["gates"].values()) else "failed-quality"):
        raise ValueError("grounded evaluation verdict drifted")
    if report["promotion_eligible"] is not (all(report["gates"].values()) and report["visually_inspected"]):
        raise ValueError("grounded evaluation promotion relationship drifted")
    return report
