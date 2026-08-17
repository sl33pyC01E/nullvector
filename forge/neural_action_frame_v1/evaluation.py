from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F

from ..safety import require_disk_floor
from ..world_action_cellular_v7.corpus import load_encoded_corpus
from .contract import CODEC_CHECKPOINT_SHA256, CORPUS, DEFAULT_OUTPUT, REPORT_FORMAT, canonical, source_sha256
from .runtime import NeuralActionFrameRuntime


def _predict(runtime, episode, *, wrong=False, alpha=1.0, batch=8):
    rows = []
    runtime.residual_alpha = alpha
    for start in range(0, len(episode["current"]), batch):
        end = start + batch
        action = episode["action"][start:end]
        if wrong:
            action = (action + 7) % 22
        frame, _ = runtime.step(
            episode["current_frame"][start:end], episode["current"][start:end], episode["previous"][start:end],
            action=action, control=episode["control"][start:end], state=episode["state"][start:end], actor_state=episode["actor_state"][start:end],
        )
        rows.append(frame)
    return torch.cat(rows)


def _contact(path, current, target, prediction):
    count = min(8, len(target))
    sheet = Image.new("RGB", (count * 256, 3 * 256))
    for index in range(count):
        for row, values in enumerate((current, target, prediction)):
            image = np.clip(values[index].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
            sheet.paste(Image.fromarray(image), (index * 256, row * 256))
    sheet.save(path)


def _metric(prediction, target):
    return float(F.l1_loss(prediction, target))


def evaluate(output: Path = DEFAULT_OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=64 * 1024**2)
    torch.set_num_threads(4)
    runtime = NeuralActionFrameRuntime.from_release(device="cuda")
    episodes, manifest = load_encoded_corpus(CORPUS)
    validation, test = episodes[4], episodes[5]
    validation_target = torch.from_numpy(validation["target_frame"]).permute(0, 3, 1, 2).float().div_(255)
    best = None
    for alpha in (0.25, 0.5, 0.75, 1.0, 1.25):
        prediction = _predict(runtime, validation, alpha=alpha)
        mae = _metric(prediction, validation_target)
        candidate = (mae, alpha)
        if best is None or candidate < best:
            best = candidate
    alpha = best[1]
    target = torch.from_numpy(test["target_frame"]).permute(0, 3, 1, 2).float().div_(255)
    current = torch.from_numpy(test["current_frame"]).permute(0, 3, 1, 2).float().div_(255)
    prediction = _predict(runtime, test, alpha=alpha)
    wrong = _predict(runtime, test, wrong=True, alpha=alpha)
    persistence_mae = _metric(current, target)
    mae = _metric(prediction, target)
    wrong_mae = _metric(wrong, target)
    metrics = {"mae": mae, "persistence_mae": persistence_mae, "improvement": 1 - mae / persistence_mae, "wrong_action_mae": wrong_mae, "correct_action_advantage": wrong_mae - mae}
    gates = {"beats_frame_persistence": metrics["improvement"] > 0, "correct_action_advantage": metrics["correct_action_advantage"] > 0}
    gates["all_passed"] = all(gates.values())
    report = {"format": REPORT_FORMAT, "status": "ready" if gates["all_passed"] else "experimental", "source_sha256": source_sha256(), "corpus_sha256": manifest["manifest_sha256"], "parents": {"action": runtime.action.report["checkpoint"]["sha256"], "codec": CODEC_CHECKPOINT_SHA256}, "selection": {"residual_alpha": alpha, "validation_mae": best[0]}, "test": metrics, "gates": gates}
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    (output / "report.json").write_bytes(canonical(report))
    _contact(output / "test_contact_sheet.png", current, target, prediction)
    return report
