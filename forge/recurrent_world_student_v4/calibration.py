from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..recurrent_world_student_v3.model import RecurrentWorldStudent
from ..world_action_clean_v9 import load
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from ..world_latent_dit.contract import ModelConfig
from .contract import CODEC, CORPUS, DEFAULT_OUTPUT, canonical


@torch.inference_mode()
def _score(model, sequence, starts, horizon, threshold, codec, norms, device):
    lm, ls, am, ass = norms
    previous = torch.from_numpy(sequence["latent"][starts - 1]).to(device)
    current = torch.from_numpy(sequence["latent"][starts]).to(device)
    previous_actor = torch.from_numpy(sequence["actor_state"][starts - 1]).to(device)
    actor = torch.from_numpy(sequence["actor_state"][starts]).to(device)
    frame = torch.from_numpy(sequence["frame"][starts]).permute(0, 3, 1, 2).float().div_(255)
    initial = frame.clone()
    for step in range(horizon):
        indices = starts + step + 1
        action = torch.from_numpy(sequence["action"][indices].astype(np.int64)).to(device)
        control = torch.from_numpy(sequence["control"][indices]).to(device)
        state = torch.from_numpy(sequence["state"][indices - 1]).to(device)
        cn, pn = (current - lm) / ls, (previous - lm) / ls
        delta = model.action(cn, pn, action, control, state, actor)
        next_latent = (cn + (delta.abs().mean(1, keepdim=True) >= threshold) * delta) * ls + lm
        an, pan = (actor - am) / ass, (previous_actor - am) / ass
        result = model.actor(an, pan, action, control, state)
        next_actor = (an + .9 * (result.gate >= .7) * (result.state - an)) * ass + am
        frame = torch.clamp(frame + codec.model.decode(next_latent).float().cpu() - codec.model.decode(current).float().cpu(), 0, 1)
        previous, current = current, next_latent
        previous_actor, actor = actor, next_actor
    target = torch.from_numpy(sequence["frame"][starts + horizon]).permute(0, 3, 1, 2).float().div_(255)
    mae = float(F.l1_loss(frame, target)); persistence = float(F.l1_loss(initial, target))
    return {"mae": mae, "persistence_mae": persistence, "improvement": 1 - mae / persistence}


def calibrate(output: Path = DEFAULT_OUTPUT):
    output = Path(output); payload = torch.load(output / "runtime.pt", map_location="cpu", weights_only=True); device = torch.device("cuda:0")
    model = RecurrentWorldStudent(ModelConfig(**payload["model_config"])); model.load_state_dict(payload["state"], strict=True); model.to(device).eval()
    codec = AdaptedWorldFrameCodec.from_checkpoint(CODEC, device="cuda"); sequences, manifest = load(CORPUS); normal = payload["normalization"]
    norms = (torch.tensor(normal["latent_mean"], device=device)[None, :, None, None], torch.tensor(normal["latent_std"], device=device)[None, :, None, None], torch.tensor(normal["actor_mean"], device=device)[None], torch.tensor(normal["actor_std"], device=device)[None])
    thresholds = (.16, .18, .20, .22, .24, .28, .32, .40)
    horizons = (4, 8, 16, 32); validation = {}
    for threshold in thresholds:
        rows = {}
        for horizon in horizons:
            starts = np.linspace(1, len(sequences[4]["latent"]) - horizon - 1, 16, dtype=np.int64)
            rows[str(horizon)] = _score(model, sequences[4], starts, horizon, threshold, codec, norms, device)
        validation[str(threshold)] = rows
    chosen = min(thresholds, key=lambda value: sum(validation[str(value)][str(horizon)]["mae"] for horizon in horizons))
    test = {}
    for horizon in horizons:
        starts = np.linspace(1, len(sequences[5]["latent"]) - horizon - 1, 24, dtype=np.int64)
        test[str(horizon)] = _score(model, sequences[5], starts, horizon, chosen, codec, norms, device)
    report = {"format": "nullvector-clean-recurrent-threshold-calibration/1.0.0", "corpus_sha256": manifest["manifest_sha256"], "checkpoint_state_sha256": payload["state_sha256"], "thresholds": list(thresholds), "validation": validation, "chosen": chosen, "test": test}
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest(); (output / "threshold_calibration.json").write_bytes(canonical(report)); return report
