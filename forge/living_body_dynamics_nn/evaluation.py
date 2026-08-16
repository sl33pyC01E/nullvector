from __future__ import annotations

import hashlib
from pathlib import Path
import uuid

import torch

from ..organism_raster_vae_v3.calibration import _canonical, _sha
from .contract import CHECKPOINT_FORMAT, DynamicsConfig, FORMAT
from .corpus import BodyTransitionCorpus
from .model import LivingBodyDynamicsNet
from .training import VALIDATION_IDENTITIES, evaluate_rows, source_sha256


@torch.inference_mode()
def evaluate(checkpoint: Path, destination: Path) -> Path:
    checkpoint = checkpoint.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source_sha256():
        raise ValueError("living dynamics evaluation checkpoint drifted")
    corpus = BodyTransitionCorpus(repeats=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LivingBodyDynamicsNet(DynamicsConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["ema_state"], strict=True)
    indices = [index for index, row in enumerate(corpus.rows) if row[0] in VALIDATION_IDENTITIES]
    metrics = evaluate_rows(model, corpus, indices, device)
    by_action = {}
    for action in range(4):
        action_indices = [index for index in indices if corpus.rows[index][2] == action]
        by_action[str(action)] = evaluate_rows(model, corpus, action_indices, device)
    by_family = {}
    for family, identity in enumerate((5, 11, 17, 23, 29)):
        family_indices = [index for index in indices if corpus.rows[index][0] == identity]
        by_family[str(family)] = evaluate_rows(model, corpus, family_indices, device)
    report = {
        "format": FORMAT, "status": "specialist_validation", "source_sha256": source_sha256(),
        "checkpoint": {"sha256": _sha(checkpoint), "segment": payload["segment"], "global_step": payload["global_step"], "ema_state_sha256": payload["ema_state_sha256"]},
        "corpus_sha256": corpus.semantic_sha256, "heldout_identities": sorted(VALIDATION_IDENTITIES),
        "metrics": metrics, "by_action": by_action, "by_family": by_family,
        "gates": {"healthy_drift_below_005": metrics["healthy_drift"] < .005, "health_mae_below_03": metrics["health_mae"] < .03, "fluid_mae_below_02": metrics["fluid_mae"] < .02, "system_mae_below_05": metrics["system_mae"] < .05, "production_promotion_allowed": False},
        "claim_boundary": {"one_step_cell_dynamics": True, "recurrent_rollout_validated": False, "game_runtime_authority": False},
    }
    report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    path = staging / "evaluation.json"
    path.write_bytes(_canonical(report))
    staging.replace(destination)
    return destination / "evaluation.json"
