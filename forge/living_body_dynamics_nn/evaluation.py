from __future__ import annotations

import hashlib
from pathlib import Path
import uuid

import numpy as np
import torch

from ..organism_raster_vae_v3.calibration import _canonical, _sha
from .contract import CHECKPOINT_FORMAT, DynamicsConfig, FORMAT
from ..creature_stage_neural_grasper_v1.feeding import FeedingState, FoodClump, absorb_food, feeder_status, metabolize_reserve
from ..living_body_substrate import LivingBody
from .corpus import BodyTransitionCorpus, collate_graphs
from .model import LivingBodyDynamicsNet
from .runtime import NeuralLivingBodyDynamicsRuntime
from .training import VALIDATION_IDENTITIES, evaluate_rows, source_sha256


@torch.inference_mode()
def _causal_feeding_metrics(
    model: LivingBodyDynamicsNet,
    corpus: BodyTransitionCorpus,
    indices: list[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predicted: dict[int, torch.Tensor] = {}
    target: dict[int, torch.Tensor] = {}
    for start in range(0, len(indices), 16):
        chosen = indices[start : start + 16]
        batch = {key: value.to(device) for key, value in collate_graphs([corpus[index] for index in chosen]).items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, _, feeding = model(batch)
        value = feeding.float()
        value[:, 6:8] = torch.sigmoid(value[:, 6:8])
        for row, index in enumerate(chosen):
            predicted[index] = value[row].cpu()
            target[index] = batch["feeding_target"][row].float().cpu()

    def select(*, scenarios: tuple[int, ...], repeats: tuple[int, ...], actions: tuple[int, ...] = (4, 6)) -> list[int]:
        return [
            index for index in indices
            if corpus.rows[index][1] in scenarios
            and corpus.rows[index][2] in actions
            and corpus.rows[index][3] in repeats
        ]

    def mean(rows: list[int], column: int, source: dict[int, torch.Tensor] = predicted) -> float:
        return float(torch.stack([source[index][column] for index in rows]).mean()) if rows else 0.0

    valid = select(scenarios=(0, 1), repeats=(0,))
    missed = select(scenarios=(0, 1), repeats=(1,))
    incompatible = select(scenarios=(0, 1), repeats=(2,))
    full = select(scenarios=(0, 1), repeats=(3,))
    feeder_ablated = select(scenarios=(2,), repeats=(0,))
    digestion_ablated = select(scenarios=(3,), repeats=(0,))
    valid_prediction = mean(valid, 0)
    valid_target = mean(valid, 0, target)
    invalid_groups = (missed, incompatible, full, feeder_ablated, digestion_ablated)
    return {
        "valid_absorption_prediction": round(valid_prediction, 9),
        "valid_absorption_target": round(valid_target, 9),
        "valid_absorption_mae": round(abs(valid_prediction - valid_target), 9),
        "missed_contact_absorption": round(mean(missed, 0), 9),
        "incompatible_food_absorption": round(mean(incompatible, 0), 9),
        "full_reserve_absorption": round(mean(full, 0), 9),
        "feeder_ablation_absorption": round(mean(feeder_ablated, 0), 9),
        "digestion_ablation_absorption": round(mean(digestion_ablated, 0), 9),
        "maximum_invalid_absorption": round(max(mean(rows, 0) for rows in invalid_groups), 9),
        "healthy_over_digestive_ablation": round(valid_prediction - mean(digestion_ablated, 0), 9),
        "healthy_route_probability": round(mean(valid, 7), 9),
        "feeder_ablation_route_probability": round(mean(feeder_ablated, 7), 9),
        "digestion_ablation_route_probability": round(mean(digestion_ablated, 7), 9),
    }


@torch.inference_mode()
def _recurrent_metrics(
    model: LivingBodyDynamicsNet,
    corpus: BodyTransitionCorpus,
    device: torch.device,
    steps: int = 32,
) -> dict[str, float]:
    runtime = NeuralLivingBodyDynamicsRuntime(model, device)
    totals = {"reserve": 0.0, "fullness": 0.0, "energy": 0.0, "health": 0.0, "systems": 0.0}
    observations = 0
    cases = 0
    for identity in sorted(VALIDATION_IDENTITIES):
        for digestive_ablation in (False, True):
            teacher = LivingBody(corpus.organisms[identity], seed=0x524543 + identity)
            neural = LivingBody(corpus.organisms[identity], seed=0x524543 + identity)
            if digestive_ablation:
                for body in (teacher, neural):
                    mask = feeder_status(body).digestive_mask
                    body.health[mask] = 0
                    body.fluid[mask] *= np.float32(.2)
            teacher_feeding = FeedingState(.65, 4.0, 24.0, 240.0, 0.0)
            neural_feeding = FeedingState(.65, 4.0, 24.0, 240.0, 0.0)
            for step in range(steps):
                feed = step % 3 == 0
                action_kind = 4 if feed else 5
                delta = .12
                activity = .55
                points = teacher.organism.cell_xy[feeder_status(teacher).feeder_mask & teacher.alive_mask]
                anchor = points[0].astype(np.float64) if len(points) else np.zeros(2, dtype=np.float64)
                profile = (1.0, 1.0, 1.0, 1.0, 1.0)
                teacher_clump = FoodClump(anchor.copy(), np.zeros(2), .55, .4, 2.1, profile)
                neural_clump = FoodClump(anchor.copy(), np.zeros(2), .55, .4, 2.1, profile)
                if feed:
                    absorb_food(teacher, teacher_feeding, teacher_clump, body_position=np.zeros(2), delta=delta)
                else:
                    metabolize_reserve(teacher, teacher_feeding, delta=delta, activity=activity)
                teacher_snapshot = teacher.tick(.1)
                transition = runtime.predict(
                    neural, neural_feeding, neural_clump, contact=feed,
                    action_kind=action_kind, activity=activity, delta=delta,
                )
                runtime.apply(neural, neural_feeding, neural_clump, transition)
                totals["reserve"] += abs(neural_feeding.reserve - teacher_feeding.reserve) / 4.0
                totals["fullness"] += abs(neural_feeding.fullness_seconds - teacher_feeding.fullness_seconds) / 240.0
                totals["energy"] += abs(neural.energy - teacher.energy) / 4.0
                totals["health"] += float(np.abs(neural.health - teacher.health).mean())
                totals["systems"] += float(np.mean([
                    abs(transition.systems[name] - teacher_snapshot.systems[name]) for name in transition.systems
                ]))
                observations += 1
            cases += 1
    return {
        "steps": steps,
        "cases": cases,
        "reserve_mae": round(totals["reserve"] / observations, 9),
        "fullness_mae": round(totals["fullness"] / observations, 9),
        "energy_mae": round(totals["energy"] / observations, 9),
        "health_mae": round(totals["health"] / observations, 9),
        "system_mae": round(totals["systems"] / observations, 9),
    }


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
    causal_feeding = _causal_feeding_metrics(model, corpus, indices, device)
    recurrent = _recurrent_metrics(model, corpus, device)
    by_action = {}
    for action in range(7):
        action_indices = [index for index in indices if corpus.rows[index][2] == action]
        by_action[str(action)] = evaluate_rows(model, corpus, action_indices, device)
    by_family = {}
    for family, identity in enumerate((5, 11, 17, 23, 29)):
        family_indices = [index for index in indices if corpus.rows[index][0] == identity]
        by_family[str(family)] = evaluate_rows(model, corpus, family_indices, device)
    gates = {
        "healthy_drift_below_005": metrics["healthy_drift"] < .005,
        "health_mae_below_03": metrics["health_mae"] < .03,
        "fluid_mae_below_02": metrics["fluid_mae"] < .02,
        "system_mae_below_05": metrics["system_mae"] < .05,
        "feeding_mae_below_03": metrics["feeding_mae"] < .03,
        "absorption_mae_below_015": metrics["absorption_mae"] < .015,
        "false_absorption_below_01": metrics["false_absorption"] < .01,
        "route_accuracy_above_97": metrics["route_accuracy"] > .97,
        "valid_absorption_signal": causal_feeding["valid_absorption_prediction"] > .005,
        "valid_absorption_mae_below_01": causal_feeding["valid_absorption_mae"] < .01,
        "invalid_absorption_below_005": causal_feeding["maximum_invalid_absorption"] < .005,
        "digestion_ablation_is_causal": causal_feeding["healthy_over_digestive_ablation"] > .005,
        "ablated_route_probability_below_05": max(
            causal_feeding["feeder_ablation_route_probability"],
            causal_feeding["digestion_ablation_route_probability"],
        ) < .5,
        "recurrent_reserve_mae_below_08": recurrent["reserve_mae"] < .08,
        "recurrent_fullness_mae_below_08": recurrent["fullness_mae"] < .08,
        "recurrent_energy_mae_below_05": recurrent["energy_mae"] < .05,
        "recurrent_health_mae_below_03": recurrent["health_mae"] < .03,
        "recurrent_system_mae_below_06": recurrent["system_mae"] < .06,
    }
    gates["production_promotion_allowed"] = all(gates.values())
    report = {
        "format": FORMAT,
        "status": "accepted_neural_authority" if gates["production_promotion_allowed"] else "rejected_neural_authority",
        "source_sha256": source_sha256(),
        "checkpoint": {"sha256": _sha(checkpoint), "segment": payload["segment"], "global_step": payload["global_step"], "ema_state_sha256": payload["ema_state_sha256"]},
        "corpus_sha256": corpus.semantic_sha256, "heldout_identities": sorted(VALIDATION_IDENTITIES),
        "metrics": metrics, "causal_feeding": causal_feeding, "recurrent": recurrent,
        "by_action": by_action, "by_family": by_family,
        "gates": gates,
        "claim_boundary": {
            "one_step_cell_dynamics": True,
            "one_step_feeding_dynamics": True,
            "recurrent_rollout_validated": gates["production_promotion_allowed"],
            "game_runtime_authority": gates["production_promotion_allowed"],
        },
    }
    report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    path = staging / "evaluation.json"
    path.write_bytes(_canonical(report))
    staging.replace(destination)
    return destination / "evaluation.json"
