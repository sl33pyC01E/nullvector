from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from ..cellular_motion.contract import MOTION_NAMES, MOTION_SPECS
from ..config import PROJECT_ROOT
from .dataset import load_corpus_manifest, validate_corpus
from .model import NeuralCellMotionUNet, neural_motion_loss


EVALUATION_FORMAT = "nullvector-neural-cell-motion-heldout-evaluation-v1"
METRIC_NAMES = (
    "loss", "displacement", "activation", "emission", "coherence",
    "temporal", "outside",
)
EVALUATION_SOURCE_FILES = (
    "forge/neural_cell_motion/evaluation.py",
    "forge/neural_cell_motion/evaluation_report.py",
    "forge/neural_cell_motion/model.py",
    "forge/neural_cell_motion/dataset.py",
    "forge/neural_cell_motion/contract.py",
)


def evaluation_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-cell-motion-evaluation-source-v1\0")
    for relative in EVALUATION_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _mean_masked(value: Tensor, occupancy: Tensor, channels: int) -> float:
    denominator = float(occupancy.sum()) * channels
    return float((value.abs() * occupancy).sum()) / max(1.0, denominator)


def _metric_bucket() -> dict[str, float]:
    return {**{name: 0.0 for name in METRIC_NAMES}, "weight": 0.0}


def _add_loss(bucket: dict[str, float], pieces: dict[str, Tensor], weight: int) -> None:
    for name in METRIC_NAMES:
        bucket[name] += float(pieces[name]) * weight
    bucket["weight"] += weight


def _finish_bucket(bucket: dict[str, float]) -> dict[str, float]:
    weight = bucket["weight"]
    if weight <= 0:
        raise ValueError("Neural motion evaluation bucket is empty.")
    return {name: round(bucket[name] / weight, 8) for name in METRIC_NAMES}


def _load_split(corpus: Path, split: str) -> list[dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("Neural motion evaluation split drifted.")
    manifest = load_corpus_manifest(corpus)
    records = [record for record in manifest["records"] if record["split"] == split]
    if len(records) != 5 or sorted(record["family_id"] for record in records) != list(range(5)):
        raise ValueError("Neural motion held-out family census drifted.")
    # A canonical manifest alone is not evidence that its referenced tensors
    # are still the source-bound, bounded artifacts that were published. Audit
    # the exact held-out shards before opening them for recurrent evaluation.
    validate_corpus(corpus, replay=False, record_ids={record["sample_id"] for record in records})
    loaded: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item["family_id"]):
        with np.load(corpus / record["path"], allow_pickle=False) as archive:
            loaded.append({
                "record": record,
                "features": archive["features"].astype(np.float32),
                "targets": archive["targets"].astype(np.float32),
                "indices": archive["indices"].astype(np.int16),
                "previous_index": archive["previous_index"].astype(np.int16),
            })
    return loaded


def _clip_rows(item: dict[str, Any], motion_id: int, facing_id: int) -> np.ndarray:
    indices = item["indices"]
    rows = np.flatnonzero((indices[:, 0] == motion_id) & (indices[:, 1] == facing_id))
    expected = MOTION_SPECS[MOTION_NAMES[motion_id]][0]
    if len(rows) != expected or not np.array_equal(indices[rows, 2], np.arange(expected, dtype=indices.dtype)):
        raise ValueError("Neural motion held-out clip registry drifted.")
    return rows


def evaluate_recurrent_split(
    corpus: Path,
    model: NeuralCellMotionUNet,
    split: str,
    *,
    device: torch.device | str = "cpu",
    precision: str = "float32",
    motion_names: Iterable[str] = MOTION_NAMES,
    facing_ids: Iterable[int] = range(8),
) -> dict[str, Any]:
    """Roll the model through complete clips on every held-out family.

    The previous model prediction, rather than the authoritative preceding
    frame, is fed back after frame zero. This exposes drift and action collapse
    that teacher-forced frame scores conceal.
    """

    corpus = Path(corpus).resolve(); device = torch.device(device)
    names = tuple(motion_names); facings = tuple(facing_ids)
    if not names or len(set(names)) != len(names) or any(name not in MOTION_SPECS for name in names):
        raise ValueError("Neural motion evaluation motion selection drifted.")
    if not facings or len(set(facings)) != len(facings) or any(type(value) is not int or not 0 <= value < 8 for value in facings):
        raise ValueError("Neural motion evaluation facing selection drifted.")
    if precision not in {"float32", "bf16"} or precision == "bf16" and device.type != "cuda":
        raise ValueError("Neural motion evaluation precision drifted.")
    items = _load_split(corpus, split); model = model.to(device); model.eval()
    aggregate = _metric_bucket(); baseline = _metric_bucket(); family_buckets = {family: _metric_bucket() for family in range(5)}; motion_buckets = {name: _metric_bucket() for name in names}
    predicted_energy = defaultdict(float); target_energy = defaultdict(float)
    loop_closure_sum = 0.0; loop_closure_count = 0; action_endpoint_sum = 0.0; action_endpoint_count = 0
    frame_count = 0; clip_count = 0; outside_max = 0.0; output_abs_max = 0.0
    autocast_enabled = precision == "bf16"
    with torch.inference_mode():
        for motion_name in names:
            motion_id = MOTION_NAMES.index(motion_name); stored_frames, _, loop = MOTION_SPECS[motion_name]
            clip_specs = [(item, facing, _clip_rows(item, motion_id, facing)) for item in items for facing in facings]
            static = torch.from_numpy(np.stack([item["features"] for item, _, _ in clip_specs])).to(device)
            occupancy = static[:, :1]; family = torch.tensor([item["record"]["family_id"] for item, _, _ in clip_specs], dtype=torch.int64, device=device)
            motion = torch.full((len(clip_specs),), motion_id, dtype=torch.int64, device=device); facing = torch.tensor([facing for _, facing, _ in clip_specs], dtype=torch.int64, device=device)
            first_rows = np.asarray([rows[0] for _, _, rows in clip_specs]); previous_rows = np.asarray([int(item["previous_index"][row]) for (item, _, _), row in zip(clip_specs, first_rows)])
            state = torch.from_numpy(np.stack([item["targets"][row] for (item, _, _), row in zip(clip_specs, previous_rows)])).to(device)
            teacher_previous = state.clone(); first_prediction: Tensor | None = None
            for frame_index in range(stored_frames):
                target = torch.from_numpy(np.stack([item["targets"][rows[frame_index]] for item, _, rows in clip_specs])).to(device)
                phase = torch.full((len(clip_specs),), frame_index / max(1, stored_frames - 1), dtype=torch.float32, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    prediction = model(static, state, family, motion, facing, phase)
                prediction = prediction.float(); target = target.float(); state = state.float()
                weight = len(clip_specs)
                for family_id in range(5):
                    mask = family == family_id
                    _, family_pieces = neural_motion_loss(prediction[mask], target[mask], state[mask], static[mask].float())
                    _, family_baseline_pieces = neural_motion_loss(teacher_previous[mask], target[mask], teacher_previous[mask], static[mask].float())
                    _add_loss(family_buckets[family_id], family_pieces, int(mask.sum()))
                    # Report macro-family quality. A large-bodied family must
                    # not dominate the headline or per-motion score merely by
                    # contributing more occupied pixels.
                    _add_loss(aggregate, family_pieces, int(mask.sum()))
                    _add_loss(baseline, family_baseline_pieces, int(mask.sum()))
                    _add_loss(motion_buckets[motion_name], family_pieces, int(mask.sum()))
                for key, channel_slice in (("displacement", slice(0, 2)), ("activation", slice(2, 3)), ("emission", slice(3, 4))):
                    predicted_energy[key] += _mean_masked(prediction[:, channel_slice], occupancy, channel_slice.stop - channel_slice.start) * weight
                    target_energy[key] += _mean_masked(target[:, channel_slice], occupancy, channel_slice.stop - channel_slice.start) * weight
                outside_max = max(outside_max, float((prediction * (1 - occupancy)).abs().max())); output_abs_max = max(output_abs_max, float(prediction.abs().max()))
                if first_prediction is None: first_prediction = prediction.clone()
                teacher_previous = target; state = prediction; frame_count += weight
            if first_prediction is None:
                raise AssertionError("Neural motion evaluator produced no frames.")
            if loop:
                loop_closure_sum += _mean_masked(state - first_prediction, occupancy, 4) * len(clip_specs); loop_closure_count += len(clip_specs)
            else:
                action_endpoint_sum += _mean_masked(state - target, occupancy, 4) * len(clip_specs); action_endpoint_count += len(clip_specs)
            clip_count += len(clip_specs)
    if frame_count <= 0 or clip_count != len(items) * len(facings) * len(names):
        raise ValueError("Neural motion evaluation coverage drifted.")
    aggregate_metrics = _finish_bucket(aggregate); baseline_metrics = _finish_bucket(baseline)
    energy = {
        key: {
            "predicted": round(predicted_energy[key] / frame_count, 8),
            "target": round(target_energy[key] / frame_count, 8),
            "ratio": round((predicted_energy[key] + 1e-12) / (target_energy[key] + 1e-12), 8),
        }
        for key in ("displacement", "activation", "emission")
    }
    result: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "split": split,
        "identity_count": len(items),
        "family_count": 5,
        "motion_count": len(names),
        "facing_count": len(facings),
        "clip_count": clip_count,
        "frame_count": frame_count,
        "motion_names": list(names),
        "facing_ids": list(facings),
        "metrics": aggregate_metrics,
        "previous_frame_baseline": baseline_metrics,
        "family_metrics": {str(family): _finish_bucket(bucket) for family, bucket in family_buckets.items()},
        "motion_metrics": {name: _finish_bucket(bucket) for name, bucket in motion_buckets.items()},
        "response_energy": energy,
        "loop_closure_mae": round(loop_closure_sum / max(1, loop_closure_count), 8),
        "action_endpoint_mae": round(action_endpoint_sum / max(1, action_endpoint_count), 8),
        "outside_support_max": round(outside_max, 8),
        "output_abs_max": round(output_abs_max, 8),
    }
    if not all(math.isfinite(float(value)) for value in (
        list(aggregate_metrics.values()) + list(baseline_metrics.values()) +
        [result["loop_closure_mae"], result["action_endpoint_mae"], result["outside_support_max"], result["output_abs_max"]] +
        [entry[field] for entry in energy.values() for field in ("predicted", "target", "ratio")]
    )):
        raise ValueError("Neural motion evaluation became non-finite.")
    return result


def acceptance_gates(evaluation: dict[str, Any]) -> dict[str, bool]:
    """Reference-relative gates that prevent a low-loss collapsed motion field."""

    evaluation = validate_evaluation_result(evaluation)
    energy = evaluation["response_energy"]
    return {
        "full_five_family_13_motion_8_facing_coverage": evaluation["identity_count"] == 5 and evaluation["family_count"] == 5 and evaluation["motion_count"] == 13 and evaluation["facing_count"] == 8 and evaluation["clip_count"] == 520 and evaluation["frame_count"] == 4720,
        "finite_metrics": all(math.isfinite(float(value)) for value in evaluation["metrics"].values()),
        "outside_support_exact_zero": evaluation["outside_support_max"] == 0,
        "bounded_output": evaluation["output_abs_max"] <= 1.000001,
        "recurrent_loss_beats_previous_frame_baseline": evaluation["metrics"]["loss"] < evaluation["previous_frame_baseline"]["loss"],
        "displacement_response_not_collapsed": .10 <= energy["displacement"]["ratio"] <= 3.0,
        "activation_response_not_collapsed": .10 <= energy["activation"]["ratio"] <= 3.0,
        "emission_response_not_collapsed": .05 <= energy["emission"]["ratio"] <= 5.0,
        "loop_closure_bounded": evaluation["loop_closure_mae"] <= .25,
        "action_endpoint_bounded": evaluation["action_endpoint_mae"] <= .35,
        "every_family_finite": all(all(math.isfinite(float(value)) for value in metrics.values()) for metrics in evaluation["family_metrics"].values()),
        "every_motion_finite": all(all(math.isfinite(float(value)) for value in metrics.values()) for metrics in evaluation["motion_metrics"].values()),
    }


def validate_evaluation_result(evaluation: Any, *, require_full: bool = True) -> dict[str, Any]:
    required = {
        "format", "split", "identity_count", "family_count", "motion_count",
        "facing_count", "clip_count", "frame_count", "motion_names",
        "facing_ids", "metrics", "previous_frame_baseline", "family_metrics",
        "motion_metrics", "response_energy", "loop_closure_mae",
        "action_endpoint_mae", "outside_support_max", "output_abs_max",
    }
    if not isinstance(evaluation, dict) or set(evaluation) != required or evaluation["format"] != EVALUATION_FORMAT or evaluation["split"] not in {"validation", "test"}:
        raise ValueError("Neural motion evaluation result contract drifted.")
    names = evaluation["motion_names"]; facings = evaluation["facing_ids"]
    if not isinstance(names, list) or len(names) != evaluation["motion_count"] or len(set(names)) != len(names) or any(name not in MOTION_SPECS for name in names):
        raise ValueError("Neural motion evaluation motion registry drifted.")
    if not isinstance(facings, list) or len(facings) != evaluation["facing_count"] or len(set(facings)) != len(facings) or any(type(value) is not int or not 0 <= value < 8 for value in facings):
        raise ValueError("Neural motion evaluation facing registry drifted.")
    expected_frames = evaluation["identity_count"] * len(facings) * sum(MOTION_SPECS[name][0] for name in names)
    expected_clips = evaluation["identity_count"] * len(facings) * len(names)
    if evaluation["identity_count"] != 5 or evaluation["family_count"] != 5 or evaluation["clip_count"] != expected_clips or evaluation["frame_count"] != expected_frames:
        raise ValueError("Neural motion evaluation coverage drifted.")
    if require_full and (names != list(MOTION_NAMES) or facings != list(range(8)) or expected_clips != 520 or expected_frames != 4720):
        raise ValueError("Neural motion evaluation is not the full authority matrix.")
    for key in ("metrics", "previous_frame_baseline"):
        if not isinstance(evaluation[key], dict) or set(evaluation[key]) != set(METRIC_NAMES) or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0 for value in evaluation[key].values()):
            raise ValueError("Neural motion evaluation metric registry drifted.")
    if set(evaluation["family_metrics"]) != {str(value) for value in range(5)} or set(evaluation["motion_metrics"]) != set(names):
        raise ValueError("Neural motion evaluation breakdown registry drifted.")
    for metrics in [*evaluation["family_metrics"].values(), *evaluation["motion_metrics"].values()]:
        if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES) or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0 for value in metrics.values()):
            raise ValueError("Neural motion evaluation breakdown drifted.")
    # Every family contributes the same complete motion/facing/frame census.
    # Motion buckets differ only by their authoritative stored-frame count.
    # Requiring both independently-derived aggregates prevents a rehashed
    # report from silently changing a nested or headline quality metric.
    motion_weight = sum(MOTION_SPECS[name][0] for name in names)
    for metric in METRIC_NAMES:
        family_mean = sum(float(evaluation["family_metrics"][str(family)][metric]) for family in range(5)) / 5
        motion_mean = sum(float(evaluation["motion_metrics"][name][metric]) * MOTION_SPECS[name][0] for name in names) / motion_weight
        aggregate = float(evaluation["metrics"][metric])
        if not math.isclose(aggregate, family_mean, rel_tol=2e-6, abs_tol=2e-7) or not math.isclose(aggregate, motion_mean, rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError("Neural motion aggregate metric is not derivable from its breakdowns.")
    if set(evaluation["response_energy"]) != {"displacement", "activation", "emission"}:
        raise ValueError("Neural motion response-energy registry drifted.")
    for entry in evaluation["response_energy"].values():
        if not isinstance(entry, dict) or set(entry) != {"predicted", "target", "ratio"} or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0 for value in entry.values()):
            raise ValueError("Neural motion response-energy values drifted.")
        expected_ratio = (float(entry["predicted"]) + 1e-12) / (float(entry["target"]) + 1e-12)
        if not math.isclose(float(entry["ratio"]), expected_ratio, rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError("Neural motion response-energy ratio drifted.")
    for key in ("loop_closure_mae", "action_endpoint_mae", "outside_support_max", "output_abs_max"):
        value = evaluation[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            raise ValueError("Neural motion evaluation scalar drifted.")
    return evaluation
