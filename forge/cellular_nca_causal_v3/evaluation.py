from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from ..cellular_nca.corpus import load_corpus
from ..cellular_nca.evaluation import _damage_system, _mae
from ..cellular_nca.teacher import make_scenarios, teacher_step
from ..cellular_nca.training import load_final_checkpoint as load_base_checkpoint
from ..cellular_nca_causal.evaluation import _relative_change, _render, _teacher_relative
from .contract import DEFAULT_OUTPUT, FORMAT, canonical, sha256_file, source_sha256
from .curriculum import ROLLOUT_STEPS
from .training import load_final


MANIFEST_NAME = "causal_v3_manifest.json"
VISUAL_NAME = "organ_causality_v3.png"


def _atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def evaluate(output: Path = DEFAULT_OUTPUT, *, device_name: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve()
    if (output / MANIFEST_NAME).exists():
        raise FileExistsError("V3 evaluation is immutable")
    device = torch.device(device_name)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(0)
    model, checkpoint, contract = load_final(output); model.to(device)
    corpus_root = Path(__file__).resolve().parents[2] / contract["corpus_parent"]["path"]
    base_model, _, _ = load_base_checkpoint(corpus_root); base_model.to(device).eval()
    arrays = load_corpus(corpus_root)["arrays"]
    static = torch.from_numpy(arrays["static"]).to(device); initial = torch.from_numpy(arrays["initial_state"]).to(device); bonds = torch.from_numpy(arrays["live_bonds"]).to(device)
    generator = torch.Generator(device=device).manual_seed(contract["seed"] ^ 0x4556414C)
    injured, live = make_scenarios(static, initial, bonds, generator)
    teacher = injured.clone(); base = injured.clone(); causal = injured.clone()
    for _ in range(32):
        teacher = teacher_step(static, teacher, live, .1); base = base_model(static, base, live); causal = model(static, causal, live)
    base_mae = _mae(base, teacher, static); causal_mae = _mae(causal, teacher, static)
    systems = (("circulation", 28, 1), ("respiration", 31, 4), ("digestion", 34, 3), ("neural", 37, 8))
    rows = []
    for name, start, readout in systems:
        rows.append({
            "system": name, "readout": readout,
            "parent_relative_change": round(_relative_change(base_model, static, initial, bonds, start, readout, 16), 8),
            "causal_relative_change": round(_relative_change(model, static, initial, bonds, start, readout, 16), 8),
            "teacher_relative_change": round(_teacher_relative(static, initial, bonds, start, readout, 16, .1), 8),
        })
    parent_error = float(np.mean([abs(row["parent_relative_change"] - row["teacher_relative_change"]) for row in rows]))
    causal_error = float(np.mean([abs(row["causal_relative_change"] - row["teacher_relative_change"]) for row in rows]))
    gates = {
        "all_values_finite": bool(np.isfinite([value for row in rows for key, value in row.items() if key.endswith("change")] + list(causal_mae.values())).all()),
        "all_four_organs_reduce_their_readout": all(row["causal_relative_change"] < -.005 for row in rows),
        "counterfactual_error_improves_over_parent": causal_error < parent_error * .75,
        "general_health_mae_below_0_02": causal_mae["health"] < .02,
        "general_fluid_mae_below_0_04": causal_mae["fluid"] < .04,
        "general_neural_mae_below_0_06": causal_mae["neural_activity"] < .06,
        "six_step_training_horizon": contract["rollout_steps"] == ROLLOUT_STEPS == 6,
    }
    visual = _render(rows)
    payload: dict[str, Any] = {
        "format": FORMAT, "status": "ready" if all(gates.values()) else "experimental",
        "source_sha256": source_sha256(), "checkpoint_sha256": sha256_file(output / f"causal_v3_segment_{contract['total_steps']:07d}.pt"),
        "ema_state_sha256": checkpoint["ema_state_sha256"], "training_steps": contract["total_steps"],
        "evaluation": {"organ_counterfactuals": rows, "parent_counterfactual_mae": round(parent_error, 8), "causal_counterfactual_mae": round(causal_error, 8), "parent_rollout_mae": base_mae, "causal_rollout_mae": causal_mae},
        "visual": {"path": VISUAL_NAME, "bytes": len(visual), "sha256": hashlib.sha256(visual).hexdigest(), "visually_inspected": False},
        "gates": gates,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    _atomic(output / VISUAL_NAME, visual); _atomic(output / MANIFEST_NAME, canonical(payload))
    return payload
