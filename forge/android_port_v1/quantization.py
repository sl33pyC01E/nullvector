from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import torch
from torch import nn

from ..mobile_action_core_v1.contract import CHECKPOINT_FORMAT as MOBILE_ACTION_FORMAT
from ..mobile_action_core_v1.data import load_sequences
from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..world_latent_dit.contract import ModelConfig
from .contract import MOBILE_ACTION, MOBILE_ACTION_SHA256, canonical, file_sha256


FORMAT = "nullvector-android-action-qdq-v1/1.0.0"


class _ActionDelta(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, current, previous, action, control, context, actor, visibility, memory): return self.model.gated_action(current, previous, action, control, context, actor, visibility, memory)


class _ActorState(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, actor, previous_actor, action, control, context, visibility, memory):
        result = self.model.actor(actor, previous_actor, action, control, context, visibility, memory); return result.state, result.gate


class _Reader(CalibrationDataReader):
    def __init__(self, rows): self.rows = iter(rows)
    def get_next(self): return next(self.rows, None)


def _rows(payload, *, heldout: bool) -> list[dict[str, np.ndarray]]:
    norm = payload["normalization"]; lm = np.asarray(norm["latent_mean"], np.float32)[None, :, None, None]; ls = np.asarray(norm["latent_std"], np.float32)[None, :, None, None]; am = np.asarray(norm["actor_mean"], np.float32)[None]; ast = np.asarray(norm["actor_std"], np.float32)[None]; rows = []
    for sequence in load_sequences():
        low, high = ((len(sequence["latent"]) - 64, len(sequence["latent"]) - 2) if heldout else (1, len(sequence["latent"]) - 65))
        for index in np.linspace(low, high, 8, dtype=int):
            rows.append({"current": ((sequence["latent"][index:index + 1] - lm) / ls).astype(np.float32), "previous": ((sequence["latent"][index - 1:index] - lm) / ls).astype(np.float32), "action": sequence["action"][index:index + 1].astype(np.int64), "control": sequence["control"][index:index + 1].astype(np.float32), "context": sequence["state"][index:index + 1].astype(np.float32), "actor": ((sequence["actor_state"][index:index + 1] - am) / ast).astype(np.float32), "previous_actor": ((sequence["actor_state"][index - 1:index] - am) / ast).astype(np.float32), "visibility": sequence["visibility"][index:index + 1].astype(np.float32), "memory": sequence["memory"][index:index + 1].astype(np.float32)})
    return rows


def _benchmark(session, feeds, steps=30):
    for _ in range(4): session.run(None, feeds)
    began = time.perf_counter()
    for _ in range(steps): session.run(None, feeds)
    return (time.perf_counter() - began) * 1000 / steps


def build(output: Path) -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if file_sha256(MOBILE_ACTION) != MOBILE_ACTION_SHA256: raise ValueError("Mobile action checkpoint drifted.")
    payload = torch.load(MOBILE_ACTION, map_location="cpu", weights_only=True)
    if payload.get("format") != MOBILE_ACTION_FORMAT or payload.get("status") != "mobile_action_ready": raise ValueError("Mobile action checkpoint is not promoted.")
    model = PerceptionRecurrentWorldStudent(ModelConfig(**payload["model_config"])); model.load_state_dict(payload["state"], strict=True); model.eval(); output.mkdir(parents=True)
    sample = _rows(payload, heldout=False)[0]; tensor = lambda name: torch.from_numpy(sample[name])
    action_fp = output / "action_delta_fp32.onnx"; actor_fp = output / "actor_state_fp32.onnx"; action_int8 = output / "action_delta_int8_qdq.onnx"
    torch.onnx.export(_ActionDelta(model), tuple(tensor(name) for name in ("current", "previous", "action", "control", "context", "actor", "visibility", "memory")), action_fp, input_names=["current", "previous", "action", "control", "context", "actor", "visibility", "memory"], output_names=["delta", "gate_logits"], opset_version=18, dynamo=False)
    torch.onnx.export(_ActorState(model), tuple(tensor(name) for name in ("actor", "previous_actor", "action", "control", "context", "visibility", "memory")), actor_fp, input_names=["actor", "previous_actor", "action", "control", "context", "visibility", "memory"], output_names=["next_actor", "actor_gate"], opset_version=18, dynamo=False)
    calibration = _rows(payload, heldout=False); action_names = ("current", "previous", "action", "control", "context", "actor", "visibility", "memory")
    quantize_static(str(action_fp), str(action_int8), _Reader([{name: row[name] for name in action_names} for row in calibration]), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=False, op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    for path in (action_fp, actor_fp, action_int8): onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    fp = ort.InferenceSession(str(action_fp), options, providers=["CPUExecutionProvider"]); quant = ort.InferenceSession(str(action_int8), options, providers=["CPUExecutionProvider"]); actor_session = ort.InferenceSession(str(actor_fp), options, providers=["CPUExecutionProvider"])
    heldout = _rows(payload, heldout=True); delta_errors = []; gate_probability_errors = []; actor_errors = []; actor_gate_errors = []
    for row in heldout:
        action_feeds = {name: row[name] for name in action_names}; expected = fp.run(None, action_feeds); actual = quant.run(None, action_feeds); delta_errors.append(float(np.abs(actual[0] - expected[0]).mean())); gate_probability_errors.append(float(np.abs(1 / (1 + np.exp(-actual[1])) - 1 / (1 + np.exp(-expected[1]))).mean()))
        actor_feeds = {name: row[name] for name in ("actor", "previous_actor", "action", "control", "context", "visibility", "memory")}; actor_actual = actor_session.run(None, actor_feeds); actor_expected = model.actor(*(tensor for tensor in (torch.from_numpy(row["actor"]), torch.from_numpy(row["previous_actor"]), torch.from_numpy(row["action"]), torch.from_numpy(row["control"]), torch.from_numpy(row["context"]), torch.from_numpy(row["visibility"]), torch.from_numpy(row["memory"]))))
        actor_errors.append(float(np.abs(actor_actual[0] - actor_expected.state.detach().numpy()).mean())); actor_gate_errors.append(float(np.abs(actor_actual[1] - actor_expected.gate.detach().numpy()).mean()))
    rollout_errors = []; rollout_maxima = []; bias_max = float(payload["inference"]["gate_logit_bias_max"]); ramp = int(payload["inference"]["gate_logit_bias_ramp_steps"])
    for sequence in load_sequences():
        start = len(sequence["latent"]) - 64; norm = payload["normalization"]; lm = np.asarray(norm["latent_mean"], np.float32)[None, :, None, None]; ls = np.asarray(norm["latent_std"], np.float32)[None, :, None, None]; am = np.asarray(norm["actor_mean"], np.float32)[None]; ast = np.asarray(norm["actor_std"], np.float32)[None]
        fp_previous = quant_previous = ((sequence["latent"][start - 1:start] - lm) / ls).astype(np.float32); fp_current = quant_current = ((sequence["latent"][start:start + 1] - lm) / ls).astype(np.float32); previous_actor = ((sequence["actor_state"][start - 1:start] - am) / ast).astype(np.float32); actor = ((sequence["actor_state"][start:start + 1] - am) / ast).astype(np.float32)
        for offset in range(16):
            index = start + offset; common = {"action": sequence["action"][index:index + 1].astype(np.int64), "control": sequence["control"][index:index + 1].astype(np.float32), "context": sequence["state"][index:index + 1].astype(np.float32), "actor": actor, "visibility": sequence["visibility"][index:index + 1].astype(np.float32), "memory": sequence["memory"][index:index + 1].astype(np.float32)}
            fp_feeds = {**common, "current": fp_current, "previous": fp_previous}; quant_feeds = {**common, "current": quant_current, "previous": quant_previous}; fp_delta, fp_logits = fp.run(None, fp_feeds); quant_delta, quant_logits = quant.run(None, quant_feeds); bias = bias_max * min(offset / ramp, 1.0) if ramp else bias_max
            fp_next = fp_current + 1 / (1 + np.exp(-(fp_logits + bias))) * fp_delta; quant_next = quant_current + 1 / (1 + np.exp(-(quant_logits + bias))) * quant_delta
            actor_feeds = {name: value for name, value in {**common, "previous_actor": previous_actor}.items() if name in ("actor", "previous_actor", "action", "control", "context", "visibility", "memory")}; proposed, actor_gate = actor_session.run(None, actor_feeds); next_actor = actor + .9 * (actor_gate >= .7) * (proposed - actor)
            fp_previous, fp_current = fp_current, fp_next.astype(np.float32); quant_previous, quant_current = quant_current, quant_next.astype(np.float32); previous_actor, actor = actor, next_actor.astype(np.float32)
        rollout_errors.append(float(np.abs(quant_current - fp_current).mean())); rollout_maxima.append(float(np.abs(quant_current).max()))
    combined_bytes = action_int8.stat().st_size + actor_fp.stat().st_size; metrics = {"action_delta_fp32_bytes": action_fp.stat().st_size, "action_delta_int8_bytes": action_int8.stat().st_size, "actor_fp32_bytes": actor_fp.stat().st_size, "combined_bytes": combined_bytes, "compression_ratio": combined_bytes / MOBILE_ACTION.stat().st_size, "quantized_delta_mae": float(np.mean(delta_errors)), "quantized_gate_probability_mae": float(np.mean(gate_probability_errors)), "actor_split_mae": float(np.mean(actor_errors)), "actor_gate_split_mae": float(np.mean(actor_gate_errors)), "horizon_16_quantized_vs_fp32_mae": float(np.mean(rollout_errors)), "horizon_16_max_abs_latent": max(rollout_maxima), "desktop_cpu_action_ms": _benchmark(quant, {name: heldout[0][name] for name in action_names}), "desktop_cpu_actor_ms": _benchmark(actor_session, {name: heldout[0][name] for name in ("actor", "previous_actor", "action", "control", "context", "visibility", "memory")})}
    gates = {"int8_model_loads": True, "combined_under_18_mib": combined_bytes < 18 * 1024**2, "delta_parity": metrics["quantized_delta_mae"] <= .02, "gate_probability_parity": metrics["quantized_gate_probability_mae"] <= .01, "actor_exact_split": metrics["actor_split_mae"] <= 1e-5 and metrics["actor_gate_split_mae"] <= 1e-5, "horizon_16_rollout_parity": metrics["horizon_16_quantized_vs_fp32_mae"] <= .08, "horizon_16_bounded": metrics["horizon_16_max_abs_latent"] <= 12}
    report = {"format": FORMAT, "status": "android_int8_candidate_ready" if all(gates.values()) else "quality_failed", "parent_sha256": MOBILE_ACTION_SHA256, "calibration_rows": len(calibration), "heldout_rows": len(heldout), "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in (action_int8, actor_fp)}, "metrics": metrics, "gates": gates, "limitations": ["Candidate requires physical Galaxy S25 Ultra QNN/NNAPI partitioning and rollout parity before replacing the FP32 APK."]}
    (output / "quantization_report.json").write_bytes(canonical(report)); return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("output", type=Path); args = parser.parse_args(argv); print(json.dumps(build(args.output), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
