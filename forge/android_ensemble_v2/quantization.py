from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .export import canonical, sha256_file


FORMAT = "nullvector-android-coupled-ensemble-int8/1.0.0"
DEFAULT_PARENT = PROJECT_ROOT / "outputs/android_ensemble_v2/export_002"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/android_ensemble_v2/int8_001"
MODEL_NAMES = ("macro", "colony", "society", "timeline", "counterfactual")


def _feeds(seed: int, count: int = 16) -> dict[str, list[dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    rows: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in MODEL_NAMES}
    for index in range(count):
        current = rng.random((1, 32, 32, 32), dtype=np.float32)
        previous = np.clip(current + rng.normal(0, .05, current.shape), 0, 1).astype(np.float32)
        global_state = rng.random((1, 44), dtype=np.float32)
        previous_global = np.clip(global_state + rng.normal(0, .05, global_state.shape), 0, 1).astype(np.float32)
        rows["macro"].append({"current": current, "previous": previous, "global_state": global_state, "previous_global": previous_global})
        features = rng.normal(0, .55, (1, 32, 64)).astype(np.float32)
        mask = np.zeros((1, 32), dtype=np.bool_); mask[:, : 5 + index % 24] = True
        rows["colony"].append({"features": features, "mask": mask})
        rows["society"].append({"features": rng.normal(0, .65, (1, 64)).astype(np.float32)})
        sequence = rng.normal(0, .5, (1, 24, 64)).astype(np.float32)
        rows["timeline"].append({"sequence": sequence})
        rows["counterfactual"].append({"sequence": np.repeat(sequence, 5, axis=0), "action": np.arange(5, dtype=np.int64)})
    return rows


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _benchmark(session: ort.InferenceSession, feed: dict[str, np.ndarray], steps: int = 12) -> float:
    for _ in range(2): session.run(None, feed)
    began = time.perf_counter()
    for _ in range(steps): session.run(None, feed)
    return (time.perf_counter() - began) * 1000 / steps


def build(output: Path = DEFAULT_OUTPUT, parent: Path = DEFAULT_PARENT, model_names: tuple[str, ...] = MODEL_NAMES) -> dict[str, object]:
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_bytes = (parent / "manifest.json").read_bytes(); manifest = json.loads(manifest_bytes)
    if manifest_bytes != canonical(manifest) or manifest.get("status") != "android_coupled_ensemble_export_ready":
        raise ValueError("FP32 Android ensemble authority is not ready")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 30); output.mkdir(parents=True)
    if not model_names or any(name not in MODEL_NAMES for name in model_names):
        raise ValueError("unknown or empty model selection")
    rows = _feeds(0x51414E44524F4944); records: dict[str, object] = {}
    for name in model_names:
        source = parent / manifest["models"][name]["path"]
        if sha256_file(source) != manifest["models"][name]["sha256"]:
            raise ValueError(f"FP32 parent drifted: {name}")
        target = output / f"{name}_int8.onnx"
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8, per_channel=True,
                         reduce_range=False, op_types_to_quantize=["MatMul", "Gemm"])
        onnx.checker.check_model(onnx.load(target))
        fp, quant = _session(source), _session(target); absolute = []; relative = []; decisions = []
        for feed in rows[name]:
            expected, actual = fp.run(None, feed), quant.run(None, feed)
            for expected_value, actual_value in zip(expected, actual):
                delta = np.abs(actual_value.astype(np.float32) - expected_value.astype(np.float32))
                absolute.append(float(delta.mean())); relative.append(float(delta.mean() / max(float(np.abs(expected_value).mean()), 1e-4)))
                if expected_value.ndim >= 2 and expected_value.shape[-1] <= 32:
                    decisions.append(float(np.mean(np.argmax(expected_value, axis=-1) == np.argmax(actual_value, axis=-1))))
        parent_bytes, quant_bytes = source.stat().st_size, target.stat().st_size
        records[name] = {
            "path": target.name, "bytes": quant_bytes, "sha256": sha256_file(target),
            "parent_sha256": sha256_file(source), "compression_ratio": quant_bytes / parent_bytes,
            "mean_absolute_error": float(np.mean(absolute)), "mean_relative_error": float(np.mean(relative)),
            "decision_agreement": float(np.mean(decisions)) if decisions else 1.0,
            "desktop_cpu_milliseconds": _benchmark(quant, rows[name][0]),
        }
    total = sum(int(row["bytes"]) for row in records.values())
    gates = {
        "all_models_load": True,
        "size_reduced": all(float(row["compression_ratio"]) < .9 for row in records.values()),
        "mean_absolute_error_below_0_03": all(float(row["mean_absolute_error"]) < .03 for row in records.values()),
        "mean_relative_error_below_0_08": all(float(row["mean_relative_error"]) < .08 for row in records.values()),
        "decision_agreement_above_0_97": all(float(row["decision_agreement"]) >= .97 for row in records.values()),
    }
    payload: dict[str, object] = {
        "format": FORMAT, "status": "android_int8_ensemble_ready" if all(gates.values()) else "quality_failed",
        "parent_manifest_sha256": manifest["manifest_sha256"], "calibration": "deterministic heldout contract probes", "model_names": list(model_names),
        "probe_rows_per_model": len(rows["macro"]), "models": records, "total_bytes": total,
        "total_mib": total / 1024**2, "gates": gates,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    (output / "manifest.json").write_bytes(canonical(payload)); return payload


def assemble(output: Path, parent: Path, parts: list[Path]) -> dict[str, object]:
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output.exists():
        raise FileExistsError(output)
    parent_manifest = json.loads((parent / "manifest.json").read_bytes()); records: dict[str, object] = {}; sources: dict[str, Path] = {}
    for part in parts:
        part = Path(part).resolve(); encoded = (part / "manifest.json").read_bytes(); payload = json.loads(encoded)
        if encoded != canonical(payload) or payload.get("status") != "android_int8_ensemble_ready" or not all(payload.get("gates", {}).values()):
            raise ValueError(f"unaccepted quantized part: {part}")
        if payload.get("parent_manifest_sha256") != parent_manifest.get("manifest_sha256"):
            raise ValueError(f"quantized parent mismatch: {part}")
        for name, record in payload["models"].items():
            if name in records or name not in MODEL_NAMES:
                raise ValueError(f"duplicate or unknown quantized model: {name}")
            source = part / record["path"]
            if sha256_file(source) != record["sha256"] or source.stat().st_size != record["bytes"]:
                raise ValueError(f"quantized artifact drifted: {name}")
            records[name], sources[name] = record, source
    if set(records) != set(MODEL_NAMES):
        raise ValueError(f"quantized ensemble is incomplete: {sorted(records)}")
    total = sum(int(row["bytes"]) for row in records.values()); gates = {
        "all_five_models": True, "under_100_mib": total < 100 * 1024**2,
        "all_part_gates": all(float(row["mean_absolute_error"]) < .03 and float(row["decision_agreement"]) >= .97 for row in records.values()),
    }
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=total * 2); output.mkdir(parents=True)
    for name in MODEL_NAMES:
        shutil.copy2(sources[name], output / records[name]["path"])
    payload = {
        "format": FORMAT, "status": "android_int8_ensemble_ready" if all(gates.values()) else "quality_failed",
        "parent_manifest_sha256": parent_manifest["manifest_sha256"], "models": records,
        "total_bytes": total, "total_mib": total / 1024**2, "gates": gates,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest(); (output / "manifest.json").write_bytes(canonical(payload)); return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--only", choices=MODEL_NAMES); parser.add_argument("--part", action="append", type=Path, default=[]); args = parser.parse_args(argv)
    if args.part:
        print(json.dumps(assemble(args.output, args.parent, args.part), indent=2, sort_keys=True)); return 0
    selected = (args.only,) if args.only else MODEL_NAMES
    print(json.dumps(build(args.output, args.parent, selected), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
