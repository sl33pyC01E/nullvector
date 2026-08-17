from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ..config import PROJECT_ROOT
from ..nature_colony_nn import NeuralColonyRuntime
from ..nature_counterfactual_nn import NeuralCounterfactualRuntime
from ..nature_macro_nn import NeuralMacroPatchRuntime
from ..nature_society_nn import NeuralSocietyRuntime
from ..nature_timeline_nn import NeuralTimelineRuntime
from ..playable_neural_runtime_v1.runtime import _component_table


FORMAT = "nullvector-android-coupled-ensemble/2.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "android_ensemble_v2" / "export_001"


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_graph(module, inputs, input_names, output_names, path: Path) -> None:
    torch.onnx.export(
        module.eval(), inputs, path, input_names=input_names, output_names=output_names,
        opset_version=18, do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))


def benchmark(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, object]:
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    session.run(None, feeds)
    began = time.perf_counter(); outputs = session.run(None, feeds); elapsed = time.perf_counter() - began
    return {"milliseconds": elapsed * 1000, "output_shapes": [list(value.shape) for value in outputs]}


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = _component_table()
    artifact = lambda name: PROJECT_ROOT / rows[name]["artifact"]["path"]

    macro = NeuralMacroPatchRuntime.from_checkpoint(artifact("macro_patch"), device="cpu").model
    colony = NeuralColonyRuntime.from_checkpoint(artifact("colony"), device="cpu").model
    society = NeuralSocietyRuntime.from_checkpoint(artifact("society"), device="cpu").model
    timeline = NeuralTimelineRuntime.from_checkpoint(artifact("timeline"), device="cpu").model
    counterfactual = NeuralCounterfactualRuntime.from_checkpoint(artifact("counterfactual"), device="cpu").model

    rng = np.random.default_rng(0x414E44524F494432)
    tensors = {
        "macro": (
            torch.from_numpy(rng.random((1, 32, 32, 32), dtype=np.float32)),
            torch.from_numpy(rng.random((1, 32, 32, 32), dtype=np.float32)),
            torch.from_numpy(rng.random((1, 44), dtype=np.float32)),
            torch.from_numpy(rng.random((1, 44), dtype=np.float32)),
        ),
        "colony": (torch.from_numpy(rng.random((1, 32, 64), dtype=np.float32)), torch.ones(1, 32, dtype=torch.bool)),
        "society": (torch.from_numpy(rng.random((1, 64), dtype=np.float32)),),
        "timeline": (torch.from_numpy(rng.random((1, 24, 64), dtype=np.float32)),),
        "counterfactual": (torch.from_numpy(rng.random((5, 24, 64), dtype=np.float32)), torch.arange(5, dtype=torch.long)),
    }
    specifications = {
        "macro": (macro, ["current", "previous", "global_state", "previous_global"], ["next_state", "next_global", "gate", "gate_logits", "global_gate", "global_gate_logits"]),
        "colony": (colony, ["features", "mask"], ["role_logits", "actions"]),
        "society": (society, ["features"], ["activity_logits", "labor_logits", "diplomacy_logits", "project_logits"]),
        "timeline": (timeline, ["sequence"], ["next_state", "event_logits", "confidence"]),
        "counterfactual": (counterfactual, ["sequence", "action"], ["next_state", "benefit", "risk"]),
    }
    parent_names = {"macro": "macro_patch", "colony": "colony", "society": "society", "timeline": "timeline", "counterfactual": "counterfactual"}
    records = {}
    for name, (module, inputs, outputs) in specifications.items():
        path = output / f"{name}_fp32.onnx"
        export_graph(module, tensors[name], inputs, outputs, path)
        feeds = {key: value.detach().cpu().numpy() for key, value in zip(inputs, tensors[name])}
        records[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "desktop_cpu_reference": benchmark(path, feeds),
            "parent": rows[parent_names[name]]["artifact"],
        }
    total = sum(record["bytes"] for record in records.values())
    payload = {
        "format": FORMAT,
        "status": "android_coupled_ensemble_export_ready",
        "precision": "fp32",
        "models": records,
        "total_bytes": total,
        "total_mib": total / 1024**2,
        "cadence_hz": {"macro": 1.0, "colony": .25, "society": .05, "timeline": .02, "counterfactual": .02},
        "roles": {
            "macro": "authoritative resource delta",
            "colony": "authoritative member role and transfer policy",
            "society": "authoritative settlement labor and construction policy",
            "timeline": "forecast observer",
            "counterfactual": "intervention observer",
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    (output / "manifest.json").write_bytes(canonical(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
