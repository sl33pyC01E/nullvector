from __future__ import annotations

import json

import onnxruntime as ort

from forge.android_foundation_v1 import build, validate


def test_android_foundation_exports_live_batched_controller(tmp_path):
    output = tmp_path / "foundation"
    report = build(output)
    assert validate(output)["semantic_sha256"] == report["semantic_sha256"]
    assert report["scope"]["frame_loop"] is False
    anatomy = json.loads((output / "foundation_anatomy.json").read_bytes())
    assert [row["family"] for row in anatomy["organisms"]] == anatomy["families"]
    assert all(row["cells"] and row["skeleton"]["muscles"] and row["appendages"] for row in anatomy["organisms"])
    assert all(len(cell["neural_style"]) == 7 for row in anatomy["organisms"] for cell in row["cells"])
    session = ort.InferenceSession(str(output / "grounded_feedback_fp32.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[0] == "batch"
    assert [value.name for value in session.get_outputs()] == ["muscle_activation", "contact_logits", "body_velocity"]
