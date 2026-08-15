from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from ..config import PROJECT_ROOT
from ..multifield_style_motion.hashing import canonical_json_bytes
from .contract import model_source_sha256
from .dataset import _read_canonical_json, sha256_file
from .evaluation import acceptance_gates, evaluate_recurrent_split, evaluation_source_sha256, validate_evaluation_result
from .model import NeuralCellMotionUNet
from .production import CONTRACT_NAME, _config_from_dict, _load_checkpoint, _semantic, checkpoint_name
from .training import _atomic_bytes


EVALUATION_REPORT_FORMAT = "nullvector-neural-cell-motion-checkpoint-evaluation-v1"


def evaluation_name(step: int) -> str:
    return f"motion_segment_{step:07d}.evaluation.json"


def evaluate_segment(output: Path, *, step: int) -> dict[str, Any]:
    output = Path(output).resolve(); contract = _read_canonical_json(output / CONTRACT_NAME)
    if contract.get("source_sha256") != model_source_sha256() or contract.get("semantic_sha256") != _semantic({key: value for key, value in contract.items() if key != "semantic_sha256"}):
        raise ValueError("Neural motion evaluation authority drifted.")
    checkpoint_path = output / checkpoint_name(step); checkpoint = _load_checkpoint(checkpoint_path, contract, expected_step=step)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Neural motion evaluation requires deterministic CUDA BF16.")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(contract["seed"]); torch.cuda.manual_seed_all(contract["seed"])
    device = torch.device("cuda", 0); model = NeuralCellMotionUNet(_config_from_dict(contract["model"])).to(device); model.load_state_dict(checkpoint["ema_state"], strict=True)
    corpus = PROJECT_ROOT / contract["corpus"]["path"]
    validation = evaluate_recurrent_split(corpus, model, "validation", device=device, precision="bf16"); validate_evaluation_result(validation)
    validation_gates = acceptance_gates(validation); final = step == contract["total_steps"]
    test = evaluate_recurrent_split(corpus, model, "test", device=device, precision="bf16") if final else None
    if test is not None: validate_evaluation_result(test)
    test_gates = acceptance_gates(test) if test is not None else None
    report: dict[str, Any] = {
        "format": EVALUATION_REPORT_FORMAT,
        "status": "passed",
        "source_sha256": model_source_sha256(),
        "evaluation_source_sha256": evaluation_source_sha256(),
        "contract_semantic_sha256": contract["semantic_sha256"],
        "corpus_semantic_sha256": contract["corpus"]["semantic_sha256"],
        "checkpoint": {"path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "step": step, "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]},
        "validation": validation,
        "test": test,
        "gates": {"validation": validation_gates, "test": test_gates},
        "promotion_eligible": final and all(validation_gates.values()) and test_gates is not None and all(test_gates.values()),
    }
    report["semantic_sha256"] = _semantic(report); _atomic_bytes(output / evaluation_name(step), canonical_json_bytes(report))
    return validate_evaluation_report(output, step=step)


def validate_evaluation_report(output: Path, *, step: int) -> dict[str, Any]:
    output = Path(output).resolve(); contract = _read_canonical_json(output / CONTRACT_NAME); path = output / evaluation_name(step); report = _read_canonical_json(path, maximum_bytes=16 * 1024 * 1024)
    if contract.get("source_sha256") != model_source_sha256() or contract.get("semantic_sha256") != _semantic({key: value for key, value in contract.items() if key != "semantic_sha256"}):
        raise ValueError("Neural motion evaluation contract drifted.")
    required = {"format", "status", "source_sha256", "evaluation_source_sha256", "contract_semantic_sha256", "corpus_semantic_sha256", "checkpoint", "validation", "test", "gates", "promotion_eligible", "semantic_sha256"}
    if set(report) != required or report["format"] != EVALUATION_REPORT_FORMAT or report["status"] != "passed" or report["source_sha256"] != model_source_sha256() or report["evaluation_source_sha256"] != evaluation_source_sha256() or report["contract_semantic_sha256"] != contract["semantic_sha256"] or report["corpus_semantic_sha256"] != contract["corpus"]["semantic_sha256"] or report["semantic_sha256"] != _semantic({key: value for key, value in report.items() if key != "semantic_sha256"}):
        raise ValueError("Neural motion evaluation report provenance drifted.")
    checkpoint_path = output / checkpoint_name(step); checkpoint = _load_checkpoint(checkpoint_path, contract, expected_step=step); checkpoint_record = report["checkpoint"]
    if not isinstance(checkpoint_record, dict) or set(checkpoint_record) != {"path", "bytes", "sha256", "step", "model_state_sha256", "ema_state_sha256"} or checkpoint_record != {"path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "step": step, "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]}:
        raise ValueError("Neural motion evaluation checkpoint binding drifted.")
    validation = validate_evaluation_result(report["validation"]); validation_gates = acceptance_gates(validation); final = step == contract["total_steps"]
    if final:
        test = validate_evaluation_result(report["test"]); test_gates = acceptance_gates(test)
    elif report["test"] is not None:
        raise ValueError("Neural motion non-final evaluation contains test data.")
    else: test_gates = None
    expected_gates = {"validation": validation_gates, "test": test_gates}; eligible = final and all(validation_gates.values()) and test_gates is not None and all(test_gates.values())
    if report["gates"] != expected_gates or report["promotion_eligible"] is not eligible:
        raise ValueError("Neural motion evaluation gate derivation drifted.")
    return {"passed": True, "step": step, "promotion_eligible": eligible, "validation_loss": validation["metrics"]["loss"], "test_loss": report["test"]["metrics"]["loss"] if report["test"] is not None else None, "semantic_sha256": report["semantic_sha256"]}
