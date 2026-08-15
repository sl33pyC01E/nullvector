from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..cellular_motion.contract import MOTION_SPECS
from ..multifield_style_motion.hashing import canonical_json_bytes
from ..safety import require_disk_floor
from .contract import MODEL_FORMAT, NeuralCellMotionConfig, model_source_sha256
from .dataset import load_corpus_manifest, sha256_file, validate_corpus
from .model import NeuralCellMotionUNet, neural_motion_loss


SMOKE_FORMAT = "nullvector-neural-cell-motion-training-smoke-v1"
SMOKE_SEED = 0x4E434D4F54494F4E


def _state_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256(b"nullvector-neural-motion-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous(); digest.update(name.encode("utf-8") + b"\0"); digest.update(str(value.dtype).encode("ascii") + b"\0"); digest.update(np.asarray(value.shape, dtype="<u8").tobytes()); digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try: torch.save(payload, temporary); os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _smoke_batch(corpus: Path) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    manifest = load_corpus_manifest(corpus)
    if len({record["family_id"] for record in manifest["records"]}) != 5: raise ValueError("Neural motion smoke requires all five families.")
    static_rows: list[np.ndarray] = []; previous_rows: list[np.ndarray] = []; target_rows: list[np.ndarray] = []; families: list[int] = []; motions: list[int] = []; facings: list[int] = []; phases: list[float] = []
    for family_id in range(5):
        record = next(item for item in manifest["records"] if item["family_id"] == family_id)
        with np.load(Path(corpus) / record["path"], allow_pickle=False) as archive:
            # Different semantic region for each family, while preserving the
            # exact recurrent predecessor recorded by the corpus.
            index = (family_id * 173 + 37) % 944; metadata = archive["indices"][index]; previous_index = int(archive["previous_index"][index])
            static_rows.append(archive["features"].astype(np.float32)); target_rows.append(archive["targets"][index].astype(np.float32)); previous_rows.append(archive["targets"][previous_index].astype(np.float32))
        motion, facing, frame = map(int, metadata[:3]); frame_count = MOTION_SPECS[list(MOTION_SPECS)[motion]][0]
        families.append(family_id); motions.append(motion); facings.append(facing); phases.append(frame / max(1, frame_count - 1))
    return torch.from_numpy(np.stack(static_rows)), torch.from_numpy(np.stack(previous_rows)), torch.from_numpy(np.stack(target_rows)), torch.tensor(families), torch.tensor(motions), torch.tensor(facings), torch.tensor(phases, dtype=torch.float32)


def run_cpu_smoke(corpus: Path, output: Path, *, steps: int = 3) -> dict[str, Any]:
    corpus = Path(corpus).resolve(); output = Path(output).resolve()
    if type(steps) is not int or not 1 <= steps <= 8: raise ValueError("Neural motion smoke step count drifted.")
    if output.exists(): raise FileExistsError(output)
    validate_corpus(corpus, replay=False); require_disk_floor(output.parent, floor_gb=100, planned_bytes=512 * 1024**2); output.mkdir(parents=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(SMOKE_SEED); np.random.seed(SMOKE_SEED & 0xffffffff)
    config = NeuralCellMotionConfig(base_channels=24, channel_multipliers=(1, 2, 3), blocks_per_level=1, condition_dim=96, attention_heads=4, dropout=0)
    model = NeuralCellMotionUNet(config); optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5); batch = _smoke_batch(corpus); history: list[dict[str, float | int]] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True); predicted = model(batch[0], batch[1], batch[3], batch[4], batch[5], batch[6]); loss, pieces = neural_motion_loss(predicted, batch[2], batch[1], batch[0]); loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)): raise FloatingPointError("Neural motion smoke became non-finite.")
        optimizer.step(); history.append({"step": step + 1, **{name: round(float(value), 8) for name, value in pieces.items()}, "gradient_norm": round(float(gradient), 8)})
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; corpus_manifest = load_corpus_manifest(corpus)
    checkpoint = {"format": MODEL_FORMAT, "model_source_sha256": model_source_sha256(), "corpus_semantic_sha256": corpus_manifest["semantic_sha256"], "config": config.to_dict(), "steps": steps, "model_state": state, "model_state_sha256": _state_sha256(state), "history": history}
    checkpoint_path = output / "smoke_checkpoint.pt"; _atomic_torch(checkpoint_path, checkpoint)
    report: dict[str, Any] = {"format": SMOKE_FORMAT, "status": "passed", "model_source_sha256": model_source_sha256(), "corpus": {"path": corpus.relative_to(Path(__file__).resolve().parents[2]).as_posix(), "semantic_sha256": corpus_manifest["semantic_sha256"]}, "config": config.to_dict(), "production_config": NeuralCellMotionConfig().to_dict(), "smoke_parameters": model.parameter_count, "production_parameters": NeuralCellMotionUNet().parameter_count, "steps": steps, "batch_families": [0, 1, 2, 3, 4], "history": history, "checkpoint": {"path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "model_state_sha256": checkpoint["model_state_sha256"]}, "gates": {"all_values_finite": all(math.isfinite(float(value)) for row in history for key, value in row.items() if key != "step"), "all_five_families_in_batch": True, "gradient_nonzero": all(row["gradient_norm"] > 0 for row in history), "loss_improved_on_fixed_batch": history[-1]["loss"] < history[0]["loss"] if steps > 1 else True, "outside_body_exact_zero": float(model(batch[0], batch[1], batch[3], batch[4], batch[5], batch[6]).detach().mul(1 - batch[0][:, :1]).abs().max()) == 0}}
    if not all(report["gates"].values()): raise ValueError("Neural motion CPU smoke failed a gate.")
    report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest(); _atomic_bytes(output / "smoke_manifest.json", canonical_json_bytes(report)); return validate_cpu_smoke(output)


def validate_cpu_smoke(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); path = output / "smoke_manifest.json"; encoded = path.read_bytes(); report = json.loads(encoded)
    required = {"format", "status", "model_source_sha256", "corpus", "config", "production_config", "smoke_parameters", "production_parameters", "steps", "batch_families", "history", "checkpoint", "gates", "semantic_sha256"}
    if encoded != canonical_json_bytes(report) or set(report) != required or report["format"] != SMOKE_FORMAT or report["status"] != "passed" or report["model_source_sha256"] != model_source_sha256() or report["semantic_sha256"] != hashlib.sha256(canonical_json_bytes({key: value for key, value in report.items() if key != "semantic_sha256"})).hexdigest() or not all(report["gates"].values()): raise ValueError("Neural motion smoke manifest drifted.")
    checkpoint_path = output / report["checkpoint"]["path"]
    if checkpoint_path.stat().st_size != report["checkpoint"]["bytes"] or sha256_file(checkpoint_path) != report["checkpoint"]["sha256"]: raise ValueError("Neural motion smoke checkpoint artifact drifted.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"format", "model_source_sha256", "corpus_semantic_sha256", "config", "steps", "model_state", "model_state_sha256", "history"} or checkpoint["format"] != MODEL_FORMAT or checkpoint["model_source_sha256"] != model_source_sha256() or checkpoint["model_state_sha256"] != _state_sha256(checkpoint["model_state"]) or checkpoint["history"] != report["history"]: raise ValueError("Neural motion smoke checkpoint semantic state drifted.")
    model = NeuralCellMotionUNet(NeuralCellMotionConfig(**{**checkpoint["config"], "channel_multipliers": tuple(checkpoint["config"]["channel_multipliers"])})); model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.parameter_count != report["smoke_parameters"] or report["production_parameters"] < 20_000_000: raise ValueError("Neural motion smoke model census drifted.")
    return {"passed": True, "steps": report["steps"], "smoke_parameters": report["smoke_parameters"], "production_parameters": report["production_parameters"], "model_state_sha256": checkpoint["model_state_sha256"], "semantic_sha256": report["semantic_sha256"]}
