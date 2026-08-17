from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torch import Tensor, nn

from ..config import PROJECT_ROOT
from ..mobile_cell_nca_v1.contract import CORPUS, CORPUS_SHA256
from ..organism_cell_vae_v1.contract import CELL_FEATURES, MAX_CELLS, canonical, sha256_file
from ..organism_cell_vae_v1.training import load_final


FORMAT = "nullvector-android-neural-organism-v1/1.0.0"
DEFAULT_RELEASE = PROJECT_ROOT / "outputs/organism_cell_vae_v1/production_v3_calibrated"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/android_neural_organism_v1/machine_heldout_44"
SAMPLE_INDEX = 44
FAMILY_NAMES = ("humanoid", "animalian", "plantlike", "anomaly", "machine")


class _RasterGraph(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(); self.model = model

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        return self.model(features, mask > .5, stochastic=False).rgba


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def _load_cellular_sample(index: int) -> dict[str, Any]:
    if _sha(CORPUS) != CORPUS_SHA256: raise ValueError("cellular corpus authority drifted")
    with np.load(CORPUS, allow_pickle=False) as archive:
        required = {"static", "initial_state", "live_bonds", "family_id", "sample_id"}
        if set(archive.files) != required: raise ValueError("cellular corpus member contract drifted")
        if not 0 <= index < len(archive["family_id"]): raise IndexError(index)
        result: dict[str, Any] = {}
        for name in required:
            selected = archive[name][index]
            result[name] = selected.item() if np.ndim(selected) == 0 else np.ascontiguousarray(selected)
        return result


def cellular_static_to_vae_features(static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project the physiology raster into the accepted cell-VAE feature vocabulary.

    Geometry and family are exact. Tissue, physiological traits, graph degree and
    side are preserved. Appendage kind is a conservative geometric hint because
    the cellular corpus predates the newer articulated appendage registry.
    """
    if static.shape != (85, 48, 48) or static.dtype != np.float32 or not np.isfinite(static).all():
        raise ValueError("cellular static tensor drifted")
    body = static[0] > .5; yx = np.argwhere(body); count = len(yx)
    if not 1 <= count <= MAX_CELLS: raise ValueError("cell count exceeds cell-VAE contract")
    values = np.zeros((1, MAX_CELLS, CELL_FEATURES), dtype=np.float32)
    mask = np.zeros((1, MAX_CELLS), dtype=np.float32)
    y = yx[:, 0]; x = yx[:, 1]; values[0, :count, 0] = x / 47 * 2 - 1; values[0, :count, 1] = y / 47 * 2 - 1
    tissue = np.argmax(static[1:15, y, x], axis=0) + 1
    tissue[static[1:15, y, x].max(axis=0) <= .5] = 0
    values[0, np.arange(count), 2 + tissue] = 1
    family = int(np.argmax(static[23:28, y, x].mean(axis=1))); values[0, :count, 17 + family] = 1
    values[0, :count, 22:30] = static[52:60, y, x].T
    values[0, :count, 30:33] = static[66:69, y, x].T
    values[0, :count, 33:37] = static[69:73, y, x].T
    # Cell-VAE appendage vocabulary: chassis, leg, arm, tail, root, frond,
    # tendril, hardpoint, wheel. Keep the core as chassis and classify only
    # peripheral cells; machines preferentially expose hardpoints/wheels.
    center_x = float(np.median(x)); center_y = float(np.median(y)); dx = x - center_x; dy = y - center_y
    radius = np.sqrt(dx * dx + dy * dy); peripheral = radius > np.quantile(radius, .60)
    kind = np.zeros(count, dtype=np.int64)
    lower = peripheral & (dy > 1); lateral = peripheral & (np.abs(dx) >= np.abs(dy))
    if family == 4:
        kind[lower] = 8; kind[lateral & ~lower] = 7
    elif family == 2:
        kind[lower] = 4; kind[lateral & ~lower] = 5
    elif family == 3:
        kind[peripheral] = 6
    else:
        kind[lower] = 1; kind[lateral & ~lower] = 2
    values[0, np.arange(count), 37 + kind] = 1
    values[0, :count, 46] = np.where(dx < -.5, -1, np.where(dx > .5, 1, 0))
    values[0, :count, 47] = 0; values[0, :count, 48] = 1
    values[0, :count, 49] = np.clip(static[52:60, y, x].max(axis=0), 0, 1)
    values[0, :count, 50] = peripheral; values[0, :count, 51] = 1; mask[0, :count] = 1
    return values, mask


def _rgba_png(value: np.ndarray) -> bytes:
    rgba = np.clip(np.rint(np.transpose(value[0], (1, 2, 0)) * 255), 0, 255).astype(np.uint8)
    from io import BytesIO
    stream = BytesIO(); Image.fromarray(rgba, "RGBA").save(stream, format="PNG", compress_level=9); return stream.getvalue()


def export_neural_organism(output: Path = DEFAULT_OUTPUT, *, sample_index: int = SAMPLE_INDEX) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    sample = _load_cellular_sample(sample_index); family_id = int(sample["family_id"])
    if sample_index == SAMPLE_INDEX and family_id != 4: raise ValueError("held-out machine identity drifted")
    features, mask = cellular_static_to_vae_features(sample["static"])
    model, _, _ = load_final(DEFAULT_RELEASE); graph = _RasterGraph(model).eval()
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}"); staging.mkdir(parents=True)
    try:
        onnx_path = staging / "organism_cell_vae_fp32.onnx"
        with torch.inference_mode(): expected = graph(torch.from_numpy(features), torch.from_numpy(mask)).numpy()
        torch.onnx.export(graph, (torch.from_numpy(features), torch.from_numpy(mask)), onnx_path,
                          input_names=("features", "mask"), output_names=("rgba",), opset_version=18,
                          do_constant_folding=True, dynamo=False)
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
        feed = {"features": features, "mask": mask}; actual = session.run(None, feed)[0]
        max_abs = float(np.max(np.abs(actual - expected)))
        for _ in range(3): session.run(None, feed)
        began = time.perf_counter(); runs = 12
        for _ in range(runs): session.run(None, feed)
        milliseconds = (time.perf_counter() - began) * 1000 / runs
        (staging / "organism_vae_features.f32").write_bytes(features.astype("<f4", copy=False).tobytes())
        (staging / "organism_vae_mask.f32").write_bytes(mask.astype("<f4", copy=False).tobytes())
        (staging / "cell_static.f32").write_bytes(sample["static"].astype("<f4", copy=False).tobytes())
        (staging / "cell_state.f32").write_bytes(sample["initial_state"].astype("<f4", copy=False).tobytes())
        (staging / "cell_bonds.f32").write_bytes(sample["live_bonds"].astype("<f4", copy=False).tobytes())
        (staging / "organism_vae_preview.png").write_bytes(_rgba_png(actual))
        artifacts = {}
        for path in sorted(staging.iterdir()): artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
        manifest = {
            "format": FORMAT, "sample_index": sample_index, "sample_id": str(sample["sample_id"]),
            "family_id": family_id, "family": FAMILY_NAMES[family_id], "held_out_from_cell_nca_training": sample_index == SAMPLE_INDEX,
            "cell_count": int(mask.sum()), "vae_release": DEFAULT_RELEASE.relative_to(PROJECT_ROOT).as_posix(),
            "vae_checkpoint_sha256": sha256_file(DEFAULT_RELEASE / "cell_vae_0001200.pt"),
            "cellular_corpus_sha256": CORPUS_SHA256, "onnx_max_abs_parity": max_abs,
            "desktop_cpu_milliseconds": milliseconds, "artifacts": artifacts,
            "gates": {"held_out_machine": sample_index == SAMPLE_INDEX and family_id == 4,
                      "cell_count_within_vae_contract": int(mask.sum()) <= MAX_CELLS,
                      "onnx_parity_below_2e_5": max_abs < 2e-5,
                      "desktop_cpu_under_50ms": milliseconds < 50,
                      "finite_rgba": bool(np.isfinite(actual).all()), "rgba_in_unit_interval": bool((actual >= 0).all() and (actual <= 1).all())},
        }
        manifest["status"] = "ready" if all(manifest["gates"].values()) else "failed"
        manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_bytes(canonical(manifest)); os.replace(staging, output)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    return validate_neural_organism(output)


def validate_neural_organism(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve(); encoded = (output / "manifest.json").read_bytes(); manifest = json.loads(encoded)
    if encoded != canonical(manifest) or manifest.get("format") != FORMAT: raise ValueError("Android organism manifest drifted")
    expected = hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"})).hexdigest()
    if manifest.get("manifest_sha256") != expected or not all(manifest.get("gates", {}).values()): raise ValueError("Android organism gates failed")
    for name, row in manifest["artifacts"].items():
        path = output / name
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]: raise ValueError(f"Android organism artifact drifted: {name}")
    return {"passed": True, "status": manifest["status"], "sample_id": manifest["sample_id"], "family": manifest["family"], "cell_count": manifest["cell_count"], "desktop_cpu_milliseconds": manifest["desktop_cpu_milliseconds"], "manifest_sha256": manifest["manifest_sha256"]}
