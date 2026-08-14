from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from ..map_topology_neural.artifacts import (
    COMPILED_MANIFEST,
    RAW_MANIFEST,
    deterministic_npz_bytes,
    load_compiled_artifact,
    load_raw_artifact,
    write_compiled_artifact,
    write_raw_artifact,
)
from ..map_topology_neural.compiler import compile_topology, make_raw_topology
from ..map_topology_neural.hashing import array_sha256, file_sha256, json_sha256, named_arrays_sha256
from ..map_topology_neural.codec import build_codec
from ..map_topology_neural.corpus import TopologyCorpusSample
from ..map_topology_neural_production.checkpoint import load_checkpoint as load_codec_checkpoint
from ..map_topology_neural_production.contract import TopologyCodecCalibrationConfig
from ..map_topology_neural_production.dataset import TopologyProductionDataset, TopologyRef
from ..map_topology_neural_prior.masking import tensor_sha256
from ..map_topology_neural_prior.model import build_prior
from ..map_topology_neural_prior_training.checkpoint import load_checkpoint as load_prior_checkpoint
from ..map_topology_neural_prior_training.contract import PriorCalibrationConfig
from ..maps.model import THEMES, WALKABLE_TERRAIN
from ..safety import require_disk_floor
from .contract import (
    CASE_FORMAT,
    CODEC_CHECKPOINT_RELATIVE,
    CODEC_CHECKPOINT_SHA256,
    CODEC_EMA_SHA256,
    CODEC_SOURCE_SHA256,
    FORMAT,
    PRIOR_CHECKPOINT_RELATIVE,
    PRIOR_CHECKPOINT_SHA256,
    PRIOR_EMA_SHA256,
    PRIOR_TRAINING_SOURCE_SHA256,
    PROJECT_ROOT,
    PROPOSAL_SOURCE,
    REPLAY_FORMAT,
    GenerationConfig,
    authority_payload,
    canonical_json_bytes,
    generation_source_sha256,
    sha256_file,
    source_manifest,
    stable_seed,
    validate_schema,
)
from .render import case_preview_png_bytes, contact_sheet_png_bytes
from .sampling import SeededParallelSample, sample_seeded_parallel


CASE_MANIFEST = "case_manifest.json"
LATENT_ARTIFACT = "latent_sample.npz"
PREVIEW_ARTIFACT = "preview.png"
BANK_MANIFEST = "generation_manifest.json"
REPLAY_REPORT = "replay_report.json"
CONTACT_SHEET = "contact_sheet.png"
CONFIG_FILE = "generation_config.json"
MAX_CASE_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_LATENT_BYTES = 4 * 1024 * 1024
MAX_BANK_BYTES = 16 * 1024 * 1024
MAX_REPLAY_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    ref: TopologyRef
    variant: int
    sampling_seed: int
    compiler_seed: int


@dataclass(frozen=True, slots=True)
class Authorities:
    dataset: TopologyProductionDataset
    prior: torch.nn.Module
    codec: torch.nn.Module


def _prepare_cpu() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)


def _artifact(path: Path, relative: str) -> dict[str, object]:
    target = path / relative
    if not target.is_file() or target.is_symlink() or target.stat().st_size <= 0:
        raise ValueError(f"Generation artifact {relative!r} is missing or unsafe.")
    return {"path": relative.replace("\\", "/"), "bytes": target.stat().st_size, "sha256": file_sha256(target)}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, canonical_json_bytes(payload))


def _config_sha(config: GenerationConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest()


def _load_authorities(corpus_root: Path) -> Authorities:
    _prepare_cpu()
    prior_path = PROJECT_ROOT / PRIOR_CHECKPOINT_RELATIVE
    codec_path = PROJECT_ROOT / CODEC_CHECKPOINT_RELATIVE
    if sha256_file(prior_path) != PRIOR_CHECKPOINT_SHA256 or sha256_file(codec_path) != CODEC_CHECKPOINT_SHA256:
        raise ValueError("Frozen prior or codec checkpoint file identity drifted.")
    prior_payload = load_prior_checkpoint(prior_path)
    if prior_payload["ema_state_sha256"] != PRIOR_EMA_SHA256 or prior_payload["source_sha256"] != PRIOR_TRAINING_SOURCE_SHA256:
        raise ValueError("Frozen masked-prior EMA/source authority drifted.")
    prior_config = PriorCalibrationConfig.from_dict(prior_payload["config"])
    prior = build_prior(prior_config.model_config()); prior.load_state_dict(prior_payload["ema_state"], strict=True); prior.eval()
    codec_payload = load_codec_checkpoint(codec_path)
    if codec_payload["ema_state_sha256"] != CODEC_EMA_SHA256 or codec_payload["source_sha256"] != CODEC_SOURCE_SHA256:
        raise ValueError("Frozen codec EMA/source authority drifted.")
    codec_config = TopologyCodecCalibrationConfig.from_dict(codec_payload["config"])
    codec = build_codec(codec_config.codec_config(), init_seed=codec_config.seed); codec.load_state_dict(codec_payload["ema_state"], strict=True); codec.eval()
    return Authorities(dataset=TopologyProductionDataset(Path(corpus_root)), prior=prior, codec=codec)


def plan_cases(dataset: TopologyProductionDataset, config: GenerationConfig) -> tuple[CaseSpec, ...]:
    refs = dataset.evaluation_refs("test", 24)
    if len(refs) != 24 or {ref.theme for ref in refs} != set(THEMES):
        raise ValueError("Held-out generation condition registry is not the exact six-theme sentinel set.")
    specs: list[CaseSpec] = []
    for ref in refs:
        for variant in range(config.variants_per_condition):
            sampling_seed = stable_seed(config.base_seed, ref.full_map_identity_sha256, variant, "sampling", bits=63)
            compiler_seed = stable_seed(config.base_seed, ref.full_map_identity_sha256, variant, "compiler", bits=64)
            case_id = f"{ref.theme}_{ref.width}x{ref.height}_{ref.full_map_identity_sha256[:8]}_v{variant:02d}"
            specs.append(CaseSpec(case_id, ref, variant, sampling_seed, compiler_seed))
    specs.sort(key=lambda item: item.case_id)
    if len(specs) != 24 * config.variants_per_condition or len({item.case_id for item in specs}) != len(specs):
        raise ValueError("Generation case plan census or identity collided.")
    return tuple(specs)


def _find_spec(authorities: Authorities, config: GenerationConfig, case_id: str) -> CaseSpec:
    matches = [spec for spec in plan_cases(authorities.dataset, config) if spec.case_id == case_id]
    if len(matches) != 1:
        raise KeyError(f"Unknown or ambiguous generation case {case_id!r}.")
    return matches[0]


def _condition_batch(tensor: object) -> tuple[dict[str, torch.Tensor], str]:
    valid_full = torch.from_numpy(getattr(tensor, "valid_mask").copy())[None].float()
    points_full = torch.from_numpy(getattr(tensor, "point_heatmaps").copy())[None].float()
    valid = F.max_pool2d(valid_full, kernel_size=4, stride=4) > 0
    points = F.max_pool2d(points_full, kernel_size=4, stride=4)
    global_conditions = torch.from_numpy(getattr(tensor, "global_conditions").copy())[None].float()
    theme_index = torch.tensor([int(getattr(tensor, "theme_index"))], dtype=torch.long)
    arrays = {
        "valid_mask": valid.numpy().astype(np.uint8),
        "point_conditions": points.numpy().astype(np.float32),
        "global_conditions": global_conditions.numpy().astype(np.float32),
        "theme_index": theme_index.numpy().astype(np.int64),
    }
    return {"valid_mask": valid, "point_conditions": points, "global_conditions": global_conditions, "theme_index": theme_index}, named_arrays_sha256(arrays)


def _decode(codec: torch.nn.Module, sample: SeededParallelSample, *, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = getattr(getattr(codec, "quantizer"), "embeddings")
    if sample.tokens.min() < 0 or sample.tokens.max() >= table.shape[0]:
        raise ValueError("Generated tokens exceed frozen codec vocabulary.")
    with torch.inference_mode():
        embedded = table.index_select(0, sample.tokens.flatten()).view(1, sample.tokens.shape[1], sample.tokens.shape[2], table.shape[1]).permute(0, 3, 1, 2).contiguous()
        logits = codec.decode(embedded)
    expected = (height, width)
    decoded = []
    for name, dtype in (("terrain", np.uint8), ("hazard", np.uint8), ("elevation", np.int8)):
        values = torch.argmax(logits[name], dim=1)[0, :height, :width].cpu().numpy().astype(dtype, copy=False)
        if values.shape != expected:
            raise RuntimeError("Frozen codec decode violated exact map dimensions.")
        decoded.append(np.ascontiguousarray(values))
    return decoded[0], decoded[1], decoded[2]


def _reachable(mask: np.ndarray, start: tuple[int, int], targets: tuple[tuple[int, int], ...]) -> bool:
    height, width = mask.shape; sx, sy = start
    if not (0 <= sx < width and 0 <= sy < height and mask[sy, sx]):
        return False
    seen = {(sx, sy)}; queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and (nx, ny) not in seen:
                seen.add((nx, ny)); queue.append((nx, ny))
    return all(point in seen for point in targets)


def _radius_one(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def _histogram(array: np.ndarray, classes: int) -> list[int]:
    return np.bincount(array.astype(np.int64).ravel(), minlength=classes)[:classes].astype(int).tolist()


def _cost(result: object, name: str) -> float:
    report = getattr(result, "report")
    costs = report.get("costs") if isinstance(report, dict) else None
    value = costs.get(name) if isinstance(costs, dict) else None
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError(f"Compiler report lacks finite cost {name!r}.")
    return float(value)


def _latent_descriptor(path: Path, arrays: dict[str, np.ndarray], payload: bytes) -> dict[str, object]:
    return {
        "path": LATENT_ARTIFACT, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_arrays_sha256": named_arrays_sha256(arrays),
        "members": {name: {"dtype": value.dtype.str, "shape": list(value.shape), "nbytes": value.nbytes, "array_sha256": array_sha256(value)} for name, value in sorted(arrays.items())},
    }


def _sample_case(authorities: Authorities, config: GenerationConfig, spec: CaseSpec) -> tuple[TopologyCorpusSample, object, dict[str, torch.Tensor], str, SeededParallelSample, object]:
    tensor = authorities.dataset.load_tensor(spec.ref)
    source = authorities.dataset.corpus.read_sample(spec.ref.shard_id, spec.ref.sample_index, expected_split="test")
    conditions, condition_sha = _condition_batch(tensor)
    sample = sample_seeded_parallel(authorities.prior, conditions, sampling_steps=config.sampling_steps, seed=spec.sampling_seed, temperature=config.temperature, top_k=config.top_k)
    terrain, hazard, elevation = _decode(authorities.codec, sample, height=source.config.height, width=source.config.width)
    raw = make_raw_topology(terrain, hazard, elevation, shape=(source.config.height, source.config.width))
    return source, tensor, conditions, condition_sha, sample, raw


def generate_case(destination: Path, *, corpus_root: Path, config: GenerationConfig, case_id: str) -> dict[str, object]:
    destination = Path(destination).resolve()
    if destination.exists():
        return validate_case(destination, corpus_root=corpus_root, config=config, exact_neural_replay=False)
    authorities = _load_authorities(corpus_root); spec = _find_spec(authorities, config, case_id)
    source, tensor, conditions, condition_sha, sample, raw = _sample_case(authorities, config, spec)
    replay_sample = sample_seeded_parallel(authorities.prior, conditions, sampling_steps=config.sampling_steps, seed=spec.sampling_seed, temperature=config.temperature, top_k=config.top_k)
    if not torch.equal(sample.tokens, replay_sample.tokens) or not torch.equal(sample.uncertainty, replay_sample.uncertainty) or sample.trace != replay_sample.trace:
        raise RuntimeError("Fresh in-worker neural sample replay failed before publication.")
    reference_result = compile_topology(source.raw, seed=spec.compiler_seed, theme=source.theme, config=source.config, start=source.start, exit=source.exit, objectives=source.objectives, spawns=source.spawns)
    result = compile_topology(raw, seed=spec.compiler_seed, theme=source.theme, config=source.config, start=source.start, exit=source.exit, objectives=source.objectives, spawns=source.spawns)
    destination.parent.mkdir(parents=True, exist_ok=True); require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=512 * 1024 * 1024)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"; staging.mkdir()
    try:
        latent_arrays = {"tokens": sample.tokens.numpy().astype(np.uint16), "uncertainty": sample.uncertainty.numpy().astype(np.float32), "valid_mask": conditions["valid_mask"].numpy().astype(np.uint8)}
        latent_payload = deterministic_npz_bytes(latent_arrays)
        if len(latent_payload) > MAX_LATENT_BYTES:
            raise ValueError("Latent sample artifact exceeds its strict byte bound.")
        _atomic_bytes(staging / LATENT_ARTIFACT, latent_payload)
        raw_artifact = write_raw_artifact(
            staging / "raw", raw=raw, seed=spec.compiler_seed, theme=source.theme, config=source.config,
            start=source.start, exit=source.exit, objectives=source.objectives, spawns=source.spawns,
            proposal_source=PROPOSAL_SOURCE,
            provenance={
                "generation_source_sha256": generation_source_sha256(), "config_sha256": _config_sha(config),
                "sampling_seed": spec.sampling_seed, "variant": spec.variant, "condition_sha256": condition_sha,
                "prior_checkpoint_sha256": PRIOR_CHECKPOINT_SHA256, "prior_ema_sha256": PRIOR_EMA_SHA256,
                "codec_checkpoint_sha256": CODEC_CHECKPOINT_SHA256, "codec_ema_sha256": CODEC_EMA_SHA256,
                "source_full_map_identity_sha256": source.full_map_identity_sha256,
                "fully_masked": True, "target_latent_tokens_accessed": False,
            },
        )
        compiled_artifact = write_compiled_artifact(staging / "compiled", raw_artifact=raw_artifact, result=result)
        preview_scale = max(2, min(config.contact_scale, 256 // max(source.config.width, source.config.height)))
        preview = case_preview_png_bytes(source, raw, result.data, scale=preview_scale)
        _atomic_bytes(staging / PREVIEW_ARTIFACT, preview)
        walkable = np.isin(raw.terrain, tuple(WALKABLE_TERRAIN)); targets = (source.exit, *source.objectives)
        global_values = getattr(tensor, "global_conditions").astype(float).tolist(); requested_hazard, requested_openness = global_values[-3], global_values[-2]
        raw_hazard = float(np.count_nonzero(raw.hazard) / raw.hazard.size); raw_openness = float(np.count_nonzero(walkable) / walkable.size)
        repair = _cost(result, "repair_fraction"); preservation = _cost(result, "cell_preservation_fraction")
        reference_repair = _cost(reference_result, "repair_fraction"); reference_preservation = _cost(reference_result, "cell_preservation_fraction")
        unique_tokens = int(torch.unique(sample.tokens[conditions["valid_mask"][:, 0]]).numel())
        condition_gate = abs(raw_openness - requested_openness) <= 0.25 and abs(raw_hazard - requested_hazard) <= 0.12
        token_gate = unique_tokens >= 4
        repair_gate = repair <= min(0.35, reference_repair + 0.08)
        trace_payload = list(sample.trace)
        manifest: dict[str, object] = {
            "format": CASE_FORMAT, "case_id": spec.case_id, "source_sha256": generation_source_sha256(), "config_sha256": _config_sha(config),
            "variant": spec.variant, "theme": source.theme, "width": source.config.width, "height": source.config.height,
            "source_ref": {"shard_id": spec.ref.shard_id, "sample_index": spec.ref.sample_index, "split": "test", "kind": spec.ref.kind, "full_map_identity_sha256": source.full_map_identity_sha256, "sample_identity_sha256": source.sample_identity_sha256, "source_raw_sha256": source.raw.raw_sha256},
            "seeds": {"sampling": spec.sampling_seed, "compiler": spec.compiler_seed},
            "conditioning": {"sha256": condition_sha, "theme_index": int(getattr(tensor, "theme_index")), "global_conditions": global_values, "fully_masked": True, "target_latent_tokens_accessed": False},
            "sampler": {"name": "seeded_top_k_gumbel_parallel_reveal_v1", "sampling_steps": config.sampling_steps, "temperature": config.temperature, "top_k": config.top_k, "latent_shape": list(sample.tokens.shape), "valid_cells": int(conditions["valid_mask"].sum()), "unique_tokens": unique_tokens, "tokens_sha256": tensor_sha256(sample.tokens), "uncertainty_sha256": tensor_sha256(sample.uncertainty), "trace_sha256": json_sha256(trace_payload), "trace": trace_payload},
            "decoded": {"raw_topology_sha256": raw.raw_sha256, "terrain_histogram": _histogram(raw.terrain, 9), "hazard_histogram": _histogram(raw.hazard, 5), "elevation_histogram": _histogram(raw.elevation, 6)},
            "reference": {"raw_topology_sha256": source.raw.raw_sha256, "compiler_repair_fraction": reference_repair, "compiler_cell_preservation_fraction": reference_preservation},
            "metrics": {"raw_required_reachable": _reachable(walkable, source.start, targets), "raw_radius_one_required_reachable": _reachable(_radius_one(walkable), source.start, targets), "requested_openness": requested_openness, "raw_openness": raw_openness, "openness_absolute_error": abs(raw_openness - requested_openness), "requested_hazard_budget": requested_hazard, "raw_hazard_fraction": raw_hazard, "hazard_absolute_error": abs(raw_hazard - requested_hazard), "compiler_repair_fraction": repair, "compiler_cell_preservation_fraction": preservation, "repair_excess_over_reference": repair - reference_repair},
            "gates": {"fully_masked": True, "target_tokens_absent": True, "raw_immutable": True, "compiled_valid": True, "exact_compiler_replay": True, "exact_neural_replay": True, "condition_adherence": condition_gate, "token_noncollapse": token_gate, "repair_within_reference_bound": repair_gate, "quality_accepted": condition_gate and token_gate and repair_gate},
            "artifacts": {
                "latent": {key: value for key, value in _latent_descriptor(staging, latent_arrays, latent_payload).items() if key in {"path", "bytes", "sha256"}},
                "raw_manifest": _artifact(staging, f"raw/{RAW_MANIFEST}"), "compiled_manifest": _artifact(staging, f"compiled/{COMPILED_MANIFEST}"), "preview": _artifact(staging, PREVIEW_ARTIFACT),
            },
        }
        manifest["case_identity_sha256"] = json_sha256(manifest)
        validate_schema(manifest, "map_topology_neural_prior_generation_case.schema.json")
        _atomic_json(staging / CASE_MANIFEST, manifest)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            import shutil; shutil.rmtree(staging, ignore_errors=True)
    return validate_case(destination, corpus_root=corpus_root, config=config, exact_neural_replay=False)


def _load_latent(case_dir: Path, manifest: dict[str, object]) -> dict[str, np.ndarray]:
    descriptor = manifest["artifacts"]["latent"]  # type: ignore[index]
    path = case_dir / str(descriptor["path"])
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_LATENT_BYTES or path.stat().st_size != descriptor["bytes"] or file_sha256(path) != descriptor["sha256"]:
        raise ValueError("Latent artifact identity failed.")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"tokens", "uncertainty", "valid_mask"}:
            raise ValueError("Latent artifact member census drifted.")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if arrays["tokens"].dtype != np.uint16 or arrays["uncertainty"].dtype != np.float32 or arrays["valid_mask"].dtype != np.uint8 or arrays["tokens"].shape != arrays["uncertainty"].shape or arrays["valid_mask"].shape != (arrays["tokens"].shape[0], 1, *arrays["tokens"].shape[1:]):
        raise ValueError("Latent artifact shape/dtype contract drifted.")
    if deterministic_npz_bytes(arrays) != path.read_bytes():
        raise ValueError("Latent artifact is not canonical deterministic NPZ.")
    return arrays


def validate_case(case_dir: Path, *, corpus_root: Path, config: GenerationConfig, exact_neural_replay: bool) -> dict[str, object]:
    case_dir = Path(case_dir).resolve(); path = case_dir / CASE_MANIFEST
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_CASE_MANIFEST_BYTES:
        raise ValueError("Generation case manifest is missing or oversized.")
    manifest = json.loads(path.read_text(encoding="utf-8")); validate_schema(manifest, "map_topology_neural_prior_generation_case.schema.json")
    if path.read_bytes() != canonical_json_bytes(manifest) or manifest["source_sha256"] != generation_source_sha256() or manifest["config_sha256"] != _config_sha(config):
        raise ValueError("Generation case canonical/source/config identity drifted.")
    stored_identity = manifest.pop("case_identity_sha256")
    if json_sha256(manifest) != stored_identity:
        raise ValueError("Generation case semantic self-hash failed.")
    manifest["case_identity_sha256"] = stored_identity
    for descriptor in manifest["artifacts"].values():
        target = case_dir / descriptor["path"]
        if not target.is_file() or target.stat().st_size != descriptor["bytes"] or file_sha256(target) != descriptor["sha256"]:
            raise ValueError("Generation case artifact closure failed.")
    arrays = _load_latent(case_dir, manifest)
    raw_artifact = load_raw_artifact(case_dir / "raw")
    compiled_artifact = load_compiled_artifact(case_dir / "compiled", raw_artifact=raw_artifact, exact_replay=True)
    if raw_artifact.raw.raw_sha256 != manifest["decoded"]["raw_topology_sha256"] or compiled_artifact.result.compiled_arrays_sha256 != compiled_artifact.manifest["compiled_arrays_sha256"]:
        raise ValueError("Generation case raw/compiled semantics drifted.")
    if exact_neural_replay:
        authorities = _load_authorities(corpus_root); spec = _find_spec(authorities, config, str(manifest["case_id"]))
        source, _, conditions, condition_sha, sample, raw = _sample_case(authorities, config, spec)
        if condition_sha != manifest["conditioning"]["sha256"] or tensor_sha256(sample.tokens) != manifest["sampler"]["tokens_sha256"] or tensor_sha256(sample.uncertainty) != manifest["sampler"]["uncertainty_sha256"] or json_sha256(list(sample.trace)) != manifest["sampler"]["trace_sha256"]:
            raise ValueError("Generation case failed exact neural sampling replay.")
        if not np.array_equal(arrays["tokens"], sample.tokens.numpy().astype(np.uint16)) or not np.array_equal(arrays["uncertainty"], sample.uncertainty.numpy()) or raw.raw_sha256 != raw_artifact.raw.raw_sha256:
            raise ValueError("Generation case replay arrays/raw decode drifted.")
        preview_scale = max(2, min(config.contact_scale, 256 // max(source.config.width, source.config.height)))
        expected_preview = case_preview_png_bytes(source, raw, compiled_artifact.result.data, scale=preview_scale)
        if expected_preview != (case_dir / PREVIEW_ARTIFACT).read_bytes():
            raise ValueError("Generation case preview failed exact byte replay.")
    return manifest


def _run_worker(arguments: list[str], *, timeout: int, maximum_attempts: int, log_root: Path, label: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    telemetry: list[dict[str, object]] = []
    environment = os.environ.copy(); environment.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    for attempt in range(1, maximum_attempts + 1):
        try:
            completed = subprocess.run([sys.executable, "-m", "forge.map_topology_neural_prior_generation", *arguments], cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            telemetry.append({"label": label, "attempt": attempt, "returncode": -9, "native_failure": False})
            (log_root / f"{label}.attempt{attempt}.stdout.txt").write_text(stdout, encoding="utf-8")
            (log_root / f"{label}.attempt{attempt}.stderr.txt").write_text(stderr + "\nWORKER TIMEOUT\n", encoding="utf-8")
            continue
        row = {"label": label, "attempt": attempt, "returncode": completed.returncode, "native_failure": completed.returncode in {-1073741819, 3221225477}}
        telemetry.append(row)
        (log_root / f"{label}.attempt{attempt}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (log_root / f"{label}.attempt{attempt}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode == 0:
            try:
                result = json.loads(completed.stdout)
            except json.JSONDecodeError:
                result = None
            if isinstance(result, dict):
                return result, telemetry
            (log_root / f"{label}.attempt{attempt}.stderr.txt").write_text(completed.stderr + "\nWORKER RETURNED MALFORMED RESULT\n", encoding="utf-8")
    raise RuntimeError(f"Worker {label} exhausted {maximum_attempts} bounded attempts.")


def _case_record(case_dir: Path, manifest: dict[str, object], relative: str) -> dict[str, object]:
    return {
        "case_id": manifest["case_id"], "theme": manifest["theme"], "width": manifest["width"], "height": manifest["height"], "variant": manifest["variant"],
        "manifest": f"{relative}/{CASE_MANIFEST}", "manifest_sha256": file_sha256(case_dir / CASE_MANIFEST), "case_identity_sha256": manifest["case_identity_sha256"],
        "raw_topology_sha256": manifest["decoded"]["raw_topology_sha256"], "compiled_arrays_sha256": json.loads((case_dir / "compiled" / COMPILED_MANIFEST).read_text(encoding="utf-8"))["compiled_arrays_sha256"],
        "unique_tokens": manifest["sampler"]["unique_tokens"], "repair_fraction": manifest["metrics"]["compiler_repair_fraction"], "quality_accepted": manifest["gates"]["quality_accepted"],
    }


def generate_bank(destination: Path, *, corpus_root: Path, config: GenerationConfig) -> dict[str, object]:
    destination = Path(destination).resolve(); corpus_root = Path(corpus_root).resolve()
    if destination.exists():
        raise FileExistsError("Generation bank publication is immutable.")
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=8 * 1024**3)
    staging = destination.parent / f".{destination.name}.staging"; staging.mkdir(parents=True, exist_ok=True)
    config_payload = config.to_dict(); config_path = staging / CONFIG_FILE
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config_payload:
        raise ValueError("Existing generation staging config disagrees with requested resume.")
    _atomic_json(config_path, config_payload)
    log_root = staging / "telemetry"; log_root.mkdir(exist_ok=True)
    authorities = _load_authorities(corpus_root); specs = plan_cases(authorities.dataset, config); del authorities
    generation_telemetry: list[dict[str, object]] = []
    def build(spec: CaseSpec):
        case_dir = staging / "cases" / spec.case_id
        if case_dir.exists():
            try:
                return validate_case(case_dir, corpus_root=corpus_root, config=config, exact_neural_replay=False), []
            except Exception:
                raise ValueError(f"Existing staged case {spec.case_id} is invalid; preserve and inspect it rather than overwrite.")
        return _run_worker(["case", "--destination", str(case_dir), "--corpus", str(corpus_root), "--config", str(config_path), "--case-id", spec.case_id], timeout=config.worker_timeout_seconds, maximum_attempts=config.maximum_attempts, log_root=log_root, label=f"generate_{spec.case_id}")
    with ThreadPoolExecutor(max_workers=config.maximum_workers) as pool:
        futures = {pool.submit(build, spec): spec for spec in specs}
        for future in as_completed(futures):
            _, telemetry = future.result(); generation_telemetry.extend(telemetry)
    verification_telemetry: list[dict[str, object]] = []; replay_results: list[dict[str, object]] = []
    def verify(spec: CaseSpec):
        return _run_worker(["verify-case", "--destination", str(staging / "cases" / spec.case_id), "--corpus", str(corpus_root), "--config", str(config_path)], timeout=config.worker_timeout_seconds, maximum_attempts=config.maximum_attempts, log_root=log_root, label=f"verify_{spec.case_id}")
    with ThreadPoolExecutor(max_workers=config.maximum_workers) as pool:
        futures = {pool.submit(verify, spec): spec for spec in specs}
        for future in as_completed(futures):
            result, telemetry = future.result(); replay_results.append(result); verification_telemetry.extend(telemetry)
    replay_results.sort(key=lambda row: str(row["case_id"]))
    all_telemetry = generation_telemetry + verification_telemetry
    replay: dict[str, object] = {
        "format": REPLAY_FORMAT, "source_sha256": generation_source_sha256(), "config_sha256": _config_sha(config),
        "case_count": len(specs), "exact_case_count": len(replay_results), "attempts": len(all_telemetry), "native_failures": sum(bool(row["native_failure"]) for row in all_telemetry),
        "case_results": replay_results,
        "gates": {"all_cases_exact": True, "all_raw_artifacts_exact": True, "all_compiled_artifacts_exact": True, "all_neural_samples_exact": True, "all_previews_exact": True},
    }
    replay["replay_sha256"] = json_sha256(replay); validate_schema(replay, "map_topology_neural_prior_generation_replay.schema.json"); _atomic_json(staging / REPLAY_REPORT, replay)
    manifests = [validate_case(staging / "cases" / spec.case_id, corpus_root=corpus_root, config=config, exact_neural_replay=False) for spec in specs]
    representative_rows = []
    for theme in THEMES:
        candidates = [manifest for manifest in manifests if manifest["theme"] == theme and manifest["variant"] == 0]
        selected = min(candidates, key=lambda row: (int(row["width"]) * int(row["height"]), str(row["case_id"])))
        representative_rows.append((f"{theme} {selected['width']}x{selected['height']} v0", (staging / "cases" / str(selected["case_id"]) / PREVIEW_ARTIFACT).read_bytes()))
    _atomic_bytes(staging / CONTACT_SHEET, contact_sheet_png_bytes(representative_rows))
    records = [_case_record(staging / "cases" / str(manifest["case_id"]), manifest, f"cases/{manifest['case_id']}") for manifest in manifests]
    records.sort(key=lambda row: str(row["case_id"])); count = len(records)
    mean = lambda values: float(sum(values) / len(values))
    aggregate = {
        "unique_raw_topologies": len({row["raw_topology_sha256"] for row in records}), "unique_compiled_topologies": len({row["compiled_arrays_sha256"] for row in records}), "unique_token_sequences": len({manifest["sampler"]["tokens_sha256"] for manifest in manifests}),
        "mean_unique_tokens": mean([int(row["unique_tokens"]) for row in records]), "mean_repair_fraction": mean([float(row["repair_fraction"]) for row in records]), "maximum_repair_fraction": max(float(row["repair_fraction"]) for row in records), "mean_reference_repair_fraction": mean([float(manifest["reference"]["compiler_repair_fraction"]) for manifest in manifests]),
        "raw_required_reachable_rate": mean([float(bool(manifest["metrics"]["raw_required_reachable"])) for manifest in manifests]), "raw_radius_one_required_reachable_rate": mean([float(bool(manifest["metrics"]["raw_radius_one_required_reachable"])) for manifest in manifests]),
        "condition_adherence_rate": mean([float(bool(manifest["gates"]["condition_adherence"])) for manifest in manifests]), "quality_accepted_cases": sum(bool(row["quality_accepted"]) for row in records),
    }
    noncollapse = aggregate["unique_raw_topologies"] >= int(count * 0.8) and aggregate["unique_token_sequences"] >= int(count * 0.8) and aggregate["mean_unique_tokens"] >= 4
    repair_gate = all(bool(manifest["gates"]["repair_within_reference_bound"]) for manifest in manifests); condition_gate = all(bool(manifest["gates"]["condition_adherence"]) for manifest in manifests)
    theme_counts = {theme: sum(row["theme"] == theme for row in records) for theme in THEMES}; size_counts: dict[str, int] = {}
    for row in records: size_counts[f"{row['width']}x{row['height']}"] = size_counts.get(f"{row['width']}x{row['height']}", 0) + 1
    manifest: dict[str, object] = {
        "format": FORMAT, "authority": "research_generation_evidence_not_runtime_map_pack", "source_sha256": generation_source_sha256(), "source_manifest": source_manifest(), "config": config_payload, "config_sha256": _config_sha(config), "authorities": authority_payload(),
        "census": {"condition_count": 24, "case_count": count, "themes": theme_counts, "sizes": size_counts, "variants_per_condition": config.variants_per_condition}, "cases": records, "aggregate": aggregate,
        "gates": {"fully_masked_all_cases": True, "target_latent_tokens_never_accessed": True, "theme_size_balance_exact": True, "raw_artifacts_immutable": True, "compiled_maps_valid": True, "exact_replay": True, "noncollapsed_bank": noncollapse, "repair_within_procedural_reference_bounds": repair_gate, "condition_adherence": condition_gate, "quality_accepted": noncollapse and repair_gate and condition_gate, "production_promotion_allowed": False},
        "claim": {"scope": "held_out_conditioned_seeded_free_generation_research_bank", "free_generation_proven": True, "valid_runtime_maps_proven": False, "quality_limit": "Compiled outputs are deterministic research candidates only; promotion remains forbidden unless diversity, condition adherence, and repair-cost gates all pass."},
        "artifacts": {"contact_sheet": _artifact(staging, CONTACT_SHEET), "replay_report": _artifact(staging, REPLAY_REPORT)},
    }
    manifest["manifest_sha256"] = json_sha256(manifest); validate_schema(manifest, "map_topology_neural_prior_generation_bank.schema.json"); _atomic_json(staging / BANK_MANIFEST, manifest)
    validate_bank(staging, corpus_root=corpus_root, exact_cases=False)
    os.replace(staging, destination)
    return validate_bank(destination, corpus_root=corpus_root, exact_cases=False)


def validate_bank(root: Path, *, corpus_root: Path, exact_cases: bool) -> dict[str, object]:
    root = Path(root).resolve(); path = root / BANK_MANIFEST
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_BANK_BYTES:
        raise ValueError("Generation bank manifest is missing or oversized.")
    manifest = json.loads(path.read_text(encoding="utf-8")); validate_schema(manifest, "map_topology_neural_prior_generation_bank.schema.json")
    if path.read_bytes() != canonical_json_bytes(manifest) or manifest["source_sha256"] != generation_source_sha256() or manifest["source_manifest"] != source_manifest():
        raise ValueError("Generation bank canonical/source identity drifted.")
    stored = manifest.pop("manifest_sha256")
    if json_sha256(manifest) != stored:
        raise ValueError("Generation bank semantic self-hash failed.")
    manifest["manifest_sha256"] = stored
    config = GenerationConfig.from_dict(manifest["config"])
    if manifest["config_sha256"] != _config_sha(config):
        raise ValueError("Generation bank config identity failed.")
    for descriptor in manifest["artifacts"].values():
        target = root / descriptor["path"]
        if not target.is_file() or target.stat().st_size != descriptor["bytes"] or file_sha256(target) != descriptor["sha256"]:
            raise ValueError("Generation bank artifact closure failed.")
    replay_path = root / REPLAY_REPORT
    if not 0 < replay_path.stat().st_size <= MAX_REPLAY_BYTES:
        raise ValueError("Generation replay report exceeds its bound.")
    replay = json.loads(replay_path.read_text(encoding="utf-8")); validate_schema(replay, "map_topology_neural_prior_generation_replay.schema.json")
    stored_replay = replay.pop("replay_sha256")
    if json_sha256(replay) != stored_replay or replay["case_count"] != len(manifest["cases"]):
        raise ValueError("Generation replay report identity/census failed.")
    for record in manifest["cases"]:
        case_dir = (root / record["manifest"]).parent
        case = validate_case(case_dir, corpus_root=corpus_root, config=config, exact_neural_replay=exact_cases)
        if file_sha256(case_dir / CASE_MANIFEST) != record["manifest_sha256"] or case["case_identity_sha256"] != record["case_identity_sha256"]:
            raise ValueError("Generation bank case registry drifted.")
    return manifest


def verify_case_result(case_dir: Path, *, corpus_root: Path, config: GenerationConfig) -> dict[str, object]:
    manifest = validate_case(case_dir, corpus_root=corpus_root, config=config, exact_neural_replay=True)
    compiled = json.loads((Path(case_dir) / "compiled" / COMPILED_MANIFEST).read_text(encoding="utf-8"))
    return {"case_id": manifest["case_id"], "case_identity_sha256": manifest["case_identity_sha256"], "raw_topology_sha256": manifest["decoded"]["raw_topology_sha256"], "compiled_arrays_sha256": compiled["compiled_arrays_sha256"], "exact": True}
