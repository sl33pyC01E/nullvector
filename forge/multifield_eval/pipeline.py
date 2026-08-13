from __future__ import annotations

from dataclasses import dataclass
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from ..config import PROJECT_ROOT
from ..multifield_diffusion import seeded_generators
from ..safety import disk_status, require_disk_floor, write_json_atomic
from .checkpoint import LoadedMultiFieldCheckpoint, load_multifield_checkpoint
from .conditions import (
    ConditionRecord,
    build_condition_grid,
    condition_batch,
    validate_grid_coverage,
)
from .rendering import (
    RENDER_FORMAT,
    aligned_fields_hash,
    bounded_postprocess_fields,
    build_contact_sheet,
    fields_to_rgba,
    palette_for_condition,
    save_npz_atomic,
    save_png_atomic,
    sha256_file,
)
from .schema import (
    GENERATION_BANK_SCHEMA,
    RAW_SAMPLE_SCHEMA,
    validate_manifest_schema,
)
from .validation import (
    ConditionTemplateBank,
    acceptance_breakdown,
    diversity_report,
    validate_generated_fields,
    validate_postprocess_delta,
)


GENERATION_BANK_FORMAT = "nullvector-multifield-generation-bank-v1"
RAW_SAMPLE_FORMAT = "nullvector-multifield-raw-sample-v1"
REPLAY_REPORT_FORMAT = "nullvector-multifield-replay-report-v1"
EVALUATION_SOURCE_FILES = (
    "forge/multifield_eval/benchmark.py",
    "forge/multifield_eval/calibration.py",
    "forge/multifield_eval/checkpoint.py",
    "forge/multifield_eval/conditions.py",
    "forge/multifield_eval/rendering.py",
    "forge/multifield_eval/schema.py",
    "forge/multifield_eval/validation.py",
    "forge/multifield_eval/pipeline.py",
    "forge/multifield_diffusion.py",
    "forge/multifield_data.py",
    "forge/multifield_metrics.py",
    "shared/schema/multifield_generation_bank.schema.json",
    "shared/schema/multifield_raw_sample.schema.json",
    "shared/schema/multifield_benchmark.schema.json",
    "shared/schema/multifield_reference_calibration.schema.json",
)


@dataclass(slots=True)
class GeneratedSample:
    record: ConditionRecord
    raw: tuple[np.ndarray, np.ndarray, np.ndarray]
    target: tuple[np.ndarray, np.ndarray, np.ndarray]
    guide: np.ndarray
    genes: np.ndarray
    corpus_seed: int
    batch_latency_seconds: float


def evaluation_source_hash(root: Path = PROJECT_ROOT) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for relative in EVALUATION_SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _autocast(bundle: LoadedMultiFieldCheckpoint):
    if bundle.precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if bundle.precision == "bf16" else torch.float16
    return torch.autocast(device_type=bundle.device.type, dtype=dtype)


def _move_batch(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_samples(
    bundle: LoadedMultiFieldCheckpoint,
    records: Sequence[ConditionRecord],
    *,
    batch_size: int = 8,
    temperature: float = 0.9,
) -> tuple[list[GeneratedSample], dict[str, Any]]:
    if not records:
        raise ValueError("Cannot generate an empty sample bank")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    legal = torch.as_tensor(
        bundle.legal_tuples, dtype=torch.long, device=bundle.device
    )
    samples: list[GeneratedSample] = []
    batch_latencies: list[float] = []
    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        subset = records[start : start + batch_size]
        cpu_batch = condition_batch(bundle, subset)
        batch = _move_batch(cpu_batch, bundle.device)
        generators = seeded_generators(
            [record.sample_seed for record in subset], bundle.device
        )
        _synchronize(bundle.device)
        batch_started = time.perf_counter()
        with _autocast(bundle):
            prediction = bundle.model.sample(
                batch["guide"],
                batch["morphology"],
                batch["subtype"],
                batch["role"],
                batch["genes"],
                temperature=temperature,
                generators=generators,
                legal_tuples=legal,
            )
        _synchronize(bundle.device)
        latency = time.perf_counter() - batch_started
        batch_latencies.append(latency)
        arrays = [values.detach().cpu().numpy().astype(np.uint8) for values in prediction]
        for index, record in enumerate(subset):
            samples.append(
                GeneratedSample(
                    record=record,
                    raw=tuple(values[index] for values in arrays),  # type: ignore[arg-type]
                    target=(
                        cpu_batch["part"][index].numpy().astype(np.uint8),
                        cpu_batch["material"][index].numpy().astype(np.uint8),
                        cpu_batch["emission"][index].numpy().astype(np.uint8),
                    ),
                    guide=cpu_batch["guide"][index].numpy().astype(np.float32),
                    genes=cpu_batch["genes"][index].numpy().astype(np.float32),
                    corpus_seed=int(cpu_batch["seed"][index]),
                    batch_latency_seconds=latency,
                )
            )
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "samples": len(samples),
        "batch_size": batch_size,
        "batches": len(batch_latencies),
        "temperature": float(temperature),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(samples) / max(elapsed, 1.0e-12),
        "mean_batch_latency_seconds": float(np.mean(batch_latencies)),
        "p95_batch_latency_seconds": float(np.quantile(batch_latencies, 0.95)),
        "mean_seconds_per_sample": elapsed / len(samples),
    }
    if bundle.device.type == "cuda":
        report.update(
            {
                "cuda_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(bundle.device)
                ),
                "cuda_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(bundle.device)
                ),
            }
        )
    return samples, report


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _raw_sample_arrays(sample: GeneratedSample) -> dict[str, np.ndarray]:
    return {
        "format": np.asarray([RAW_SAMPLE_FORMAT]),
        "part": sample.raw[0],
        "material": sample.raw[1],
        "emission": sample.raw[2],
        "guide": sample.guide,
        "genes": sample.genes,
        "target_part": sample.target[0],
        "target_material": sample.target[1],
        "target_emission": sample.target[2],
        "morphology": np.asarray([sample.record.morphology], dtype=np.uint8),
        "subtype": np.asarray([sample.record.subtype], dtype=np.uint8),
        "role": np.asarray([sample.record.role], dtype=np.uint8),
        "source_index": np.asarray([sample.record.source_index], dtype=np.int64),
        "corpus_seed": np.asarray([sample.corpus_seed], dtype=np.uint32),
        "sample_seed": np.asarray([sample.record.sample_seed], dtype=np.uint64),
    }


def write_generation_bank(
    bundle: LoadedMultiFieldCheckpoint,
    destination: Path,
    *,
    mode: str = "stratified",
    samples_per_condition: int = 1,
    base_seed: int | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    temperature: float = 0.9,
    max_postprocess_delta: float = 0.03,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    manifest_path = destination / "generation_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Generation manifest already exists; outputs are immutable: {manifest_path}"
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "Generation destination is non-empty. Use a new run directory so prior "
            "raw neural outputs are never overwritten."
        )
    records = build_condition_grid(
        bundle,
        mode=mode,
        samples_per_condition=samples_per_condition,
        base_seed=base_seed,
        limit=limit,
    )
    planned_bytes = len(records) * 512 * 1024 + 32 * 1024 * 1024
    disk_before = require_disk_floor(destination, planned_bytes=planned_bytes)
    raw_dir = destination / "raw"
    compiled_dir = destination / "compiled"
    raw_dir.mkdir(parents=True, exist_ok=True)
    compiled_dir.mkdir(parents=True, exist_ok=True)

    templates = ConditionTemplateBank.build(bundle)
    generated, performance = generate_samples(
        bundle,
        records,
        batch_size=batch_size,
        temperature=temperature,
    )
    source_hash = evaluation_source_hash()
    sample_entries: list[dict[str, Any]] = []
    raw_validation: list[dict[str, Any]] = []
    raw_images: list[np.ndarray] = []
    compiled_images: list[np.ndarray] = []
    processed_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for sample in generated:
        record = sample.record
        palette = palette_for_condition(sample.corpus_seed, record)
        raw_rgba, raw_emission = fields_to_rgba(*sample.raw, palette)
        raw_npz = raw_dir / f"{record.sample_id}.npz"
        raw_png = raw_dir / f"{record.sample_id}_rgba.png"
        raw_emission_png = raw_dir / f"{record.sample_id}_emission.png"
        raw_manifest = raw_dir / f"{record.sample_id}.json"
        validation = validate_generated_fields(
            *sample.raw,
            guide=sample.guide,
            target=sample.target,
            record=record,
            legal_tuples=bundle.legal_tuples,
            templates=templates,
        )
        save_npz_atomic(raw_npz, **_raw_sample_arrays(sample))
        save_png_atomic(raw_png, raw_rgba)
        save_png_atomic(raw_emission_png, raw_emission)
        raw_payload = {
            "format": RAW_SAMPLE_FORMAT,
            "render_format": RENDER_FORMAT,
            "condition": record.metadata(),
            "raw_fields_sha256": aligned_fields_hash(*sample.raw),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "canonical_ema_hash": bundle.payload["canonical_ema_hash"],
            "corpus_sha256": bundle.corpus.file_sha256,
            "training_source_hash": bundle.payload["training_source_hash"],
            "evaluation_source_hash": source_hash,
            "guide_policy": bundle.guide_policy.metadata(),
            "legal_tuple_fingerprint": bundle.payload["legal_tuple_fingerprint"],
            "temperature": float(temperature),
            "artifacts": {
                "fields": _artifact(raw_npz, destination),
                "rgba": _artifact(raw_png, destination),
                "emission": _artifact(raw_emission_png, destination),
            },
            "validation": validation,
        }
        # This manifest is deliberately published before bounded postprocess.
        validate_manifest_schema(raw_payload, RAW_SAMPLE_SCHEMA)
        write_json_atomic(raw_manifest, raw_payload)

        processed, postprocess = bounded_postprocess_fields(
            *sample.raw,
            max_delta_fraction=max_postprocess_delta,
        )
        delta_validation = validate_postprocess_delta(
            sample.raw, processed, postprocess, bundle.legal_tuples
        )
        if not delta_validation["valid"]:
            raise RuntimeError(
                f"Postprocess contract failed for {record.sample_id}: "
                f"{delta_validation['errors']}"
            )
        compiled_rgba, compiled_emission = fields_to_rgba(*processed, palette)
        compiled_npz = compiled_dir / f"{record.sample_id}.npz"
        compiled_png = compiled_dir / f"{record.sample_id}_rgba.png"
        compiled_emission_png = compiled_dir / f"{record.sample_id}_emission.png"
        save_npz_atomic(
            compiled_npz,
            format=np.asarray([postprocess["format"]]),
            part=processed[0],
            material=processed[1],
            emission=processed[2],
            raw_fields_sha256=np.asarray([postprocess["raw_fields_sha256"]]),
            processed_fields_sha256=np.asarray(
                [postprocess["processed_fields_sha256"]]
            ),
        )
        save_png_atomic(compiled_png, compiled_rgba)
        save_png_atomic(compiled_emission_png, compiled_emission)

        raw_images.append(raw_rgba)
        compiled_images.append(compiled_rgba)
        raw_validation.append(validation)
        processed_fields.append(processed)
        sample_entries.append(
            {
                "condition": record.metadata(),
                "raw_manifest": _artifact(raw_manifest, destination),
                "raw_fields_sha256": raw_payload["raw_fields_sha256"],
                "compiled_fields_sha256": postprocess["processed_fields_sha256"],
                "compiled_artifacts": {
                    "fields": _artifact(compiled_npz, destination),
                    "rgba": _artifact(compiled_png, destination),
                    "emission": _artifact(compiled_emission_png, destination),
                },
                "raw_validation": validation,
                "postprocess": postprocess,
                "postprocess_validation": delta_validation,
            }
        )

    raw_sheet_path = destination / "raw_contact_sheet.png"
    compiled_sheet_path = destination / "compiled_contact_sheet.png"
    columns = min(8, len(records))
    save_png_atomic(
        raw_sheet_path,
        np.asarray(
            build_contact_sheet(
                raw_images,
                records,
                validation=raw_validation,
                columns=columns,
            )
        ),
    )
    save_png_atomic(
        compiled_sheet_path,
        np.asarray(
            build_contact_sheet(
                compiled_images,
                records,
                validation=raw_validation,
                columns=columns,
            )
        ),
    )
    manifest = {
        "format": GENERATION_BANK_FORMAT,
        "status": "ready",
        "created_unix_seconds": time.time(),
        "provenance": bundle.provenance(),
        "evaluation_source_hash": source_hash,
        "grid": validate_grid_coverage(records, mode),
        "generation": {
            "mode": mode,
            "samples_per_condition": samples_per_condition,
            "base_seed": (
                int(bundle.payload["fixed_validation"]["generation_seed"])
                if base_seed is None
                else int(base_seed)
            ),
            "temperature": float(temperature),
            "batch_size": batch_size,
            "legal_tuple_constrained": True,
            "raw_is_authoritative": True,
            "postprocess_max_delta_fraction": max_postprocess_delta,
        },
        "condition_template_fingerprint": templates.fingerprint,
        "performance": performance,
        "validation": {
            "acceptance": acceptance_breakdown(records, raw_validation),
            "diversity": diversity_report([sample.raw for sample in generated], records),
            "compiled_diversity": diversity_report(processed_fields, records),
        },
        "disk": {
            "before": disk_before.to_dict(),
            "after": disk_status(destination).to_dict(),
        },
        "contact_sheets": {
            "raw": _artifact(raw_sheet_path, destination),
            "compiled": _artifact(compiled_sheet_path, destination),
        },
        "samples": sample_entries,
    }
    validate_manifest_schema(manifest, GENERATION_BANK_SCHEMA)
    write_json_atomic(manifest_path, manifest)
    return manifest


def _safe_artifact(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Manifest artifact escapes its run directory: {relative}")
    return candidate


def _record_from_metadata(values: Mapping[str, Any]) -> ConditionRecord:
    return ConditionRecord(
        ordinal=int(values["ordinal"]),
        grid_mode=str(values["grid_mode"]),
        source_index=int(values["source_index"]),
        variation=int(values["variation"]),
        sample_seed=int(values["sample_seed"]),
        morphology=int(values["morphology_id"]),
        subtype=int(values["subtype_id"]),
        role=int(values["role_id"]),
    )


def replay_generation_bank(
    manifest_path: Path,
    *,
    checkpoint_path: Path | None = None,
    corpus_path: Path | None = None,
    device: str | None = None,
    precision: str | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != GENERATION_BANK_FORMAT:
        raise ValueError("Generation manifest has an unsupported format")
    validate_manifest_schema(payload, GENERATION_BANK_SCHEMA)
    root = manifest_path.parent
    recorded_checkpoint = Path(payload["provenance"]["checkpoint"])
    recorded_environment = payload["provenance"]["evaluation_environment"]
    replay_device = str(recorded_environment["device_type"]) if device is None else device
    replay_precision = str(recorded_environment["precision"]) if precision is None else precision
    if replay_device != str(recorded_environment["device_type"]):
        raise ValueError(
            "Exact replay requires the recorded device type "
            f"{recorded_environment['device_type']!r}, got {replay_device!r}."
        )
    if replay_precision != str(recorded_environment["precision"]):
        raise ValueError(
            "Exact replay requires the recorded precision "
            f"{recorded_environment['precision']!r}, got {replay_precision!r}."
        )
    bundle = load_multifield_checkpoint(
        checkpoint_path or recorded_checkpoint,
        corpus_path=corpus_path,
        device=replay_device,
        precision=replay_precision,
    )
    provenance_errors = []
    for name, observed in (
        ("checkpoint_sha256", bundle.checkpoint_sha256),
        ("canonical_ema_hash", bundle.payload["canonical_ema_hash"]),
        ("training_source_hash", bundle.payload["training_source_hash"]),
    ):
        if str(payload["provenance"][name]) != str(observed):
            provenance_errors.append(f"{name} differs")
    if str(payload["provenance"]["corpus"]["file_sha256"]) != bundle.corpus.file_sha256:
        provenance_errors.append("corpus_sha256 differs")
    if payload.get("evaluation_source_hash") != evaluation_source_hash():
        provenance_errors.append("evaluation_source_hash differs")
    active_environment = bundle.provenance()["evaluation_environment"]
    for name in (
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "device_type",
        "precision",
        "gpu_name",
        "deterministic_algorithms",
    ):
        if recorded_environment.get(name) != active_environment.get(name):
            provenance_errors.append(f"evaluation_environment.{name} differs")
    if provenance_errors:
        raise ValueError(
            "Replay provenance mismatch: " + ", ".join(provenance_errors)
        )
    records = [
        _record_from_metadata(entry["condition"]) for entry in payload["samples"]
    ]
    generated, performance = generate_samples(
        bundle,
        records,
        batch_size=int(payload["generation"]["batch_size"]),
        temperature=float(payload["generation"]["temperature"]),
    )
    comparisons: list[dict[str, Any]] = []
    for sample, entry in zip(generated, payload["samples"]):
        raw_manifest_path = _safe_artifact(
            root, entry["raw_manifest"]["path"]
        )
        raw_manifest_artifact_exact = (
            sha256_file(raw_manifest_path) == entry["raw_manifest"]["sha256"]
            and raw_manifest_path.stat().st_size == int(entry["raw_manifest"]["bytes"])
        )
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        validate_manifest_schema(raw_manifest, RAW_SAMPLE_SCHEMA)
        raw_npz_path = _safe_artifact(
            root, raw_manifest["artifacts"]["fields"]["path"]
        )
        raw_artifacts_exact = all(
            sha256_file(_safe_artifact(root, artifact["path"])) == artifact["sha256"]
            and _safe_artifact(root, artifact["path"]).stat().st_size
            == int(artifact["bytes"])
            for artifact in raw_manifest["artifacts"].values()
        )
        with np.load(raw_npz_path, allow_pickle=False) as archive:
            recorded_raw = (
                archive["part"],
                archive["material"],
                archive["emission"],
            )
            conditioning_exact = bool(
                np.array_equal(archive["guide"], sample.guide)
                and np.array_equal(archive["genes"], sample.genes)
                and np.array_equal(archive["target_part"], sample.target[0])
                and np.array_equal(archive["target_material"], sample.target[1])
                and np.array_equal(archive["target_emission"], sample.target[2])
                and int(archive["sample_seed"][0]) == sample.record.sample_seed
                and int(archive["source_index"][0]) == sample.record.source_index
            )
        palette = palette_for_condition(sample.corpus_seed, sample.record)
        replay_rgba, replay_emission = fields_to_rgba(*sample.raw, palette)
        rgba_path = _safe_artifact(
            root, raw_manifest["artifacts"]["rgba"]["path"]
        )
        emission_path = _safe_artifact(
            root, raw_manifest["artifacts"]["emission"]["path"]
        )
        recorded_rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
        recorded_emission = np.asarray(Image.open(emission_path).convert("RGBA"))
        processed, postprocess = bounded_postprocess_fields(
            *sample.raw,
            max_delta_fraction=float(
                payload["generation"]["postprocess_max_delta_fraction"]
            ),
        )
        compiled_path = _safe_artifact(
            root, entry["compiled_artifacts"]["fields"]["path"]
        )
        compiled_artifacts_exact = all(
            sha256_file(_safe_artifact(root, artifact["path"])) == artifact["sha256"]
            and _safe_artifact(root, artifact["path"]).stat().st_size
            == int(artifact["bytes"])
            for artifact in entry["compiled_artifacts"].values()
        )
        with np.load(compiled_path, allow_pickle=False) as archive:
            recorded_compiled = (
                archive["part"],
                archive["material"],
                archive["emission"],
            )
        replay_compiled_rgba, replay_compiled_emission = fields_to_rgba(
            *processed, palette
        )
        compiled_rgba_path = _safe_artifact(
            root, entry["compiled_artifacts"]["rgba"]["path"]
        )
        compiled_emission_path = _safe_artifact(
            root, entry["compiled_artifacts"]["emission"]["path"]
        )
        checks = {
            "sample_id": sample.record.sample_id,
            "raw_manifest_artifact_exact": raw_manifest_artifact_exact,
            "raw_artifacts_exact": raw_artifacts_exact,
            "conditioning_exact": conditioning_exact,
            "raw_fields_exact": all(
                np.array_equal(first, second)
                for first, second in zip(sample.raw, recorded_raw)
            ),
            "raw_hash_exact": aligned_fields_hash(*sample.raw)
            == entry["raw_fields_sha256"],
            "raw_rgba_exact": np.array_equal(replay_rgba, recorded_rgba),
            "raw_emission_exact": np.array_equal(
                replay_emission, recorded_emission
            ),
            "compiled_fields_exact": all(
                np.array_equal(first, second)
                for first, second in zip(processed, recorded_compiled)
            ),
            "compiled_hash_exact": postprocess["processed_fields_sha256"]
            == entry["compiled_fields_sha256"],
            "compiled_artifacts_exact": compiled_artifacts_exact,
            "compiled_rgba_exact": np.array_equal(
                replay_compiled_rgba,
                np.asarray(Image.open(compiled_rgba_path).convert("RGBA")),
            ),
            "compiled_emission_exact": np.array_equal(
                replay_compiled_emission,
                np.asarray(Image.open(compiled_emission_path).convert("RGBA")),
            ),
        }
        checks["exact"] = all(
            value for key, value in checks.items() if key != "sample_id"
        )
        comparisons.append(checks)
    report = {
        "format": REPLAY_REPORT_FORMAT,
        "status": "exact" if all(item["exact"] for item in comparisons) else "mismatch",
        "manifest": str(manifest_path),
        "samples": len(comparisons),
        "exact_samples": int(sum(item["exact"] for item in comparisons)),
        "provenance": bundle.provenance(),
        "performance": performance,
        "comparisons": comparisons,
    }
    if report_path is not None:
        write_json_atomic(Path(report_path).resolve(), report)
    return report
