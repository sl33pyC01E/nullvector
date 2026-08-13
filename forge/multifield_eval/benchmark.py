from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..multifield_data import MorphologyCorpusDataset, select_condition_bank
from ..safety import write_json_atomic
from ..train_multifield import evaluate_full_mask
from .checkpoint import LoadedMultiFieldCheckpoint
from .conditions import build_condition_grid, validate_grid_coverage
from .pipeline import evaluation_source_hash, generate_samples
from .rendering import aligned_fields_hash
from .schema import BENCHMARK_SCHEMA, validate_manifest_schema
from .validation import (
    ConditionTemplateBank,
    acceptance_breakdown,
    calibrate_reference_fields,
    diversity_report,
    validate_generated_fields,
)


BENCHMARK_FORMAT = "nullvector-multifield-checkpoint-benchmark-v2"


def benchmark_checkpoint(
    bundle: LoadedMultiFieldCheckpoint,
    *,
    grid_mode: str = "stratified",
    samples_per_condition: int = 2,
    generation_limit: int | None = None,
    generation_batch_size: int = 8,
    full_mask_examples: int = 256,
    full_mask_batch_size: int = 32,
    temperature: float = 0.9,
    base_seed: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if full_mask_examples <= 0 or full_mask_batch_size <= 0:
        raise ValueError("full-mask counts and batch size must be positive")
    started = time.perf_counter()
    full_mask_indices = select_condition_bank(
        bundle.corpus,
        bundle.validation_indices,
        min(full_mask_examples, len(bundle.validation_indices)),
        seed=int(bundle.payload["fixed_validation"]["full_mask_seed"]),
    )
    full_mask_dataset = MorphologyCorpusDataset(
        bundle.corpus,
        full_mask_indices,
        guide_policy=bundle.guide_policy,
    )
    full_mask_loader = DataLoader(
        full_mask_dataset,
        batch_size=full_mask_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bundle.device.type == "cuda",
    )
    class_weights = {
        name: values.to(bundle.device)
        for name, values in bundle.payload["class_weights"].items()
    }
    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)
    full_mask_started = time.perf_counter()
    full_mask = evaluate_full_mask(
        bundle.model,
        full_mask_loader,
        device=bundle.device,
        precision=bundle.precision,
        class_weights=class_weights,
        legal_tuples=bundle.legal_tuples,
        fixed_seed=int(bundle.payload["fixed_validation"]["full_mask_seed"]),
        field_weights=(
            float(bundle.payload["config"]["field_part_weight"]),
            float(bundle.payload["config"]["field_material_weight"]),
            float(bundle.payload["config"]["field_emission_weight"]),
        ),
    )
    full_mask_latency = time.perf_counter() - full_mask_started
    full_mask_memory = {}
    if bundle.device.type == "cuda":
        full_mask_memory = {
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(bundle.device)
            ),
            "cuda_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(bundle.device)
            ),
        }

    records = build_condition_grid(
        bundle,
        mode=grid_mode,
        samples_per_condition=samples_per_condition,
        base_seed=base_seed,
        limit=generation_limit,
    )
    templates = ConditionTemplateBank.build(bundle)
    generated, generation_performance = generate_samples(
        bundle,
        records,
        batch_size=generation_batch_size,
        temperature=temperature,
    )
    sample_reports = [
        validate_generated_fields(
            *sample.raw,
            guide=sample.guide,
            target=sample.target,
            record=sample.record,
            legal_tuples=bundle.legal_tuples,
            templates=templates,
        )
        for sample in generated
    ]
    raw_validity = {
        "hard_valid_rate": float(
            np.mean([report.get("accepted", False) for report in sample_reports])
        ),
        "condition_exact_match_rate": float(
            np.mean([report.get("condition_exact_match", False) for report in sample_reports])
        ),
        "condition_in_distribution_rate": float(
            np.mean([report.get("condition_in_distribution", False) for report in sample_reports])
        ),
        "mean_tuple_validity": float(
            np.mean([report.get("tuples", {}).get("valid_fraction", 0.0) for report in sample_reports])
        ),
        "margin_safe_rate": float(
            np.mean([report.get("margins", {}).get("safe", False) for report in sample_reports])
        ),
        "mean_scaffold_coverage": float(
            np.mean(
                [
                    report.get("topology", {}).get("scaffold_coverage_radius_2", 0.0)
                    for report in sample_reports
                ]
            )
        ),
        "mean_silhouette_iou_to_source": float(
            np.mean(
                [
                    report.get("source_similarity", {}).get("silhouette_iou", 0.0)
                    for report in sample_reports
                ]
            )
        ),
    }
    report = {
        "format": BENCHMARK_FORMAT,
        "status": "ready",
        "created_unix_seconds": time.time(),
        "provenance": bundle.provenance(),
        "evaluation_source_hash": evaluation_source_hash(),
        "condition_template_fingerprint": templates.fingerprint,
        "reference_calibration": calibrate_reference_fields(
            bundle, templates=templates
        ),
        "full_mask": {
            "examples_requested": full_mask_examples,
            "examples_evaluated": len(full_mask_dataset),
            "batch_size": full_mask_batch_size,
            "elapsed_seconds": full_mask_latency,
            "examples_per_second": len(full_mask_dataset)
            / max(full_mask_latency, 1.0e-12),
            **full_mask_memory,
            "metrics": full_mask,
        },
        "generation": {
            "grid": validate_grid_coverage(records, grid_mode),
            "temperature": temperature,
            "legal_tuple_constrained": True,
            "performance": generation_performance,
            "raw_validity": raw_validity,
            "diversity": diversity_report(
                [sample.raw for sample in generated], records
            ),
            "acceptance": acceptance_breakdown(records, sample_reports),
            "samples": [
                {
                    "condition": sample.record.metadata(),
                    "raw_fields_sha256": aligned_fields_hash(*sample.raw),
                    "validation": validation,
                }
                for sample, validation in zip(generated, sample_reports)
            ],
        },
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    validate_manifest_schema(report, BENCHMARK_SCHEMA)
    if output_path is not None:
        write_json_atomic(Path(output_path).resolve(), report)
    return report
