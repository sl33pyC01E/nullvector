from __future__ import annotations

import hashlib
import contextlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..multifield_data import MorphologyCorpusDataset
from ..multifield_eval.benchmark import BENCHMARK_FORMAT
from ..multifield_eval.checkpoint import load_multifield_checkpoint
from ..multifield_diffusion import seeded_generators
from ..multifield_eval.conditions import (
    ConditionRecord,
    build_condition_grid,
    condition_batch,
)
from ..multifield_eval.pipeline import (
    GENERATION_BANK_FORMAT,
    evaluation_source_hash,
)
from ..multifield_eval.rendering import aligned_fields_hash
from ..multifield_eval.schema import (
    BENCHMARK_SCHEMA,
    GENERATION_BANK_SCHEMA,
    RAW_SAMPLE_SCHEMA,
    validate_manifest_schema,
)
from ..multifield_eval.validation import (
    ConditionTemplateBank,
    calibrate_reference_fields,
    validate_generated_fields,
)
from ..safety import require_disk_floor, write_json_atomic


CONDITIONING_AUDIT_FORMAT = "nullvector-multifield-condition-diagnostic-v3"
CONDITIONING_AUDIT_SCHEMA = "multifield_conditioning_audit.schema.json"
RAW_SAMPLE_FORMAT = "nullvector-multifield-raw-sample-v1"
EXPECTED_NPZ_KEYS = frozenset(
    {
        "format",
        "part",
        "material",
        "emission",
        "guide",
        "genes",
        "target_part",
        "target_material",
        "target_emission",
        "morphology",
        "subtype",
        "role",
        "source_index",
        "corpus_seed",
        "sample_seed",
    }
)
AUDIT_SOURCE_FILES = (
    "forge/multifield_conditioning/__init__.py",
    "forge/multifield_conditioning/__main__.py",
    "forge/multifield_conditioning/audit.py",
    "forge/multifield_conditioning/cli.py",
    "shared/schema/multifield_conditioning_audit.schema.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditioning_audit_source_hash(root: Path = PROJECT_ROOT) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for relative in AUDIT_SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant is forbidden: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Artifact escapes immutable bank root: {relative!r}")
    return candidate


def _verify_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    if set(artifact) != {"path", "bytes", "sha256"}:
        raise ValueError("Artifact record keys are not exact")
    path = _safe_path(root, str(artifact["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Recorded artifact is missing: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != int(artifact["bytes"]):
        raise ValueError(f"Artifact byte count changed: {path}")
    observed_hash = _sha256_file(path)
    if observed_hash != str(artifact["sha256"]):
        raise ValueError(f"Artifact SHA-256 changed: {path}")
    return path


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


def _exact_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
) -> np.ndarray:
    values = np.asarray(arrays[name])
    if values.shape != shape or values.dtype != np.dtype(dtype):
        raise ValueError(
            f"NPZ member {name!r} must be {np.dtype(dtype)} {shape}, "
            f"got {values.dtype} {values.shape}"
        )
    return values


def _load_raw_npz(path: Path) -> dict[str, np.ndarray]:
    # Refuse excessive members or expansion before NumPy allocates arrays.
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != len(EXPECTED_NPZ_KEYS):
            raise ValueError("Raw NPZ member count is not exact")
        if any(member.file_size > 4 * 1024 * 1024 for member in members):
            raise ValueError("Raw NPZ contains an oversized member")
        if sum(member.file_size for member in members) > 16 * 1024 * 1024:
            raise ValueError("Raw NPZ expansion exceeds the audit bound")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_NPZ_KEYS:
            raise ValueError("Raw NPZ key set is not exact")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if arrays["format"].shape != (1,) or arrays["format"].dtype.kind != "U":
        raise ValueError("Raw NPZ format member is malformed")
    if str(arrays["format"][0]) != RAW_SAMPLE_FORMAT:
        raise ValueError("Raw NPZ format is unsupported")
    _exact_array(arrays, "part", (48, 48), np.uint8)
    _exact_array(arrays, "material", (48, 48), np.uint8)
    _exact_array(arrays, "emission", (48, 48), np.uint8)
    _exact_array(arrays, "guide", (8, 48, 48), np.float32)
    _exact_array(arrays, "genes", (24,), np.float32)
    _exact_array(arrays, "target_part", (48, 48), np.uint8)
    _exact_array(arrays, "target_material", (48, 48), np.uint8)
    _exact_array(arrays, "target_emission", (48, 48), np.uint8)
    _exact_array(arrays, "morphology", (1,), np.uint8)
    _exact_array(arrays, "subtype", (1,), np.uint8)
    _exact_array(arrays, "role", (1,), np.uint8)
    _exact_array(arrays, "source_index", (1,), np.int64)
    _exact_array(arrays, "corpus_seed", (1,), np.uint32)
    _exact_array(arrays, "sample_seed", (1,), np.uint64)
    if not np.isfinite(arrays["guide"]).all() or not np.isfinite(arrays["genes"]).all():
        raise ValueError("Raw NPZ contains non-finite conditioning values")
    return arrays


def _mcnemar_exact_p(generated_only: int, reference_only: int) -> float:
    discordant = generated_only + reference_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(generated_only, reference_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_classification_statistics(
    generated_correct: Sequence[bool],
    reference_correct: Sequence[bool],
    generated_prediction: Sequence[int],
    reference_prediction: Sequence[int],
) -> dict[str, Any]:
    lengths = {
        len(generated_correct),
        len(reference_correct),
        len(generated_prediction),
        len(reference_prediction),
    }
    if len(lengths) != 1 or not generated_correct:
        raise ValueError("Paired diagnostic vectors must be non-empty and equal length")
    generated = np.asarray(generated_correct, dtype=bool)
    reference = np.asarray(reference_correct, dtype=bool)
    generated_ids = np.asarray(generated_prediction, dtype=np.int64)
    reference_ids = np.asarray(reference_prediction, dtype=np.int64)
    count = len(generated)
    both_correct = int((generated & reference).sum())
    generated_only = int((generated & ~reference).sum())
    reference_only = int((~generated & reference).sum())
    both_wrong = int((~generated & ~reference).sum())
    generated_matches = both_correct + generated_only
    reference_matches = both_correct + reference_only
    p_value = _mcnemar_exact_p(generated_only, reference_only)
    return {
        "samples": count,
        "generated_matches": generated_matches,
        "generated_rate": generated_matches / count,
        "reference_matches": reference_matches,
        "reference_rate": reference_matches / count,
        "generated_minus_reference_rate": (generated_matches - reference_matches)
        / count,
        "prediction_agreement": int((generated_ids == reference_ids).sum()),
        "prediction_agreement_rate": float((generated_ids == reference_ids).mean()),
        "both_correct": both_correct,
        "generated_only_correct": generated_only,
        "reference_only_correct": reference_only,
        "both_wrong": both_wrong,
        "retention_given_reference_correct": both_correct / max(reference_matches, 1),
        "correction_given_reference_wrong": generated_only
        / max(count - reference_matches, 1),
        "mcnemar_two_sided_exact_p": p_value,
        "generated_significantly_worse_at_0_01": bool(
            reference_only > generated_only and p_value < 0.01
        ),
    }


def _condition_vectors(
    generated: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    axis: str,
) -> dict[str, Any]:
    if axis == "all":
        generated_correct = [bool(item["exact_condition_match"]) for item in generated]
        reference_correct = [bool(item["exact_condition_match"]) for item in reference]
        generated_prediction = [
            int(item["predicted_morphology_id"]) * 1000
            + int(item["predicted_subtype_id"]) * 10
            + int(item["predicted_role_id"])
            for item in generated
        ]
        reference_prediction = [
            int(item["predicted_morphology_id"]) * 1000
            + int(item["predicted_subtype_id"]) * 10
            + int(item["predicted_role_id"])
            for item in reference
        ]
    else:
        generated_correct = [bool(item[f"{axis}_match"]) for item in generated]
        reference_correct = [bool(item[f"{axis}_match"]) for item in reference]
        generated_prediction = [int(item[f"predicted_{axis}_id"]) for item in generated]
        reference_prediction = [int(item[f"predicted_{axis}_id"]) for item in reference]
    return paired_classification_statistics(
        generated_correct,
        reference_correct,
        generated_prediction,
        reference_prediction,
    )


def _verify_provenance(
    bank: Mapping[str, Any], benchmark: Mapping[str, Any], bundle: Any
) -> None:
    bank_provenance = bank["provenance"]
    benchmark_provenance = benchmark["provenance"]
    active = bundle.provenance()
    for name in (
        "checkpoint_sha256",
        "canonical_ema_hash",
        "training_source_hash",
        "legal_tuple_fingerprint",
        "published_next_epoch",
        "global_step",
    ):
        if bank_provenance[name] != active[name]:
            raise ValueError(f"Bank checkpoint provenance changed at {name!r}")
        if benchmark_provenance[name] != active[name]:
            raise ValueError(f"Benchmark checkpoint provenance changed at {name!r}")
    if bank_provenance["corpus"]["file_sha256"] != bundle.corpus.file_sha256:
        raise ValueError("Bank corpus SHA-256 differs from the checkpoint corpus")
    if benchmark_provenance["corpus"]["file_sha256"] != bundle.corpus.file_sha256:
        raise ValueError("Benchmark corpus SHA-256 differs from the checkpoint corpus")
    if bank_provenance["split"] != benchmark_provenance["split"]:
        raise ValueError("Bank and benchmark split provenance differ")
    active_evaluation_hash = evaluation_source_hash()
    if bank["evaluation_source_hash"] != active_evaluation_hash:
        raise ValueError("Bank evaluation source hash differs from the active evaluator")
    if benchmark["evaluation_source_hash"] != active_evaluation_hash:
        raise ValueError("Benchmark evaluation source hash differs from the active evaluator")


def _intervention_summary(
    baseline: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    baseline_cpu = tuple(values.detach().cpu() for values in baseline)
    candidate_cpu = tuple(values.detach().cpu() for values in candidate)
    count = int(baseline_cpu[0].shape[0])
    per_sample = []
    for index in range(count):
        changed = (
            (baseline_cpu[0][index] != candidate_cpu[0][index])
            | (baseline_cpu[1][index] != candidate_cpu[1][index])
            | (baseline_cpu[2][index] != candidate_cpu[2][index])
        )
        baseline_visible = baseline_cpu[0][index] != 0
        candidate_visible = candidate_cpu[0][index] != 0
        intersection = int((baseline_visible & candidate_visible).sum().item())
        union = int((baseline_visible | candidate_visible).sum().item())
        per_sample.append(
            {
                "categorical_hamming": float(changed.float().mean().item()),
                "part_hamming": float(
                    (baseline_cpu[0][index] != candidate_cpu[0][index])
                    .float()
                    .mean()
                    .item()
                ),
                "material_hamming": float(
                    (baseline_cpu[1][index] != candidate_cpu[1][index])
                    .float()
                    .mean()
                    .item()
                ),
                "emission_hamming": float(
                    (baseline_cpu[2][index] != candidate_cpu[2][index])
                    .float()
                    .mean()
                    .item()
                ),
                "silhouette_iou": intersection / max(union, 1),
                "exactly_unchanged": not bool(changed.any().item()),
            }
        )

    def mean(name: str) -> float:
        return float(np.mean([float(item[name]) for item in per_sample]))

    categorical = [float(item["categorical_hamming"]) for item in per_sample]
    unchanged = sum(bool(item["exactly_unchanged"]) for item in per_sample)
    return {
        "samples": count,
        "exactly_unchanged": unchanged,
        "exactly_changed": count - unchanged,
        "mean_categorical_hamming": mean("categorical_hamming"),
        "minimum_categorical_hamming": min(categorical),
        "maximum_categorical_hamming": max(categorical),
        "mean_part_hamming": mean("part_hamming"),
        "mean_material_hamming": mean("material_hamming"),
        "mean_emission_hamming": mean("emission_hamming"),
        "mean_silhouette_iou": mean("silhouette_iou"),
        "condition_effect_detected": unchanged < count,
        "per_sample": per_sample,
    }


@torch.inference_mode()
def _same_noise_interventions(
    checkpoint_path: Path,
    *,
    corpus_path: Path,
    device: str,
    precision: str,
) -> dict[str, Any]:
    bundle = load_multifield_checkpoint(
        checkpoint_path,
        corpus_path=corpus_path,
        device=device,
        precision=precision,
    )
    # Eight cells cover every role and cycle through all five families. Each
    # baseline/counterfactual call receives freshly reset identical generators.
    grid = build_condition_grid(bundle, mode="stratified", samples_per_condition=1)
    records = [grid[(role % 5) * 8 + role] for role in range(8)]
    cpu_batch = condition_batch(bundle, records)
    batch = {
        name: values.to(bundle.device, non_blocking=bundle.device.type == "cuda")
        for name, values in cpu_batch.items()
    }
    legal = torch.as_tensor(bundle.legal_tuples, dtype=torch.long, device=bundle.device)
    seeds = [record.sample_seed for record in records]

    def sample(
        morphologies: torch.Tensor,
        subtypes: torch.Tensor,
        roles: torch.Tensor,
        genes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bundle.precision == "fp32":
            context = contextlib.nullcontext()
        else:
            dtype = torch.bfloat16 if bundle.precision == "bf16" else torch.float16
            context = torch.autocast(device_type=bundle.device.type, dtype=dtype)
        with context:
            return bundle.model.sample(
                batch["guide"],
                morphologies,
                subtypes,
                roles,
                genes,
                temperature=0.9,
                generators=seeded_generators(seeds, bundle.device),
                legal_tuples=legal,
            )

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)
    started = time.perf_counter()
    baseline = sample(
        batch["morphology"], batch["subtype"], batch["role"], batch["genes"]
    )
    control = sample(
        batch["morphology"], batch["subtype"], batch["role"], batch["genes"]
    )
    control_summary = _intervention_summary(baseline, control)
    if control_summary["exactly_unchanged"] != len(records):
        raise RuntimeError("Same-noise baseline control did not replay exactly")

    local_subtype = batch["subtype"] % 4
    next_family = (batch["morphology"] + 1) % bundle.model.morphology_count
    variants = {
        "morphology_with_family_local_subtype": sample(
            next_family,
            next_family * 4 + local_subtype,
            batch["role"],
            batch["genes"],
        ),
        "subtype_within_family": sample(
            batch["morphology"],
            batch["morphology"] * 4 + ((local_subtype + 1) % 4),
            batch["role"],
            batch["genes"],
        ),
        "role": sample(
            batch["morphology"],
            batch["subtype"],
            (batch["role"] + 1) % bundle.model.role_count,
            batch["genes"],
        ),
        "genes_inverted": sample(
            batch["morphology"],
            batch["subtype"],
            batch["role"],
            1.0 - batch["genes"],
        ),
    }
    summaries = {
        name: _intervention_summary(baseline, candidate)
        for name, candidate in variants.items()
    }
    elapsed = time.perf_counter() - started
    result = {
        "status": "ready",
        "contract": "same-noise-axis-sensitivity-v1",
        "interpretation": (
            "Causal sensitivity probe only: a changed output proves that an axis "
            "affects generation, not that an out-of-distribution counterfactual is valid."
        ),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "device": str(bundle.device),
        "precision": bundle.precision,
        "gpu_name": (
            torch.cuda.get_device_name(bundle.device)
            if bundle.device.type == "cuda"
            else None
        ),
        "samples": len(records),
        "records": [record.metadata() for record in records],
        "temperature": 0.9,
        "same_seed_control": control_summary,
        "axes": summaries,
        "all_axes_effect_detected": all(
            values["condition_effect_detected"] for values in summaries.values()
        ),
        "elapsed_seconds": elapsed,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(bundle.device))
            if bundle.device.type == "cuda"
            else 0
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(bundle.device))
            if bundle.device.type == "cuda"
            else 0
        ),
    }
    return result


def audit_conditioning_bank(
    bank_manifest_path: Path,
    *,
    benchmark_path: Path,
    checkpoint_path: Path | None = None,
    corpus_path: Path | None = None,
    output_path: Path | None = None,
    intervention_device: str | None = None,
    intervention_precision: str = "bf16",
) -> dict[str, Any]:
    bank_manifest_path = Path(bank_manifest_path).resolve()
    benchmark_path = Path(benchmark_path).resolve()
    bank_root = bank_manifest_path.parent
    bank = _read_json(bank_manifest_path)
    benchmark = _read_json(benchmark_path)
    validate_manifest_schema(bank, GENERATION_BANK_SCHEMA)
    validate_manifest_schema(benchmark, BENCHMARK_SCHEMA)
    if bank.get("format") != GENERATION_BANK_FORMAT or bank.get("status") != "ready":
        raise ValueError("Generation bank is not a ready supported bank")
    if benchmark.get("format") != BENCHMARK_FORMAT or benchmark.get("status") != "ready":
        raise ValueError("Benchmark is not a ready supported benchmark")

    recorded_checkpoint = Path(str(bank["provenance"]["checkpoint"])).resolve()
    checkpoint = recorded_checkpoint if checkpoint_path is None else Path(checkpoint_path).resolve()
    bundle = load_multifield_checkpoint(
        checkpoint,
        corpus_path=corpus_path,
        device="cpu",
        precision="fp32",
    )
    if checkpoint != recorded_checkpoint and bundle.checkpoint_sha256 != str(
        bank["provenance"]["checkpoint_sha256"]
    ):
        raise ValueError("Relocated checkpoint does not match the recorded checkpoint")
    _verify_provenance(bank, benchmark, bundle)
    templates = ConditionTemplateBank.build(bundle)
    if templates.fingerprint != bank["condition_template_fingerprint"]:
        raise ValueError("Condition template fingerprint differs from the bank")

    generated_classifications: list[dict[str, Any]] = []
    reference_classifications: list[dict[str, Any]] = []
    sample_comparisons: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in bank["samples"]:
        condition = entry["condition"]
        record = _record_from_metadata(condition)
        if record.sample_id != condition["sample_id"] or record.sample_id in seen_ids:
            raise ValueError("Bank contains a duplicate or inconsistent sample id")
        seen_ids.add(record.sample_id)
        raw_manifest_path = _verify_artifact(bank_root, entry["raw_manifest"])
        raw_manifest = _read_json(raw_manifest_path)
        validate_manifest_schema(raw_manifest, RAW_SAMPLE_SCHEMA)
        if raw_manifest["condition"] != condition:
            raise ValueError(f"Raw condition differs for {record.sample_id}")
        if raw_manifest["validation"] != entry["raw_validation"]:
            raise ValueError(f"Raw validation copy differs for {record.sample_id}")
        for name in ("checkpoint_sha256", "canonical_ema_hash", "corpus_sha256"):
            expected = {
                "checkpoint_sha256": bundle.checkpoint_sha256,
                "canonical_ema_hash": bundle.payload["canonical_ema_hash"],
                "corpus_sha256": bundle.corpus.file_sha256,
            }[name]
            if raw_manifest[name] != expected:
                raise ValueError(f"Raw provenance differs at {name} for {record.sample_id}")
        fields_path = _verify_artifact(bank_root, raw_manifest["artifacts"]["fields"])
        _verify_artifact(bank_root, raw_manifest["artifacts"]["rgba"])
        _verify_artifact(bank_root, raw_manifest["artifacts"]["emission"])
        arrays = _load_raw_npz(fields_path)
        scalar_expectations = {
            "morphology": record.morphology,
            "subtype": record.subtype,
            "role": record.role,
            "source_index": record.source_index,
            "sample_seed": record.sample_seed,
            "corpus_seed": int(bundle.corpus.seeds[record.source_index]),
        }
        for name, expected in scalar_expectations.items():
            if int(arrays[name][0]) != expected:
                raise ValueError(f"Raw scalar {name} differs for {record.sample_id}")
        if not np.array_equal(arrays["genes"], bundle.corpus.genes[record.source_index]):
            raise ValueError(f"Genes differ from held-out source for {record.sample_id}")
        target = (
            bundle.corpus.part_owner[record.source_index],
            bundle.corpus.material[record.source_index],
            bundle.corpus.emission_level[record.source_index],
        )
        for name, observed, expected in zip(
            ("target_part", "target_material", "target_emission"),
            (arrays["target_part"], arrays["target_material"], arrays["target_emission"]),
            target,
        ):
            if not np.array_equal(observed, expected):
                raise ValueError(f"{name} differs from held-out source for {record.sample_id}")
        expected_guide = MorphologyCorpusDataset(
            bundle.corpus, [record.source_index], guide_policy=bundle.guide_policy
        )[0]["guide"].numpy()
        if not np.array_equal(arrays["guide"], expected_guide):
            raise ValueError(f"Guide differs from held-out source for {record.sample_id}")
        generated_fields = (arrays["part"], arrays["material"], arrays["emission"])
        if aligned_fields_hash(*generated_fields) != raw_manifest["raw_fields_sha256"]:
            raise ValueError(f"Raw field digest differs for {record.sample_id}")
        recomputed = validate_generated_fields(
            *generated_fields,
            guide=arrays["guide"],
            target=target,
            record=record,
            legal_tuples=bundle.legal_tuples,
            templates=templates,
        )
        if recomputed != raw_manifest["validation"]:
            raise ValueError(f"Raw validation does not recompute for {record.sample_id}")
        generated_condition = recomputed["condition_adherence"]
        reference_condition = templates.classify(*target, record)
        generated_classifications.append(generated_condition)
        reference_classifications.append(reference_condition)
        sample_comparisons.append(
            {
                "sample_id": record.sample_id,
                "source_index": record.source_index,
                "raw_fields_sha256": raw_manifest["raw_fields_sha256"],
                "condition": {
                    "morphology_id": record.morphology,
                    "subtype_id": record.subtype,
                    "role_id": record.role,
                },
                "generated_prediction": {
                    axis: int(generated_condition[f"predicted_{axis}_id"])
                    for axis in ("morphology", "subtype", "role")
                },
                "reference_prediction": {
                    axis: int(reference_condition[f"predicted_{axis}_id"])
                    for axis in ("morphology", "subtype", "role")
                },
                "generated_exact": bool(generated_condition["exact_condition_match"]),
                "reference_exact": bool(reference_condition["exact_condition_match"]),
            }
        )

    if len(sample_comparisons) != int(bank["grid"]["samples"]):
        raise ValueError("Bank grid count differs from its sample list")
    paired = {
        axis: _condition_vectors(generated_classifications, reference_classifications, axis)
        for axis in ("morphology", "subtype", "role", "all")
    }
    reference_calibration = calibrate_reference_fields(bundle, templates=templates)
    metrics = benchmark["full_mask"]["metrics"]
    counterfactual = {
        "examples": int(benchmark["full_mask"]["examples_evaluated"]),
        "combined": {
            "preference_rate": float(metrics["validation_condition_preference_rate"]),
            "mean_nll_margin": float(metrics["validation_condition_nll_margin"]),
        },
        "morphology_subtype": {
            "preference_rate": float(
                metrics["validation_morphology_subtype_preference_rate"]
            ),
            "mean_nll_margin": float(
                metrics["validation_morphology_subtype_nll_margin"]
            ),
        },
        "role": {
            "preference_rate": float(metrics["validation_role_preference_rate"]),
            "mean_nll_margin": float(metrics["validation_role_nll_margin"]),
        },
        "genes": {
            "preference_rate": float(metrics["validation_genes_preference_rate"]),
            "mean_nll_margin": float(metrics["validation_genes_nll_margin"]),
        },
    }
    counterfactual_support = all(
        values["preference_rate"] >= 0.95 and values["mean_nll_margin"] > 0.0
        for name, values in counterfactual.items()
        if name != "examples"
    )
    if intervention_device is None:
        intervention = {
            "status": "not_run",
            "reason": "No same-noise intervention device was requested.",
        }
        intervention_support = True
    else:
        intervention = _same_noise_interventions(
            bundle.path,
            corpus_path=bundle.corpus.path,
            device=intervention_device,
            precision=intervention_precision,
        )
        intervention_support = bool(intervention["all_axes_effect_detected"])
    reference_exact = reference_calibration["diagnostic_condition_adherence"][
        "exact_match"
    ]
    proxy_ceiling_limited = bool(
        reference_exact["subtype"]["rate"] < 0.5
        or reference_exact["all"]["rate"] < 0.5
    )
    paired_regression = any(
        paired[axis]["generated_significantly_worse_at_0_01"]
        for axis in paired
    )
    condition_use_supported = counterfactual_support and intervention_support
    if proxy_ceiling_limited and not paired_regression and condition_use_supported:
        conclusion = "weak_proxy_dominates_no_conditioning_regression_detected"
        training_change_warranted = False
    elif paired_regression:
        conclusion = "paired_conditioning_regression_detected"
        training_change_warranted = True
    else:
        conclusion = "conditioning_evidence_inconclusive"
        training_change_warranted = False

    report = {
        "format": CONDITIONING_AUDIT_FORMAT,
        "status": "ready",
        "source_provenance": {
            "conditioning_audit_source_hash": conditioning_audit_source_hash(),
            "evaluation_source_hash": evaluation_source_hash(),
            "bank_manifest": str(bank_manifest_path),
            "bank_manifest_sha256": _sha256_file(bank_manifest_path),
            "benchmark": str(benchmark_path),
            "benchmark_sha256": _sha256_file(benchmark_path),
        },
        "checkpoint": {
            "path": str(bundle.path),
            "sha256": bundle.checkpoint_sha256,
            "canonical_ema_hash": str(bundle.payload["canonical_ema_hash"]),
            "training_source_hash": str(bundle.payload["training_source_hash"]),
            "published_next_epoch": int(bundle.payload["next_epoch"]),
            "global_step": int(bundle.payload["global_step"]),
            "corpus_sha256": bundle.corpus.file_sha256,
            "split_fingerprint": str(bundle.payload["split"]["fingerprint"]),
            "condition_template_fingerprint": templates.fingerprint,
        },
        "bank": {
            "samples": len(sample_comparisons),
            "grid": dict(bank["grid"]),
            "hard_valid_rate": float(
                bank["validation"]["acceptance"]["overall"]["acceptance_rate"]
            ),
            "exact_unique_fraction": float(
                bank["validation"]["diversity"]["exact_unique_fraction"]
            ),
        },
        "reference_calibration": reference_calibration,
        "paired_reference_normalized": paired,
        "denoising_counterfactual": counterfactual,
        "same_noise_intervention": intervention,
        "decision": {
            "proxy_ceiling_limited": proxy_ceiling_limited,
            "paired_regression_detected": paired_regression,
            "counterfactual_condition_use_supported": condition_use_supported,
            "conclusion": conclusion,
            "training_change_warranted": training_change_warranted,
            "training_run_started": False,
            "rationale": (
                "Nearest-centroid exact match is normalized against untouched held-out "
                "references and exact source/output pairs. A new training objective is "
                "not selected unless the paired test detects a significant regression."
            ),
        },
        "samples": sample_comparisons,
    }
    validate_manifest_schema(report, CONDITIONING_AUDIT_SCHEMA)
    if output_path is not None:
        destination = Path(output_path).resolve()
        if destination.exists():
            raise FileExistsError(
                f"Conditioning audits are immutable and will not be overwritten: {destination}"
            )
        require_disk_floor(destination.parent, planned_bytes=16 * 1024 * 1024)
        write_json_atomic(destination, report)
    return report
