from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..multifield_data import (
    GuidePolicy,
    MorphologyCorpus,
    compute_legal_tuples,
    legal_tuple_fingerprint,
    stratified_corpus_split,
)
from ..safety import write_json_atomic
from .schema import CALIBRATION_SCHEMA, validate_manifest_schema
from .validation import ConditionTemplateBank, calibrate_reference_fields


@dataclass(slots=True)
class CorpusCalibrationContext:
    corpus: MorphologyCorpus
    training_indices: np.ndarray
    validation_indices: np.ndarray
    guide_policy: GuidePolicy
    legal_tuples: np.ndarray
    payload: dict[str, Any]


def calibrate_morphology_corpus(
    corpus_path: Path,
    *,
    validation_fraction: float = 0.08,
    split_seed: int = 0x5A17,
    guide_policy: GuidePolicy = GuidePolicy(),
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Calibrate hard gates on the complete held-out reference partition."""

    corpus = MorphologyCorpus.load(
        Path(corpus_path).resolve(), verify_hash=True, verify_source=True
    )
    split = stratified_corpus_split(
        corpus,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    legal_tuples = compute_legal_tuples(corpus, split.training)
    context = CorpusCalibrationContext(
        corpus=corpus,
        training_indices=split.training,
        validation_indices=split.validation,
        guide_policy=guide_policy,
        legal_tuples=legal_tuples,
        payload={"split": split.metadata()},
    )
    templates = ConditionTemplateBank.build(context)
    calibration = calibrate_reference_fields(context, templates=templates)
    report = {
        **calibration,
        "status": "calibrated",
        "corpus": corpus.metadata(),
        "split": split.metadata(),
        "guide_policy": guide_policy.metadata(),
        "legal_tuple_count": int(len(legal_tuples)),
        "legal_tuple_fingerprint": legal_tuple_fingerprint(legal_tuples),
        "condition_template_fingerprint": templates.fingerprint,
    }
    from .pipeline import evaluation_source_hash

    report["evaluation_source_hash"] = evaluation_source_hash()
    validate_manifest_schema(report, CALIBRATION_SCHEMA)
    if output_path is not None:
        destination = Path(output_path).resolve()
        if destination.exists():
            raise FileExistsError(
                f"Calibration reports are immutable and will not be overwritten: {destination}"
            )
        write_json_atomic(destination, report)
    return report
