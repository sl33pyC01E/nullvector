from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from ..maps.model import MapData
from .catalog import LegalClassMasks, build_legal_class_masks, validate_decoration_fields
from .features import EncodedFeatures, encode_features, validate_encoded_features


@dataclass(frozen=True, slots=True)
class FoundationResult:
    features: EncodedFeatures
    legal_masks: LegalClassMasks
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class FoundationCase:
    data: MapData
    protected_backbone: np.ndarray
    required_clearance: np.ndarray
    decoration_forbidden: np.ndarray
    public_seed: int


def build_foundation(
    data: MapData,
    *,
    protected_backbone: np.ndarray,
    required_clearance: np.ndarray,
    decoration_forbidden: np.ndarray,
    public_seed: int,
) -> FoundationResult:
    """Build and validate the complete in-memory pre-model contract without disk I/O."""
    features = encode_features(
        data,
        protected_backbone=protected_backbone,
        required_clearance=required_clearance,
        decoration_forbidden=decoration_forbidden,
        public_seed=public_seed,
    )
    feature_report = validate_encoded_features(
        data,
        features,
        protected_backbone=protected_backbone,
        required_clearance=required_clearance,
        decoration_forbidden=decoration_forbidden,
    )
    legal_masks = build_legal_class_masks(
        data,
        protected_backbone=protected_backbone,
        required_clearance=required_clearance,
        decoration_forbidden=decoration_forbidden,
    )
    shape = data.shape
    empty_fields = {
        "variant": np.zeros(shape, dtype=np.uint8),
        "decal": np.zeros(shape, dtype=np.uint8),
        "prop": np.zeros(shape, dtype=np.uint8),
        "emission": np.zeros(shape, dtype=np.uint8),
    }
    legality_report = validate_decoration_fields(
        data,
        protected_backbone=protected_backbone,
        required_clearance=required_clearance,
        decoration_forbidden=decoration_forbidden,
        **empty_fields,
    )
    report = {
        "passed": bool(feature_report["passed"] and legality_report["passed"]),
        "map_id": data.map_id,
        "theme": data.theme,
        "feature_contract_sha256": features.channel_manifest_sha256,
        "feature_tensor_sha256": features.tensor_sha256,
        "catalog_sha256": legal_masks.catalog_sha256,
        "legal_masks_sha256": legal_masks.masks_sha256,
        "feature_validation": feature_report,
        "empty_selection_validation": legality_report,
    }
    return FoundationResult(features, legal_masks, report)


def fuzz_foundation(
    cases: Iterable[FoundationCase], *, require_all_themes: bool = False
) -> dict[str, object]:
    """Exercise supplied provenance masks twice per case, entirely in memory."""
    failures: list[dict[str, object]] = []
    signatures: set[str] = set()
    themes: dict[str, int] = {}
    case_count = 0
    for index, case in enumerate(cases):
        case_count += 1
        themes[case.data.theme] = themes.get(case.data.theme, 0) + 1
        try:
            first = build_foundation(
                case.data,
                protected_backbone=case.protected_backbone,
                required_clearance=case.required_clearance,
                decoration_forbidden=case.decoration_forbidden,
                public_seed=case.public_seed,
            )
            second = build_foundation(
                case.data,
                protected_backbone=case.protected_backbone,
                required_clearance=case.required_clearance,
                decoration_forbidden=case.decoration_forbidden,
                public_seed=case.public_seed,
            )
        except Exception as error:
            failures.append(
                {
                    "index": index,
                    "map_id": case.data.map_id,
                    "failures": ["exception"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        signatures.add(first.features.tensor_sha256)
        case_failures: list[str] = []
        if not first.report["passed"] or not second.report["passed"]:
            case_failures.append("validation")
        if first.features.tensor_sha256 != second.features.tensor_sha256:
            case_failures.append("feature_replay")
        if first.legal_masks.masks_sha256 != second.legal_masks.masks_sha256:
            case_failures.append("legal_mask_replay")
        if case_failures:
            failures.append(
                {
                    "index": index,
                    "map_id": case.data.map_id,
                    "failures": case_failures,
                }
            )
    if require_all_themes:
        from ..maps.model import THEMES

        missing = [theme for theme in THEMES if themes.get(theme, 0) == 0]
        if missing:
            failures.append({"index": None, "map_id": None, "failures": [f"missing_theme:{x}" for x in missing]})
    return {
        "passed": not failures,
        "case_count": case_count,
        "unique_feature_tensors": len(signatures),
        "per_theme": themes,
        "failures": failures,
        "disk_io": False,
    }
