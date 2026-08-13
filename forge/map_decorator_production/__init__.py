"""Process-isolated production corpus and CUDA training for neural map decoration."""

from .contract import (
    CORPUS_FORMAT_VERSION,
    OBJECTIVE_BUCKETS,
    PRODUCTION_CONTRACT_SHA256,
    SENTINEL_DIMENSIONS,
    SIZE_PROFILES,
    CorpusConfig,
    ObjectiveBucket,
    SizeProfile,
    production_contract_manifest,
)
from .teacher import (
    ProductionSample,
    build_production_sample,
    canonical_full_map_identity,
    semantic_teacher_targets,
)

__all__ = [
    "CORPUS_FORMAT_VERSION",
    "OBJECTIVE_BUCKETS",
    "PRODUCTION_CONTRACT_SHA256",
    "SENTINEL_DIMENSIONS",
    "SIZE_PROFILES",
    "CorpusConfig",
    "ObjectiveBucket",
    "SizeProfile",
    "ProductionSample",
    "build_production_sample",
    "canonical_full_map_identity",
    "production_contract_manifest",
    "semantic_teacher_targets",
]
