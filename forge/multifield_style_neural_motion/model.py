from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..multifield_style.model import LoadedGenerationBank, LoadedSourceSample
from ..neural_rig_bridge import NeuralRigBinding


@dataclass(frozen=True, slots=True)
class NeuralMotionCandidate:
    sample: LoadedSourceSample
    source_entry: Mapping[str, Any]
    raw_archive_path: Path
    raw_manifest_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_entry", MappingProxyType(dict(self.source_entry)))


@dataclass(frozen=True, slots=True)
class NeuralMotionSource:
    bank: LoadedGenerationBank
    corpus_path: Path
    corpus_sha256: str
    corpus_bytes: int
    corpus_source_sha256: str
    split_fingerprint: str
    split_seed: int
    validation_fraction: float
    legal_tuples: np.ndarray
    legal_tuple_fingerprint: str
    candidates_by_family: Mapping[str, tuple[NeuralMotionCandidate, ...]]

    def __post_init__(self) -> None:
        if self.legal_tuples.dtype != np.uint8 or self.legal_tuples.ndim != 2 or self.legal_tuples.shape[1:] != (3,):
            raise ValueError("Neural motion legal tuple table must be uint8 [N, 3]")
        self.legal_tuples.flags.writeable = False
        object.__setattr__(
            self,
            "candidates_by_family",
            MappingProxyType(
                {family: tuple(candidates) for family, candidates in self.candidates_by_family.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class SelectedNeuralIdentity:
    candidate: NeuralMotionCandidate
    binding: NeuralRigBinding


@dataclass(frozen=True, slots=True)
class NeuralStyleParent:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_bytes: int
    manifest: Mapping[str, Any]
    palettes: Mapping[str, Mapping[str, Any]]
    palette_artifacts: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(
            self,
            "palettes",
            MappingProxyType(
                {sample_id: MappingProxyType(dict(palette)) for sample_id, palette in self.palettes.items()}
            ),
        )
        object.__setattr__(
            self,
            "palette_artifacts",
            MappingProxyType(
                {sample_id: MappingProxyType(dict(record)) for sample_id, record in self.palette_artifacts.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class NeuralPresentationFrame:
    layers: Mapping[str, np.ndarray]
    palette: Mapping[str, Any]
    palette_sha256: str
    categorical_sha256: str
    aligned_fields_sha256: str
    bound_frame_sha256: str
    motion_frame_sha256: str
    presentation_sha256: tuple[str, ...]
    gates: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class NeuralIdentityPayload:
    family: str
    sample_id: str
    file_payloads: Mapping[str, bytes]
    identity_manifest: Mapping[str, Any]
    frame_count: int
    clip_count: int
