from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..multifield_style_neural_motion.model import NeuralMotionSource, NeuralStyleParent
from ..multifield_style_neural_motion.source import load_neural_motion_source
from ..multifield_style_neural_motion.style_parent import load_neural_style_parent
from ..neural_rig_repair.constants import (
    BANK_SCHEMA,
    EXPECTED_CLIP_COUNT,
    EXPECTED_FRAME_COUNT,
    EXPECTED_SAMPLE_COUNT,
    MAX_BANK_BYTES,
    MAX_JSON_BYTES,
    MAX_STRESS_REPORT_BYTES,
    MAX_STRESS_SHARD_BYTES,
    PROJECT_ROOT,
)
from ..neural_rig_repair.hashing import canonical_json_bytes, sha256_bytes, source_hash as repair_source_hash
from ..neural_rig_repair.model import RepairSource
from ..neural_rig_repair.schema import load_schema_json, resolve_artifact_record
from ..neural_rig_repair.source import load_repair_source
from ..neural_rig_repair.stress import load_stress_report, load_stress_shard


DEFAULT_REPAIR_BANK = (
    PROJECT_ROOT
    / "outputs"
    / "neural_rig_repair_v2_sharded_replay_20260813"
    / "repair_bank_manifest.json"
)


@dataclass(frozen=True, slots=True)
class RepairStyleAuthority:
    bank_path: Path
    bank: Mapping[str, Any]
    repair_source: RepairSource
    neural_source: NeuralMotionSource
    style_parent: NeuralStyleParent
    motion_audits: Mapping[int, Mapping[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bank", MappingProxyType(dict(self.bank)))
        object.__setattr__(
            self,
            "motion_audits",
            MappingProxyType({int(key): MappingProxyType(dict(value)) for key, value in self.motion_audits.items()}),
        )


def _verify_self_hash(payload: Mapping[str, Any], key: str) -> None:
    unsigned = dict(payload)
    stored = unsigned.pop(key, None)
    if stored != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError(f"repair style authority {key} mismatch")


def load_repair_style_authority(
    bank_path: Path = DEFAULT_REPAIR_BANK,
) -> RepairStyleAuthority:
    """Load the sealed repair bank and its two byte-exact motion sweeps.

    Loading is intentionally strict but does not rerender 75,520 frames.  The
    exporter independently reconstructs every clip it consumes and compares
    that result with this sealed audit.
    """

    bank_path = Path(bank_path).resolve()
    bank_root = bank_path.parent
    bank = load_schema_json(
        bank_path,
        maximum_bytes=MAX_BANK_BYTES,
        label="repair style authority bank",
        schema=BANK_SCHEMA,
    )
    _verify_self_hash(bank, "bank_sha256")
    if (
        bank["status"] != "ready"
        or bank["source"]["repair_source_sha256"] != repair_source_hash()
        or bank["build_contract"]["sample_count"] != EXPECTED_SAMPLE_COUNT
        or bank["build_contract"]["clip_count"] != EXPECTED_CLIP_COUNT
        or bank["build_contract"]["frame_count"] != EXPECTED_FRAME_COUNT
        or bank["gates"].get("independent_process_sharded_motion_replay_exact") is not True
        or any(value is not True for value in bank["gates"].values())
    ):
        raise ValueError("repair style authority bank is not the sealed all-80 bank")

    generation_path = resolve_artifact_record(
        PROJECT_ROOT,
        bank["source"]["generation_manifest"],
        label="repair style generation manifest",
        maximum_bytes=MAX_JSON_BYTES,
    )
    style_path = resolve_artifact_record(
        PROJECT_ROOT,
        bank["source"]["style_manifest"],
        label="repair style parent manifest",
        maximum_bytes=MAX_JSON_BYTES,
    )
    repair_source = load_repair_source(generation_path, style_path)
    neural_source = load_neural_motion_source(generation_path)
    style_parent = load_neural_style_parent(style_path, neural_source)

    reports = []
    for artifact_name in ("motion_stress", "motion_replay"):
        report_path = resolve_artifact_record(
            bank_root,
            bank["artifacts"][artifact_name],
            label=f"repair style {artifact_name}",
            maximum_bytes=MAX_STRESS_REPORT_BYTES,
        )
        reports.append((report_path, load_stress_report(report_path, verify_shards=True)))
    if canonical_json_bytes(reports[0][1]) != canonical_json_bytes(reports[1][1]):
        raise ValueError("repair style motion authorities are not byte-exact")

    motion_audits: dict[int, Mapping[str, Any]] = {}
    report_path, report = reports[0]
    for shard_record in report["shards"]:
        shard_path = resolve_artifact_record(
            report_path.parent,
            shard_record,
            label="repair style stress shard",
            maximum_bytes=MAX_STRESS_SHARD_BYTES,
        )
        shard = load_stress_shard(shard_path)
        for sample in shard["samples"]:
            ordinal = int(sample["ordinal"])
            if ordinal in motion_audits:
                raise ValueError("repair style authority duplicated an identity")
            motion_audits[ordinal] = sample["motion_audit"]
    if set(motion_audits) != set(range(EXPECTED_SAMPLE_COUNT)):
        raise ValueError("repair style authority does not cover all 80 identities")
    for ordinal, (sample, audit) in enumerate(
        zip(repair_source.samples, (motion_audits[index] for index in range(EXPECTED_SAMPLE_COUNT)), strict=True)
    ):
        if (
            sample.ordinal != ordinal
            or audit["sample_id"] != sample.sample_id
            or audit["clip_count"] != 104
            or audit["frame_count"] != 944
            or any(value is not True for value in audit["gates"].values())
        ):
            raise ValueError("repair style sample motion authority drifted")

    return RepairStyleAuthority(
        bank_path=bank_path,
        bank=bank,
        repair_source=repair_source,
        neural_source=neural_source,
        style_parent=style_parent,
        motion_audits=motion_audits,
    )
