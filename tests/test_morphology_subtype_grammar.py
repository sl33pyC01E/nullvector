from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from forge.morphology import generate_motion_clip, genome_from_seed, validate_motion_clip, validate_specimen
from forge.morphology_subtype_grammar import (
    DEFAULT_OUTPUT,
    VARIANT_NAMES,
    _build,
    build_bank,
    render_subtype_specimen,
    specialize_genome,
    validate_bank,
)


def test_all_family_subtype_role_combinations_preserve_current_sprite_contract() -> None:
    hashes = set()
    for family_id in range(5):
        for variant in range(4):
            for role_id in range(8):
                seed = 0x51500000 + family_id * 0x10000 + variant * 0x1000 + role_id * 131
                base = replace(genome_from_seed(seed, family_id), silhouette_variant=variant, subtype_id=family_id * 4 + variant, role_id=role_id)
                specialized = specialize_genome(base)
                assert specialized.family == family_id and specialized.silhouette_variant == variant
                assert specialized.subtype_id == family_id * 4 + variant
                assert specialized.asymmetry in (-1, 0, 1) and specialized.x_offset == 0
                specimen = render_subtype_specimen(base)
                assert validate_specimen(specimen) == []
                hashes.add(specimen.manifest["hashes"]["semantic_sha256"])
    assert len(hashes) == 160


def test_subtype_chassis_are_classifiable_without_sacrificing_soft_symmetry() -> None:
    _, report = _build()
    assert report["status"] == "ready" and all(report["gates"].values())
    assert min(item["accuracy"] for item in report["classification"]) >= 0.75
    assert min(row["recall"] for item in report["classification"] for row in item["per_class"]) >= 0.5
    assert min(report["aggregate"]["mean_horizontal_symmetry_by_family"]) >= 0.68
    assert report["variant_vocab"] == [list(values) for values in VARIANT_NAMES]


def test_every_subtype_retains_breathing_locomotion_and_action_motion() -> None:
    clip_hashes = set()
    for family_id in range(5):
        for variant in range(4):
            seed = 0x4D4F0000 + family_id * 0x10000 + variant * 0x1000
            base = replace(genome_from_seed(seed, family_id), silhouette_variant=variant, subtype_id=family_id * 4 + variant)
            specimen = render_subtype_specimen(base)
            for motion, facing in (("idle_breathe", "north"), ("locomote", "southeast"), ("attack", "west"), ("cast", "northeast")):
                clip = generate_motion_clip(specimen, motion, facing=facing)
                assert validate_motion_clip(clip) == []
                assert clip.manifest["metrics"]["unique_semantic_frames"] >= 2
                assert clip.manifest["metrics"]["max_changed_pixel_fraction"] > 0
                clip_hashes.add(clip.sha256)
    assert len(clip_hashes) == 5 * 4 * 4


def test_bank_is_canonical_and_exactly_replayable(tmp_path: Path) -> None:
    destination = tmp_path / "subtypes"
    result = build_bank(destination)
    assert result["passed"] is True and result["sample_count"] == 160
    assert validate_bank(destination / "morphology_subtype_grammar.json") == result
    manifest = destination / "morphology_subtype_grammar.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["classification"][0]["accuracy"] = 1.0
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_bank(manifest)


def test_published_bank_replays_when_present() -> None:
    manifest = DEFAULT_OUTPUT / "morphology_subtype_grammar.json"
    if manifest.is_file():
        assert validate_bank(manifest)["passed"] is True
