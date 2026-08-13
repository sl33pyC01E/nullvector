from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge.multifield_style_neural_motion.rendering import render_neural_motion_frame
from forge.neural_rig_repair.binding import bind_repair_plan
from forge.neural_rig_repair.constants import MAX_PLAN_BYTES
from forge.neural_rig_repair.planner import load_repair_plan
from forge.neural_rig_repair.schema import resolve_artifact_record
from forge.neural_rig_repair_style import load_repair_style_authority, reconstruct_clip
from forge.neural_rig_repair_style.pilot import PILOT_ORDINALS


def _binding(authority, ordinal: int):
    record = authority.bank["plans"][ordinal]
    path = resolve_artifact_record(
        authority.bank_path.parent,
        record["artifact"],
        label="repair style test plan",
        maximum_bytes=MAX_PLAN_BYTES,
    )
    return bind_repair_plan(
        authority.repair_source,
        authority.repair_source.samples[ordinal],
        load_repair_plan(path),
        verify_exact_plan=True,
    )


@pytest.fixture(scope="module")
def authority():
    return load_repair_style_authority()


def test_reconstructs_one_loop_per_family_and_styles_without_field_changes(authority):
    for ordinal in PILOT_ORDINALS:
        binding = _binding(authority, ordinal)
        audit = next(
            clip
            for clip in authority.motion_audits[ordinal]["clips"]
            if clip["motion"] == "idle_breathe" and clip["facing"] == "northeast"
        )
        clip = reconstruct_clip(binding, audit)
        assert len(clip.frames) == 9
        assert clip.frames[0].fields.sha256 == clip.frames[-1].fields.sha256
        sample = authority.neural_source.bank.samples[ordinal]
        palette = authority.style_parent.palettes[binding.sample_id]
        palette_artifact = authority.style_parent.palette_artifacts[binding.sample_id]
        first = render_neural_motion_frame(
            clip.frames[0],
            sample.condition,
            sample.fields.aligned_sha256,
            palette,
            palette_artifact["sha256"],
        )
        last = render_neural_motion_frame(
            clip.frames[-1],
            sample.condition,
            sample.fields.aligned_sha256,
            palette,
            palette_artifact["sha256"],
        )
        assert all(first.gates.values())
        assert first.presentation_sha256 == last.presentation_sha256
        assert all(
            np.array_equal(first.layers[name], last.layers[name])
            for name in first.layers
        )


def test_rejects_audit_for_the_wrong_identity(authority):
    binding = _binding(authority, 0)
    wrong_audit = authority.motion_audits[16]["clips"][0]
    try:
        reconstruct_clip(binding, wrong_audit)
    except ValueError as error:
        assert "linkage" in str(error)
    else:
        raise AssertionError("wrong-identity motion audit was accepted")
