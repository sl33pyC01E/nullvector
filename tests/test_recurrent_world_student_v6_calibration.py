from __future__ import annotations
import inspect
from forge.recurrent_world_student_v6 import calibration


def test_trust_bias_is_selected_on_validation_before_test():
    source=inspect.getsource(calibration.calibrate)
    assert source.index("sequences[4]")<source.index("chosen_bias,chosen_ramp=min")<source.index("sequences[5]")
    assert 'motion_ratio"]>=.25' in source
    assert "gate_logit_bias_ramp_steps" in source


def test_calibration_compares_persistence_and_frozen_v5_parent():
    source=inspect.getsource(calibration.calibrate)
    assert "all_test_horizons_beat_persistence" in source
    assert "all_test_horizons_beat_v5_parent" in source
    assert "candidate_vs_v5_parent" in source
