from __future__ import annotations

from forge.nature_neural_runtime_v2.runtime import _component_table


def test_live_nature_runtime_component_release_is_complete():
    rows = _component_table()
    assert {"locomotion_25d", "behavior", "colony", "society", "timeline", "counterfactual"} <= rows.keys()
    for name in ("locomotion_25d", "behavior", "colony", "society", "timeline", "counterfactual"):
        assert all(rows[name]["quality_gates"].values())
