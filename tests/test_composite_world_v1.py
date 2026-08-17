from __future__ import annotations
from forge.composite_world_v1.contract import ARTIFACTS
def test_composite_has_visual_actor_and_body_authorities():
    assert {"action_dit","world_vae","actor_state","organism_vae","physiology"}<=set(ARTIFACTS)
