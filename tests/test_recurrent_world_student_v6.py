from __future__ import annotations
import inspect
from forge.recurrent_world_student_v6 import contract,evaluation,training


def test_v6_binds_selected_perception_parent_and_natural_corpus():
    assert contract.PARENT_SHA256=="2fcc958bdcc513ae72ddf5424887503cc47fbbaa7060b93470f153651f98fadf"
    assert "selected_v1_update1700" in contract.PARENT.as_posix()
    assert "world_action_natural_v10" in contract.CORPUS.as_posix()


def test_v6_trains_recurrent_rollouts_with_learned_trust():
    source=inspect.getsource(training.train)
    assert "range(plan.rollout_steps)" in source
    assert "proposal*truth" in source
    assert "previous,current=current.detach(),next_latent.detach()" in source


def test_v6_selects_without_touching_test_and_requires_long_horizons():
    source=inspect.getsource(evaluation.evaluate)
    assert "min(rows" in source
    assert source.index("min(rows")<source.index("sequences[5]")
    assert "(1,2,4,8,16,32)" in source
    assert "all_horizons_beat_persistence" in source
