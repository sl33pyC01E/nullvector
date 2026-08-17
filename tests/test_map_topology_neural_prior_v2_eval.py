from __future__ import annotations

import numpy as np

from forge.map_topology_neural_prior_v2_eval.evaluate import _radius_one, _reachable


def test_reachability_requires_every_target() -> None:
    mask=np.zeros((7,7),dtype=bool);mask[1,1:6]=True;mask[1:6,5]=True
    assert _reachable(mask,(1,1),((5,1),(5,5)))
    assert not _reachable(mask,(1,1),((5,1),(1,5)))


def test_radius_one_rejects_single_cell_corridor() -> None:
    mask=np.zeros((9,9),dtype=bool);mask[3:6,1:8]=True
    eroded=_radius_one(mask)
    assert eroded[4,4]
    assert not eroded[3,4]
    thin=np.zeros((9,9),dtype=bool);thin[4,1:8]=True
    assert not _radius_one(thin).any()
