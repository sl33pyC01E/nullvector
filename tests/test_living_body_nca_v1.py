from __future__ import annotations

import numpy as np
import pytest
import torch

from forge.creature_stage_developmental import develop
from forge.creature_stage_developmental.genomes import review_genomes
from forge.cellular_nca.contract import DIRECTION_XY
from forge.living_body_nca_v1 import LivingBodyNCARuntime, rasterize_body
from forge.living_body_nca_v1.evaluation import _audit_once, _damage_system
from forge.living_body_nca_v1.contract import DEFAULT_AUTHORITY
from forge.living_body_substrate import LivingBody


def test_all_reviewed_chassis_rasterize_into_native_causal_contract() -> None:
    families=set()
    for genome in review_genomes():
        body=LivingBody(develop(genome),seed=genome.seed);row=rasterize_body(body);families.add(body.family)
        assert row.static.shape==(85,48,48) and row.state.shape==(12,48,48) and row.live_bonds.shape==(8,48,48)
        assert row.static.dtype==row.state.dtype==row.live_bonds.dtype==np.float32
        assert int(row.static[0].sum())==body.organism.cell_count and np.array_equal(row.state[11]>0,row.static[0]>0)
        assert np.isfinite(row.static).all() and np.isfinite(row.state).all()
    assert families==set(range(5))


def test_runtime_applies_bounded_neural_cell_update() -> None:
    class Model(torch.nn.Module):
        def forward(self,static,state,bonds):
            result=state.clone();result[:,0]=torch.clamp(result[:,0]-.02*static[:,:1],0,1);result[:,6]=torch.clamp(result[:,6]+.01*static[:,:1],0,1);return result
    body=LivingBody(develop(review_genomes()[0]),seed=7);before=body.health.copy();runtime=LivingBodyNCARuntime(Model(),torch.device("cpu"),blend=.5);snapshot=runtime.step("entity",body)
    assert float(body.health.mean())<float(before.mean()) and float(body.scar.mean())>0
    assert not snapshot.dead and not snapshot.incapacitated and np.isfinite(body.health).all()


def test_damage_is_synchronized_into_recurrent_state() -> None:
    body=LivingBody(develop(review_genomes()[1]),seed=8);first=rasterize_body(body);body.impact((0,0),4,.6);second=rasterize_body(body,first.state);y=second.canvas_xy[:,1];x=second.canvas_xy[:,0]
    assert np.allclose(second.state[0,y,x],body.health) and float(second.state[7,y,x].max())>0


def test_live_bonds_are_reciprocal_and_dead_cells_do_not_conduct() -> None:
    body = LivingBody(develop(review_genomes()[2]), seed=9)
    body.alive_mask[body.organism.appendage_index == 0] = False
    row = rasterize_body(body)
    reverse = {direction: index for index, direction in enumerate(DIRECTION_XY)}
    for channel, (dx, dy) in enumerate(DIRECTION_XY):
        reflected = np.roll(
            row.live_bonds[reverse[(-dx, -dy)]],
            shift=(-dy, -dx),
            axis=(0, 1),
        )
        assert np.array_equal(row.live_bonds[channel], reflected)
    assert not np.any(row.live_bonds[:, row.state[11] == 0])


def test_targeted_damage_is_local_and_dummy_audit_replays() -> None:
    class Identity(torch.nn.Module):
        def forward(self, static, state, bonds):
            return state

    body = LivingBody(develop(review_genomes()[0]), seed=10)
    before = body.health.copy()
    assert _damage_system(body, "neural") > 0
    changed = body.health != before
    assert changed.any()
    assert all(body.organ[index] == "brain" for index in np.flatnonzero(changed))
    runtime = LivingBodyNCARuntime(Identity(), torch.device("cpu"), blend=.5)
    first = _audit_once(runtime, 2)
    second = _audit_once(runtime, 2)
    assert first == second
    assert len([row for row in first if row["ablation"] is None]) == 10


def test_real_runtime_fails_closed_until_authority_is_ready(tmp_path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        LivingBodyNCARuntime.from_output(tmp_path / "missing", device="cpu")


def test_selected_authority_is_the_default() -> None:
    assert DEFAULT_AUTHORITY.name == "nca_causal_v3_selected"
