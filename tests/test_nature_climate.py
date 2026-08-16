from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import ClimateSystem,NatureWorld,SEASONS


def test_climate_cycle_is_deterministic_varied_and_bounded() -> None:
    left=ClimateSystem(91);right=ClimateSystem(91);samples=[left.sample(time) for time in np.linspace(0,720,200)];assert samples==[right.sample(time) for time in np.linspace(0,720,200)];assert {sample.season for sample in samples}==set(SEASONS);assert all(0<=value<=1 for sample in samples for value in (sample.light,sample.rainfall,sample.heat,sample.phase_flux,sample.toxin))


def test_climate_changes_real_fields_and_records_events() -> None:
    world=NatureWorld(seed=5,size=32);before=world.fields.copy()
    for _ in range(800):world.step(.25,publish=False)
    assert not np.array_equal(before,world.fields)
    assert world.climate.current.season in SEASONS
    assert any(event.get("type")=="climate" for event in world.events)
