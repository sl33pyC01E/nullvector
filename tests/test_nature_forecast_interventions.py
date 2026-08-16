from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import NatureWorld
from forge.nature_sim_v2.adventure import AdventureState
from forge.nature_sim_v2.forecast_interventions import INTERVENTIONS,apply_intervention,intervention_offers


def _case():
    world=NatureWorld(seed=83,size=32);world.seed_founders(variants_per_family=1);entity=next(iter(world.organisms.values()));entity.position[:]=16;adventure=AdventureState(seed=91,size=32)
    for name in adventure.inventory:adventure.inventory[name]=20
    return world,adventure,entity


def test_intervention_offers_are_distinct_and_forecast_tuned():
    for event in ("birth","death","predation","mutation","colony","climate","construction","discovery","migration","quiet"):
        offers=intervention_offers(event,epoch=17);assert len(offers)==3 and len({item.intervention_id for item in offers})==3;assert event in offers[0].tuned_for


def test_every_intervention_changes_physical_or_ecological_authority():
    for offer in INTERVENTIONS:
        world,adventure,entity=_case();fields=world.fields.copy();material=world.materials.mass.copy();score=adventure.score;message=apply_intervention(world,adventure,entity,offer,forecast_event=offer.tuned_for[0]);event=world.events[-1]
        assert event["type"]=="neural_intervention" and event["matched"] is True and event["affected"]>0
        assert not np.array_equal(fields,world.fields) or not np.array_equal(material,world.materials.mass)
        assert adventure.score>score and message.startswith("FORECAST MATCH")


def test_intervention_cost_failure_is_atomic():
    world,adventure,entity=_case();offer=INTERVENTIONS[0];adventure.inventory[offer.costs[0][0]]=0;before=dict(adventure.inventory)
    try:apply_intervention(world,adventure,entity,offer,forecast_event="birth")
    except ValueError as exc:assert "INTERVENTION NEEDS" in str(exc)
    else:raise AssertionError("unfunded intervention was accepted")
    assert adventure.inventory==before and not world.events
