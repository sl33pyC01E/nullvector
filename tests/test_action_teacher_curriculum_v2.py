from forge.action_teacher_v1.contract import ACTIONS
from types import SimpleNamespace
from forge.action_teacher_v1.curriculum_v2 import _ability_actor,_apply

def test_balanced_curriculum_names_every_action_once_per_cycle():
    schedule=tuple(ACTIONS[index%len(ACTIONS)] for index in range(len(ACTIONS)*3));assert set(schedule)==set(ACTIONS);assert all(schedule.count(name)==3 for name in ACTIONS);assert callable(_apply)


def test_ability_actor_prefers_current_when_it_has_the_requested_slot(monkeypatch):
    creatures=[SimpleNamespace(entity_id=4),SimpleNamespace(entity_id=9)]
    demo=SimpleNamespace(selected=9,world=SimpleNamespace(organisms={item.entity_id:item for item in creatures}),adventure=SimpleNamespace(bonus=lambda _name:0))
    monkeypatch.setattr("forge.action_teacher_v1.curriculum_v2.entity_abilities",lambda entity,equipment_damage=0:list(range(4 if entity.entity_id==9 else 2)))
    assert _ability_actor(demo,creatures,3).entity_id==9
    assert _ability_actor(demo,creatures,4) is None
