from forge.action_teacher_v1.contract import ACTIONS
from forge.action_teacher_v1.curriculum_v2 import _apply

def test_balanced_curriculum_names_every_action_once_per_cycle():
    schedule=tuple(ACTIONS[index%len(ACTIONS)] for index in range(len(ACTIONS)*3));assert set(schedule)==set(ACTIONS);assert all(schedule.count(name)==3 for name in ACTIONS);assert callable(_apply)
