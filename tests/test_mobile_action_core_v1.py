from forge.mobile_action_core_v1.contract import MobileActionConfig, config_dict
from forge.recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from forge.world_latent_dit.contract import ModelConfig


def test_mobile_action_parameter_budget() -> None:
    mobile = PerceptionRecurrentWorldStudent(ModelConfig(**config_dict(MobileActionConfig())))
    desktop = PerceptionRecurrentWorldStudent(ModelConfig(width=512, layers=8, heads=8, patch=4))
    assert mobile.parameter_count < desktop.parameter_count * .35
    assert mobile.parameter_count > 1_000_000
