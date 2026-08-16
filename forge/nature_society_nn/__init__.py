from .contract import ModelConfig,TrainingConfig
from .model import SocietyStrategist

__all__=["ModelConfig","TrainingConfig","SocietyStrategist","NeuralSocietyRuntime"]


def __getattr__(name):
    if name=="NeuralSocietyRuntime":
        from .runtime import NeuralSocietyRuntime
        return NeuralSocietyRuntime
    raise AttributeError(name)
