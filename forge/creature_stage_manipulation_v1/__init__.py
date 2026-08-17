from .contract import (
    CONTROLLER, CONTROLLER_SHA256, FORMAT,
    LIMB_POSE_CONTROLLER, LIMB_POSE_CONTROLLER_SHA256,
)

__all__ = [
    "CONTROLLER", "CONTROLLER_SHA256", "FORMAT",
    "LIMB_POSE_CONTROLLER", "LIMB_POSE_CONTROLLER_SHA256",
    "ManipulationStep", "NeuralManipulationArena",
]


def __getattr__(name: str):
    # Keep the package import acyclic: the neural limb dataset imports the
    # low-level articulation module, while the arena imports the neural
    # runtime. Arena symbols are loaded only when a caller requests them.
    if name in {"ManipulationStep", "NeuralManipulationArena"}:
        from .arena import ManipulationStep, NeuralManipulationArena
        return {"ManipulationStep": ManipulationStep, "NeuralManipulationArena": NeuralManipulationArena}[name]
    raise AttributeError(name)
