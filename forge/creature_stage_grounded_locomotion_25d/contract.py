from __future__ import annotations

from dataclasses import dataclass
import math


FORMAT="nullvector-grounded-locomotion-2.5d/1.0.0"


@dataclass(frozen=True,slots=True)
class Grounded25DConfig:
    frames:int=120
    substeps:int=2
    constraint_iterations:int=6
    dt:float=1/30
    maximum_speed:float=3.2
    acceleration:float=11.0
    body_damping:float=.88
    node_damping:float=.76
    contact_compliance:float=.035
    ground_scale:float=.12
    stride_length:float=1.15

    def __post_init__(self)->None:
        if not 30<=self.frames<=720 or not 1<=self.substeps<=4 or not 3<=self.constraint_iterations<=12:
            raise ValueError("2.5D locomotion discrete config drifted")
        for value in (self.dt,self.maximum_speed,self.acceleration,self.body_damping,self.node_damping,self.contact_compliance,self.ground_scale,self.stride_length):
            if not math.isfinite(value) or value<=0:raise ValueError("2.5D locomotion continuous config drifted")

