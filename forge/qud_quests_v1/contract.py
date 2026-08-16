from __future__ import annotations

from dataclasses import dataclass
import math


FORMAT="nullvector-emergent-quest-journal/1.0.0"
KINDS=("discoveries","crafts","buildings","grafts","births","colonies","predations","artifacts","score")


@dataclass(slots=True)
class QuestEntry:
    quest_id:str
    activity_id:str
    issuer:str
    description:str
    metric:str
    baseline:float
    target:float
    rewards:tuple[tuple[str,float],...]
    progress:float=0.0
    complete:bool=False

    def __post_init__(self)->None:
        if not self.quest_id or not self.activity_id or not self.issuer or self.metric not in KINDS:raise ValueError("quest identity drifted")
        if not math.isfinite(self.baseline) or not math.isfinite(self.target) or self.target<=0:raise ValueError("quest target drifted")
