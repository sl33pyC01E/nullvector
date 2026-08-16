from __future__ import annotations

import hashlib
import numpy as np

from .contract import QuestEntry


ACTIVITY_METRIC={
    "explore":"discoveries","map":"discoveries","study_anomaly":"discoveries","recover_relic":"artifacts",
    "craft":"crafts","build":"buildings","graft":"grafts","heal":"score","breed":"births","found_colony":"colonies",
    "hunt":"predations","defend":"predations","raid":"predations","forage":"score","trade":"score","negotiate":"score",
}


class QuestJournal:
    def __init__(self)->None:self.entries:dict[str,QuestEntry]={};self.reputation:dict[str,float]={};self.completed=0

    @staticmethod
    def metrics(world,adventure)->dict[str,float]:
        return {"discoveries":float(len(adventure.discoveries)),"crafts":float(adventure.craft_count),"buildings":float(len(adventure.buildings)),"grafts":float(sum(event.get("type")=="graft" for event in world.events)),"births":float(world.births),"colonies":float(len(world.colonies)),"predations":float(world.predation_events),"artifacts":float(len(adventure.artifacts)),"score":float(adventure.score)}

    def accept(self,activity,metrics:dict[str,float])->QuestEntry:
        existing=next((entry for entry in self.entries.values() if entry.activity_id==activity.activity_id),None)
        if existing is not None:return existing
        metric=ACTIVITY_METRIC[activity.kind];target=1 if metric in ("grafts","buildings","colonies","artifacts") else max(1,round(1+activity.difficulty*3));digest=hashlib.sha256(f"{activity.activity_id}:{metric}:{target}".encode()).hexdigest();entry=QuestEntry("q-"+digest[:14],activity.activity_id,activity.issuer,activity.description,metric,metrics[metric],float(target),activity.reward_materials);self.entries[entry.quest_id]=entry;return entry

    def accept_nearest(self,society,world,entity,adventure)->str:
        available=[activity for activity in society.activities.values() if not any(entry.activity_id==activity.activity_id for entry in self.entries.values())]
        if not available:return "NO SETTLEMENT CONTRACTS // HISTORY ADVANCES EVERY 60 TICKS"
        activity=min(available,key=lambda item:float(np.linalg.norm(world._delta(entity.position,np.asarray(item.location,float)))));entry=self.accept(activity,self.metrics(world,adventure));return f"ACCEPTED // {entry.description.upper()} // {entry.metric.upper()} {int(entry.target)}"

    def observe(self,world,adventure)->tuple[QuestEntry,...]:
        metrics=self.metrics(world,adventure);finished=[]
        for entry in self.entries.values():
            if entry.complete:continue
            entry.progress=max(0,min(entry.target,metrics[entry.metric]-entry.baseline))
            if entry.progress>=entry.target:
                entry.complete=True;finished.append(entry);self.completed+=1;self.reputation[entry.issuer]=self.reputation.get(entry.issuer,0)+.08
                for name,amount in entry.rewards:adventure.inventory[name if name in adventure.inventory else "knowledge"]+=amount
                adventure.score+=75
        return tuple(finished)

    def active(self,limit:int=3)->tuple[QuestEntry,...]:return tuple(entry for entry in self.entries.values() if not entry.complete)[:limit]
