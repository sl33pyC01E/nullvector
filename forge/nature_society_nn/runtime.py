from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from ..nature_sim_v2.climate import SEASONS
from ..nature_world_scale_v1.atlas import BIOMES
from ..qud_society_v1.architecture import PURPOSES
from ..qud_society_v1.contract import ACTIVITIES,CULTURAL_TRAITS,TECHNOLOGIES
from .contract import CHECKPOINT_FORMAT,DIPLOMACY,LABOR_SECTORS,PROJECTS,ModelConfig,source_sha256
from .model import SocietyStrategist


@dataclass(frozen=True,slots=True)
class SocietyDecision:
    activity:str
    labor:tuple[float,...]
    diplomacy:str
    project:str


def extract_features(faction,settlement,world,*,biome:str|None=None)->np.ndarray:
    state=world.climate.current;stress=max(0,.28-state.rainfall)+state.toxin+max(0,state.phase_flux-.5);row=np.zeros(64,np.float32);row[:8]=faction.cultural_traits;row[8:18]=[float(name in faction.technologies) for name in TECHNOLOGIES];row[18+faction.family]=1;stock=settlement.stockpiles;members=[item for item in world.organisms.values() if item.alive and item.genome.lineage_id==faction.lineage_id];integrity=float(np.mean([item.body.systems()["integrity"] for item in members])) if members else 0.0;row[23:39]=(min(1,settlement.population/24),min(1,settlement.wealth/4),min(1,settlement.food/3),min(1,settlement.power/3),integrity,min(1,stress+.15),float(settlement.shortages>0),min(1,faction.knowledge),min(1,stock.get("mineral",0)/3),min(1,stock.get("medicine",0)),min(1,stock.get("parts",0)),min(1,stock.get("water",0)/3),min(1,len(settlement.buildings)/20),faction.cohesion,min(1,settlement.projects_completed/8),min(1,world.tick_index/5000));biome=biome or getattr(world,"biome",None) or BIOMES[world.seed%len(BIOMES)];row[39+BIOMES.index(biome)]=1;row[47+SEASONS.index(state.season)]=1;row[53]=float(state.event is not None);row[54]=min(1,stress);relations=np.asarray(tuple(faction.relations.values()) or (0,),np.float32);row[55:59]=(relations.min(),relations.max(),relations.mean(),relations.std());counts={name:0 for name in PURPOSES}
    for building in settlement.buildings:counts[building.purpose]+=1
    row[59:64]=(counts["habitat"]/8,counts["workshop"]/8,counts["clinic"]/8,counts["granary"]/8,sum(counts[name] for name in ("observatory","graft_house","battery_hall","shrine","market"))/12);return row


class NeuralSocietyRuntime:
    def __init__(self,model,device):self.model,self.device=model,device;self.cache={}
    @classmethod
    def from_checkpoint(cls,path:Path,device:str="cuda"):
        target=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu");payload=torch.load(path,map_location="cpu",weights_only=True)
        report=payload.get("report",{});quality=(report.get("heldout_activity_accuracy",0)>=.85 and report.get("heldout_diplomacy_accuracy",0)>=.95 and report.get("heldout_project_accuracy",0)>=.95 and report.get("heldout_labor_mae",1)<=.02)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256() or payload.get("runtime_precision")!="bfloat16" or not quality:raise ValueError("society checkpoint provenance or quality drifted")
        model=SocietyStrategist(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"],strict=True);return cls(model.to(target).eval(),target)
    @torch.inference_mode()
    def decide(self,faction,settlement,world,*,biome:str|None=None)->SocietyDecision:
        signature=(world.tick_index//60,settlement.population,round(settlement.food,1),settlement.shortages,len(settlement.buildings),round(faction.knowledge,1));cached=self.cache.get(settlement.settlement_id)
        if cached is not None and cached[0]==signature:return cached[1]
        features=torch.from_numpy(extract_features(faction,settlement,world,biome=biome)).to(self.device).unsqueeze(0);activity,labor,diplomacy,project=self.model(features);weights=torch.softmax(labor[0],-1).float().cpu().numpy();decision=SocietyDecision(ACTIVITIES[int(activity.argmax(-1))],tuple(map(float,weights)),DIPLOMACY[int(diplomacy.argmax(-1))],PROJECTS[int(project.argmax(-1))]);self.cache[settlement.settlement_id]=(signature,decision);return decision
