from __future__ import annotations
from collections import Counter,deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np,torch
from .contract import CHECKPOINT_FORMAT,EVENTS,FEATURES,SEQUENCE,ModelConfig,source_sha256
from .model import TimelineTransformer
from ..nature_sim_v2.contract import INTENTS
from ..nature_sim_v2.climate import SEASONS

@dataclass(frozen=True,slots=True)
class TimelineForecast:event:str;confidence:float;population_delta:float;resource_delta:float;state:tuple[float,...]

def extract_world_features(world,society=None)->np.ndarray:
    snapshot=world.snapshot();living=[item for item in world.organisms.values() if item.alive];pop=max(1,len(living));families=np.asarray(snapshot.family_counts,np.float32)/pop;systems=np.mean([list(item.body.systems().values()) for item in living],axis=0) if living else np.zeros(7);counts=Counter(item.intent for item in living);intents=np.asarray([counts[name]/pop for name in INTENTS]);climate=world.climate.current;climate_values=np.asarray((climate.light,climate.rainfall,climate.heat,climate.phase_flux,climate.toxin,climate.phase),np.float32);season=np.eye(len(SEASONS),dtype=np.float32)[SEASONS.index(climate.season)];area=world.size*world.size;row=np.concatenate((np.asarray((snapshot.population/world.max_population,),np.float32),families,np.asarray((min(1,snapshot.lineage_count/50),min(1,snapshot.colony_count/20),min(1,snapshot.births/world.max_population),min(1,snapshot.deaths/world.max_population),min(1,snapshot.predation_events/world.max_population),min(1,snapshot.mutation_count/world.max_population)),np.float32),np.asarray(snapshot.resource_totals,np.float32)/area,climate_values,season,np.asarray(systems,np.float32),intents));return np.pad(row,(0,FEATURES-len(row))).astype(np.float32)

class NeuralTimelineRuntime:
    def __init__(self,model,device,report):self.model=model;self.device=device;self.report=report;self.history=deque(maxlen=SEQUENCE)
    @classmethod
    def from_checkpoint(cls,path:Path,*,device="cuda"):
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");payload=torch.load(Path(path),map_location="cpu",weights_only=False)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("timeline checkpoint provenance drifted")
        model=TimelineTransformer(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"]);model.to(target).eval();return cls(model,target,payload["report"])
    def observe(self,world,society=None)->TimelineForecast:
        current=extract_world_features(world,society);self.history.append(current);rows=list(self.history);sequence=np.stack(([rows[0]]*(SEQUENCE-len(rows)))+rows)
        with torch.inference_mode():state,logits,confidence=self.model(torch.from_numpy(sequence[None]).to(self.device));prob=torch.softmax(logits.float(),-1);index=int(prob.argmax(-1));certainty=float(prob[0,index]);pred=state[0].float().cpu().numpy()
        return TimelineForecast(EVENTS[index],min(certainty,float(confidence[0])),float(pred[0]-current[0]),float(pred[12:22].mean()-current[12:22].mean()),tuple(float(v) for v in pred))
