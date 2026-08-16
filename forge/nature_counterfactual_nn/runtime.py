from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np,torch
from .contract import ACTIONS,CHECKPOINT_FORMAT,ModelConfig,SEQUENCE,source_sha256
from .model import CounterfactualTransformer

@dataclass(frozen=True,slots=True)
class Counterfactual:
    action:str
    benefit:float
    risk:float
    population_delta:float
    resource_delta:float

class NeuralCounterfactualRuntime:
    def __init__(self,model,device,report):self.model=model;self.device=device;self.report=report
    @classmethod
    def from_checkpoint(cls,path:Path,*,device="cuda"):
        target=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu");payload=torch.load(Path(path),map_location="cpu",weights_only=False)
        if payload.get("format")!=CHECKPOINT_FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("counterfactual checkpoint provenance drifted")
        model=CounterfactualTransformer(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"]);model.to(target).eval();return cls(model,target,payload["report"])
    def evaluate(self,history,actions=ACTIONS):
        rows=list(history)
        if not rows:raise ValueError("counterfactual history is empty")
        sequence=np.stack(([rows[0]]*(SEQUENCE-len(rows)))+rows[-SEQUENCE:]).astype(np.float32);current=sequence[-1];indices=np.asarray([ACTIONS.index(action) for action in actions],np.int64);batch=np.repeat(sequence[None],len(indices),axis=0)
        with torch.inference_mode():state,benefit,risk=self.model(torch.from_numpy(batch).to(self.device),torch.from_numpy(indices).to(self.device));state=state.float().cpu().numpy();benefit=benefit.float().cpu().numpy();risk=risk.float().cpu().numpy()
        result=[]
        for row,action,value,danger in zip(state,actions,benefit,risk):result.append(Counterfactual(action,float(value),float(danger),float(row[0]-current[0]),float(row[12:22].mean()-current[12:22].mean())))
        return tuple(result)
