from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..creature_stage_developmental import develop
from ..creature_stage_grounded_locomotion_25d import Grounded25DConfig,simulate_25d
from ..creature_stage_morphology_v2 import morphology_review_genomes
from .data import VALIDATION_IDENTITIES,control_program,load_corpus
from .training import evaluate_arrays,load_model


@torch.inference_mode()
def evaluate(checkpoint:Path,corpus_path:Path,output:Path,*,device:str="cuda")->dict[str,object]:
    corpus=load_corpus(corpus_path);model,payload=load_model(checkpoint,device=device,ema=True);target=torch.device(device)
    metrics=evaluate_arrays(model,corpus,target);physics=[]
    for identity in VALIDATION_IDENTITIES:
        sequence=int(np.flatnonzero((corpus.identity==identity)&(corpus.program==0))[0])
        tensors={name:torch.from_numpy(getattr(corpus,name)[sequence:sequence+1]).to(target) for name in ("global_static","appendage_meta","appendage_mask","muscle_meta","muscle_owner","muscle_mask","dynamic")}
        out=model(tensors["global_static"],tensors["appendage_meta"],tensors["appendage_mask"],tensors["muscle_meta"],tensors["muscle_owner"].long(),tensors["muscle_mask"],tensors["dynamic"])
        contact=(torch.sigmoid(out.contact_logits)[0].cpu().numpy()>=.5);muscle=out.muscle[0].cpu().numpy();organism=develop(morphology_review_genomes()[identity]);appendages=len(organism.genome.appendages);muscles=len(organism.muscles)
        rollout=simulate_25d(organism,Grounded25DConfig(frames=corpus.dynamic.shape[1]),control=lambda f,t:control_program(0,f,t),contact_gate=lambda f,p,o,m,c=contact:c[f,:appendages],muscle_gate=lambda f,p,o,a=muscle:a[f,:muscles])
        physics.append({"identity":identity,"path_length":rollout.path_length,"lateral_extent":rollout.lateral_extent,"depth_extent":rollout.depth_extent,"contact_slip":rollout.maximum_contact_slip,"edge_strain":rollout.maximum_edge_strain,"chassis_tilt":rollout.maximum_chassis_tilt_degrees,"appendage_motion":rollout.appendage_motion_px})
    gates={"contact_f1":metrics["contact_f1"]>=.96,"muscle_mae":metrics["muscle_mae"]<=.10,"velocity_mae":metrics["velocity_mae"]<=.25,"all_depth_motion":min(v["depth_extent"] for v in physics)>.15,"all_lateral_motion":min(v["lateral_extent"] for v in physics)>.15,"contacts_hold":max(v["contact_slip"] for v in physics)<.08,"tethers_hold":max(v["edge_strain"] for v in physics)<.45,"chassis_upright":max(v["chassis_tilt"] for v in physics)<8,"appendages_move":min(v["appendage_motion"] for v in physics)>.12}
    report={"format":"nullvector-neural-locomotion-2.5d-evaluation/1.0.0","checkpoint":str(checkpoint),"corpus_sha256":corpus.semantic_sha256,"metrics":metrics,"physics":physics,"gates":gates,"passed":all(gates.values())}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n","utf-8");return report

