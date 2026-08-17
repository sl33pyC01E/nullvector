from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..recurrent_world_student_v6.calibration import _score
from ..recurrent_world_student_v6.training import _normalizers
from ..world_action_natural_v10 import load
from ..world_frame_decoder_adapt_v1 import AdaptedWorldFrameCodec
from ..world_latent_dit.contract import ModelConfig
from .contract import CHECKPOINT_FORMAT,CODEC,CORPUS,DEFAULT_OUTPUT,PARENT_SHA256,REPORT_FORMAT,canonical,file_sha256,source_sha256,state_sha256
from .training import _pixel_metrics

def evaluate(output:Path=DEFAULT_OUTPUT):
    output=Path(output);paths=sorted(output.glob("milestone_*.pt"));sequences,manifest=load(CORPUS);rows=[]
    for path in paths:
        p=torch.load(path,map_location="cpu",weights_only=True)
        if p.get("format")!=CHECKPOINT_FORMAT or p.get("source_sha256")!=source_sha256() or p.get("corpus_sha256")!=manifest["manifest_sha256"] or p.get("parent_sha256")!=PARENT_SHA256:raise ValueError("V7 milestone drifted")
        rows.append((p["selection_score"],p["update"],path,p))
    if not rows:raise FileNotFoundError("no V7 milestones")
    _,update,path,payload=min(rows,key=lambda row:(row[0],row[1]));device=torch.device("cuda:0");model=PerceptionRecurrentWorldStudent(ModelConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"]);model.to(device).eval();codec=AdaptedWorldFrameCodec.from_checkpoint(CODEC,device="cuda");norms=_normalizers(payload,device);bias=float(payload["inference"]["gate_logit_bias_max"]);ramp=int(payload["inference"]["gate_logit_bias_ramp_steps"]);test_latent={str(h):_score(model,sequences[5],norms,device,h,bias,ramp,48) for h in (1,2,4,8,16,32)};test_pixel={str(h):_pixel_metrics(model,codec,sequences[5],norms,device,h,bias,ramp,24) for h in (4,8,16,32)};gates={"all_latent_horizons_beat_persistence":all(r["improvement"]>0 for r in test_latent.values()),"all_pixel_horizons_beat_persistence":all(r["improvement"]>0 for r in test_pixel.values()),"pixel_motion_at_32":test_pixel["32"]["motion_ratio"]>=.15,"under_half_gpu_memory":payload["runtime"]["peak_reserved_bytes"]<12*1024**3};gates["all_passed"]=all(gates.values());state=payload["ema_state"];release={"format":CHECKPOINT_FORMAT,"status":"ready" if gates["all_passed"] else "experimental","source_sha256":source_sha256(),"corpus_sha256":manifest["manifest_sha256"],"parent_sha256":PARENT_SHA256,"codec_sha256":payload["codec_sha256"],"selected_milestone":{"update":update,"sha256":file_sha256(path)},"model_config":payload["model_config"],"normalization":payload["normalization"],"inference":payload["inference"],"state":state,"state_sha256":state_sha256(state),"validation":payload["validation"],"test":{"latent":test_latent,"pixel":test_pixel},"gates":gates,"plan":payload["plan"],"runtime":payload["runtime"]};tmp=output/".runtime.pt.tmp";torch.save(release,tmp);tmp.replace(output/"runtime.pt");report={k:v for k,v in release.items() if k not in ("state","normalization")};report["format"]=REPORT_FORMAT;report["checkpoint_sha256"]=file_sha256(output/"runtime.pt");report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(output/"evaluation.json").write_bytes(canonical(report));return report
