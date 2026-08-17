from __future__ import annotations
import copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from ..action_teacher_v1 import validate_trajectory
from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..world_frame_vae.contract import ModelConfig as VAEConfig
from ..world_frame_vae.model import WorldFrameVAE
from ..world_frame_vae_refiner.contract import ModelConfig
from ..world_frame_vae_refiner.model import PixelCellRefiner
from ..world_frame_vae_refiner.training import _loss
from .contract import BASE,CHECKPOINT_FORMAT,DEFAULT_OUTPUT,Plan,REPORT_FORMAT,canonical,file_sha256,source_sha256,state_sha256
EPISODES=tuple(PROJECT_ROOT/f"outputs/action_teacher_v1/curriculum-v1-{letter}" for letter in "abcdef")
def _load_frames():
    rows=[];sources=[]
    for root in EPISODES:
        manifest=validate_trajectory(root)
        with np.load(root/manifest["artifact"]["path"],allow_pickle=False) as archive:rows.append(archive["frame"].copy())
        sources.append({"session_id":manifest["session_id"],"manifest_sha256":manifest["manifest_sha256"],"arrays_sha256":manifest["arrays_sha256"],"frames":len(rows[-1])})
    return np.concatenate(rows),sources
def _base(device):
    payload=torch.load(BASE,map_location="cpu",weights_only=True);report=json.loads((BASE.parent/"report.json").read_text("utf-8"))
    if payload.get("source_sha256")!=report.get("source_sha256") or payload.get("report")!=report or state_sha256(payload["ema_state"])!=payload.get("ema_sha256"):raise ValueError("bound world VAE provenance drifted")
    model=WorldFrameVAE(VAEConfig(**payload["model_config"]));model.load_state_dict(payload["ema_state"],strict=True);return model.to(device).eval(),report
def _contact(path,target,base,refined):
    count=min(8,len(target));sheet=Image.new("RGB",(256*count,768))
    for index in range(count):
        for row,images in enumerate((target,base,refined)):sheet.paste(Image.fromarray(np.clip(images[index]*255,0,255).astype(np.uint8)),(index*256,row*256))
    sheet.save(path)
def train(output:Path=DEFAULT_OUTPUT,plan:Plan=Plan()):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);require_disk_floor(output.parent,floor_gb=100,planned_bytes=512*1024**2)
    if not torch.cuda.is_available():raise RuntimeError("bound refiner requires CUDA")
    torch.set_num_threads(2);torch.manual_seed(plan.seed);rng=np.random.default_rng(plan.seed);device=torch.device("cuda:0");torch.cuda.set_per_process_memory_fraction(.45,0);frames,sources=_load_frames();targets=torch.from_numpy(frames).permute(0,3,1,2).float()/255;split=len(targets)-60;base_model,base_report=_base(device);bases=[]
    with torch.inference_mode():
        for start in range(0,len(targets),8):mean,_=base_model.encode(targets[start:start+8].to(device));bases.append(base_model.decode(mean).float().cpu())
    bases=torch.cat(bases);model=PixelCellRefiner(ModelConfig()).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=plan.learning_rate,weight_decay=1e-4,fused=True);history=[];start=0
    for step in range(plan.segment,plan.updates+1,plan.segment):
        path=output/f"refiner_{step:07d}.pt"
        if path.exists():payload=torch.load(path,map_location="cpu",weights_only=True);start=step
    if start:model.load_state_dict(payload["model_state"]);ema.load_state_dict(payload["ema_state"]);optimizer.load_state_dict(payload["optimizer_state"]);rng.bit_generator.state=payload["rng_state"];history=list(payload["history"])
    for end in range(start+plan.segment,plan.updates+1,plan.segment):
        began=time.perf_counter();torch.cuda.reset_peak_memory_stats(device);model.train()
        for update in range(end-plan.segment+1,end+1):
            indices=rng.integers(0,split,plan.batch_size);ys=rng.integers(0,257-plan.crop,plan.batch_size);xs=rng.integers(0,257-plan.crop,plan.batch_size);base=torch.stack([bases[i,:,y:y+plan.crop,x:x+plan.crop] for i,y,x in zip(indices,ys,xs)]).to(device);target=torch.stack([targets[i,:,y:y+plan.crop,x:x+plan.crop] for i,y,x in zip(indices,ys,xs)]).to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):prediction=model(base);loss,parts=_loss(prediction,target)
            loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
            with torch.no_grad():torch._foreach_mul_(list(ema.parameters()),plan.ema_decay);torch._foreach_add_(list(ema.parameters()),list(model.parameters()),alpha=1-plan.ema_decay)
            if update==1 or update%100==0:history.append({"update":update,"loss":round(float(loss),7),"mae":round(float(parts[0]),7),"edge":round(float(parts[2]),7),"gradient":round(gradient,7)})
        raw={name:value.detach().cpu() for name,value in model.state_dict().items()};smooth={name:value.detach().cpu() for name,value in ema.state_dict().items()};payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"base_checkpoint_sha256":file_sha256(BASE),"base_source_sha256":base_report["source_sha256"],"update":end,"plan":plan.to_dict(),"model_state":raw,"ema_state":smooth,"model_state_sha256":state_sha256(raw),"ema_state_sha256":state_sha256(smooth),"optimizer_state":optimizer.state_dict(),"rng_state":rng.bit_generator.state,"history":history,"runtime":{"seconds":round(time.perf_counter()-began,6),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))}};temporary=output/f".refiner-{end}.tmp";torch.save(payload,temporary);os.replace(temporary,output/f"refiner_{end:07d}.pt");print(json.dumps({"update":end,"loss":history[-1]["loss"],**payload["runtime"]}),flush=True)
    model.load_state_dict(payload["ema_state"]);model.to(device).eval();validation=targets[split:];base_validation=bases[split:];refined=[]
    with torch.inference_mode():
        for start in range(0,len(validation),8):refined.append(model(base_validation[start:start+8].to(device)).float().cpu())
    refined=torch.cat(refined);base_mae=float(F.l1_loss(base_validation,validation));mae=float(F.l1_loss(refined,validation));base_mse=float(F.mse_loss(base_validation,validation));mse=float(F.mse_loss(refined,validation));dx=lambda value:(value[:,:,:,1:]-value[:,:,:,:-1],value[:,:,1:]-value[:,:,:-1]);bdx,bdy=dx(base_validation);rdx,rdy=dx(refined);tdx,tdy=dx(validation);base_edge=float(F.l1_loss(bdx,tdx)+F.l1_loss(bdy,tdy));edge=float(F.l1_loss(rdx,tdx)+F.l1_loss(rdy,tdy));metrics={"base_mae":base_mae,"refined_mae":mae,"mae_improvement":1-mae/base_mae,"base_psnr_db":-10*math.log10(base_mse),"refined_psnr_db":-10*math.log10(mse),"base_edge_mae":base_edge,"refined_edge_mae":edge,"edge_improvement":1-edge/base_edge};gates={"mae_improves":metrics["mae_improvement"]>.25,"edge_improves":metrics["edge_improvement"]>0,"psnr_above_31":metrics["refined_psnr_db"]>31};gates["all_passed"]=all(gates.values());checkpoint=output/f"refiner_{plan.updates:07d}.pt";report={"format":REPORT_FORMAT,"status":"ready" if gates["all_passed"] else "experimental","source_sha256":source_sha256(),"base":{"path":str(BASE.relative_to(PROJECT_ROOT)).replace('\\','/'),"sha256":file_sha256(BASE),"source_sha256":base_report["source_sha256"]},"checkpoint":{"path":checkpoint.name,"bytes":checkpoint.stat().st_size,"sha256":file_sha256(checkpoint),"ema_state_sha256":payload["ema_state_sha256"]},"parameters":model.parameter_count if hasattr(model,"parameter_count") else sum(p.numel() for p in model.parameters()),"sources":sources,"metrics":metrics,"gates":gates};report["report_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(output/"report.json").write_bytes(canonical(report));_contact(output/"heldout_contact_sheet.png",validation.permute(0,2,3,1).numpy(),base_validation.permute(0,2,3,1).numpy(),refined.permute(0,2,3,1).numpy());return report
