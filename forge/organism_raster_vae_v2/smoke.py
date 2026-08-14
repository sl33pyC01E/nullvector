from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FORMAT, OrganismVAEV2Config, authority, canonical_json_bytes, sha256_file, source_manifest, source_sha256
from .dataset import OrganismRasterCorpusV2
from .model import HierarchicalOrganismRasterVAE, HierarchicalOutput, hierarchical_loss, metrics


SEED: Final[int] = 0x4F524756414532
MANIFEST_NAME: Final[str] = "organism_vae_v2_manifest.json"; CHECKPOINT_NAME: Final[str] = "checkpoint.pt"
CONTACT_NAME: Final[str] = "reconstruction_and_organs.png"; FUSION_NAME: Final[str] = "hierarchical_fusion.png"; MUTATION_NAME: Final[str] = "hierarchical_mutation.png"
MAX_CHECKPOINT_BYTES: Final[int] = 768 * 1024 * 1024


def _png(image: Image.Image) -> bytes:
    buffer=BytesIO(); image.save(buffer,"PNG",optimize=False,compress_level=9); return buffer.getvalue()


def _atomic(path: Path, payload: bytes) -> None:
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.tmp-",dir=path.parent); temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def _artifact(path: Path) -> dict[str,Any]: return {"path":path.name,"bytes":path.stat().st_size,"sha256":sha256_file(path)}


def _batch(corpus: OrganismRasterCorpusV2, indices: list[int], device: torch.device) -> dict[str,Tensor|list[str]]:
    result=default_collate([corpus[index] for index in indices]); return {key:(value.to(device) if isinstance(value,Tensor) else value) for key,value in result.items()}


def _tensors(batch: dict[str,Tensor|list[str]]) -> dict[str,Tensor]: return {key:value for key,value in batch.items() if isinstance(value,Tensor)}


def _forward(model: HierarchicalOrganismRasterVAE,batch:dict[str,Tensor],**kwargs:Any)->HierarchicalOutput:
    return model(batch["living_field"],batch["family"],batch["subtype"],batch["role"],batch["genes"],batch["style"],**kwargs)


def _rgba_image(rgba: Tensor,scale:int=3)->Image.Image:
    value=rgba.detach().cpu().clamp(0,1).permute(1,2,0).numpy(); value[...,:3]*=value[...,3:4]; return Image.fromarray(np.rint(value*255).astype(np.uint8)).resize((48*scale,48*scale),Image.Resampling.NEAREST)


def _system_image(values:Tensor,roles:Tensor,alpha:Tensor,scale:int=3)->Image.Image:
    palette=torch.tensor(((250,73,86),(90,210,255),(255,178,71),(206,91,255),(255,238,110),(91,255,161),(255,104,190),(108,238,205)),dtype=torch.float32)/255; role_values=roles.detach().cpu(); importance=torch.tensor((0.0,4.0,1.0,2.2))[role_values.long().clamp(0,3)]; score=values.detach().cpu()*(.15+importance); strength,system=score.max(0); strength=(strength/4.15).clamp(0,1); rgb=palette[system].permute(2,0,1)*(.25+.75*strength[None]); return _rgba_image(torch.cat((rgb,alpha.detach().cpu().clamp(0,1)[None])),scale)


def _concat(outputs:list[HierarchicalOutput])->HierarchicalOutput:
    names=HierarchicalOutput.__dataclass_fields__; return HierarchicalOutput(**{name:torch.cat([getattr(output,name).detach().cpu() for output in outputs],dim=0) for name in names})


def _evaluate(corpus:OrganismRasterCorpusV2,model:HierarchicalOrganismRasterVAE)->tuple[dict[str,float],dict[str,Tensor],HierarchicalOutput]:
    model.eval(); outputs=[]; batches=[]
    with torch.inference_mode():
        for start in range(0,len(corpus),5):
            batch=_tensors(_batch(corpus,list(range(start,min(len(corpus),start+5))),torch.device("cpu"))); outputs.append(_forward(model,batch,sample=False)); batches.append(batch)
    combined={name:torch.cat([batch[name] for batch in batches],dim=0) for name in batches[0]}; output=_concat(outputs); return metrics(output,combined),combined,output


def _contact(corpus:OrganismRasterCorpusV2,model:HierarchicalOrganismRasterVAE)->tuple[bytes,list[str]]:
    indices=[corpus.indices_by_family[family][offset] for family in range(5) for offset in (0,1)]; batch=_tensors(_batch(corpus,indices,torch.device("cpu"))); model.eval()
    with torch.inference_mode(): output=_forward(model,batch,sample=False)
    scale=3; cell=144; canvas=Image.new("RGB",(24+4*(cell+8),58+len(indices)*(cell+18)),(4,8,16)); draw=ImageDraw.Draw(canvas); draw.text((12,8),"HIERARCHICAL ORGANISM VAE V2 // RASTER + PHYSIOLOGY",fill=(164,242,255)); draw.text((12,23),"TARGET RGBA | V2 RASTER | TARGET SYSTEMS | V2 SYSTEMS",fill=(78,154,181))
    for row,index in enumerate(indices):
        y=56+row*(cell+18); images=(_rgba_image(batch["rgba"][row]),_rgba_image(output.rgba[row]),_system_image(batch["physiology"][row],batch["system_role"][row],batch["occupancy"][row]),_system_image(output.physiology[row],output.system_role_logits[row].argmax(1),output.occupancy_logits[row,0].sigmoid()))
        for column,image in enumerate(images): canvas.paste(image.convert("RGB"),(12+column*(cell+8),y))
        draw.text((12,y+cell+2),corpus.samples[index].sample_id,fill=(108,155,174))
    return _png(canvas),[corpus.samples[index].sample_id for index in indices]


def _fusion(corpus:OrganismRasterCorpusV2,model:HierarchicalOrganismRasterVAE)->tuple[bytes,list[dict[str,Any]]]:
    cell=144; steps=7; canvas=Image.new("RGB",(24+steps*(cell+6),42+5*(cell+22)),(4,8,16)); draw=ImageDraw.Draw(canvas); draw.text((12,8),"HIERARCHICAL NEURAL FUSION // COARSE + FINE LATENTS",fill=(175,247,143)); draw.text((12,23),"same-family endpoints / continuous anatomy, detail, palette and condition",fill=(76,143,112)); records=[]; model.eval()
    with torch.inference_mode():
        for family in range(5):
            indices=corpus.indices_by_family[family][:2]; batch=_tensors(_batch(corpus,indices,torch.device("cpu"))); condition=model.condition_vector(batch["family"],batch["subtype"],batch["role"],batch["genes"],batch["style"]); cm,_,fm,_=model.encode(batch["living_field"],condition); y=40+family*(cell+22); frames=[]
            for step in range(steps):
                alpha=step/(steps-1); output=model.decode(cm[:1]*(1-alpha)+cm[1:]*alpha,fm[:1]*(1-alpha)+fm[1:]*alpha,condition[:1]*(1-alpha)+condition[1:]*alpha); frames.append(output.rgba); canvas.paste(_rgba_image(output.rgba[0]).convert("RGB"),(12+step*(cell+6),y)); draw.text((12+step*(cell+6),y+cell+2),f"{alpha:.2f}",fill=(99,177,135))
            adjacent=[float((frames[index+1]-frames[index]).abs().mean()) for index in range(steps-1)]; records.append({"family":family,"left":corpus.samples[indices[0]].sample_id,"right":corpus.samples[indices[1]].sample_id,"endpoint_rgba_l1":float((frames[-1]-frames[0]).abs().mean()),"maximum_adjacent_rgba_l1":max(adjacent)})
    return _png(canvas),records


def _mutation(corpus:OrganismRasterCorpusV2,model:HierarchicalOrganismRasterVAE)->tuple[bytes,list[dict[str,Any]]]:
    cell=144; labels=("BASE","COARSE-","COARSE+","FINE-","FINE+","MIXED"); canvas=Image.new("RGB",(24+6*(cell+6),42+5*(cell+22)),(4,8,16)); draw=ImageDraw.Draw(canvas); draw.text((12,8),"HIERARCHICAL MUTATION // CHASSIS VS CELLULAR DETAIL",fill=(255,165,237)); draw.text((12,23),"coarse edits reshape chassis / fine edits alter appendages, cells and palette",fill=(155,87,145)); generator=torch.Generator().manual_seed(SEED^0x4D555441); records=[]; model.eval()
    with torch.inference_mode():
        for family in range(5):
            index=corpus.indices_by_family[family][0]; batch=_tensors(_batch(corpus,[index],torch.device("cpu"))); condition=model.condition_vector(batch["family"],batch["subtype"],batch["role"],batch["genes"],batch["style"]); cm,_,fm,_=model.encode(batch["living_field"],condition); cn=torch.randn(cm.shape,generator=generator); cn=F.avg_pool2d(cn,3,1,1); fn=torch.randn(fm.shape,generator=generator); fn=F.avg_pool2d(fn,3,1,1); variants=((cm,fm),(cm-cn*3.0,fm),(cm+cn*3.0,fm),(cm,fm-fn*2.5),(cm,fm+fn*2.5),(cm+cn*2.2,fm-fn*1.8)); frames=[]; y=40+family*(cell+22)
            for column,(coarse,fine) in enumerate(variants):
                output=model.decode(coarse,fine,condition); frames.append(output.rgba); canvas.paste(_rgba_image(output.rgba[0]).convert("RGB"),(12+column*(cell+6),y)); draw.text((12+column*(cell+6),y+cell+2),labels[column],fill=(190,105,179))
            records.append({"family":family,"sample_id":corpus.samples[index].sample_id,"maximum_rgba_l1_from_base":max(float((frame-frames[0]).abs().mean()) for frame in frames[1:])})
    return _png(canvas),records


def _state_hash(state:dict[str,Tensor])->str:return tensor_state_sha256({name:value.detach().cpu() for name,value in state.items()})


def _load_checkpoint(path:Path)->tuple[HierarchicalOrganismRasterVAE,dict[str,Any]]:
    if not path.is_file() or path.is_symlink() or not 0<path.stat().st_size<=MAX_CHECKPOINT_BYTES:raise ValueError("Organism VAE v2 checkpoint missing or oversized.")
    payload=torch.load(path,map_location="cpu",weights_only=True); keys={"format","source_sha256","source_manifest","authority","config","steps","batch_size","seed","model_state","model_state_sha256","history"}
    if not isinstance(payload,dict) or set(payload)!=keys or payload["format"]!=CHECKPOINT_FORMAT or payload["source_sha256"]!=source_sha256() or payload["source_manifest"]!=source_manifest() or payload["authority"]!=authority():raise ValueError("Organism VAE v2 checkpoint provenance drifted.")
    config=OrganismVAEV2Config(**payload["config"]);model=HierarchicalOrganismRasterVAE(config);model.load_state_dict(payload["model_state"],strict=True)
    if _state_hash(model.state_dict())!=payload["model_state_sha256"]:raise ValueError("Organism VAE v2 state hash failed.")
    if type(payload["steps"]) is not int or not 1<=payload["steps"]<=100_000 or type(payload["batch_size"]) is not int or not 1<=payload["batch_size"]<=45 or not isinstance(payload["history"],list) or len(payload["history"])!=payload["steps"]:raise ValueError("Organism VAE v2 training census drifted.")
    if any(not isinstance(row,dict) or row.get("step")!=index or any(isinstance(value,float) and not math.isfinite(value) for value in row.values()) for index,row in enumerate(payload["history"],1)):raise ValueError("Organism VAE v2 history drifted.")
    return model,payload


def run_smoke(output:Path,*,steps:int=2048,batch_size:int=15)->dict[str,Any]:
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Organism VAE v2 publication is immutable.")
    if type(steps) is not int or not 8<=steps<=20_000 or type(batch_size) is not int or not 1<=batch_size<=45:raise ValueError("Organism VAE v2 training bounds drifted.")
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=1024**3)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("Organism VAE v2 requires deterministic CUDA BF16.")
    if torch.cuda.mem_get_info(0)[0]<8*1024**3:raise RuntimeError("Organism VAE v2 requires 8 GiB free VRAM.")
    device=torch.device("cuda",0);torch.cuda.reset_peak_memory_stats(device);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED&0xffffffff)
    corpus=OrganismRasterCorpusV2();config=OrganismVAEV2Config();model=HierarchicalOrganismRasterVAE(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);latent_generator=torch.Generator(device=device).manual_seed(SEED^0x4C4154);order_generator=torch.Generator().manual_seed(SEED^0x4F5244);order=torch.randperm(len(corpus),generator=order_generator).tolist();cursor=0;history=[];started=time.perf_counter()
    for step in range(steps):
        if cursor+batch_size>len(order):order=torch.randperm(len(corpus),generator=order_generator).tolist();cursor=0
        indices=order[cursor:cursor+batch_size];cursor+=batch_size;batch=_tensors(_batch(corpus,indices,device));model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):output_value=_forward(model,batch,generator=latent_generator,sample=True)
        loss,pieces=hierarchical_loss(output_value,batch,config,beta_scale=min(1,(step+1)/max(64,steps//5)));loss.float().backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)):raise FloatingPointError("Organism VAE v2 became non-finite.")
        optimizer.step();history.append({"step":step+1,**{name:round(float(value),8) for name,value in pieces.items()},"gradient_norm":round(float(gradient),8)})
    training_seconds=time.perf_counter()-started;state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()};cpu_model=HierarchicalOrganismRasterVAE(config);cpu_model.load_state_dict(state);evaluation,_,_=_evaluate(corpus,cpu_model);contact,contact_ids=_contact(corpus,cpu_model);fusion,fusion_records=_fusion(corpus,cpu_model);mutation,mutation_records=_mutation(corpus,cpu_model);staging=output.parent/f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}";staging.mkdir(parents=True)
    try:
        payload={"format":CHECKPOINT_FORMAT,"source_sha256":source_sha256(),"source_manifest":source_manifest(),"authority":authority(),"config":config.to_dict(),"steps":steps,"batch_size":batch_size,"seed":SEED,"model_state":state,"model_state_sha256":_state_hash(state),"history":history};torch.save(payload,staging/CHECKPOINT_NAME);_load_checkpoint(staging/CHECKPOINT_NAME);_atomic(staging/CONTACT_NAME,contact);_atomic(staging/FUSION_NAME,fusion);_atomic(staging/MUTATION_NAME,mutation);artifacts={name:_artifact(staging/filename) for name,filename in (("checkpoint",CHECKPOINT_NAME),("reconstruction",CONTACT_NAME),("fusion",FUSION_NAME),("mutation",MUTATION_NAME))};positive={"upstream_and_v1_bound":True,"identity_census_45":len(corpus)==45,"finite_training":all(math.isfinite(float(row["loss"])) for row in history),"model_updated":history[0]["loss"]!=history[-1]["loss"],"fusion_endpoints_distinct":all(row["endpoint_rgba_l1"]>1e-4 for row in fusion_records),"fusion_steps_continuous":all(row["maximum_adjacent_rgba_l1"]<row["endpoint_rgba_l1"] for row in fusion_records),"mutations_measurable":all(row["maximum_rgba_l1_from_base"]>1e-4 for row in mutation_records),"posterior_nonzero":min(evaluation["coarse_latent_std_mean"],evaluation["fine_latent_std_mean"])>0};gates={**positive,"production_promotion_allowed":False};runtime={"device":torch.cuda.get_device_name(device),"precision":"bf16-autocast-float32-loss","training_seconds":training_seconds,"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),"parameters":sum(parameter.numel() for parameter in model.parameters()),"artifact_render_device":"cpu"};manifest={"format":FORMAT,"status":"passed" if all(positive.values()) else "failed","source_sha256":source_sha256(),"source_manifest":source_manifest(),"authority":authority(),"config":config.to_dict(),"sample_count":len(corpus),"steps":steps,"batch_size":batch_size,"seed":SEED,"metrics":evaluation,"loss_start":history[0]["loss"],"loss_end":history[-1]["loss"],"history_sha256":hashlib.sha256(canonical_json_bytes(history)).hexdigest(),"model_state_sha256":payload["model_state_sha256"],"artifacts":artifacts,"contact_sample_ids":contact_ids,"fusion":fusion_records,"mutations":mutation_records,"runtime":runtime,"gates":gates,"claim_boundary":{"hierarchical_continuous_rasterizer":True,"coarse_and_fine_fusion":True,"coarse_and_fine_mutation":True,"generative_prior_trained":False,"production_promotion_allowed":False}}
        if manifest["status"]!="passed":raise ValueError("Organism VAE v2 gates failed.")
        manifest["manifest_sha256"]=hashlib.sha256(canonical_json_bytes(manifest)).hexdigest();_atomic(staging/MANIFEST_NAME,canonical_json_bytes(manifest));require_disk_floor(output.parent,floor_gb=100,planned_bytes=0);os.replace(staging,output)
    except BaseException:
        if staging.exists():os.replace(staging,output.parent/f"{staging.name}.failed-{time.time_ns()}")
        raise
    return validate_smoke(output)


def validate_smoke(output:Path)->dict[str,Any]:
    output=Path(output).resolve();path=output/MANIFEST_NAME
    if not path.is_file() or path.is_symlink() or not 0<path.stat().st_size<=8*1024**2:raise ValueError("Organism VAE v2 manifest missing or oversized.")
    encoded=path.read_bytes();manifest=json.loads(encoded)
    if encoded!=canonical_json_bytes(manifest):raise ValueError("Organism VAE v2 manifest is not canonical JSON.")
    stored=manifest.pop("manifest_sha256",None)
    if stored!=hashlib.sha256(canonical_json_bytes(manifest)).hexdigest():raise ValueError("Organism VAE v2 manifest hash failed.")
    manifest["manifest_sha256"]=stored;required={"format","status","source_sha256","source_manifest","authority","config","sample_count","steps","batch_size","seed","metrics","loss_start","loss_end","history_sha256","model_state_sha256","artifacts","contact_sample_ids","fusion","mutations","runtime","gates","claim_boundary","manifest_sha256"}
    if set(manifest)!=required or manifest["format"]!=FORMAT or manifest["status"]!="passed" or manifest["source_sha256"]!=source_sha256() or manifest["source_manifest"]!=source_manifest() or manifest["authority"]!=authority():raise ValueError("Organism VAE v2 manifest contract drifted.")
    if manifest["config"]!=OrganismVAEV2Config(**manifest["config"]).to_dict() or manifest["sample_count"]!=45 or manifest["seed"]!=SEED:raise ValueError("Organism VAE v2 config/census drifted.")
    for record in manifest["artifacts"].values():
        if set(record)!={"path","bytes","sha256"} or Path(record["path"]).name!=record["path"]:raise ValueError("Organism VAE v2 artifact descriptor drifted.")
        artifact=output/record["path"]
        if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size!=record["bytes"] or sha256_file(artifact)!=record["sha256"]:raise ValueError("Organism VAE v2 artifact identity failed.")
    model,payload=_load_checkpoint(output/CHECKPOINT_NAME)
    if payload["model_state_sha256"]!=manifest["model_state_sha256"] or payload["steps"]!=manifest["steps"] or payload["batch_size"]!=manifest["batch_size"] or hashlib.sha256(canonical_json_bytes(payload["history"])).hexdigest()!=manifest["history_sha256"]:raise ValueError("Organism VAE v2 checkpoint semantics drifted.")
    expected_claim={"hierarchical_continuous_rasterizer":True,"coarse_and_fine_fusion":True,"coarse_and_fine_mutation":True,"generative_prior_trained":False,"production_promotion_allowed":False}
    if manifest["claim_boundary"]!=expected_claim or manifest["gates"].get("production_promotion_allowed") is not False or not all(value for key,value in manifest["gates"].items() if key!="production_promotion_allowed"):raise ValueError("Organism VAE v2 gates/claim drifted.")
    corpus=OrganismRasterCorpusV2();evaluation,_,_=_evaluate(corpus,model);contact,ids=_contact(corpus,model);fusion,fusion_records=_fusion(corpus,model);mutation,mutation_records=_mutation(corpus,model)
    if evaluation!=manifest["metrics"] or ids!=manifest["contact_sample_ids"] or fusion_records!=manifest["fusion"] or mutation_records!=manifest["mutations"]:raise ValueError("Organism VAE v2 semantic replay failed.")
    for payload_bytes,name in ((contact,"reconstruction"),(fusion,"fusion"),(mutation,"mutation")):
        if hashlib.sha256(payload_bytes).hexdigest()!=manifest["artifacts"][name]["sha256"]:raise ValueError("Organism VAE v2 visual replay failed.")
    return manifest
