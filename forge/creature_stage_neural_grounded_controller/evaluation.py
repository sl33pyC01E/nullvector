from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_developmental.contract import FAMILIES, TISSUES
from ..creature_stage_grounded_locomotion.physics import GroundedCycle
from ..multifield_style_motion.hashing import artifact_record_from_bytes, canonical_json_bytes, deterministic_npz_bytes, sha256_file
from .contract import EVALUATION_FORMAT, MAX_APPENDAGES, MAX_MUSCLES, source_sha256
from .dataset import ControllerCorpus, build_corpus
from .model import NeuralGroundedController
from .physics import simulate_controlled_cycle
from .training import load_model


EVALUATION_FILE = "forge/creature_stage_neural_grounded_controller/evaluation.py"
TISSUE_COLORS = ((244,111,126),(238,237,204),(255,69,113),(248,188,144),(113,158,177),(72,225,246),(234,52,101),(123,216,244),(241,169,63),(253,244,105),(164,102,230),(115,229,92),(185,78,255),(177,195,207),(255,131,61))
MODE_COLORS = {"passive":(100,120,132),"step":(72,229,255),"drag":(151,245,82),"float":(199,93,255),"wheel":(255,188,70)}


def evaluation_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-grounded-controller-evaluation-v1\0")
    digest.update(source_sha256().encode("ascii") + b"\0")
    digest.update((PROJECT_ROOT / EVALUATION_FILE).read_bytes())
    return digest.hexdigest()


def _font(size: int):
    for path in (Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file(): return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _predict(model: NeuralGroundedController, corpus: ControllerCorpus, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    muscle, contact, body = [], [], []
    with torch.inference_mode():
        for start in range(0, corpus.samples, 144):
            indices = torch.arange(start, min(start + 144, corpus.samples)); batch = corpus.batch(indices, device)
            output = model(batch["owner_input"], batch["global_input"], batch["owner_meta"], batch["owner_mask"], batch["muscle_meta"], batch["muscle_owner"], batch["muscle_mask"])
            muscle.append(output.muscle_activation.float().cpu().numpy()); contact.append(torch.sigmoid(output.contact_logits).float().cpu().numpy()); body.append(output.body_velocity.float().cpu().numpy())
    count = corpus.samples // 72
    return np.concatenate(muscle).reshape(count,72,MAX_MUSCLES), np.concatenate(contact).reshape(count,72,MAX_APPENDAGES), np.concatenate(body).reshape(count,72)


def _simulate(corpus: ControllerCorpus, muscle: np.ndarray, contact_probability: np.ndarray) -> tuple[GroundedCycle, ...]:
    cycles = []
    organisms = tuple(corpus.teacher.organisms[index] for index in corpus.teacher.split_indices("validation"))
    for index, organism in enumerate(organisms):
        appendages, muscles = len(organism.genome.appendages), len(organism.muscles)
        cycles.append(simulate_controlled_cycle(organism, (contact_probability[index,:,:appendages] >= .5).astype(np.bool_), muscle[index,:,:muscles].astype(np.float32)))
    return tuple(cycles)


def _cycle_arrays(corpus: ControllerCorpus, cycles: tuple[GroundedCycle, ...], muscle: np.ndarray, contact: np.ndarray) -> dict[str, np.ndarray]:
    max_cells = corpus.teacher.arrays["cells_local"].shape[2]; max_nodes = corpus.teacher.arrays["nodes_local"].shape[2]
    cells = np.zeros((len(cycles),72,max_cells,2),np.float32); nodes=np.zeros((len(cycles),72,max_nodes,2),np.float32)
    body=np.zeros((len(cycles),72),np.float32)
    organisms = tuple(corpus.teacher.organisms[index] for index in corpus.teacher.split_indices("validation"))
    for index,(organism,cycle) in enumerate(zip(organisms,cycles,strict=True)):
        cells[index,:,:organism.cell_count]=np.stack([frame.cells_local for frame in cycle.frames])
        nodes[index,:,:len(organism.skeleton_nodes)]=np.stack([frame.nodes_local for frame in cycle.frames])
        body[index]=[frame.body_velocity_x for frame in cycle.frames]
    return {"cells":cells,"nodes":nodes,"body_velocity":body,"muscle":muscle.astype(np.float32),"contact_probability":contact.astype(np.float32),"identity":np.asarray(corpus.teacher.split_indices("validation"),np.uint8)}


def _metrics(corpus: ControllerCorpus, cycles: tuple[GroundedCycle, ...], muscle: np.ndarray, contact: np.ndarray,
             ablated: tuple[GroundedCycle, ...]) -> dict[str, float]:
    cell_errors=[]; node_errors=[]; appendage_errors=[]; target_dist=[]; predicted_dist=[]; ablated_errors=[]
    organisms = tuple(corpus.teacher.organisms[index] for index in corpus.teacher.split_indices("validation"))
    for local,(identity,organism,cycle,zero) in enumerate(zip(corpus.teacher.split_indices("validation"),organisms,cycles,ablated,strict=True)):
        count=organism.cell_count; nodes=len(organism.skeleton_nodes)
        target_cells=corpus.teacher.arrays["cells_local"][identity,:,:count]
        predicted_cells=np.stack([frame.cells_local for frame in cycle.frames]); zero_cells=np.stack([frame.cells_local for frame in zero.frames])
        target_nodes=corpus.teacher.arrays["nodes_local"][identity,:,:nodes]; predicted_nodes=np.stack([frame.nodes_local for frame in cycle.frames])
        cell_errors.append(np.abs(predicted_cells-target_cells).reshape(-1)); node_errors.append(np.abs(predicted_nodes-target_nodes).reshape(-1)); ablated_errors.append(np.abs(zero_cells-target_cells).reshape(-1))
        owner=corpus.teacher.arrays["appendage_owner"][identity,:count]>=0; appendage_errors.append(np.abs(predicted_cells[:,owner]-target_cells[:,owner]).reshape(-1))
        target_dist.append(float(corpus.teacher.arrays["body_world_x"][identity,-1]-corpus.teacher.arrays["body_world_x"][identity,0])); predicted_dist.append(cycle.distance_px)
    mm=corpus.muscle_mask.numpy(); om=corpus.owner_mask.numpy(); mt=corpus.muscle_target.numpy().reshape(len(cycles),72,MAX_MUSCLES); ct=corpus.contact_target.numpy().reshape(len(cycles),72,MAX_APPENDAGES)>=.5
    hard=contact>=.5; active=np.broadcast_to(om.reshape(len(cycles),72,MAX_APPENDAGES),hard.shape)
    tp=int((hard&ct&active).sum());fp=int((hard&~ct&active).sum());fn=int((~hard&ct&active).sum())
    distance_ratio=np.asarray(predicted_dist)/np.maximum(np.asarray(target_dist),1e-8)
    result={
        "cell_position_mae_px":float(np.concatenate(cell_errors).mean()),"node_position_mae_px":float(np.concatenate(node_errors).mean()),"appendage_position_mae_px":float(np.concatenate(appendage_errors).mean()),
        "muscle_activation_mae":float(np.abs(muscle.reshape(-1,MAX_MUSCLES)-mt.reshape(-1,MAX_MUSCLES))[mm].mean()),
        "contact_f1":2*tp/max(1,2*tp+fp+fn),"contact_iou":tp/max(1,tp+fp+fn),
        "loop_seam_max_px":max(c.loop_seam_max_abs for c in cycles),"maximum_contact_slip_px":max(c.maximum_contact_slip_px for c in cycles),"maximum_edge_strain":max(c.maximum_edge_strain for c in cycles),"vertical_axis_max_degrees":max(c.vertical_axis_max_degrees for c in cycles),
        "mean_advance_ratio":float(distance_ratio.mean()),"minimum_advance_ratio":float(distance_ratio.min()),"maximum_advance_ratio":float(distance_ratio.max()),
        "ablation_cell_position_mae_px":float(np.concatenate(ablated_errors).mean()),
    }
    result["neural_control_improvement_over_ablation"]=(result["ablation_cell_position_mae_px"]-result["cell_position_mae_px"])/max(result["ablation_cell_position_mae_px"],1e-8)
    return {name:round(value,9) for name,value in result.items()}


def _gates(metrics: dict[str,float]) -> dict[str,bool]:
    return {"finite":all(math.isfinite(v) for v in metrics.values()),"contact_schedule_exact":metrics["contact_iou"]>=.999,"muscle_accuracy":metrics["muscle_activation_mae"]<=.10,"cell_trajectory_accuracy":metrics["cell_position_mae_px"]<=.25,"appendage_trajectory_accuracy":metrics["appendage_position_mae_px"]<=.30,"node_trajectory_accuracy":metrics["node_position_mae_px"]<=.25,"physics_loop_closure":metrics["loop_seam_max_px"]<.002,"contacts_hold":metrics["maximum_contact_slip_px"]<.05,"tethers_hold":metrics["maximum_edge_strain"]<.12,"vertical_axis_preserved":metrics["vertical_axis_max_degrees"]<5,"all_entities_advance":metrics["minimum_advance_ratio"]>=.70 and metrics["maximum_advance_ratio"]<=1.35,"neural_controller_is_causal":metrics["neural_control_improvement_over_ablation"]>=.30}


def _render_frame(corpus: ControllerCorpus, cycles: tuple[GroundedCycle,...], frame_index:int)->Image.Image:
    panel_w,panel_h,header=300,280,62; image=Image.new("RGB",(panel_w*5,header+panel_h),(3,8,14));draw=ImageDraw.Draw(image,"RGBA")
    draw.rectangle((0,0,image.width,header),fill=(6,17,25,255));draw.text((18,10),"NEURAL MUSCLE + CONTACT POLICY // PHYSICS-EXECUTED UNSEEN GRAFTS",font=_font(20),fill=(220,243,248,255));draw.text((image.width-170,16),f"PHASE {frame_index:02d}/71",font=_font(14),fill=(77,226,247,255))
    organisms = tuple(corpus.teacher.organisms[index] for index in corpus.teacher.split_indices("validation"))
    for col,(identity,organism,cycle) in enumerate(zip(corpus.teacher.split_indices("validation"),organisms,cycles,strict=True)):
        x0,y0=col*panel_w,header; frame=cycle.frames[frame_index];record=corpus.teacher.manifest["cycles"][identity];accent=MODE_COLORS[cycle.primary_mode]
        cx,cy,scale=x0+150,y0+151,4.25;ground=cy+cycle.ground_y*scale
        draw.rectangle((x0,y0,x0+panel_w-1,y0+panel_h-1),outline=(26,65,78,255));draw.text((x0+9,y0+8),record["genome_id"].upper(),font=_font(11),fill=(220,239,244,255));draw.text((x0+9,y0+26),f"{FAMILIES[int(corpus.teacher.arrays['family'][identity])].upper()} // NEURAL {cycle.primary_mode.upper()} POLICY",font=_font(8),fill=(*accent,255));draw.line((x0+10,ground,x0+panel_w-10,ground),fill=(*accent,130),width=1);draw.line((cx,y0+48,cx,ground),fill=(95,210,231,38),width=1)
        target=corpus.teacher.arrays["cells_local"][identity,frame_index,:organism.cell_count]
        for x,y in target:
            px,py=cx+float(x)*scale,cy+float(y)*scale;draw.rectangle((px-.7,py-.7,px+.7,py+.7),outline=(214,250,255,55))
        for left,right in organism.skeleton_edges:
            a,b=frame.nodes_local[int(left)],frame.nodes_local[int(right)];draw.line((cx+a[0]*scale,cy+a[1]*scale,cx+b[0]*scale,cy+b[1]*scale),fill=(81,215,239,120),width=2)
        for (x,y),tissue in zip(frame.cells_local,organism.tissue,strict=True):
            px,py=cx+float(x)*scale,cy+float(y)*scale;color=TISSUE_COLORS[int(tissue)];draw.rectangle((px-1.45,py-1.45,px+1.45,py+1.45),fill=(*color,225))
        terminals=[]
        for appendage in range(len(organism.genome.appendages)):
            edges=np.flatnonzero(organism.skeleton_edge_appendage==appendage);terminals.append(int(organism.skeleton_edges[int(edges[-1]),1]))
        for appendage in np.flatnonzero(frame.contact_active):
            tip=frame.nodes_local[terminals[int(appendage)]];anchor=frame.contact_anchor_world[appendage].copy();anchor[0]-=frame.body_world_x
            draw.line((cx+tip[0]*scale,cy+tip[1]*scale,cx+anchor[0]*scale,cy+anchor[1]*scale),fill=(*accent,220),width=2);draw.ellipse((cx+anchor[0]*scale-4,cy+anchor[1]*scale-2,cx+anchor[0]*scale+4,cy+anchor[1]*scale+2),fill=(*accent,235))
        draw.text((x0+9,y0+panel_h-20),"SOLID neural-physics  /  GHOST authority",font=_font(8),fill=(138,168,176,255))
    return image


def _artifacts(corpus:ControllerCorpus,cycles:tuple[GroundedCycle,...],stage:Path)->dict[str,dict[str,Any]]:
    frames=stage/"frames";frames.mkdir()
    for frame in range(72):_render_frame(corpus,cycles,frame).save(frames/f"frame_{frame:03d}.png",compress_level=9)
    sheet=Image.new("RGB",(1500,342*4),(3,8,14))
    for row,frame in enumerate((0,18,36,54)):sheet.paste(_render_frame(corpus,cycles,frame),(0,row*342))
    sheet_path=stage/"controller_contact_sheet.png";sheet.save(sheet_path,compress_level=9);mp4=stage/"controller_motion.mp4";gif=stage/"controller_motion.gif"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate","18","-i",str(frames/"frame_%03d.png"),"-c:v","libx264","-pix_fmt","yuv420p",str(mp4)],check=True)
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate","18","-i",str(frames/"frame_%03d.png"),"-vf","fps=18,scale=1100:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",str(gif)],check=True);shutil.rmtree(frames)
    return {name:artifact_record_from_bytes(path.name,path.read_bytes()) for name,path in (("contact_sheet",sheet_path),("gif",gif),("mp4",mp4))}


def evaluate(checkpoint:Path,output:Path,*,device:str="cuda",visually_inspected:bool=False)->dict[str,Any]:
    checkpoint,output=Path(checkpoint).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=2*1024**3);target_device=torch.device(device);corpus=build_corpus(split="validation",device=target_device)
    candidates={};candidate_payload={}
    ablated=_simulate(corpus,np.zeros((5,72,MAX_MUSCLES),np.float32),np.zeros((5,72,MAX_APPENDAGES),np.float32))
    for name,ema in (("raw",False),("ema",True)):
        model,payload=load_model(checkpoint,ema=ema,device=target_device);muscle,contact,body=_predict(model,corpus,target_device);cycles=_simulate(corpus,muscle,contact)
        metrics=_metrics(corpus,cycles,muscle,contact,ablated);gates=_gates(metrics);candidates[name]={"metrics":metrics,"gates":gates,"passed_gate_count":sum(gates.values())};candidate_payload[name]=(muscle,contact,body,cycles,payload)
    selected=max(candidates,key=lambda n:(candidates[n]["passed_gate_count"],-candidates[n]["metrics"]["cell_position_mae_px"],-candidates[n]["metrics"]["muscle_activation_mae"]));muscle,contact,body,cycles,payload=candidate_payload[selected];gates=candidates[selected]["gates"]
    stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True)
    try:
        archive=deterministic_npz_bytes(_cycle_arrays(corpus,cycles,muscle,contact));(stage/"controlled_cycles.npz").write_bytes(archive);artifacts=_artifacts(corpus,cycles,stage);artifacts["cycles"]=artifact_record_from_bytes("controlled_cycles.npz",archive)
        report={"format":EVALUATION_FORMAT,"status":"passed" if all(gates.values()) else "failed-quality","source_sha256":source_sha256(),"evaluation_source_sha256":evaluation_source_sha256(),"checkpoint":{"path":checkpoint.relative_to(PROJECT_ROOT).as_posix(),"sha256":sha256_file(checkpoint),"updates":payload["updates"],"model_state_sha256":payload["model_state_sha256"],"ema_state_sha256":payload["ema_state_sha256"]},"scope":{"split":"untouched_grafted_sentinels","families":5,"identities":list(corpus.teacher.split_indices("validation")),"frames_per_identity":72,"physics_executed":True,"rasterizer_used":False},"selection":{"selected":selected,"criterion":"gate_count_then_cell_then_muscle","candidates":candidates},"metrics":candidates[selected]["metrics"],"gates":gates,"visually_inspected":bool(visually_inspected),"promotion_eligible":all(gates.values()) and bool(visually_inspected),"artifacts":artifacts};report["semantic_sha256"]=hashlib.sha256(canonical_json_bytes(report)).hexdigest();(stage/"evaluation_manifest.json").write_bytes(canonical_json_bytes(report));os.replace(stage,output)
    except BaseException:shutil.rmtree(stage,ignore_errors=True);raise
    return validate_evaluation(output, replay=False)


def validate_evaluation(output: Path, *, replay: bool = True, device: str = "cpu") -> dict[str, Any]:
    output = Path(output).resolve(); manifest = output / "evaluation_manifest.json"
    if manifest.is_symlink() or not manifest.is_file() or not 0 < manifest.stat().st_size <= 2 * 1024**2:
        raise ValueError("grounded controller evaluation manifest missing or oversized")
    raw = manifest.read_bytes(); report = json.loads(raw)
    if raw != canonical_json_bytes(report):
        raise ValueError("grounded controller evaluation manifest is not canonical")
    semantic = report.pop("semantic_sha256")
    if semantic != hashlib.sha256(canonical_json_bytes(report)).hexdigest():
        raise ValueError("grounded controller evaluation semantic hash drifted")
    report["semantic_sha256"] = semantic
    if (
        report["format"] != EVALUATION_FORMAT or report["source_sha256"] != source_sha256()
        or report["evaluation_source_sha256"] != evaluation_source_sha256()
        or report["scope"] != {"split":"untouched_grafted_sentinels","families":5,"identities":[1,3,5,7,9],"frames_per_identity":72,"physics_executed":True,"rasterizer_used":False}
        or report["status"] != ("passed" if all(report["gates"].values()) else "failed-quality")
        or report["promotion_eligible"] is not (all(report["gates"].values()) and report["visually_inspected"])
    ):
        raise ValueError("grounded controller evaluation authority drifted")
    checkpoint = PROJECT_ROOT / report["checkpoint"]["path"]
    if sha256_file(checkpoint) != report["checkpoint"]["sha256"]:
        raise ValueError("grounded controller evaluation checkpoint drifted")
    for artifact in report["artifacts"].values():
        path = output / artifact["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("grounded controller evaluation artifact drifted")
    selected = max(report["selection"]["candidates"], key=lambda name:(report["selection"]["candidates"][name]["passed_gate_count"],-report["selection"]["candidates"][name]["metrics"]["cell_position_mae_px"],-report["selection"]["candidates"][name]["metrics"]["muscle_activation_mae"]))
    if selected != report["selection"]["selected"] or report["metrics"] != report["selection"]["candidates"][selected]["metrics"] or report["gates"] != report["selection"]["candidates"][selected]["gates"]:
        raise ValueError("grounded controller candidate selection drifted")
    if replay:
        target_device = torch.device(device); corpus = build_corpus(split="validation", device=target_device)
        ablated = _simulate(corpus,np.zeros((5,72,MAX_MUSCLES),np.float32),np.zeros((5,72,MAX_APPENDAGES),np.float32))
        replay_candidates={}; replay_arrays={}
        for name,ema in (("raw",False),("ema",True)):
            model,_=load_model(checkpoint,ema=ema,device=target_device);muscle,contact,body=_predict(model,corpus,target_device);cycles=_simulate(corpus,muscle,contact);metrics=_metrics(corpus,cycles,muscle,contact,ablated);gates=_gates(metrics);replay_candidates[name]={"metrics":metrics,"gates":gates,"passed_gate_count":sum(gates.values())};replay_arrays[name]=deterministic_npz_bytes(_cycle_arrays(corpus,cycles,muscle,contact))
        if replay_candidates != report["selection"]["candidates"] or replay_arrays[selected] != (output/report["artifacts"]["cycles"]["path"]).read_bytes():
            raise ValueError("grounded controller exact replay drifted")
    return report


def main()->None:
    parser=argparse.ArgumentParser(description="Evaluate neural contact/muscle controller through grounded physics");parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--device",default="cuda");parser.add_argument("--visually-inspected",action="store_true");args=parser.parse_args();report=evaluate(args.checkpoint,args.output,device=args.device,visually_inspected=args.visually_inspected);print(json.dumps({"status":report["status"],"selected":report["selection"]["selected"],"metrics":report["metrics"],"gates":report["gates"],"semantic_sha256":report["semantic_sha256"]},indent=2))


if __name__=="__main__":main()
