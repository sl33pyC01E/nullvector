from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from ..creature_stage_developmental.development import develop
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_grounded_locomotion.physics import simulate_grounded_cycle
from ..safety import require_disk_floor
from .physics import simulate_target_field_cycle
from .runtime import NeuralTargetFieldRuntime

AUDIT_FORMAT="nullvector-neural-grounded-target-field-audit/1.0.0"

def _canonical(value)->bytes:
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode()

def _worker(checkpoint:Path,identity:int,output:Path)->None:
    organism=develop(review_genomes()[identity]);teacher=simulate_grounded_cycle(organism)
    runtime=NeuralTargetFieldRuntime.from_checkpoint(checkpoint,device="cpu")
    first=simulate_target_field_cycle(organism,runtime);second=simulate_target_field_cycle(organism,runtime)
    if first.identity_sha256!=second.identity_sha256: raise ValueError("target field cycle replay drifted")
    row={"identity":identity,"genome_id":organism.genome.genome_id,"organism_sha256":organism.identity_sha256,"cycle_sha256":first.identity_sha256,
        "distance_ratio":first.distance_px/teacher.distance_px,"maximum_contact_slip_px":first.maximum_contact_slip_px,"maximum_edge_strain":first.maximum_edge_strain,
        "vertical_axis_max_degrees":first.vertical_axis_max_degrees,"loop_seam_max_abs":first.loop_seam_max_abs}
    output.write_bytes(_canonical(row))

def audit(checkpoint:Path,output:Path,*,attempts:int=3,timeout_seconds:int=180)->dict:
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve();require_disk_floor(output.parent,floor_gb=100,planned_bytes=32*1024**2)
    if output.exists(): raise FileExistsError(output)
    stage=output.parent/f".{output.name}.tmp-{uuid.uuid4().hex}";stage.mkdir(parents=True);telemetry=[];rows=[];environment=os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES":"-1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","PYTHONHASHSEED":"0"})
    for identity in range(len(review_genomes())):
        result=stage/f"identity_{identity:02d}.json";accepted=False
        for attempt in range(1,attempts+1):
            command=[sys.executable,"-m","forge.creature_stage_neural_target_field_v1.audit","--worker",str(identity),str(checkpoint),str(result)]
            started=time.perf_counter()
            try: completed=subprocess.run(command,cwd=Path(__file__).resolve().parents[2],env=environment,capture_output=True,text=True,timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                telemetry.append({"identity":identity,"attempt":attempt,"returncode":None,"seconds":time.perf_counter()-started,"timeout":True,"stderr":str(exc)[-1000:]});continue
            telemetry.append({"identity":identity,"attempt":attempt,"returncode":completed.returncode,"seconds":time.perf_counter()-started,"timeout":False,"stderr":completed.stderr[-1000:]})
            if completed.returncode==0 and result.is_file(): accepted=True;break
        if not accepted: raise RuntimeError(f"target field audit identity {identity} exhausted retries")
        rows.append(json.loads(result.read_bytes()))
    metrics={"distance_ratio_min":min(row["distance_ratio"] for row in rows),"distance_ratio_max":max(row["distance_ratio"] for row in rows),
        "maximum_contact_slip_px":max(row["maximum_contact_slip_px"] for row in rows),"maximum_edge_strain":max(row["maximum_edge_strain"] for row in rows),
        "vertical_axis_max_degrees":max(row["vertical_axis_max_degrees"] for row in rows),"loop_seam_max_abs":max(row["loop_seam_max_abs"] for row in rows)}
    gates={"all_identities":len(rows)==10,"advance":metrics["distance_ratio_min"]>.75 and metrics["distance_ratio_max"]<1.3,
        "physics":metrics["maximum_contact_slip_px"]<.05 and metrics["maximum_edge_strain"]<.12 and metrics["vertical_axis_max_degrees"]<5 and metrics["loop_seam_max_abs"]<.002}
    artifacts=[]
    for identity in range(len(rows)):
        path=stage/f"identity_{identity:02d}.json";raw=path.read_bytes();artifacts.append({"path":path.name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
    gates["all_passed"]=all(gates.values());raw=checkpoint.read_bytes();report={"format":AUDIT_FORMAT,"status":"passed" if gates["all_passed"] else "failed-quality",
        "checkpoint":{"path":str(checkpoint),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()},"metrics":metrics,"gates":gates,"identities":rows,"artifacts":artifacts,"telemetry":telemetry}
    report["semantic_sha256"]=hashlib.sha256(_canonical(report)).hexdigest();(stage/"audit_report.json").write_bytes(_canonical(report))
    if not gates["all_passed"]: raise ValueError("target field exhaustive audit failed quality")
    os.replace(stage,output);validate_audit(output,checkpoint=checkpoint);return report

def validate_audit(root:Path,*,checkpoint:Path|None=None)->dict:
    root=Path(root).resolve();raw=(root/"audit_report.json").read_bytes();report=json.loads(raw)
    if raw!=_canonical(report): raise ValueError("target audit report is not canonical")
    semantic=report.pop("semantic_sha256")
    if semantic!=hashlib.sha256(_canonical(report)).hexdigest(): raise ValueError("target audit semantic hash drifted")
    report["semantic_sha256"]=semantic
    if report.get("format")!=AUDIT_FORMAT or report.get("status")!="passed": raise ValueError("target audit status drifted")
    rows=report.get("identities",[])
    if [row.get("identity") for row in rows]!=list(range(10)) or len({row.get("genome_id") for row in rows})!=10: raise ValueError("target audit identity census drifted")
    metrics={"distance_ratio_min":min(row["distance_ratio"] for row in rows),"distance_ratio_max":max(row["distance_ratio"] for row in rows),
        "maximum_contact_slip_px":max(row["maximum_contact_slip_px"] for row in rows),"maximum_edge_strain":max(row["maximum_edge_strain"] for row in rows),
        "vertical_axis_max_degrees":max(row["vertical_axis_max_degrees"] for row in rows),"loop_seam_max_abs":max(row["loop_seam_max_abs"] for row in rows)}
    if report.get("metrics")!=metrics: raise ValueError("target audit aggregate drifted")
    gates={"all_identities":True,"advance":metrics["distance_ratio_min"]>.75 and metrics["distance_ratio_max"]<1.3,
        "physics":metrics["maximum_contact_slip_px"]<.05 and metrics["maximum_edge_strain"]<.12 and metrics["vertical_axis_max_degrees"]<5 and metrics["loop_seam_max_abs"]<.002}
    gates["all_passed"]=all(gates.values())
    if report.get("gates")!=gates or not gates["all_passed"]: raise ValueError("target audit gate drifted")
    artifacts=report.get("artifacts",[])
    if [item.get("path") for item in artifacts]!=[f"identity_{i:02d}.json" for i in range(10)]: raise ValueError("target audit artifact census drifted")
    for item,row in zip(artifacts,rows,strict=True):
        path=root/item["path"];payload=path.read_bytes()
        if len(payload)!=item["bytes"] or hashlib.sha256(payload).hexdigest()!=item["sha256"] or payload!=_canonical(row): raise ValueError("target audit worker artifact drifted")
    checkpoint=Path(checkpoint or report["checkpoint"]["path"]).resolve();payload=checkpoint.read_bytes()
    if len(payload)!=report["checkpoint"]["bytes"] or hashlib.sha256(payload).hexdigest()!=report["checkpoint"]["sha256"]: raise ValueError("target audit checkpoint drifted")
    if not all(any(item["identity"]==identity and item["returncode"]==0 for item in report.get("telemetry",[])) for identity in range(10)): raise ValueError("target audit successful-attempt coverage drifted")
    return report

def main()->None:
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=int);parser.add_argument("checkpoint",type=Path);parser.add_argument("output",type=Path);args=parser.parse_args()
    if args.worker is not None: _worker(args.checkpoint,args.worker,args.output)
    else: print(json.dumps(audit(args.checkpoint,args.output),indent=2,sort_keys=True))

if __name__=="__main__": main()
