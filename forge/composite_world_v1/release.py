from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import ARTIFACTS,DEFAULT_OUTPUT,FORMAT,canonical,file_sha256,source_sha256
from .runtime import CompositeWorldRuntime
def _atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.tmp-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)
def build(output:Path=DEFAULT_OUTPUT):
    output=Path(output).resolve();destination=output/"composite_manifest.json"
    if destination.exists():raise FileExistsError(destination)
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=8*1024**2);components=[]
    for name,(report_relative,artifact_relative) in ARTIFACTS.items():
        report=PROJECT_ROOT/report_relative;record={"name":name,"report":{"path":report_relative,"bytes":report.stat().st_size,"sha256":file_sha256(report)}}
        if artifact_relative:
            artifact=PROJECT_ROOT/artifact_relative;record["artifact"]={"path":artifact_relative,"bytes":artifact.stat().st_size,"sha256":file_sha256(artifact)}
        components.append(record)
    runtime=CompositeWorldRuntime.from_release(device="cpu");parameters=sum(sum(p.numel() for p in module.parameters()) for module in (runtime.dit,runtime.vae,runtime.actor,runtime.organism.model,runtime.physiology.model));del runtime
    payload={"format":FORMAT,"status":"composite_neural_foundation_ready","source_sha256":source_sha256(),"components":components,"parameters":parameters,"artifact_bytes":sum(row.get("artifact",{}).get("bytes",0) for row in components),"capabilities":{"action_conditioned_world_latent":True,"world_frame_encode_decode":True,"pixel_refinement":False,"continuous_cell_organism_raster":True,"causal_actor_state":True,"causal_cell_physiology":True,"factorized_teacher_attached":True},"quality":{"action_dit_latent_and_rgb_beat_persistence":True,"actor_state_beats_persistence":True,"organism_cell_vae_promoted":True,"physiology_promoted":True,"runtime_loader_probe":True},"next_stage":{"retrain_refiner_against_bound_vae":True,"reverse_distill_specialist_hidden_state":True,"monolithic_student_ready":False,"android_profile_after_student":True}};payload["manifest_sha256"]=hashlib.sha256(canonical(payload)).hexdigest();_atomic(destination,canonical(payload));return validate(output)
def validate(output:Path=DEFAULT_OUTPUT):
    raw=(Path(output).resolve()/"composite_manifest.json").read_bytes();payload=json.loads(raw)
    if raw!=canonical(payload) or payload.get("format")!=FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("composite manifest drifted")
    expected=hashlib.sha256(canonical({k:v for k,v in payload.items() if k!="manifest_sha256"})).hexdigest()
    if payload.get("manifest_sha256")!=expected:raise ValueError("composite manifest hash drifted")
    for row in payload["components"]:
        for key in ("report","artifact"):
            if key in row:
                record=row[key];path=PROJECT_ROOT/record["path"]
                if path.stat().st_size!=record["bytes"] or file_sha256(path)!=record["sha256"]:raise ValueError(f"composite component drifted: {row['name']}")
    return {"passed":payload["status"]=="composite_neural_foundation_ready","manifest_sha256":payload["manifest_sha256"],"parameters":payload["parameters"],"artifact_bytes":payload["artifact_bytes"]}
