from __future__ import annotations

import hashlib,json,os
from pathlib import Path
import tempfile
from typing import Any

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import COMPONENTS,DEFAULT_OUTPUT,FORMAT,MANIFEST_NAME,REQUIRED_DOMAINS,canonical,sha256_file,source_sha256


def _atomic(path:Path,value:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True);descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.tmp-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def _report_ready(report:dict[str,Any])->bool:
    gates=report.get("gates")
    if isinstance(gates,dict) and gates:return all(value is True for value in gates.values())
    if report.get("promotion_allowed") is True or report.get("passed") is True:return True
    return report.get("status") in {"ready","passed","quality_passed"}


def _parameters(report:dict[str,Any])->int|None:
    for key in ("parameters","parameter_count","refiner_parameters"):
        if isinstance(report.get(key),int):return int(report[key])
    for parent in ("model","training"):
        value=report.get(parent)
        if isinstance(value,dict):
            for key in ("parameters","parameter_count"):
                if isinstance(value.get(key),int):return int(value[key])
    return None


def build(output:Path=DEFAULT_OUTPUT)->dict[str,Any]:
    output=Path(output).resolve();destination=output/MANIFEST_NAME
    if destination.exists():raise FileExistsError(destination)
    require_disk_floor(output.parent,floor_gb=100,planned_bytes=16*1024**2);rows=[]
    for name,domain,classification,report_relative,artifact_relative in COMPONENTS:
        report_path=PROJECT_ROOT/report_relative
        if not report_path.is_file():raise FileNotFoundError(report_relative)
        raw=report_path.read_bytes();report=json.loads(raw);evidence_ready=_report_ready(report);artifact=None
        if artifact_relative is not None:
            artifact_path=PROJECT_ROOT/artifact_relative
            if not artifact_path.is_file():raise FileNotFoundError(artifact_relative)
            artifact={"path":artifact_relative,"bytes":artifact_path.stat().st_size,"sha256":sha256_file(artifact_path)}
        if classification=="ready" and not evidence_ready:raise ValueError(f"ready component {name} lacks passing evidence")
        if classification=="experimental" and evidence_ready and name!="neural_map_topology":raise ValueError(f"experimental component {name} unexpectedly passes all evidence")
        rows.append({"name":name,"domain":domain,"classification":classification,"report":{"path":report_relative,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"format":report.get("format"),"status":report.get("status"),"evidence_ready":evidence_ready},"artifact":artifact,"parameter_count":_parameters(report)})
    covered={row["domain"] for row in rows};missing=sorted(set(REQUIRED_DOMAINS)-covered);ready_domains=sorted({row["domain"] for row in rows if row["classification"]=="ready"});candidate_domains=sorted({row["domain"] for row in rows if row["classification"]=="validated_candidate"});experimental=[row["name"] for row in rows if row["classification"]=="experimental"]
    payload={"format":FORMAT,"status":"ensemble_scaffold_complete","source_sha256":source_sha256(),"components":rows,"summary":{"component_count":len(rows),"required_domains":list(REQUIRED_DOMAINS),"missing_domains":missing,"ready_domains":ready_domains,"candidate_domains":candidate_domains,"experimental_components":experimental,"runtime_artifact_bytes":sum(row["artifact"]["bytes"] for row in rows if row["artifact"]),"known_parameter_count":sum(row["parameter_count"] or 0 for row in rows),"coverage_complete":not missing,"all_components_promotion_ready":all(row["classification"]=="ready" for row in rows),"composite_distillation_ready":not missing and not experimental},"next_blockers":["Replace or repair the failed cellular temporal action model.","Promote metric-only behavior, colony, society, timeline, counterfactual, frame-codec, and latent-DiT artifacts with explicit replay gates.","Export compact runtime-only weights before mobile profiling."]};payload["manifest_sha256"]=hashlib.sha256(canonical(payload)).hexdigest();_atomic(destination,canonical(payload));return validate(output)


def validate(output:Path=DEFAULT_OUTPUT)->dict[str,Any]:
    output=Path(output).resolve();raw=(output/MANIFEST_NAME).read_bytes();payload=json.loads(raw)
    if raw!=canonical(payload) or payload.get("format")!=FORMAT or payload.get("source_sha256")!=source_sha256():raise ValueError("neural foundation manifest drifted")
    expected=hashlib.sha256(canonical({k:v for k,v in payload.items() if k!="manifest_sha256"})).hexdigest()
    if payload.get("manifest_sha256")!=expected:raise ValueError("neural foundation hash drifted")
    for row in payload["components"]:
        report=PROJECT_ROOT/row["report"]["path"]
        if report.stat().st_size!=row["report"]["bytes"] or sha256_file(report)!=row["report"]["sha256"]:raise ValueError(f"foundation report drifted: {row['name']}")
        if row["artifact"]:
            artifact=PROJECT_ROOT/row["artifact"]["path"]
            if artifact.stat().st_size!=row["artifact"]["bytes"] or sha256_file(artifact)!=row["artifact"]["sha256"]:raise ValueError(f"foundation artifact drifted: {row['name']}")
    return {"passed":payload["summary"]["coverage_complete"],"promotion_ready":payload["summary"]["all_components_promotion_ready"],"distillation_ready":payload["summary"]["composite_distillation_ready"],"manifest_sha256":payload["manifest_sha256"],"components":payload["summary"]["component_count"]}
