from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ..android_ensemble_v2.export import sha256_file
from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, ModelConfig, TrainingConfig, canonical, config_dict, corpus_source_sha256, source_sha256
from .model import MobileCoordinatorStudent


DEFAULT_CORPUS = PROJECT_ROOT / "outputs/mobile_coordinator_student_v1/corpus_001"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mobile_coordinator_student_v1/production_001"
INPUTS = ("current", "previous", "global_state", "previous_global", "members", "member_mask", "society", "sequence")
TARGETS = ("target_macro", "target_macro_global", "target_role", "target_member_action", "target_activity", "target_labor", "target_diplomacy", "target_project", "target_timeline", "target_event", "target_confidence", "target_counter_state", "target_benefit", "target_risk")


def _atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, temporary); os.replace(temporary, path)


def load_corpus(root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    root = Path(root).resolve(); encoded = (root / "manifest.json").read_bytes(); manifest = json.loads(encoded)
    if encoded != canonical(manifest) or manifest.get("source_sha256") != corpus_source_sha256() or manifest.get("status") != "ready": raise ValueError("coordinator corpus manifest drifted")
    expected = hashlib.sha256(canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})).hexdigest()
    if manifest.get("manifest_sha256") != expected: raise ValueError("coordinator corpus identity drifted")
    path = root / manifest["artifact"]["path"]
    if path.stat().st_size != manifest["artifact"]["bytes"] or sha256_file(path) != manifest["artifact"]["sha256"]: raise ValueError("coordinator corpus artifact drifted")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(INPUTS + TARGETS): raise ValueError("coordinator corpus member closure drifted")
        arrays = {name: archive[name] for name in archive.files}
    return arrays, manifest


def _batch(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    values = []
    for name in INPUTS + TARGETS:
        value = arrays[name][indices]
        tensor = torch.from_numpy(value)
        if name == "member_mask": tensor = tensor.bool()
        else: tensor = tensor.float()
        values.append(tensor.to(device, non_blocking=True))
    return tuple(values)


def _loss(outputs: tuple[torch.Tensor, ...], batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict[str, float]]:
    current = batch[0]; targets = batch[len(INPUTS):]
    tm, tg, tr, ta, tact, tlabor, tdip, tproj, tt, tevent, tconf, tcs, tb, trisk = targets
    pm, pg, pr, pa, pact, plabor, pdip, pproj, pt, pevent, pconf, pcs, pb, prisk = outputs
    smooth = torch.nn.functional.smooth_l1_loss; ce = torch.nn.functional.cross_entropy
    valid = batch[5].flatten()
    losses = {
        "macro": smooth((pm-current)*100, (tm-current)*100), "macro_global": smooth((pg-batch[2])*24, (tg-batch[2])*24),
        "colony_role": ce(pr.flatten(0,1)[valid], tr.argmax(-1).flatten()[valid]), "colony_action": smooth(pa[batch[5]], ta[batch[5]]),
        "society": (ce(pact,tact.argmax(-1))+ce(plabor,tlabor.argmax(-1))+ce(pdip,tdip.argmax(-1))+ce(pproj,tproj.argmax(-1)))/4,
        "timeline": smooth((pt-batch[7][:,-1])*8,(tt-batch[7][:,-1])*8), "timeline_event": ce(pevent,tevent.argmax(-1)), "timeline_conf": smooth(pconf,tconf),
        "counter_state": smooth((pcs-batch[7][:,-1,None])*5,(tcs-batch[7][:,-1,None])*5), "counter_value": smooth(pb,tb)+smooth(prisk,trisk),
        "counter_rank": ce(pb * 6, tb.argmax(-1)),
    }
    total = losses["macro"]*4 + losses["macro_global"] + losses["colony_role"]*.5 + losses["colony_action"] + losses["society"]*.65 + losses["timeline"] + losses["timeline_event"]*.5 + losses["timeline_conf"] + losses["counter_state"] + losses["counter_value"]*2 + losses["counter_rank"]
    return total, {name: float(value.detach()) for name, value in losses.items()}


@torch.inference_mode()
def evaluate(model: nn.Module, arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, float]:
    metrics: dict[str, list[float]] = {name: [] for name in ("macro_mae","persistence_mae","global_mae","role","activity","labor","diplomacy","project","timeline_mae","event","counter_state_mae","counter_best")}
    for start in range(0, len(indices), 32):
        batch = _batch(arrays, indices[start:start+32], device); outputs = model(*batch[:len(INPUTS)]); targets = batch[len(INPUTS):]
        pm,pg,pr,pa,pact,plabor,pdip,pproj,pt,pevent,pconf,pcs,pb,prisk=outputs;tm,tg,tr,ta,tact,tlabor,tdip,tproj,tt,tevent,tconf,tcs,tb,trisk=targets
        metrics["macro_mae"].append(float((pm-tm).abs().mean()));metrics["persistence_mae"].append(float((batch[0]-tm).abs().mean()));metrics["global_mae"].append(float((pg-tg).abs().mean()))
        valid=batch[5]; metrics["role"].append(float((pr.argmax(-1)[valid]==tr.argmax(-1)[valid]).float().mean()));metrics["activity"].append(float((pact.argmax(-1)==tact.argmax(-1)).float().mean()));metrics["labor"].append(float((plabor.argmax(-1)==tlabor.argmax(-1)).float().mean()));metrics["diplomacy"].append(float((pdip.argmax(-1)==tdip.argmax(-1)).float().mean()));metrics["project"].append(float((pproj.argmax(-1)==tproj.argmax(-1)).float().mean()))
        metrics["timeline_mae"].append(float((pt-tt).abs().mean()));metrics["event"].append(float((pevent.argmax(-1)==tevent.argmax(-1)).float().mean()));metrics["counter_state_mae"].append(float((pcs-tcs).abs().mean()));metrics["counter_best"].append(float((pb.argmax(-1)==tb.argmax(-1)).float().mean()))
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def train(output: Path = DEFAULT_OUTPUT, corpus: Path = DEFAULT_CORPUS, *, plan: TrainingConfig = TrainingConfig(), config: ModelConfig = ModelConfig(), device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True); require_disk_floor(output, floor_gb=100, planned_bytes=1 << 30)
    arrays, corpus_manifest = load_corpus(corpus); count = len(arrays["current"]); rng = np.random.default_rng(plan.seed); order = rng.permutation(count); validation = order[-max(128,count//8):]; training = order[:-len(validation)]
    selected = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"); torch.manual_seed(plan.seed); model = MobileCoordinatorStudent(config).to(selected); ema = copy.deepcopy(model).eval(); optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=plan.weight_decay, fused=selected.type=="cuda")
    scaler = None; history=[]; began=time.perf_counter(); generator=np.random.default_rng(plan.seed^0x545241494E)
    for step in range(1,plan.steps+1):
        indices=generator.choice(training,size=plan.batch_size,replace=True); batch=_batch(arrays,indices,selected); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(selected.type,dtype=torch.bfloat16,enabled=selected.type=="cuda"): outputs=model(*batch[:len(INPUTS)]); loss,parts=_loss(outputs,batch)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        with torch.no_grad():
            for target,source in zip(ema.parameters(),model.parameters()): target.lerp_(source,1-plan.ema_decay)
        if step==1 or step%100==0: history.append({"step":step,"loss":float(loss.detach()),**parts})
        if step%400==0 or step==plan.steps:
            metrics=evaluate(ema,arrays,validation,selected); payload={"format":CHECKPOINT_FORMAT,"status":"training","source_sha256":source_sha256(),"corpus_manifest_sha256":corpus_manifest["manifest_sha256"],"model_config":config_dict(config),"training_config":config_dict(plan),"step":step,"state":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"metrics":metrics};_atomic(output/f"coordinator_{step:07d}.pt",payload);_atomic(output/"latest.pt",payload)
    metrics=evaluate(model,arrays,validation,selected); ema_metrics=evaluate(ema,arrays,validation,selected); gates={"macro_not_worse_than_persistence":metrics["macro_mae"]<=metrics["persistence_mae"]*1.10,"role_agreement":metrics["role"]>=.85,"society_agreement":min(metrics[name] for name in ("activity","labor","diplomacy","project"))>=.80,"timeline_event_agreement":metrics["event"]>=.75,"counterfactual_best_agreement":metrics["counter_best"]>=.80}
    final={"format":CHECKPOINT_FORMAT,"status":"mobile_coordinator_ready" if all(gates.values()) else "quality_failed","source_sha256":source_sha256(),"corpus_manifest_sha256":corpus_manifest["manifest_sha256"],"model_config":config_dict(config),"training_config":config_dict(plan),"step":plan.steps,"state":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"metrics":metrics,"ema_metrics":ema_metrics,"deployment_state":"state","gates":gates,"parameters":sum(p.numel() for p in model.parameters()),"seconds":time.perf_counter()-began};_atomic(output/"final.pt",final)
    report={key:value for key,value in final.items() if key not in ("state","ema","optimizer")};report["checkpoint"]={"path":"final.pt","bytes":(output/"final.pt").stat().st_size,"sha256":sha256_file(output/"final.pt")};report["manifest_sha256"]=hashlib.sha256(canonical(report)).hexdigest();(output/"report.json").write_bytes(canonical(report));return report


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True);build_parser=sub.add_parser("corpus");build_parser.add_argument("--output",type=Path,default=DEFAULT_CORPUS);build_parser.add_argument("--samples",type=int,default=1536);build_parser.add_argument("--device",default="cuda");train_parser=sub.add_parser("train");train_parser.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS);train_parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);train_parser.add_argument("--steps",type=int,default=3600);train_parser.add_argument("--batch-size",type=int,default=24);train_parser.add_argument("--device",default="cuda");args=parser.parse_args(argv)
    if args.command=="corpus":
        from .corpus import build
        result=build(args.output,samples=args.samples,device=args.device)
    else: result=train(args.output,args.corpus,plan=TrainingConfig(steps=args.steps,batch_size=args.batch_size),device=args.device)
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
