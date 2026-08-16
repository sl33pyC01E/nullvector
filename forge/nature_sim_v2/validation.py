from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .world import NatureWorld


def run_single_world(*,seed:int,steps:int,delta:float)->dict[str,object]:
    world=NatureWorld(seed=seed);world.seed_founders(variants_per_family=3)
    minimum_family=[10**9]*5;maximum_population=0
    for index in range(steps):
        world.step(delta,publish=False)
        if index%20==19 or index==steps-1:
            snapshot=world.snapshot();maximum_population=max(maximum_population,snapshot.population);minimum_family=[min(old,new) for old,new in zip(minimum_family,snapshot.family_counts)]
    final=world.snapshot();living=[o for o in world.organisms.values() if o.alive];intents=Counter(o.intent for o in living);total=max(1,sum(intents.values()))
    diversity={"genotype_count":len({o.genome.semantic_sha256() for o in living}),"max_generation":max((o.genome.developmental.generation for o in living),default=0),"intent_counts":dict(sorted(intents.items())),"intent_entropy":-sum((count/total)*math.log2(count/total) for count in intents.values()),"moving_fraction":sum(float(np.linalg.norm(o.velocity))>.02 for o in living)/max(1,len(living))}
    return {"final":asdict(final),"maximum_population":maximum_population,"minimum_family_counts":minimum_family,"diversity":diversity}


def _worker_command(seed:int,steps:int,delta:float,output:Path)->list[str]:
    return [sys.executable,"-m","forge.nature_sim_v2.validation_worker","--seed",str(seed),"--steps",str(steps),"--delta",str(delta),"--output",str(output)]


def run_long_horizon(*,seed:int=0x4E4154555245,steps:int=1200,delta:float=.25,output:Path|None=None,max_attempts:int=3)->dict[str,object]:
    if not 100<=steps<=100_000:raise ValueError("nature horizon drifted")
    telemetry=[];runs=[]
    with tempfile.TemporaryDirectory(prefix="nullvector-nature-validation-") as temporary:
        root=Path(temporary)
        for replay in range(2):
            published=None
            for attempt in range(1,max_attempts+1):
                target=root/f"replay-{replay}-attempt-{attempt}.json";environment=os.environ.copy();environment.update({"CUDA_VISIBLE_DEVICES":"-1","PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
                completed=subprocess.run(_worker_command(seed,steps,delta,target),cwd=Path(__file__).resolve().parents[2],env=environment,capture_output=True,text=True,timeout=max(180,steps*1.5))
                record={"replay":replay,"attempt":attempt,"returncode":completed.returncode,"stdout":completed.stdout[-2000:],"stderr":completed.stderr[-4000:],"published":target.is_file()};telemetry.append(record)
                if completed.returncode==0 and target.is_file():published=json.loads(target.read_text("utf-8"));break
            if published is None:raise RuntimeError(f"nature validation replay {replay} exhausted isolated attempts")
            runs.append(published)
    exact=runs[0]==runs[1];final=runs[0]["final"];diversity=runs[0]["diversity"]
    gates={"exact_replay":exact,"finite_bounded_population":0<final["population"]<=180 and runs[0]["maximum_population"]<=180,"births_present":final["births"]>0,"multiple_lineages":final["lineage_count"]>=5,"multiple_families":sum(v>0 for v in final["family_counts"])>=3,"colonies_present":final["colony_count"]>0,"resources_nonnegative":min(final["resource_totals"])>=0,"activity_did_not_collapse":diversity["moving_fraction"]>=.20 and diversity["intent_entropy"]>=.50,"structural_offspring_present":diversity["max_generation"]>=1 and diversity["genotype_count"]>final["lineage_count"]}
    report={"format":"nullvector-nature-sim-v2-validation/1.1.0","seed":seed,"steps":steps,"delta":delta,**runs[0],"gates":gates,"passed":all(gates.values()),"worker_telemetry":telemetry}
    if output is not None:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);stage=output.with_suffix(output.suffix+f".tmp-{os.getpid()}");stage.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n","utf-8");os.replace(stage,output)
    return report

