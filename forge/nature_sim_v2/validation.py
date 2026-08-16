from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .world import NatureWorld


def run_long_horizon(*, seed: int=0x4E4154555245, steps: int=1200, delta: float=.25, output: Path|None=None) -> dict[str,object]:
    if not 100<=steps<=100_000: raise ValueError("nature horizon drifted")
    left=NatureWorld(seed=seed);right=NatureWorld(seed=seed)
    left.seed_founders(variants_per_family=3);right.seed_founders(variants_per_family=3)
    minimum_family=[10**9]*5;maximum_population=0
    for _ in range(steps):
        a=left.step(delta);b=right.step(delta)
        if a.semantic_sha256!=b.semantic_sha256:raise AssertionError("nature exact replay diverged")
        maximum_population=max(maximum_population,a.population)
        minimum_family=[min(old,new) for old,new in zip(minimum_family,a.family_counts)]
    final=left.snapshot()
    gates={
        "exact_replay":final.semantic_sha256==right.snapshot().semantic_sha256,
        "finite_bounded_population":0<final.population<=left.max_population and maximum_population<=left.max_population,
        "births_present":final.births>0,
        "multiple_lineages":final.lineage_count>=5,
        "multiple_families":sum(v>0 for v in final.family_counts)>=3,
        "colonies_present":final.colony_count>0,
        "resources_nonnegative":min(final.resource_totals)>=0,
    }
    report={"format":"nullvector-nature-sim-v2-validation/1.0.0","seed":seed,"steps":steps,"delta":delta,"final":asdict(final),"maximum_population":maximum_population,"minimum_family_counts":minimum_family,"gates":gates,"passed":all(gates.values())}
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n","utf-8")
    return report

