from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..organism_raster_vae_v3.calibration import _canonical
from .calibration import _evaluate,_graph_contact,_load_parent,source_sha256
from .dataset import GraphTokenCorpus
from .model import GraphTokenRasterVAE


def render(checkpoint_path: Path,output: Path,report: Path) -> None:
    device=torch.device("cuda");payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if payload.get("source_sha256")!=source_sha256():raise ValueError("graph visual worker source drifted")
    model=GraphTokenRasterVAE();model.load_state_dict(payload["ema_state"],strict=True);model=model.to(device).eval();parent,_,_=_load_parent(device);corpus=GraphTokenCorpus();validation=[index for index,(identity,_) in enumerate(corpus.rows) if identity in {5,11,17,23,29}];baseline,baseline_capture=_evaluate(parent,corpus,validation,device,False);metrics,captures=_evaluate(model,corpus,validation,device,True);image=_graph_contact(captures,baseline_capture);output.parent.mkdir(parents=True,exist_ok=True);image.save(output,compress_level=7);decoded=np.asarray(Image.open(output).convert("RGB"));cell=192;panels=[int(np.count_nonzero(decoded[59+r*(cell+36):59+r*(cell+36)+cell,14+c*(cell+18):14+c*(cell+18)+cell].max(2)>24)) for r in range(5) for c in range(3)];complete=all(value>300 for value in panels);report.write_bytes(_canonical({"format":"nullvector-graph-token-visual-replay/1.0.0","baseline_metrics":baseline,"metrics":metrics,"panel_nonblack_pixels":panels,"complete":complete}))


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--report",type=Path,required=True);args=parser.parse_args();render(args.checkpoint,args.output,args.report)


if __name__=="__main__":main()
