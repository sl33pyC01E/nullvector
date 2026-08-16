from __future__ import annotations
import argparse,json
from pathlib import Path
from .contract import TrainingConfig
from .data import build_corpus
from .evaluation import evaluate
from .training import train

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("corpus");c.add_argument("--output",type=Path,required=True)
    t=sub.add_parser("train");t.add_argument("--corpus",type=Path,required=True);t.add_argument("--output",type=Path,required=True);t.add_argument("--updates",type=int,default=2400);t.add_argument("--device",default="cuda")
    e=sub.add_parser("evaluate");e.add_argument("--corpus",type=Path,required=True);e.add_argument("--checkpoint",type=Path,required=True);e.add_argument("--output",type=Path,required=True);e.add_argument("--device",default="cuda")
    a=p.parse_args()
    if a.command=="corpus":result=build_corpus(a.output);print(result.semantic_sha256)
    elif a.command=="train":print(json.dumps(train(a.corpus,a.output,training=TrainingConfig(updates=a.updates),device=a.device),indent=2))
    else:print(json.dumps(evaluate(a.checkpoint,a.corpus,a.output,device=a.device),indent=2))
if __name__=="__main__":main()

