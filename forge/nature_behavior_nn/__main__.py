from __future__ import annotations

import argparse,json
from pathlib import Path

from .contract import ModelConfig,TrainingConfig
from .corpus import build_corpus
from .training import train


def main()->None:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    corpus=sub.add_parser("corpus");corpus.add_argument("output",type=Path);corpus.add_argument("--worlds",type=int,default=12);corpus.add_argument("--steps",type=int,default=260)
    fit=sub.add_parser("train");fit.add_argument("corpus",type=Path);fit.add_argument("output",type=Path);fit.add_argument("--updates",type=int,default=1200);fit.add_argument("--batch-size",type=int,default=384);fit.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.command=="corpus":result=build_corpus(args.output,worlds=args.worlds,steps=args.steps)
    else:result=train(args.corpus,args.output,training=TrainingConfig(updates=args.updates,batch_size=args.batch_size),device=args.device)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":main()
