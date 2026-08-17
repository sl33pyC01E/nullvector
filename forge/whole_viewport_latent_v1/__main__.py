import argparse,json
from pathlib import Path
from .contract import DEFAULT_CORPUS,DEFAULT_OUTPUT,TrainingConfig
from .training import train

parser=argparse.ArgumentParser();parser.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS);parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);parser.add_argument("--steps",type=int,default=4000);parser.add_argument("--batch-size",type=int,default=8);parser.add_argument("--rollout-steps",type=int,default=4);parser.add_argument("--device",default="cuda");args=parser.parse_args();print(json.dumps(train(corpus=args.corpus,output=args.output,training=TrainingConfig(steps=args.steps,batch_size=args.batch_size,rollout_steps=args.rollout_steps),device=args.device),indent=2))
