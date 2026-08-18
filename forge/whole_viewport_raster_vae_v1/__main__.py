import argparse,json
from pathlib import Path
from .contract import DEFAULT_CORPUS,DEFAULT_OUTPUT,TrainingPlan
from .training import train

parser=argparse.ArgumentParser();parser.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS);parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);parser.add_argument("--updates",type=int,default=2400);parser.add_argument("--batch-size",type=int,default=4);args=parser.parse_args();print(json.dumps(train(corpus=args.corpus,output=args.output,plan=TrainingPlan(updates=args.updates,batch_size=args.batch_size)),indent=2))
