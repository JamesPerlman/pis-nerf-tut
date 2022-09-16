from argparse import ArgumentParser
import tensorflow as tf

from src.dataset import Dataset
from src.encode import encoder_fn
from src.nerf import NeRFModel
from src.nerf_trainer import NeRFTrainer

from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--dir", required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    dataset = Dataset(args.dir)
