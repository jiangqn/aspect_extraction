import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from model import TECNN

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Trainer(object):

    def __init__(self, config):
        self._config = config

    def run(self):
        pass

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epoches', type=int, default=10)
parser.add_argument('--layers', type=list, default=[[[128, 5], [128, 3]], [[256, 5]], [[256, 5]], [[256, 5]]])
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.5)

config = parser.parse_args()

trainer = Trainer(config)
trainer.run()