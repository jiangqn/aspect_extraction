import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Trainer(object):

    def __init__(self, config):
        self._config = config

    def run(self):
        pass

parser = argparse.ArgumentParser()

config = parser.parse_args()

trainer = Trainer(config)
trainer.run()