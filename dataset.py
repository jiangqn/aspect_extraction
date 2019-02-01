import torch
from torch.utils.data import Dataset
import numpy as np

class DecnnDataset(Dataset):

    def __init__(self, path=None):
        data = np.load(path)
        self._sentences = torch.from_numpy(data['sentences']).long()
        self._labels = torch.from_numpy(data['labels']).long()
        self._len = self._sentences.size(0)

    def __getitem__(self, index):
        return self._sentences[index], self._labels[index]

    def __len__(self):
        return self._len

class GcaeDataset(Dataset):

    def __init__(self, path):
        data = np.load(path)
        self._sentences = torch.from_numpy(data['sentences']).long()
        self._terms = torch.from_numpy(data['terms']).long()
        self._labels = torch.from_numpy(data['labels']).long()
        self._len = self._labels.size(0)

    def __getitem__(self, index):
        return self._sentences[index], self._terms[index], self._labels[index]

    def __len__(self):
        return self._len