import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from model.gcae import GCAE
import os
from torch.utils.data import DataLoader
from dataset import GcaeDataset

class GcaeTrainer(object):

    def __init__(self, config):
        self._config = config
        self._paths = {}
        base_path = self._config.base_path
        self._paths['train_data'] = os.path.join(base_path, 'train.npz')
        self._paths['dev_data'] = os.path.join(base_path, 'test.npz')

    def _make_model(self):
        embedding = nn.Embedding(self._config.vocab_size, self._config.embed_size)
        embedding.weight.data.copy_(torch.from_numpy(np.load(self._paths['glove_path'])))
        # embedding.weight.requires_grad = False
        model = GCAE(
            embedding=embedding,
            kernel_num=self._config.kernel_num,
            kernel_sizes=self._config.kernel_sizes,
            aspect_embedding=embedding,
            aspect_kernel_num=100,
            aspect_kernel_sizes=[4],
            dropout=self._config.dropout
        )
        return model

    def _make_data(self):
        train_dataset = GcaeDataset(self._paths['train_data'])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=2
        )
        dev_dataset = GcaeDataset(self._paths['dev_data'])
        dev_loader = DataLoader(
            dataset=dev_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=2
        )
        return train_loader, dev_loader

    def run(self):
        model = self._make_model()
        model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate, weight_decay=self._config.l2_reg)
        for epoch in range(self._config.num_epoches):
            train_total_cases = 0
            train_correct_cases = 0
            for data in train_data:
                src, _, _, _, term, _, labels, _, _, _ = data
                src, term, labels = src.cuda(), term.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(src, term)
                _, predicts = outputs.max(dim=1)
                train_total_cases += labels.shape[0]
                train_correct_cases += (predicts == labels).sum().item()
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self._config.clip)
                optimizer.step()
            dev_total_cases = 0
            dev_correct_cases = 0
            with torch.no_grad():
                for data in dev_data:
                    src, _, _, _, term, _, labels, _, _, _ = data
                    src, term, labels = src.cuda(), term.cuda(), labels.cuda()
                    outputs = model(src, term)
                    _, predicts = outputs.max(dim=1)
                    dev_total_cases += labels.shape[0]
                    dev_correct_cases += (predicts == labels).sum().item()
            train_accuracy = train_correct_cases / train_total_cases
            dev_accuracy = dev_correct_cases / dev_total_cases
        for i in range(self._config.num_epoches):
            print('[epoch %2d] [train_accuracy %.4f] [dev_accuracy %.4f]' % (i, train_results[i], dev_results[i]))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--vocab_size', type=int, default=4180)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--l2_reg', type=float, default=0.002)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--num_epoches', type=int, default=30)
parser.add_argument('--base_path', type=str, default='./data/restaurant')
parser.add_argument('--kernel_num', type=int, default=100)
parser.add_argument('--kernel_sizes', type=list, default=[3, 4, 5, 6])
parser.add_argument('--aspect_kernel_num', type=int, default=100)
parser.add_argument('--aspect_kernel_sizes', type=list, default=[4])

config = parser.parse_args()
trainer = GcaeTrainer(config)
trainer.run()