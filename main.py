import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from model.tecnn import TECNN
from torch.utils.data import DataLoader
from dataset import TecnnDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Trainer(object):

    def __init__(self, config):
        self._config = config
        self._paths = {}
        self._paths['train_data'] = self._config.base_path + 'train.npz'
        self._paths['dev_data'] = self._config.base_path + 'dev.npz'

    def _make_model(self):
        embedding = nn.Embedding(4602, 300)
        model = TECNN(
            embedding=embedding,
            dropout=self._config.dropout,
            layers=self._config.layers
        )
        return model

    def run(self):
        train_dataset = TecnnDataset(self._paths['train_data'])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=2
        )
        dev_dataset = TecnnDataset(self._paths['dev_data'])
        dev_loader = DataLoader(
            dataset=dev_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=2
        )
        model = self._make_model()
        model = model.cuda()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate)
        for epoch in range(1, self._config.num_epoches + 1):
            total_samples = 0
            total_loss = 0
            total_acc = 0
            for step, data in enumerate(train_loader):
                optimizer.zero_grad()
                sentences, labels = data
                sentences, labels = sentences.cuda(), labels.cuda()
                logits = model(sentences)
                loss = self._loss(sentences, labels, logits, criterion)
                acc = self._accuracy(sentences, labels, logits)
                total_samples += sentences.size(0)
                total_loss += loss * sentences.size(0)
                total_acc += acc * sentences.size(0)
                loss.backward()
                optimizer.step()
            train_loss = total_loss / total_samples
            train_acc = total_acc / total_samples
            dev_loss, dev_acc = self.eval(model, criterion, dev_loader)
            print('[epoch %2d] [train_loss %.4f] [train_acc %.4f] [dev_loss %.4f] [dev_acc %.4f]' %
                  (epoch, train_loss, train_acc, dev_loss, dev_acc))

    def eval(self, model, criterion, data_loader):
        total_samples = 0
        total_loss = 0
        total_acc = 0
        for data in data_loader:
            sentences, labels = data
            sentences, labels = sentences.cuda(), labels.cuda()
            with torch.no_grad():
                logits = model(sentences)
                loss = self._loss(sentences, labels, logits, criterion)
                acc = self._accuracy(sentences, labels, logits)
                total_samples += sentences.size(0)
                total_loss += loss * sentences.size(0)
                total_acc += acc * sentences.size(0)
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc

    def _loss(self, sentences, labels, logits, criterion):
        sentences = sentences.view(-1)
        labels = labels.view(-1)
        logits = logits.view(-1, logits.size(-1))
        losses = criterion(logits, labels).masked_select(sentences != 0)
        loss = losses.mean()
        return loss

    def _accuracy(self, sentences, labels, logits):
        sentences = sentences.view(-1)
        labels = labels.view(-1)
        predicts = logits.max(dim=-1, keepdim=False)[1]
        predicts = predicts.view(-1)
        results = (predicts == labels).masked_select(sentences != 0)
        accuracy = results.float().mean().item()
        return accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='./data/official_data/processed_data/restaurant/')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epoches', type=int, default=100)
parser.add_argument('--layers', type=list, default=[[[128, 5], [128, 3]], [[256, 5]], [[256, 5]], [[256, 5]]])
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--dropout', type=float, default=0.5)

config = parser.parse_args()

trainer = Trainer(config)
trainer.run()