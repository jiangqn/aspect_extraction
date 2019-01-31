import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from model.decnn import DECNN
from torch.utils.data import DataLoader
from dataset import DecnnDataset

class DecnnTrainer(object):

    def __init__(self, config):
        self._config = config
        self._paths = {}
        self._paths['train_data'] = self._config.base_path + 'train.npz'
        self._paths['dev_data'] = self._config.base_path + 'dev.npz'
        self._paths['glove_path'] = self._config.base_path + 'glove.npy'

    def _make_model(self):
        embedding = nn.Embedding(
            num_embeddings=self._config.vocab_size,
            embedding_dim=self._config.embed_size
        )
        embedding.weight.data.copy_(torch.from_numpy(np.load(self._paths['glove_path'])))
        # embedding.weight.requires_grad = False
        model = DECNN(
            embedding=embedding,
            dropout=self._config.dropout,
            layers=self._config.layers
        )
        return model

    def _make_data(self):
        train_dataset = DecnnDataset(self._paths['train_data'])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=2
        )
        dev_dataset = DecnnDataset(self._paths['dev_data'])
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
        train_loader, dev_loader = self._make_data()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate)
        max_f1_score = 0
        for epoch in range(1, self._config.num_epoches + 1):
            total_samples = 0
            total_loss = 0
            total_f1_score = 0
            for step, data in enumerate(train_loader):
                optimizer.zero_grad()
                sentences, labels = data
                sentences, labels = sentences.cuda(), labels.cuda()
                logits = model(sentences)
                loss = self._loss(sentences, labels, logits, criterion)
                f1_score = self._f1_score(sentences, labels, logits)
                total_samples += sentences.size(0)
                total_loss += loss * sentences.size(0)
                total_f1_score += f1_score * sentences.size(0)
                loss.backward()
                optimizer.step()
            train_loss = total_loss / total_samples
            train_f1_score = total_f1_score / total_samples
            dev_loss, dev_f1_score = self.eval(model, criterion, dev_loader)
            max_f1_score = max(max_f1_score, dev_f1_score)
            print('[epoch %3d] [train_loss %.4f] [train_f1_score %.4f] [dev_loss %.4f] [dev_f1_score %.4f]' %
                  (epoch, train_loss, train_f1_score, dev_loss, dev_f1_score))
        print('max_f1_score: %.4f' % max_f1_score)

    def eval(self, model, criterion, data_loader):
        total_samples = 0
        total_loss = 0
        total_f1_score = 0
        for data in data_loader:
            sentences, labels = data
            sentences, labels = sentences.cuda(), labels.cuda()
            with torch.no_grad():
                logits = model(sentences)
                loss = self._loss(sentences, labels, logits, criterion)
                acc = self._f1_score(sentences, labels, logits)
                total_samples += sentences.size(0)
                total_loss += loss * sentences.size(0)
                total_f1_score += acc * sentences.size(0)
        avg_loss = total_loss / total_samples
        avg_f1_score = total_f1_score / total_samples
        return avg_loss, avg_f1_score

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

    def _f1_score(self, sentences, labels, logits):
        sentences = sentences.view(-1)
        labels = labels.view(-1).masked_select(sentences != 0)
        predicts = logits.max(dim=-1, keepdim=False)[1]
        predicts = predicts.view(-1).masked_select(sentences != 0)
        TP = torch.min(labels, predicts).sum().item()
        FP = torch.min(1 - labels, predicts).sum().item()
        FN = torch.min(labels, 1 - predicts).sum().item()
        f1_score = (2 * TP) / (2 * TP + FP + FN)
        return f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='./data/official_data/processed_data/restaurant/')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epoches', type=int, default=100)
parser.add_argument('--vocab_size', type=int, default=4602)
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--layers', type=list, default=[[[128, 5], [128, 3]], [[256, 5]], [[256, 5]], [[256, 5]]])
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--dropout', type=float, default=0.5)

config = parser.parse_args()

trainer = DecnnTrainer(config)
trainer.run()