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
        self._paths['dev_data'] = os.path.join(base_path, 'dev.npz')
        self._paths['glove_path'] = self._config.base_path + '../glove.npy'
        self._paths['model_path'] = self._config.base_path + 'gcae.pkl'

    def _make_model(self):
        embedding = nn.Embedding(self._config.vocab_size, self._config.embed_size)
        embedding.weight.data.copy_(torch.from_numpy(np.load(self._paths['glove_path'])))
        # embedding.weight.requires_grad = False
        model = GCAE(
            embedding=embedding,
            kernel_num=self._config.kernel_num,
            kernel_sizes=self._config.kernel_sizes,
            aspect_embedding=embedding,
            aspect_kernel_num=self._config.aspect_kernel_num,
            aspect_kernel_sizes=self._config.aspect_kernel_sizes,
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
        train_loader, dev_loader = self._make_data()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate, weight_decay=self._config.l2_reg)
        max_acc = 0
        for epoch in range(self._config.num_epoches):
            total_samples = 0
            total_loss = 0
            total_acc = 0
            for data in train_loader:
                sentences, terms, labels = data
                sentences, terms, labels = sentences.cuda(), terms.cuda(), labels.cuda()
                optimizer.zero_grad()
                logits = model(sentences, terms)
                loss = criterion(logits, labels)
                loss.backward()
                total_samples += labels.size(0)
                total_loss += labels.size(0) * loss
                total_acc += labels.size(0) * self._accuracy(logits, labels)
                optimizer.step()
            train_loss = total_loss / total_samples
            train_acc = total_acc / total_samples
            dev_loss, dev_acc = self.eval(model, criterion, dev_loader)
            print('[epoch %3d] [train_loss %.4f] [train_acc %.4f] [dev_loss %.4f] [dev_acc %.4f]' %
                  (epoch, train_loss, train_acc, dev_loss, dev_acc))
            if dev_acc > max_acc:
                torch.save(model, self._paths['model_path'])
                max_acc = max(max_acc, dev_acc)
        print('max_acc %.4f' % max_acc)

    def _accuracy(self, logits, labels):
        predicts = logits.max(dim=-1, keepdim=False)[1]
        accuracy = (predicts == labels).float().mean().item()
        return accuracy

    def eval(self, model, criterion, data_loader):
        total_samples = 0
        total_loss = 0
        total_acc = 0
        for data in data_loader:
            sentences, terms, labels = data
            sentences, terms, labels = sentences.cuda(), terms.cuda(), labels.cuda()
            with torch.no_grad():
                logits = model(sentences, terms)
                loss = criterion(logits, labels)
                total_samples += labels.size(0)
                total_loss += labels.size(0) * loss
                total_acc += labels.size(0) * self._accuracy(logits, labels)
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epoches', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--l2_reg', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--vocab_size', type=int, default=4602)
parser.add_argument('--kernel_num', type=int, default=100)
parser.add_argument('--kernel_sizes', type=list, default=[3, 4, 5, 6])
parser.add_argument('--aspect_kernel_num', type=int, default=100)
parser.add_argument('--aspect_kernel_sizes', type=list, default=[3])
parser.add_argument('--base_path', type=str, default='./data/official_data/processed_data/restaurant/classification/')

config = parser.parse_args()

trainer = GcaeTrainer(config)
trainer.run()