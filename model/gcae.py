import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class GCAE(nn.Module):

    def __init__(self, embedding, kernel_num, kernel_sizes, aspect_embedding, aspect_kernel_num, aspect_kernel_sizes, dropout):
        super(GCAE, self).__init__()
        self._embedding = embedding
        embed_size = embedding.embedding_dim
        self._aspect_embedding = aspect_embedding
        aspect_embed_size = aspect_embedding.embedding_dim
        self._sentence_conv = nn.ModuleList(
            nn.Conv1d(
                in_channels=embed_size,
                out_channels=kernel_num,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ) for kernel_size in kernel_sizes
        )
        self._sentence_conv_gate = nn.ModuleList(
            nn.Conv1d(
                in_channels=embed_size,
                out_channels=kernel_num,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ) for kernel_size in kernel_sizes
        )
        self._aspect_conv = nn.ModuleList(
            nn.Conv1d(
                in_channels=aspect_embed_size,
                out_channels=aspect_kernel_num,
                kernel_size=aspect_kernel_size,
                padding=aspect_embed_size // 2
            ) for aspect_kernel_size in aspect_kernel_sizes
        )
        self._aspect_linear = nn.Linear(len(aspect_kernel_sizes) * aspect_kernel_num, kernel_num)
        self._dropout = nn.Dropout(dropout)
        self._linear = nn.Linear(len(kernel_sizes) * kernel_num, 2)

    def forward(self, sentence, aspect):
        # sentence: Tensor (batch_size, sentence_length)
        # aspect: Tensor (batch_size, aspect_length)
        sentence = self._embedding(sentence)
        aspect = self._aspect_embedding(aspect)
        aspect = torch.cat([
            torch.max(
                F.relu(conv(aspect.transpose(1, 2))),
                dim=-1
            )[0] for conv in self._aspect_conv
        ], dim=1)
        aspect = self._aspect_linear(aspect)
        sentence = torch.cat([
            torch.max(
                torch.tanh(conv(sentence.transpose(1, 2))) * F.relu(conv_gate(sentence.transpose(1, 2)) + aspect.unsqueeze(2)),
                dim=-1
            )[0] for conv, conv_gate in zip(self._sentence_conv, self._sentence_conv_gate)
        ], dim=1)
        sentence = self._dropout(sentence)
        logits = self._linear(sentence)
        return logits