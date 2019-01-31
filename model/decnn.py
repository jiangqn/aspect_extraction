import torch
import torch.nn as nn
import torch.nn.functional as F

class DECNN(nn.Module):

    def __init__(self, embedding, dropout, layers):
        super(DECNN, self).__init__()
        self._embedding = embedding
        self._dropout = nn.Dropout(dropout)
        self._layers = nn.ModuleList()
        input_size = embedding.embedding_dim
        for layer_kernels in layers:
            layer = nn.ModuleList()
            output_size = 0
            for kernel in layer_kernels:
                conv = nn.Conv1d(
                    in_channels=input_size,
                    out_channels=kernel[0],
                    kernel_size=kernel[1],
                    padding=kernel[1] // 2
                )
                layer.append(conv)
                output_size += kernel[0]
            self._layers.append(layer)
            input_size = output_size
        self._linear = nn.Linear(input_size, 2)

    def forward(self, src):
        src = self._embedding(src)
        src = self._dropout(src).transpose(1, 2)
        for layer in self._layers:
            layer_outputs = []
            for conv in layer:
                output = F.relu(conv(src))
                layer_outputs.append(output)
            src = torch.cat(layer_outputs, dim=1)
            src = self._dropout(src)
        src = src.transpose(1, 2)
        logits = self._linear(src)
        return logits