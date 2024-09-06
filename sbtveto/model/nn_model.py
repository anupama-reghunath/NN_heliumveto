import torch
import torch.nn as nn

from torch import nn


class NN(nn.Module):
    """
    Basic class for a customisable NN model in pytorch
    """
    def __init__(self, input_dim = 8005, output_dim = 3, hidden_sizes = [128, 64, 32, 16, 8], dropout = 0.1):
        super(NN, self).__init__()
        self._layers = []
        self._batch_norms = []

        for i in range(len(hidden_sizes) -1):
            if i == 0:
                self._layers.append(nn.Linear(input_dim, hidden_sizes[i+1]))
            else:
                self._layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

            self._batch_norms.append(nn.BatchNorm1d(hidden_sizes[i+1]))

        self._layers = nn.ModuleList(self._layers)
        self._batch_norms = nn.ModuleList(self._batch_norms)
        self._dropout = nn.Dropout(dropout)
        self._output_layer = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):

        for i in range(len(self._layers)):
            x = self._batch_norms[i](torch.relu(self._layers[i](x)))
            x = self._dropout(x)
        x = self._output_layer(x)
        return x