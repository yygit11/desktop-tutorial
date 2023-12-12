import sys

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.bn1d0 = nn.BatchNorm1d(739)
        self.bn1d1 = nn.BatchNorm1d(512)
        self.SA = nn.MultiheadAttention(4, 4)
        self.Lin1 = nn.Linear(739, 128)
        self.Lin2 = nn.Linear(512, 32)
        self.Lin3 = nn.Linear(32, 2)
        self.Fla = nn.Flatten()
        self.relu = nn.ReLU()

    def _forward_impl(self, x):
        x = x.to(torch.float32)
        x = self.bn1d0(x)
        x = self.Lin1(x)
        x = self.relu(x)

        x = x.unsqueeze(-1)
        x = torch.concat((x, x, x, x), -1)
        x, _ = self.SA(x, x, x)
        x = self.Fla(x)
        x = self.relu(x)
        x = self.bn1d1(x)

        x = self.Lin2(x)
        x = self.relu(x)
        x = self.Lin3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def mymodel(**kwargs):
    model = Model(**kwargs)
    return model
