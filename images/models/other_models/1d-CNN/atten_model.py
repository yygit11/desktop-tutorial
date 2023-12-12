import sys

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.bn1d0 = nn.BatchNorm1d(739)
        # self.bn1d1 = nn.BatchNorm1d(256)
        self.lin1 = nn.Linear(739, 256)
        self.lin2 = nn.Linear(128, 16)
        self.lin3 = nn.Linear(16, 2)
        self.conv1 = nn.Conv1d(256, 128, kernel_size=4)
        # self.conv2 = nn.Conv1d(128, 64, kernel_size=2)
        self.relu = nn.ReLU()
        self.fla = nn.Flatten()

    def _forward_impl(self, x):
        x = self.bn1d0(x)
        x = self.lin1(x)
        x = self.relu(x)

        x = x.unsqueeze(-1)
        x = torch.concat((x, x, x, x), -1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.fla(x)
        # x = self.bn1d1(x)

        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def mymodel(**kwargs):
    model = Model(**kwargs)
    return model
