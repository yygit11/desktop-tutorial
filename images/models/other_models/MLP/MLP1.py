import sys
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from torchvision import models

#训练集和验证集
gp_data = pd.read_csv("../../gp_data/FD.csv")
gp_data_train = pd.concat([gp_data.iloc[:50, :], gp_data.iloc[72:123, :]]).reset_index(drop=True)
tx_data_train = pd.read_csv('../../densenet121/data_feature/train_data0.csv')
X_train =pd.concat([gp_data_train,tx_data_train],axis=1).values
data_l = pd.read_csv("../../gp_data/label.csv")
Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
X_train = torch.tensor(X_train,dtype=torch.float32)
Y_train = torch.tensor(Y_train,dtype=torch.float32)
#模型
net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(739, 512),
                    nn.ReLU(),
                    nn.Linear(512,256),
                    nn.ReLU(),
                    nn.Linear(256, 2),
                    )
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)
lr, num_epochs = 0.1, 3
loss = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#训练
trains_ = torch.split(X_train, 16)
targets = torch.split(Y_train, 16)
for epoch in range(num_epochs):
    print('epoch:',epoch)
    metric = d2l.Accumulator(3)
    for i in range(len(targets)):
        X = trains_[i]
        y = targets[i].long()
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()
        else:
            l.sum().backward()
            optimizer(X.shape[0])
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    print("train_loss:", metric[0] / metric[2])
    print("train_acc:", metric[1] / metric[2])
sys.exit()

