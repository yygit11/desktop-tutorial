import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import atten_model

#数据
gp_data = pd.read_csv("../../gp_data/FD.csv")
gp_data_train = pd.concat([gp_data.iloc[:50, :], gp_data.iloc[72:123, :]]).reset_index(drop=True)
gp_data_val = pd.concat([gp_data.iloc[50:72, :], gp_data.iloc[123:, :]]).reset_index(drop=True)
tx_data_train = pd.read_csv('../../densenet121/data_feature/train_data0.csv')
tx_data_val = pd.read_csv('../../densenet121/data_feature/val_data0.csv')
X_train =pd.concat([gp_data_train,tx_data_train],axis=1).values
X_val =pd.concat([gp_data_val,tx_data_val],axis=1).values
data_l = pd.read_csv("../../gp_data/label.csv")
Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
Y_val = pd.concat([data_l.iloc[50:72, :], data_l.iloc[123:, :]]).values.reshape(-1)

X_train = torch.tensor(X_train,dtype=torch.float32)
Y_train = torch.tensor(Y_train,dtype=torch.float32).long()
X_val = torch.tensor(X_val,dtype=torch.float32)
Y_val = torch.tensor(Y_val,dtype=torch.float32).long()
train_iter = d2l.load_array((X_train, Y_train), 16)
test_iter = d2l.load_array((X_val, Y_val), 16)
# 模型
net = atten_model.mymodel()
# 超参数
batch_size, lr, num_epochs = 32, 0.1, 100
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

#训练
losses = []
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    #                     legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
        # test_acc = d2l.evaluate_accuracy(net, test_iter)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
        losses.append(train_metrics[0])
    train_loss, train_acc = train_metrics
    print('train_loss:', train_loss)
    print('train_acc:', train_acc)

# for i in range(20):
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
losses = pd.DataFrame(np.array(losses))
losses.to_csv('attention_loss.csv')
# d2l.plt.show()
# torch.save(net.state_dict(), 'best_models'+str(i)+'.pth.tar')
