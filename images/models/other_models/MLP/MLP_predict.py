import sys
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import nn
from d2l import torch as d2l
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)  # 寻找数组中最大的maxk个数的序号,然后排列
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
#训练集和验证集
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
#测试集
gp_data_test = pd.read_csv("../../gp_data2/FD.csv")
tx_data_test = pd.read_csv('../../densenet121/data_feature/test_data0.csv')
X_test = pd.concat([gp_data_test,tx_data_test],axis=1).values
Y_test = pd.read_csv("../../gp_data2/label.csv").values.reshape(-1)
X_test = torch.tensor(X_test,dtype=torch.float32)
Y_test = torch.tensor(Y_test,dtype=torch.float32).long()
#加载模型
pretrained_dict = torch.load('./best_models.pth.tar')  # 加载模型训练结果
model = nn.Sequential(
    nn.BatchNorm1d(739),
    nn.Linear(739, 128), nn.ReLU(),
    nn.Linear(128,2)
    )
model.load_state_dict(pretrained_dict)
# model.eval()
#预测
with torch.no_grad():
    output = model(X_train)
    output2 = model(X_val)
    output3 = model(X_test)
_, pred = output.topk(1, 1, True, True)
pred = pred.t().squeeze()
_, pred2 = output2.topk(1, 1, True, True)
pred = pred.t().squeeze()
_, pred3 = output3.topk(1, 1, True, True)
pred = pred.t().squeeze()
print(pred)
train_score = metrics.accuracy_score(Y_train, pred)
print(train_score)
train_F1 = metrics.f1_score(Y_train, pred)
print(train_F1)
val_score = metrics.accuracy_score(Y_val, pred2)
print(val_score)
val_F1 = metrics.f1_score(Y_val, pred2)
print(val_F1)
test_score = metrics.accuracy_score(Y_test, pred3)
print(test_score)
test_F1 = metrics.f1_score(Y_test, pred3)
print(test_F1)
# c = np.concatenate((c, pred.numpy()))