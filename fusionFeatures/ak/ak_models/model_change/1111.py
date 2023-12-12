import sys
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics

model = nn.Sequential()
model.add_module('dense', nn.Linear(739, 32))
model.add_module('re_lu', nn.ReLU())
model.add_module('dense_1', nn.Linear(32, 32))
model.add_module('re_lu_1', nn.ReLU())
model.add_module('dense_2', nn.Linear(32, 1))
model.add_module('classhead', nn.Sigmoid())
model.load_state_dict(torch.load('./ak_torchModel.pth.tar'))
model = model.cuda()

data = pd.read_csv('./data_test.csv')
data = torch.Tensor(data.values)
data = data.cuda()
pred = model(data)
pred = pred.cpu().detach().numpy()
for i in range(len(pred)):
    if pred[i] < 0.5:
        pred[i] = 0
    else:
        pred[i] = 1

Y_test = pd.read_csv("../../../gp_data2/label.csv")
data_l = pd.read_csv("../../../gp_data/label.csv")
Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
Y_val = pd.concat([data_l.iloc[50:72, :], data_l.iloc[123:, :]]).values.reshape(-1)
test_score = metrics.accuracy_score(Y_test, pred)

print(test_score)
