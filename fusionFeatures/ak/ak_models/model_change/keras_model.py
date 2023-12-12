import sys

import pandas as pd
import torch
import torch.nn as nn
import torchsummary
from keras import Sequential
from keras.layers import Dense, Input, Normalization, ReLU, Activation
import autokeras as ak
from keras.models import Model
from keras.models import load_model

# keras模型手动搭建
# model = Sequential()
# model.add(Input(shape=(739,), name='input_1'))
# model.add(Normalization(axis=1))
# model.add(Dense(32, name='dense'))
# model.add(ReLU(32, name='relu'))
# model.add(Dense(32, name='dense_1'))
# model.add(ReLU(32, name='relu_1'))
# model.add(Dense(1, name='dense_2'))
# model.add(Activation('sigmoid',name='classification_head_1'))
# 加载参数
# model.load_weights('./FD5.h5')
# model.summary()
# model.save('test.h5')

# keras模型参数
model = load_model("./FD5.h5")
# print(model.layers[3].get_weights())
# print(model.layers[3])
# print(model.layers[3].get_weights()[0])  # w
# print(model.layers[3].get_weights()[0].shape)  # w
# print(model.layers[3].get_weights()[1])  # b
# print(model.layers[3].get_weights()[1].shape)  # b
# 第三层
k_dict3_w = model.layers[3].get_weights()[0]
k_dict3_w = torch.tensor(k_dict3_w.T)
k_dict3_b = model.layers[3].get_weights()[1]
k_dict3_b = torch.tensor(k_dict3_b)
print(k_dict3_w.shape)
print(k_dict3_b.shape)
# 第五层
k_dict5_w = model.layers[5].get_weights()[0]
k_dict5_w = torch.tensor(k_dict5_w.T)
k_dict5_b = model.layers[5].get_weights()[1]
k_dict5_b = torch.tensor(k_dict5_b)
print(k_dict5_w.shape)
print(k_dict5_b.shape)
# 第七层
k_dict7_w = model.layers[7].get_weights()[0]
k_dict7_w = torch.tensor(k_dict7_w.T)
k_dict7_b = model.layers[7].get_weights()[1]
k_dict7_b = torch.tensor(k_dict7_b)
print(k_dict7_w.shape)
print(k_dict7_b.shape)
print('------------------------------------------------')

# torch模型搭建
# torch模型
data = pd.read_csv('./data_test.csv')
data = torch.Tensor(data.values)
data = data.cuda()

net = nn.Sequential()
net.add_module('dense', nn.Linear(739, 32))
net.add_module('re_lu', nn.ReLU())
net.add_module('dense_1', nn.Linear(32, 32))
net.add_module('re_lu_1', nn.ReLU())
net.add_module('dense_2', nn.Linear(32, 1))
net.add_module('classhead', nn.Sigmoid())
net = net.cuda()
# sys.exit()
# torchsummary.summary(net,(16,739))
# 模型参数更新
t_dict = net.state_dict()
# 参数更新
t_dict['dense.weight'] = k_dict3_w
t_dict['dense.bias'] = k_dict3_b
t_dict['dense_1.weight'] = k_dict5_w
t_dict['dense_1.bias'] = k_dict5_b
t_dict['dense_2.weight'] = k_dict7_w
t_dict['dense_2.bias'] = k_dict7_b
net.load_state_dict(t_dict)
b = net(data)
b = b.cpu().detach().numpy()
for i in range(len(b)):
    if b[i] < 0.5:
        b[i] = 0
    else:
        b[i] = 1
Y_test = pd.read_csv("../../../gp_data2/label.csv") #label
from sklearn import metrics
test_score = metrics.accuracy_score(Y_test, b)
print(test_score)
torch.save(net.state_dict(),'./ak_torchModel.pth.tar')