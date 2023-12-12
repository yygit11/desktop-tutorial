import sys

import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn

# 构建模型
for i in range(10):
    print(i)
    pretrained_dict = torch.load('./models/model_best'+str(i)+'.pth.tar')  # 加载模型训练结果
    model = models.swin_b()  # 构建一个模型
    model.head = nn.Linear(in_features=1024, out_features=2, bias=True)
    model_dict = model.state_dict()  # 查看模型的参数字典
    model_dict.update(pretrained_dict['state_dict'])  # 更新参数字典为加载模型的参数
    model.load_state_dict(model_dict)
    model.eval()  # 评估模式，预测时候用
    model.head = nn.Sequential()
    print(model)
    # 预测
    a = torch.load("../data2/train_data.pth")
    b = torch.load("../data2/train_label.pth")
    c = torch.load("../data2/val_data.pth")
    d = torch.load("../data2/val_label.pth")
    e = torch.load("../data2/test_data.pth")
    f = torch.load("../data2/test_label.pth")
    # print(b)
    # print(d)
    # print(f)
    with torch.no_grad():
        output1 = model(a)
        print(output1.shape)
        train_data = pd.DataFrame(output1.numpy())
        train_data.to_csv('./data_feature/train_data'+str(i)+'.csv')
        output2 = model(c)
        print(output2.shape)
        train_data = pd.DataFrame(output2.numpy())
        train_data.to_csv('./data_feature/val_data' + str(i) + '.csv')
        output3 = model(e)
        print(output3.shape)
        train_data = pd.DataFrame(output3.numpy())
        train_data.to_csv('./data_feature/test_data' + str(i) + '.csv')


