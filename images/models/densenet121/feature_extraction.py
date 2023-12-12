import sys

import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn

# 构建模型
for i in range(1):
    print(i)
    pretrained_dict = torch.load('./models/model_best'+str(i)+'.pth.tar')  # 加载模型训练结果
    model = models.densenet121()  # 构建一个模型
    model.classifier = nn.Linear(model.classifier.in_features, 2) #densenet121
    model_dict = model.state_dict()  # 查看模型的参数字典
    model_dict.update(pretrained_dict['state_dict'])  # 更新参数字典为加载模型的参数
    model.load_state_dict(model_dict)
    for j in model_dict:
        print(j)
    for j in pretrained_dict['state_dict']:
        print(j)
    model.eval()  # 评估模式，预测时候用
    sys.exit()
    model.features[-6] = nn.Sequential()
    model.features[-5] = nn.Sequential()
    model.features[-4] = nn.Sequential()
    model.features[-3] = nn.Sequential()
    model.features[-2] = nn.Sequential()
    model.features[-1] = nn.Sequential()
    model.classifier = nn.Sequential()
    # model.classifier = nn.Sequential()
    # model.features.denseblock4 = nn.Sequential()
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
        train_data.to_csv('./data_feature3/train_data'+str(i)+'.csv')
        output2 = model(c)
        print(output2.shape)
        train_data = pd.DataFrame(output2.numpy())
        train_data.to_csv('./data_feature3/val_data' + str(i) + '.csv')
        output3 = model(e)
        print(output3.shape)
        train_data = pd.DataFrame(output3.numpy())
        train_data.to_csv('./data_feature3/test_data' + str(i) + '.csv')


