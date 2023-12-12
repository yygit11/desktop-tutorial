import sys

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn

#tensor([100.0000,  93.1818,  76.9737])
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


# 构建模型
acc_all = []
for i in range(10):
    print(i)
    pretrained_dict = torch.load('./model_best'+str(i)+'.pth.tar')  # 加载模型训练结果
    model = models.swin_b()  # 构建一个模型
    model.head = nn.Linear(in_features=1024, out_features=2, bias=True)
    model_dict = model.state_dict()  # 查看模型的参数字典
    model_dict.update(pretrained_dict['state_dict'])  # 更新参数字典为加载模型的参数
    model.load_state_dict(model_dict)
    model.eval()  # 评估模式，预测时候用
    # print(model)
    # 预测
    a = torch.load("../data/train_data.pth")
    b = torch.load("../data/train_label.pth")
    c = torch.load("../data/val_data.pth")
    d = torch.load("../data/val_label.pth")
    e = torch.load("../data/test_data.pth")
    f = torch.load("../data/test_label.pth")
    with torch.no_grad():
        output = model(a)
        acc1, acc5 = accuracy(output, b, topk=(1, 2))
        output = model(c)
        acc2, acc5 = accuracy(output, d, topk=(1, 2))
        output = model(e)
        acc3, acc5 = accuracy(output, f, topk=(1, 2))
        acc = torch.concat([acc1,acc2,acc3],dim=0)
        print(acc)
        sys.exit()
        acc = acc.numpy().tolist()
        acc_all.append(acc)
acc_all = pd.DataFrame(acc_all,columns=['train_acc','val_acc','test_acc'])
print(acc_all)
acc_all.to_csv('acc_all.csv')


