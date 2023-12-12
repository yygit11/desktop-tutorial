import sys

import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn

#tensor([100.0000,  90.9091,  70.1754])
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
acc_all = np.array([])
for i in range(1):
    print(i)
    pretrained_dict = torch.load('./models/model_best'+str(i)+'.pth.tar')  # 加载模型训练结果
    model = models.convnext_base()  # 构建一个模型
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
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
        acc1, acc5 = accuracy(output, d, topk=(1, 2))
        output = model(c)
        acc2, acc5 = accuracy(output, d, topk=(1, 2))
        output = model(e)
        # acc3, acc5 = accuracy(output, f, topk=(1, 2))
        acc = torch.concat([acc1,acc2],dim=0)
        print(acc)
    # acc = acc.numpy()
    # acc_all = np.concatenate((acc_all,acc),axis=0)
    # print(acc)


