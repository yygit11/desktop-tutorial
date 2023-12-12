import torch
import torchvision.models as models
import torch.nn as nn
from sklearn import metrics
import numpy as np


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
pretrained_dict = torch.load('./models/model_best9.pth.tar',map_location='cpu')  # 加载模型训练结果
model = models.swin_b()  # 构建一个模型
model.head = nn.Linear(in_features=1024, out_features=2, bias=True)
model_dict = model.state_dict()  # 查看模型的参数字典
model_dict.update(pretrained_dict['state_dict'])  # 更新参数字典为加载模型的参数
model.load_state_dict(model_dict)
model.eval()  # 评估模式，预测时候用
# 预测`
a = torch.load("../data/val_data.pth")
b = torch.load("../data/val_label.pth")
c = np.array([])
images = torch.split(a,16)
for i in range(len(images)):
    print(i)
    with torch.no_grad():
        output = model(images[i])
    _, pred = output.topk(1, 1, True, True)  # 寻找数组中最大的maxk个数的序号,然后排列
    pred = pred.t().squeeze()
    c = np.concatenate((c, pred.numpy()))
train_score = metrics.accuracy_score(b.numpy(), c)
print(train_score)
train_F1 = metrics.f1_score(b.numpy(), c)
print(train_F1)

