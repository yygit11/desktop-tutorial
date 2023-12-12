# 多层感知机
import torch
from torch import nn
from d2l import torch as d2l
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#从零开始实现

# #获取数据集
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
#
# #初始化模型参数
# num_inputs, num_outpus,num_hiddens = 784,10,256  #单隐藏层包含256个隐藏元素
# W1 = nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)) #第一层 初始层随机
# b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True)) #偏差为隐藏层个数
# W2 = nn.Parameter(torch.randn(num_hiddens,num_outpus,requires_grad=True)) #第二层 前一个隐藏层出的输
# b2 = nn.Parameter(torch.zeros(num_outpus,requires_grad=True))
# params = [W1,b1,W2,b2]
#
# #rule函数
# def relu(X):
#     a = torch.zeros_like(X)  #zeros_like数据类型一样，元素为0
#     return torch.max(X,a)
#
# #实现模型
# def net(X):
#     X = X.reshape((-1,num_inputs))
#     H = relu(X @ W1 + b1)  #@矩阵乘法简写
#     return (H @ W2 + b2)
#
# num_epochs,lr = 10,0.1
# loss = nn.CrossEntropyLoss(reduction='none')
# updater = torch.optim.SGD(params,lr = lr)
# d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
# d2l.plt.show()








#简洁实现
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 5
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
        print(train_metrics)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, print(train_loss)
    assert train_acc <= 1 and train_acc > 0.7, print(train_acc)
    assert test_acc <= 1 and test_acc > 0.7, print(test_acc)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# d2l.plt.show()