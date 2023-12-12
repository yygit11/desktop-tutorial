import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data1 = pd.read_csv('FD0_train.csv')
data2 = pd.read_csv('MLP_loss.csv')
data3 = pd.read_csv('Dense_loss.csv')
data4 = pd.read_csv('1d-cnn_loss.csv')
y1 = data1.iloc[:,-1].values
y2 = data2.iloc[:,-1].values
y3 = data3.iloc[:,-1].values
y4 = data4.iloc[:,-1].values
x = data1.iloc[:,-2].values
plt.figure(figsize=(4.2, 3.1), dpi=300)
plt.xlabel('epochs', fontsize=10)
plt.ylabel('loss', fontsize=10)
# plt.ylim(0,0.7)
# plt.xlim(0,100)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.tight_layout(pad=0.3)
line1, = plt.plot(x, y1,linewidth=1.5,label='NAS')
line2, = plt.plot(x, y2,linewidth=1.5,label='MLP')
line3, = plt.plot(x, y3,linewidth=1.5,label='Dense')
line4, = plt.plot(x, y4,linewidth=1.5,label='1D-CNN')
plt.legend()
plt.savefig("./images/all_loss.png", dpi=300)
plt.show()



# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# y2 = [1, 4, 9, 16, 25, 36, 49, 64, 81] # 绘制的曲线属性设置
# line1, = plt.plot(x, y, color='r', marker='d', linestyle='--', markersize=6, alpha=0.5, linewidth=3)
# line2, = plt.plot(x, y2, color='g', marker='*', linestyle='-', markersize=6, alpha=0.5, linewidth=3)
# plt.plot(x, y, 'rd--') # 可以使用这种方式进行画图的属性设置 # x,y坐标轴名称设置,可以同时设置标签的字体大小颜色等
# plt.xlabel(u'x坐标轴', fontsize=14, color='r')
# plt.ylabel(u'y坐标轴', fontsize=14, color='b')
# # 显示曲线图像
# plt.show()