import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('train_loss.csv')
data2 = pd.read_csv('val_loss.csv')
y = data.iloc[:,-1].values
x = data.iloc[:,-2].values
y2 = data2.iloc[:,-1].values
plt.figure(figsize=(4.2, 3.1), dpi=300)
plt.xlabel('epochs', fontsize=10)
plt.ylabel('loss', fontsize=10)
plt.yticks(np.arange(0.1,1,0.2),fontsize=10)
plt.xticks(fontsize=10)
plt.ylim(-0.05,1.05)
plt.tight_layout(pad=0.3)
plt.plot(x, y,linewidth=1.5,label='train_loss')
plt.plot(x, y2,linewidth=1.5,label='val_loss',color='g',linestyle='--')
plt.legend()
plt.savefig("loss.tif", dpi=300)
# plt.show()


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