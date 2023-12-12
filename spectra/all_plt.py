import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.text as text

font = dict(family='times new roman', weight='bold')

data = pd.read_csv("gp_data/R.csv", header=None)
data1 = pd.read_csv("gp_data/R.csv", header=None).iloc[1:, :].values
data2 = pd.read_csv("gp_data2/R.csv").values
data_x = data.iloc[0, :]

# axes = plt.figure(figsize=(15,10)) # figsize=(15,8)
fig, axes = plt.subplots(figsize=(10, 7), dpi=300)
axes.spines["top"].set_linewidth(1.8)
axes.spines["bottom"].set_linewidth(1.8)
axes.spines["left"].set_linewidth(1.8)
axes.spines["right"].set_linewidth(1.8)

p1 = axes.plot(data_x, data2.T, color='red', label="pear2")
p2 = axes.plot(data_x, data1.T, color='blue', label="pear1")
# plt.setp(p1[1:], label="_")
# plt.setp(p2[1:], label="_")
plt.legend(fontsize=24, prop=font)

axes.set_xlabel("Wavelength (nm)", fontsize=26, fontdict=font)
axes.set_ylabel("Reflectance", fontsize=26, fontdict=font)
axes.set_xticks(np.arange(900, 1701, 200), fontsize=26)
axes.set_yticks(np.arange(0.1, 0.7, 0.1), fontsize=26)

labels = plt.legend(['pear2', 'pear1'], loc=3,fontsize=24).get_texts()
[label.set_fontname('Times New Roman') for label in labels]
axes.set_title("Pear1 and Pear2 Raw Spectral Reflectance", fontsize=28, fontdict=font)

plt.xlim()
plt.ylim()
plt.xticks(np.arange(900, 1800, 200), fontsize=26)
plt.yticks(fontsize=26)
x1_label = axes.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = axes.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

# 2.4 坐标轴刻度字体颜色设置
axes.tick_params(axis='both',
                 labelsize=26,  # y轴字体大小设置
                 direction='in',  # y轴标签方向设置,
                 )

# plt.savefig("1.tif", dpi=300)
plt.show()
