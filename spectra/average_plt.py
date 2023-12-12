import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.text as text

font=dict(family='times new roman',weight='bold')

data = pd.read_csv("./data/trainVal/R.csv",header=None)
data_x = data.iloc[0,:]

data1 = pd.read_csv("./data/trainVal/MC.csv").iloc[1:,:]
data2 = pd.read_csv("./data/Test/MC.csv").iloc[1:,:]

data1_0 = data1.iloc[0:73,:].mean()
data1_1 = data1.iloc[73:,:].mean()
data2_0 = data2.iloc[0:213,:].mean()
data2_1 = data2.iloc[213:468,:].mean()

# axes = plt.figure(figsize=(15,10)) # figsize=(15,8)
fig,axes =plt.subplots(figsize=(10,7),dpi=300)

# axes.plot(data_x,data_2.T,color="red",linewidth=2.0)
# axes.plot(data_x,data_1.T,color="orange",linewidth=2.0)
# axes.plot(data_x,data_0.T,color="green",linewidth=2.0)

axes.spines["top"].set_linewidth(1.8)
axes.spines["bottom"].set_linewidth(1.8)
axes.spines["left"].set_linewidth(1.8)
axes.spines["right"].set_linewidth(1.8)

axes.plot(data_x,data1_0.T,color="red")
axes.plot(data_x,data1_1.T,color="orange")
axes.plot(data_x,data2_0.T,color="blue")
axes.plot(data_x,data2_1.T,color="green")

axes.set_xlabel("WaveLength (nm)",fontsize=26,fontdict=font)
axes.set_ylabel("Reflectance",fontsize=26,fontdict=font)
# axes.set_xticks(np.arange(900,1701,200),fontsize=26,weight="bold")
# axes.set_yticks(np.arange(0.12,0.6,0.05),fontsize=26,weight="bold")
# labels = plt.legend(['Disease','Asymptomatic','Healthy'],fontsize=24).get_texts()

labels = plt.legend(['pear1 diease', 'pear1 health', 'pear2 diease','pear2 health'], loc=3,fontsize=24).get_texts()
# labels = plt.legend(['diease', 'health'], loc=3,fontsize=24).get_texts()

[label.set_fontname('Times New Roman') for label in labels]
axes.set_title("Pear1 and Pear2 Average Spectral Reflectance",fontsize=28,fontdict=font)


x1_label = axes.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = axes.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

  # 2.4 坐标轴刻度字体颜色设置
axes.tick_params(axis='both',
                 labelsize=26, # y轴字体大小设置
                 direction='in', # y轴标签方向设置,
                  )

plt.savefig("./resultImg/MC_Avg.tif",dpi=300)
# plt.show()