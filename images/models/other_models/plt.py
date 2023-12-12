import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data0 = pd.read_csv('NAS/NAS_loss.csv')
data1 = pd.read_csv('MLP/MLP_loss.csv')
data2 = pd.read_csv('Attention/attention_loss.csv')
data3 = pd.read_csv('1d-CNN/1d-cnn_loss.csv')
y = data0.iloc[:,-1].values
x = data1.iloc[:,-2].values
y1 = data1.iloc[:,-1].values
y2 = data2.iloc[:,-1].values
y3 = data3.iloc[:,-1].values
plt.figure(figsize=(4.2, 3.1), dpi=300)
plt.xlabel('epochs', fontsize=10)
plt.ylabel('loss', fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.tight_layout(pad=0.3)
plt.plot(x, y,linewidth=1.5,label='Model(NAS)')
plt.plot(x, y1,linewidth=1.5,label='MLP')
plt.plot(x, y2,linewidth=1.5,label='SAFCN')
plt.plot(x, y3,linewidth=1.5,label='1DCNN')
plt.legend()
plt.savefig("accuracy.png", dpi=300)
plt.show()