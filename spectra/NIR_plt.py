import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.text as text

def PlotSpectrum(spec, columns):

    font = dict(family='times new roman', weight='bold')
    fig, axes = plt.subplots(figsize=(10, 7), dpi=300)
    axes.spines["top"].set_linewidth(1.8)
    axes.spines["bottom"].set_linewidth(1.8)
    axes.spines["left"].set_linewidth(1.8)
    axes.spines["right"].set_linewidth(1.8)
    axes.tick_params(axis='both',
                     labelsize=26,  # y轴字体大小设置
                     direction='in',  # y轴标签方向设置,
                     )

    for i in range(spec.shape[0]):
        plt.plot(columns, spec[i, :], linewidth=0.6)

    axes.set_xlabel("Wavelength (nm)", fontsize=26, fontdict=font)
    axes.set_ylabel("Reflectance", fontsize=26, fontdict=font)
    # axes.set_xticks(np.arange(900, 1701, 200), fontsize=26)
    # axes.set_yticks(np.arange(0.1, 0.7, 0.1), fontsize=26)

    plt.xlim()
    plt.ylim()
    plt.xticks(np.arange(900, 1800, 200), fontsize=26)
    plt.yticks(fontsize=22)

    return plt


data = pd.read_csv("./data/trainVal/SG+SNV.csv")
x = data.values
# x = np.mean(x,axis=0).reshape(1,228) #平均
y = data.columns.values.astype(float)
pp = PlotSpectrum(x, y)
pp.savefig("./resultImg/SG+SNV.tif", dpi=300)
# pp.show()