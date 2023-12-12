"""
均值中心化
"""
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../data/Test/R.csv")
y = data.columns
data_y = np.array(data.iloc[0:,:])

def MeanCentering(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
        :return: data after MeanScaler :(n_samples, n_features)
    """
    for i in range(data.shape[0]):
        Mean = np.mean(data[i])
        data[i] = data[i] - Mean
    return data

after_data = MeanCentering(data_y)

pd.DataFrame(after_data).to_csv("../data/Test/MC.csv", header=y, index=0)

# plt.figure()
# plt.plot(data_x,after_data.T)
# plt.xlabel("WaveLength:(nm)")
# plt.ylabel("Reflectance")
# plt.title("Mean_centering")
# # plt.savefig('../Pre_Image/Mean_centering',dpi=300)
# plt.show()