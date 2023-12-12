import sys

import pandas as pd
import tensorflow as tf
import autokeras as ak
from keras.callbacks import TensorBoard
# 数据
a = []
b = []
for i in range(1):
    print(i)
    gp_data = pd.read_csv("../gp_data/FD.csv")
    gp_data_train = pd.concat([gp_data.iloc[:50, :], gp_data.iloc[72:123, :]]).reset_index(drop=True)
    tx_data_train = pd.read_csv('../densenet121/data_feature2/train_data0.csv')
    X_train =pd.concat([gp_data_train,tx_data_train],axis=1).values
    data_l = pd.read_csv("../gp_data/label.csv")
    Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
    # 模型
    clf = ak.StructuredDataClassifier(
        overwrite=True, max_trials=10
    )
    clf.fit(
        X_train,
        Y_train,
        epochs=100,
        callbacks=[TensorBoard(log_dir='tmp/gp_densenet121/2/FD' + str(i))]
    )
    # model = clf.export_model()
    # model.summary()
    # model.save("./ak_models/gp_densenet121/2/FD" + str(i) + ".h5")
