import sys

import pandas as pd
import tensorflow as tf
import autokeras as ak
from keras.callbacks import TensorBoard
# 数据
gp_data = pd.read_csv("../../gp_data/FD.csv")
gp_data_train = pd.concat([gp_data.iloc[:50, :], gp_data.iloc[72:123, :]]).reset_index(drop=True)
gp_data_val = pd.concat([gp_data.iloc[50:72, :], gp_data.iloc[123:, :]]).reset_index(drop=True)
tx_data_train = pd.read_csv('../../densenet121/data_feature/train_data0.csv')
tx_data_val = pd.read_csv('../../densenet121/data_feature/val_data0.csv')
X_train =pd.concat([gp_data_train,tx_data_train],axis=1).values
X_val =pd.concat([gp_data_val,tx_data_val],axis=1).values
data_l = pd.read_csv("../../gp_data/label.csv")
Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
Y_val = pd.concat([data_l.iloc[50:72, :], data_l.iloc[123:, :]]).values.reshape(-1)
# 模型
clf = ak.StructuredDataClassifier(
    overwrite=True, max_trials=10
)
clf.fit(
    X_train,
    Y_train,
    epochs=100,
    callbacks=[TensorBoard(log_dir='tmp')]
)
model = clf.export_model()
model.summary()
# model.save("FD.h5")
