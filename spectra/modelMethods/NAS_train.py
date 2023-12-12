import sys

import pandas as pd
import tensorflow as tf
import autokeras as ak
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/trainVal/MC+SNV.csv')
data_l = pd.read_csv('../data/trainVal/label.csv')
X_train = pd.concat([data.iloc[:50, :], data.iloc[72:123, :]]).values
X_val = pd.concat([data.iloc[50:72, :], data.iloc[123:, :]]).values
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
    callbacks=[TensorBoard(log_dir='./NAS/tmp/MC+SNV')]
)
model = clf.export_model()
model.summary()
model.save("./NAS/models/MC+SNV.h5")
