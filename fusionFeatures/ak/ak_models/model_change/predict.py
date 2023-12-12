import sys
import numpy as np
import pandas as pd
import autokeras as ak
import torch
from sklearn import metrics
from keras.models import load_model
from keras.models import Model

test_acc_array = np.array([])
a = []

# 训练集和验证集
gp_data = pd.read_csv("../../../gp_data/FD.csv")
gp_data_train = pd.concat([gp_data.iloc[:50, :], gp_data.iloc[72:123, :]]).reset_index(drop=True)
gp_data_val = pd.concat([gp_data.iloc[50:72, :], gp_data.iloc[123:, :]]).reset_index(drop=True)
tx_data_train = pd.read_csv('../../../densenet121/data_feature/train_data0.csv')
tx_data_val = pd.read_csv('../../../densenet121/data_feature/val_data0.csv')
X_train = pd.concat([gp_data_train, tx_data_train], axis=1).values
X_val = pd.concat([gp_data_val, tx_data_val], axis=1).values
data_l = pd.read_csv("../../../gp_data/label.csv")
Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
Y_val = pd.concat([data_l.iloc[50:72, :], data_l.iloc[123:, :]]).values.reshape(-1)
# 测试集
gp_data_test = pd.read_csv("../../../gp_data2/FD.csv")
tx_data_test = pd.read_csv('../../../densenet121/data_feature/test_data0.csv')
X_test = pd.concat([gp_data_test, tx_data_test], axis=1).values
Y_test = pd.read_csv("../../../gp_data2/label.csv")


clf = load_model('./test.h5')
clf.summary()
predict_train = clf.predict(X_val)
# predict_val = clf.predict(X_val)
# predict_test = clf.predict(X_test)
for i in range(len(predict_train)):
    if predict_train[i][0] < 0.5:
        predict_train[i][0] = 0
    else:
        predict_train[i][0] = 1
score = metrics.accuracy_score(Y_val, predict_train)
# test_F1 = metrics.f1_score(Y_test, predict_test)
# test_acc_array = np.append(test_acc_array, test_score)
# print(score)
# sys.exit()

dense1_layer_model = Model(inputs=clf.input, outputs=clf.get_layer('classification_head_1').output)
dense1_output = dense1_layer_model.predict(x=X_train, batch_size=16)
# print("[get output by layers name]")
print(dense1_output)
print(dense1_output.shape)
# a = pd.DataFrame(dense1_output)
# print(torch.tensor(a.values))
# a.to_csv('data_train.csv', index=None)
