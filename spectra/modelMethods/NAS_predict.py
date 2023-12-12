import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

#训练集和验证集
data = pd.read_csv('../data/trainVal/MC+SNV.csv')
data_l = pd.read_csv('../data/trainVal/label.csv')
X_train = pd.concat([data.iloc[:50, :], data.iloc[72:123, :]]).values
X_val = pd.concat([data.iloc[50:72, :], data.iloc[123:, :]]).values
Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
Y_val = pd.concat([data_l.iloc[50:72, :], data_l.iloc[123:, :]]).values.reshape(-1)
#测试集
X_test = pd.read_csv('../data/Test/MC+SNV.csv').values
Y_test = pd.read_csv('../data/Test/label.csv').values
clf = load_model("./NAS/models/MC+SNV.h5")
clf.summary()
predict_train = clf.predict(X_train)
for i in range(len(predict_train)):
    if predict_train[i][0] < 0.5:
        predict_train[i][0] = 0
    else:
        predict_train[i][0] = 1
predict_val = clf.predict(X_val)
for i in range(len(predict_val)):
    if predict_val[i][0] < 0.5:
        predict_val[i][0] = 0
    else:
        predict_val[i][0] = 1
predict_test = clf.predict(X_test)
for i in range(len(predict_test)):
    if predict_test[i][0] < 0.5:
        predict_test[i][0] = 0
    else:
        predict_test[i][0] = 1
train_score = metrics.accuracy_score(Y_train, predict_train)
train_F1 = metrics.f1_score(Y_train, predict_train)
val_score = metrics.accuracy_score(Y_val, predict_val)
val_F1 = metrics.f1_score(Y_val, predict_val)
test_score = metrics.accuracy_score(Y_test, predict_test)
test_F1 = metrics.f1_score(Y_test, predict_test)
print(metrics.classification_report(Y_val, predict_val, digits=4))
print(metrics.classification_report(Y_test, predict_test, digits=4))
b = []
b.append(train_F1)
b.append(train_score)
b.append(val_F1)
b.append(val_score)
b.append(test_F1)
b.append(test_score)
print(b)

