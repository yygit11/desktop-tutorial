import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

X_train = torch.load("../../data/train_data.pth").numpy()
Y_train = torch.load("../../data/train_label.pth").numpy()
X_val = torch.load("../../data/val_data.pth").numpy()
Y_val = torch.load("../../data/val_label.pth").numpy()
X_test = torch.load("../../data/test_data.pth").numpy()
Y_test = torch.load("../../data/test_label.pth").numpy()
clf = load_model("./NAS/models/Image.h5")
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

