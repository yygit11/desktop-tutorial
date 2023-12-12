import sys

import pandas as pd
import tensorflow as tf
import autokeras as ak
import torch
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

X_train = torch.load("../../data/train_data.pth").numpy()
Y_train = torch.load("../../data/train_label.pth").numpy()
X_val = torch.load("../../data/val_data.pth").numpy()
Y_val = torch.load("../../data/val_label.pth").numpy()
# 模型
clf = ak.ImageClassifier(
    overwrite=True, max_trials=10
)
clf.fit(
    X_train,
    Y_train,
    # validation_data=(X_val, Y_val),
    epochs=100,
    callbacks=[TensorBoard(log_dir='./NAS/tmp/Image')]
)
model = clf.export_model()
model.summary()
model.save("./NAS/models/Image.h5")
