import sys

import numpy as np
import pandas as pd
import autokeras as ak
import torch
from keras.callbacks import TensorBoard

# 数据
a = []
b = []
for i in range(10):
    print(i)
    gp_data = pd.read_csv("../gp_data/FD.csv")
    gp_data_train = pd.concat([gp_data.iloc[:50, :], gp_data.iloc[72:123, :]]).reset_index(drop=True)
    gp_data_val = pd.concat([gp_data.iloc[50:72, :], gp_data.iloc[123:, :]]).reset_index(drop=True)
    tx_data_train = pd.read_csv('../densenet121/data_feature/train_data0.csv')
    tx_data_val = pd.read_csv('../densenet121/data_feature/val_data0.csv')
    X_train = pd.concat([gp_data_train, tx_data_train], axis=1).values
    X_val = pd.concat([gp_data_val, tx_data_val], axis=1).values
    data_l = pd.read_csv("../gp_data/label.csv")
    Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
    Y_val = pd.concat([data_l.iloc[50:72, :], data_l.iloc[123:, :]]).values.reshape(-1)

    # 模型
    X_train.resize(X_train.shape[0], 1, X_train.shape[1])
    X_val.resize(X_val.shape[0], 1, X_val.shape[1])
    input_node = ak.Input()
    output_node = ak.Normalization()(input_node)
    output_node = ak.ConvBlock(num_layers=2, num_blocks=1)(output_node)
    # output_node = ak.RNNBlock(num_layers=2)(output_node)
    # output_node = ak.ConvBlock(num_layers=1, num_blocks=1)(output_node)
    # output_node = ak.RNNBlock(num_layers=1)(output_node)
    output_node = ak.ClassificationHead()(output_node)

    clf = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=10)
    clf.fit(
        X_train,
        Y_train,
        epochs=100,
        validation_data=(X_val, Y_val),
        # callbacks=[TensorBoard(log_dir='tmp/gp_densenet121/3/FD' + str(i))]
    )
    model = clf.export_model()
    model.summary()
    model.save("./ak_models2/gp_densenet121/spcnn_model" + str(i) + ".h5")
