import sys
import numpy as np
import pandas as pd
import autokeras as ak
from sklearn import metrics
from keras.models import load_model
from keras.models import Model

train_acc_array = np.array([])
val_acc_array = np.array([])
test_acc_array = np.array([])
a = []
# 训练集和验证集
for i in range(0,2,1):
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
    # 测试集
    gp_data_test = pd.read_csv("../gp_data2/FD.csv")
    tx_data_test = pd.read_csv('../densenet121/data_feature/test_data0.csv')
    X_test = pd.concat([gp_data_test, tx_data_test], axis=1).values
    Y_test = pd.read_csv("../gp_data2/label.csv")
    # X_train.resize(X_train.shape[0],1,X_train.shape[1])
    # X_val.resize(X_val.shape[0],1,X_val.shape[1])
    # X_test.resize(X_test.shape[0], 1, X_test.shape[1])

    # clf = load_model('./ak_models/model_change/test.h5')
    # clf = load_model("./ak_models2/gp_densenet121/spcnn_model"+str(i)+".h5")
    # clf = load_model("./ak_models2/gp_densenet121/rnn_model" + str(i) + ".h5")
    clf = load_model("./ak_models2/gp_densenet121/dense/model.h5")
    # clf = load_model("./ak_models2/gp_densenet121/cnn+rnn/rnn_cnn_model22.h5")
    clf.summary()


    # config = model.get_config()
    # from keras import Sequential
    # clf = Sequential.from_config(config)
    # clf.load_weights("./ak_models/model_change/111.h5")

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
    train_acc_array = np.append(train_acc_array, train_score)
    val_score = metrics.accuracy_score(Y_val, predict_val)
    val_F1 = metrics.f1_score(Y_val, predict_val)
    val_acc_array = np.append(val_acc_array, val_score)
    test_score = metrics.accuracy_score(Y_test, predict_test)
    test_F1 = metrics.f1_score(Y_test, predict_test)
    test_acc_array = np.append(test_acc_array, test_score)
    b = []
    b.append(train_F1)
    b.append(train_score)
    b.append(val_F1)
    b.append(val_score)
    b.append(test_F1)
    b.append(test_score)
    a.append(b)
acc_pd = pd.DataFrame(a, columns=['train_F1', 'train_acc', 'val_F1', 'val_acc', 'test_F1', 'test_acc'])
print(acc_pd)
# acc_pd.to_csv('./test.csv')
# dense1_layer_model = Model(inputs=clf.input, outputs=clf.get_layer('normalization').output)
# dense1_output = dense1_layer_model.predict(x=X_train, batch_size=16)
#
# print("[get output by layers name]")
# print(dense1_output.shape)
# print(dense1_output[0])
