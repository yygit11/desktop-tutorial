import sys

from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold

# 训练集，验证集
data = pd.read_csv('../data/trainVal/MC.csv')
data_l = pd.read_csv('../data/trainVal/label.csv')
X_train = pd.concat([data.iloc[:50, :], data.iloc[72:123, :]]).values
X_val = pd.concat([data.iloc[50:72, :], data.iloc[123:, :]]).values
Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
Y_val = pd.concat([data_l.iloc[50:72, :], data_l.iloc[123:, :]]).values.reshape(-1)
clf = svm.SVC(C=100,gamma=5)
clf.fit(X_train, Y_train)

from sklearn.metrics import confusion_matrix, auc
from sklearn import metrics

print(metrics.classification_report(Y_train, clf.predict(X_train), digits=4))
print(metrics.accuracy_score(Y_train, clf.predict(X_train)))
print(confusion_matrix(Y_train, clf.predict(X_train)))
print(metrics.classification_report(Y_val, clf.predict(X_val), digits=4))
print(metrics.accuracy_score(Y_val, clf.predict(X_val)))
print(confusion_matrix(Y_val, clf.predict(X_val)))

X_test = pd.read_csv('../data/Test/MC.csv').values
Y_test = pd.read_csv('../data/Test/label.csv').values
print(metrics.classification_report(Y_test, clf.predict(X_test), digits=4))
print(metrics.accuracy_score(Y_test, clf.predict(X_test)))
print(confusion_matrix(Y_test, clf.predict(X_test)))
print(metrics.f1_score(Y_test, clf.predict(X_test)))

# y_score = clf.decision_function(X_test)
# fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_score)
# auc2 = auc(fpr, tpr)  # 计算auc面积，auc值越接近1模型性能越好
# # 绘制ROC曲线
# plt.title('ROC Curve')  # 标题
# plt.xlabel('FPR', fontsize=14)  # x轴标签
# plt.ylabel('TPR', fontsize=14)  # y轴标签
# plt.plot(fpr, tpr, label='AUC = %0.4f' % auc2, c='r')  # 划线
# plt.legend(fontsize=12)  # 图例
#
# plt.show()
