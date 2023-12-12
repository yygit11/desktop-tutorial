import sys

from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import KFold

# 训练
data = pd.read_csv('../data/trainVal/R.csv').values
data_l = pd.read_csv('../data/trainVal/label.csv').values
clf = svm.SVC()
param_grid = {'C': [10, 40, 70, 100, 130, 160, 190, 220, 250], 'gamma': [0.1, 0.5, 1, 5, 10]}
# clf = XGBClassifier()
# param_grid = {'max_depth': [5,10,15,20,25], 'gamma': [0.1, 0.5, 1, 5, 10]}
# clf = RandomForestClassifier()
# param_grid = {'n_estimators': [10, 70, 130, 190, 250], 'max_depth': [5, 10, 15, 20, 25]}
# clf = RandomForestClassifier()
# param_grid = {'n_estimators': [10, 70, 130, 190, 250], 'max_depth': [5, 10, 15, 20, 25]}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(data, data_l)
print(grid_search.score(data, data_l))
print(grid_search.best_params_, grid_search.best_score_)  # 最优参数
clf = grid_search.best_estimator_  # 最优模型
# scores = cross_val_score(clf,data,data_l,cv=5,scoring='accuracy')
# print(scores)
# print(scores.mean())
# sys.exit()
from sklearn.metrics import confusion_matrix, auc
from sklearn import metrics

data = pd.read_csv('../data/trainVal/R.csv')
data_l = pd.read_csv('../data/trainVal/label.csv')
X_train, X_val, Y_train, Y_val = train_test_split(data, data_l, train_size=0.8,random_state=1)

print(metrics.classification_report(Y_train, clf.predict(X_train), digits=4))
print(metrics.accuracy_score(Y_train, clf.predict(X_train)))
print(confusion_matrix(Y_train, clf.predict(X_train)))
print(metrics.classification_report(Y_val, clf.predict(X_val), digits=4))
print(metrics.accuracy_score(Y_val, clf.predict(X_val)))
print(confusion_matrix(Y_val, clf.predict(X_val)))

X_test = pd.read_csv('../data/Test/R.csv').values
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
