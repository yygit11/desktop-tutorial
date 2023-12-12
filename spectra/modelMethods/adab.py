from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
#训练集，验证集
data = pd.read_csv('../data/trainVal/R.csv')
data_l = pd.read_csv('../data/trainVal/label.csv')
X_train = pd.concat([data.iloc[:50, :], data.iloc[72:123, :]]).values
X_val = pd.concat([data.iloc[50:72, :], data.iloc[123:, :]]).values
Y_train = pd.concat([data_l.iloc[:50, :], data_l.iloc[72:123, :]]).values.reshape(-1)
Y_val = pd.concat([data_l.iloc[50:72, :], data_l.iloc[123:, :]]).values.reshape(-1)
# tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=15)
clf = AdaBoostClassifier(n_estimators=50,learning_rate=0.1,random_state=0)
clf.fit(X_train, Y_train)

from sklearn.metrics import confusion_matrix
from sklearn import metrics
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