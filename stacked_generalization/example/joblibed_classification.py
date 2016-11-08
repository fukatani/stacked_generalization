import os
import sys

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_random_state
from stacked_generalization.lib.joblibed import JoblibedClassifier


iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# Joblibed model
rf = RandomForestClassifier(n_estimators=40,
                            criterion='gini',
                            random_state=1)
clf = JoblibedClassifier(rf, "rf")


train_idx, test_idx = list(StratifiedKFold(iris.target, 3))[0]

xs_train = iris.data[train_idx]
y_train = iris.target[train_idx]
xs_test = iris.data[test_idx]
y_test = iris.target[test_idx]


print("First fit and prediction (not cached).")
clf.fit(xs_train, y_train, train_idx)
score = clf.score(xs_test, y_test, test_idx)
print('Classfier score: {0}'.format(score))

print("Second fit and prediction (load cache).")
clf.fit(xs_train, y_train, train_idx)
score = clf.score(xs_test, y_test, test_idx)
print('Classfier score: {0}'.format(score))
