from sklearn import datasets
from sklearn.utils.validation import check_random_state
from stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.cross_validation import StratifiedKFold

iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# Stage 1 model
bclf = LogisticRegression(random_state=1)
# Stage 0 models
clfs = [RandomForestClassifier(n_estimators=40, criterion = 'gini', random_state=1),
        ExtraTreesClassifier(n_estimators=30, criterion = 'gini', random_state=3),
        GradientBoostingClassifier(n_estimators=25, random_state=1),
        RidgeClassifier(random_state=1),
        ]

sl = StackedClassifier(bclf,
                       clfs,
                       n_folds=3,
                       verbose=0,
                       stack_by_proba=True,
                       oob_score_flag=True,
                       )

# cross validation
cv_score = 0
n_folds = 3
for train_idx, test_idx in StratifiedKFold(iris.target, n_folds):
    xs_train = iris.data[train_idx]
    y_train = iris.target[train_idx]
    xs_test = iris.data[test_idx]
    y_test = iris.target[test_idx]

    sl.fit(xs_train, y_train)
    print('oob_score: {0}'.format(sl.oob_score_))
    cv_score += sl.score(xs_test, y_test)
cv_score /= n_folds

print('cv_score: {0}'.format(cv_score))
